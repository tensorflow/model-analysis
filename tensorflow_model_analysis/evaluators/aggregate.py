# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Public API for performing evaluations using the EvalMetricsGraph."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

# Standard Imports
import apache_beam as beam
import numpy as np
from scipy import mean
from scipy.stats import sem
from scipy.stats import t

from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.eval_metrics_graph import eval_metrics_graph
from tensorflow_model_analysis.eval_saved_model import dofn
from tensorflow_model_analysis.evaluators import counter_util
from tensorflow_model_analysis.post_export_metrics import metric_keys
from tensorflow_model_analysis.slicer import slicer
from typing import Any, Dict, Generator, Iterable, List, Optional, Text, Tuple, Union

_SAMPLE_ID = '___SAMPLE_ID'


@beam.ptransform_fn
@beam.typehints.with_input_types(
    beam.typehints.Tuple[slicer.BeamSliceKeyType, slicer.BeamExtractsType])
@beam.typehints.with_output_types(
    beam.typehints.Tuple[slicer.BeamSliceKeyType, beam.typehints
                         .List[beam.typehints.Any]])
def ComputePerSliceMetrics(  # pylint: disable=invalid-name
    slice_result: beam.pvalue.PCollection,
    eval_shared_model: types.EvalSharedModel,
    desired_batch_size: Optional[int] = None,
    num_bootstrap_samples: Optional[int] = 1,
    random_seed: Optional[int] = None,
) -> beam.pvalue.PCollection:
  """PTransform for computing, aggregating and combining metrics.

  Args:
    slice_result: Incoming PCollection consisting of slice key and extracts.
    eval_shared_model: Shared model parameters for EvalSavedModel.
    desired_batch_size: Optional batch size for batching in Aggregate.
    num_bootstrap_samples: Number of replicas to use in calculating uncertainty
      using bootstrapping.  If 1 is provided (default), aggregate metrics will
      be calculated with no uncertainty. If num_bootstrap_samples is > 0,
      multiple samples of each slice will be calculated using the Poisson
      bootstrap method. To calculate standard errors, num_bootstrap_samples
      should be 20 or more in order to provide useful data. More is better, but
      you pay a performance cost.
    random_seed: Seed to use for testing, because nondeterministic tests stink.

  Returns:
    DoOutputsTuple. The tuple entries are
    PCollection of (slice key, metrics) and
    PCollection of (slice key, plot metrics).
  """
  # TODO(ckuhn): Remove this workaround per discussions in CL/227944001
  slice_result.element_type = beam.typehints.Any

  compute_with_sampling = False
  if not num_bootstrap_samples:
    num_bootstrap_samples = 1
  if num_bootstrap_samples < 1:
    raise ValueError(
        'num_bootstrap_samples should be > 0, got %d' % num_bootstrap_samples)

  if num_bootstrap_samples > 1:
    slice_result_sampled = slice_result | 'FanoutBootstrap' >> beam.ParDo(
        _FanoutBootstrapFn(num_bootstrap_samples))
    compute_with_sampling = True

  output_results = (
      slice_result
      | 'CombinePerSlice' >> beam.CombinePerKey(
          _AggregateCombineFn(
              eval_shared_model=eval_shared_model,
              desired_batch_size=desired_batch_size,
              compute_with_sampling=False))
      | 'InterpretOutput' >> beam.ParDo(
          _ExtractOutputDoFn(eval_shared_model=eval_shared_model)))
  if compute_with_sampling:
    output_results = (
        slice_result_sampled
        | 'CombinePerSliceWithSamples' >> beam.CombinePerKey(
            _AggregateCombineFn(
                eval_shared_model=eval_shared_model,
                desired_batch_size=desired_batch_size,
                compute_with_sampling=True,
                seed_for_testing=random_seed))
        | 'InterpretSampledOutput' >> beam.ParDo(
            _ExtractOutputDoFn(eval_shared_model=eval_shared_model))
        | beam.GroupByKey()
        | beam.ParDo(_MergeBootstrap(), beam.pvalue.AsIter(output_results)))
  # Separate metrics and plots.
  return (output_results
          | beam.ParDo(_SeparateMetricsAndPlotsFn()).with_outputs(
              _SeparateMetricsAndPlotsFn.OUTPUT_TAG_PLOTS,
              main=_SeparateMetricsAndPlotsFn.OUTPUT_TAG_METRICS))


@beam.typehints.with_input_types(
    beam.typehints.Tuple[slicer.BeamSliceKeyType, slicer.BeamExtractsType])
@beam.typehints.with_output_types(
    beam.typehints.Tuple[slicer.BeamSliceKeyType, slicer.BeamExtractsType])
class _FanoutBootstrapFn(beam.DoFn):
  """For each bootstrap sample you want, we fan out an additional slice."""

  def __init__(self, num_bootstrap_samples: int):
    self._num_bootstrap_samples = num_bootstrap_samples

  def process(
      self, element: Tuple[slicer.SliceKeyType, types.Extracts]
  ) -> Generator[Tuple[slicer.SliceKeyType, types.Extracts], None, None]:
    slice_key, value = element
    for i in range(0, self._num_bootstrap_samples):
      # Prepend the sample ID to the original slice key.
      slice_key_as_list = list(slice_key)
      slice_key_as_list.insert(0, (_SAMPLE_ID, i))
      augmented_slice_key = tuple(slice_key_as_list)
      # This fans out the pipeline, but because we are reducing in a
      # CombinePerKey, we shouldn't have to deal with a great increase in
      # network traffic.
      # TODO(b/120421778): Create benchmarks to better understand performance
      # implications and tradeoffs.
      yield (augmented_slice_key, value)


class _MergeBootstrap(beam.DoFn):
  """Merge the bootstrap values and fit a T-distribution to get confidence."""

  def process(
      self, element: Tuple[slicer.SliceKeyType, List[Dict[Text, Any]]],
      unsampled_results: List[Tuple[slicer.SliceKeyType, Dict[Text, Any]]]
  ) -> Generator[Tuple[slicer.SliceKeyType, Dict[Text, Any]], None, None]:
    """Merge the bootstrap values.

    Args:
      element: The element is the tuple that contains slice key and a list of
        the metrics dict. It's the output of the GroupByKey step. All the
        metrics that under the same slice key are generated by
        poisson-bootstrap.
      unsampled_results: The unsampled_results is passed in as a side input.
        It's a tuple that contains the slice key and the metrics dict from a run
        of the slice with no sampling (ie, all examples in the set are
        represented exactly once.) This should be identical to the values
        obtained without sampling.

    Yields:
      A tuple of slice key and the metrics dict which contains the unsampled
      value, as well as value with confidence interval data.

    Raises:
      ValueError if the key of metrics inside element does not equal to the
      key of metrics in unsampled_results.
    """
    slice_key, metrics = element
    # metrics should be a list of dicts, but the dataflow runner has a quirk
    # that requires specific casting.
    metrics = list(metrics)
    if len(metrics) == 1:
      yield slice_key, metrics[0]
      return

    # Group the same metrics into one list.
    metrics_dict = {}
    for metric in metrics:
      for metrics_name in metric:
        if metrics_name not in metrics_dict:
          metrics_dict[metrics_name] = []
        metrics_dict[metrics_name].append(metric[metrics_name])

    unsampled_metrics_dict = {}
    for unsampled_slice_key, unsampled_metrics in unsampled_results:
      # To find the corresponding unsampled metrics based on the slice_key of
      # input element. If not found, the ValueError will be raised below.
      if unsampled_slice_key == slice_key:
        unsampled_metrics_dict = unsampled_metrics
        break

    # The key set of the two metrics dicts must be identical.
    if set(metrics_dict.keys()) != set(unsampled_metrics_dict.keys()):
      raise ValueError('Keys of two metrics do not match: sampled_metrics: %s. '
                       'unsampled_metrics: %s' %
                       (metrics_dict.keys(), unsampled_metrics_dict.keys()))

    metrics_with_confidence = {}
    for metrics_name in metrics_dict:
      metrics_with_confidence[metrics_name] = _calculate_confidence_interval(
          metrics_dict[metrics_name], unsampled_metrics_dict[metrics_name])

    yield slice_key, metrics_with_confidence


def _calculate_confidence_interval(
    sampling_data_list: List[Union[int, float, np.ndarray]],
    unsampled_data: Union[int, float, np.ndarray]):
  """Calculate the confidence interval of the data.

  Args:
    sampling_data_list: A list of number or np.ndarray.
    unsampled_data: Individual number or np.ndarray. The format of the
      unsampled_data should match the format of the element inside
      sampling_data_list.

  Returns:
    Confidence Interval value stored inside
    types.ValueWithConfidenceInterval.
  """
  if isinstance(sampling_data_list[0], (np.ndarray, list)):
    merged_data = sampling_data_list[0][:]
    if isinstance(sampling_data_list[0], np.ndarray):
      merged_data = merged_data.astype(object)
    for index in range(len(merged_data)):
      merged_data[index] = _calculate_confidence_interval(
          [data[index] for data in sampling_data_list], unsampled_data[index])
    return merged_data
  else:
    # Data has to be numeric. That means throw out nan values.
    sampling_data_list = [
        data for data in sampling_data_list if not np.isnan(data)
    ]
    n_samples = len(sampling_data_list)
    if n_samples:
      confidence = 0.95
      sample_mean = mean(sampling_data_list)
      std_err = sem(sampling_data_list)
      t_stat = t.ppf((1 + confidence) / 2, n_samples - 1)
      upper_bound = sample_mean + t_stat * std_err
      lower_bound = sample_mean - t_stat * std_err
      # Set [mean, lower_bound, upper_bound] for each metric component.
      return types.ValueWithConfidenceInterval(sample_mean, lower_bound,
                                               upper_bound, unsampled_data)
    else:
      return types.ValueWithConfidenceInterval(
          float('nan'), float('nan'), float('nan'), float('nan'))


def _add_metric_variables(  # pylint: disable=invalid-name
    left: types.MetricVariablesType,
    right: types.MetricVariablesType) -> types.MetricVariablesType:
  """Returns left and right metric variables combined."""
  if left is not None and right is not None:
    if len(left) != len(right):
      raise ValueError('metric variables lengths should match, but got '
                       '%d and %d' % (len(left), len(right)))
    return [x + y for x, y in zip(left, right)]
  elif left is not None:
    return left
  else:
    return right


class _AggState(object):
  """Combine state for AggregateCombineFn.

  There are two parts to the state: the metric variables (the actual state),
  and a list of FeaturesPredictionsLabels. See _AggregateCombineFn for why
  we need this.
  """

  __slots__ = ['metric_variables', 'fpls']

  def __init__(self):
    self.metric_variables = None  # type: Optional[types.MetricVariablesType]
    self.fpls = []  # type: List[types.FeaturesPredictionsLabels]

  def copy_from(  # pylint: disable=invalid-name
      self, other: '_AggState') -> None:
    if other.metric_variables:
      self.metric_variables = other.metric_variables
    self.fpls = other.fpls

  def __iadd__(self, other: '_AggState') -> '_AggState':
    self.metric_variables = _add_metric_variables(self.metric_variables,
                                                  other.metric_variables)
    self.fpls.extend(other.fpls)
    return self

  def add_fpl(  # pylint: disable=invalid-name
      self, fpl: types.FeaturesPredictionsLabels) -> None:
    self.fpls.append(fpl)

  def add_metrics_variables(  # pylint: disable=invalid-name
      self, metric_variables: types.MetricVariablesType) -> None:
    self.metric_variables = _add_metric_variables(self.metric_variables,
                                                  metric_variables)


@beam.typehints.with_input_types(slicer.BeamExtractsType)
@beam.typehints.with_output_types(beam.typehints.List[beam.typehints.Any])
class _AggregateCombineFn(beam.CombineFn):
  """Aggregate combine function.

  This function really does three things:
    1. Batching of FeaturesPredictionsLabels.
    3. "Partial reduction" of these batches by sending this through the
       "intro metrics" step.
    3. The "normal" combining of MetricVariables.

  What we really want to do is conceptually the following:
  Predictions | GroupByKey() | KeyAwareBatchElements()
              | ParDo(IntroMetrics()) | CombineValues(CombineMetricVariables()).

  but there's no way to KeyAwareBatchElements in Beam, and no way to do partial
  reductions either. Hence, this CombineFn has to do the work of batching,
  partial reduction (intro metrics), and actual combining, all in one.

  We do this by accumulating FeaturesPredictionsLabels in the combine state
  until we accumulate a large enough batch, at which point we send them
  through the "intro metrics" step. When merging, we merge the metric variables
  and accumulate FeaturesPredictionsLabels accordingly. We do one final
  "intro metrics" and merge step before producing the final output value.

  See also:
  BEAM-3737: Key-aware batching function
  (https://issues.apache.org/jira/browse/BEAM-3737).
  """

  # This needs to be large enough to allow for efficient TF invocations during
  # batch flushing, but shouldn't be too large as it also acts as cap on the
  # maximum memory usage of the computation.
  _DEFAULT_DESIRED_BATCH_SIZE = 1000

  def __init__(self,
               eval_shared_model: types.EvalSharedModel,
               desired_batch_size: Optional[int] = None,
               compute_with_sampling: Optional[bool] = False,
               seed_for_testing: Optional[int] = None) -> None:
    self._eval_shared_model = eval_shared_model
    self._eval_metrics_graph = None  # type: eval_metrics_graph.EvalMetricsGraph
    self._seed_for_testing = seed_for_testing
    # We really want the batch size to be adaptive like it is in
    # beam.BatchElements(), but there isn't an easy way to make it so.
    # TODO(b/73789023): Figure out how to make this batch size dynamic.
    if desired_batch_size and desired_batch_size > 0:
      self._desired_batch_size = desired_batch_size
    else:
      self._desired_batch_size = self._DEFAULT_DESIRED_BATCH_SIZE

    self._compute_with_sampling = compute_with_sampling

    # Metrics.
    self._combine_batch_size = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE, 'combine_batch_size')
    self._model_load_seconds = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE, 'model_load_seconds')
    self._num_compacts = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, 'num_compacts')

  def _start_bundle(self) -> None:
    # There's no initialisation method for CombineFns.
    # See BEAM-3736: Add SetUp() and TearDown() for CombineFns.
    # Default to eval_saved_model dofn to preserve legacy assumption
    # of eval_saved_model.
    # TODO(ihchen): Update all callers and make this an error condition to not
    # have construct_fn specified.
    if self._eval_shared_model.construct_fn is None:
      construct_fn = dofn.make_construct_fn(
          eval_saved_model_path=self._eval_shared_model.model_path,
          add_metrics_callbacks=self._eval_shared_model.add_metrics_callbacks,
          include_default_metrics=True,
          additional_fetches=None)
      self._eval_metrics_graph = (
          self._eval_shared_model.shared_handle.acquire(
              construct_fn(self._model_load_seconds)))
    else:
      self._eval_metrics_graph = (
          self._eval_shared_model.shared_handle.acquire(
              self._eval_shared_model.construct_fn(self._model_load_seconds)))

  def _poissonify(
      self, accumulator: _AggState) -> List[types.FeaturesPredictionsLabels]:
    # pylint: disable=line-too-long
    """Creates a bootstrap resample of the data in an accumulator.

    Given a set of data, it will be represented in the resample set a number of
    times, that number of times is drawn from Poisson(1).
    See
    http://www.unofficialgoogledatascience.com/2015/08/an-introduction-to-poisson-bootstrap26.html
    for a detailed explanation of the technique.

    Args:
      accumulator: Accumulator containing FPLs from a sample

    Returns:
      A list of FPLs representing a bootstrap resample of the accumulator items.
    """
    if self._seed_for_testing:
      np.random.seed(self._seed_for_testing)

    result = []
    if accumulator.fpls:
      poisson_counts = np.random.poisson(1, len(accumulator.fpls))
      for i, fpl in enumerate(accumulator.fpls):
        result.extend([fpl] * poisson_counts[i])
    return result

  def _maybe_do_batch(self, accumulator: _AggState,
                      force: bool = False) -> None:
    """Maybe intro metrics and update accumulator in place.

    Checks if accumulator has enough FPLs for a batch, and if so, does the
    intro metrics for the batch and updates accumulator in place.

    Args:
      accumulator: Accumulator. Will be updated in place.
      force: Force intro metrics even if accumulator has less FPLs than the
        batch size.
    """

    if self._eval_metrics_graph is None:
      self._start_bundle()
    batch_size = len(accumulator.fpls)
    if force or batch_size >= self._desired_batch_size:
      if accumulator.fpls:
        self._combine_batch_size.update(batch_size)
        fpls_for_metrics = accumulator.fpls
        if self._compute_with_sampling:
          # If we are computing with multiple bootstrap replicates, use fpls
          # generated by the Poisson bootstrapping technique.
          fpls_for_metrics = self._poissonify(accumulator)
        if fpls_for_metrics:
          accumulator.add_metrics_variables(
              self._eval_metrics_graph.metrics_reset_update_get_list(
                  fpls_for_metrics))
        else:
          # Call to metrics_reset_update_get_list does a reset prior to the
          # metrics update, but does not handle empty updates. Explicitly
          # calling just reset here, to make the flow clear.
          self._eval_metrics_graph.reset_metric_variables()
        del accumulator.fpls[:]

  def create_accumulator(self) -> _AggState:
    return _AggState()

  def add_input(self, accumulator: _AggState,
                elem: types.Extracts) -> _AggState:
    accumulator.add_fpl(elem[constants.FEATURES_PREDICTIONS_LABELS_KEY])
    self._maybe_do_batch(accumulator)
    return accumulator

  def merge_accumulators(self, accumulators: Iterable[_AggState]) -> _AggState:
    result = self.create_accumulator()
    for acc in accumulators:
      result += acc
      # Compact within the loop to avoid accumulating too much data.
      #
      # During the "map" side of combining merging happens with memory limits
      # but on the "reduce" side it's across all bundles (for a given key).
      #
      # So we could potentially accumulate get num_bundles * batch_size
      # elements if we don't process the batches within the loop, which
      # could cause OOM errors (b/77293756).
      self._maybe_do_batch(result)
    return result

  def compact(self, accumulator: _AggState) -> _AggState:
    self._maybe_do_batch(accumulator, force=True)  # Guaranteed compaction.
    self._num_compacts.inc(1)
    return accumulator

  def extract_output(
      self, accumulator: _AggState) -> Optional[types.MetricVariablesType]:
    # It's possible that the accumulator has not been fully flushed, if it was
    # not produced by a call to compact (which is not guaranteed across all Beam
    # Runners), so we defensively flush it here again, before we extract data
    # from it, to ensure correctness.
    self._maybe_do_batch(accumulator, force=True)
    return accumulator.metric_variables


class _SeparateMetricsAndPlotsFn(beam.DoFn):
  """Separates metrics and plots into two separate PCollections."""
  OUTPUT_TAG_METRICS = 'tag_metrics'
  OUTPUT_TAG_PLOTS = 'tag_plots'

  def process(self,
              element: Tuple[slicer.SliceKeyType, types.MetricVariablesType]):
    (slice_key, results) = element
    slicing_metrics = {}
    plots = {}
    for k, v in results.items():  # pytype: disable=attribute-error
      if metric_keys.is_plot_key(k):
        plots[k] = v
      else:
        slicing_metrics[k] = v
    yield (slice_key, slicing_metrics)
    if plots:
      yield beam.pvalue.TaggedOutput(self.OUTPUT_TAG_PLOTS, (slice_key, plots))  # pytype: disable=bad-return-type


@beam.typehints.with_input_types(
    beam.typehints.Tuple[slicer.BeamSliceKeyType, beam.typehints
                         .List[beam.typehints.Any]])
# No typehint for output type, since it's a multi-output DoFn result that
# Beam doesn't support typehints for yet (BEAM-3280).
class _ExtractOutputDoFn(beam.DoFn):
  """A DoFn that extracts the metrics output."""

  def __init__(self, eval_shared_model: types.EvalSharedModel) -> None:
    self._eval_shared_model = eval_shared_model
    self._model_load_seconds = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE, 'model_load_seconds')
    # This keeps track of the number of times the poisson bootstrap encounters
    # an empty set of elements for a slice sample. Should be extremely rare in
    # practice, keeping this counter will help us understand if something is
    # misbehaving.
    self._num_bootstrap_empties = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, 'num_bootstrap_empties')

  def start_bundle(self) -> None:
    # There's no initialisation method for CombineFns.
    # See BEAM-3736: Add SetUp() and TearDown() for CombineFns.
    # Default to eval_saved_model dofn to preserve legacy assumption
    # of eval_saved_model.
    # TODO(ihchen): Update all callers and make this an error condition to not
    # have construct_fn specified.
    counter_util.update_beam_counters(
        self._eval_shared_model.add_metrics_callbacks)
    if self._eval_shared_model.construct_fn is None:
      construct_fn = dofn.make_construct_fn(
          eval_saved_model_path=self._eval_shared_model.model_path,
          add_metrics_callbacks=self._eval_shared_model.add_metrics_callbacks,
          include_default_metrics=True,
          additional_fetches=None)
      self._eval_saved_model = (
          self._eval_shared_model.shared_handle.acquire(
              construct_fn(self._model_load_seconds)))
    else:
      self._eval_saved_model = (
          self._eval_shared_model.shared_handle.acquire(
              self._eval_shared_model.construct_fn(self._model_load_seconds)))

  def process(
      self, element: Tuple[slicer.SliceKeyType, types.MetricVariablesType]
  ) -> Generator[Tuple[slicer.SliceKeyType, Dict[Text, Any]], None, None]:
    (slice_key, metric_variables) = element
    if metric_variables:
      self._eval_saved_model.set_metric_variables(metric_variables)
      result = self._eval_saved_model.get_metric_values()

      # If slice key contains uncertainty sample ID, remove it from the key.
      if len(slice_key) and _SAMPLE_ID in slice_key[0]:
        slice_key = slice_key[1:]
      yield (slice_key, result)
    else:
      # Increase a counter for empty bootstrap samples. When sampling is not
      # enabled, this should never be exected. The slice extractor/fanout only
      # emits slices that match examples, and if the slice matches examples, it
      # will never be empty.
      self._num_bootstrap_empties.inc(1)
