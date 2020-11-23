# Lint as: python3
# Copyright 2019 Google LLC
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
"""Metrics and plots evaluation."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import copy
import datetime
from typing import Any, Dict, Iterable, Iterator, List, NamedTuple, Optional, Set, Text, Tuple, Type, Union
import apache_beam as beam
import numpy as np

from tensorflow_model_analysis import config
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import model_util
from tensorflow_model_analysis import types
from tensorflow_model_analysis import util
from tensorflow_model_analysis.eval_saved_model import constants as eval_constants
from tensorflow_model_analysis.evaluators import counter_util
from tensorflow_model_analysis.evaluators import eval_saved_model_util
from tensorflow_model_analysis.evaluators import evaluator
from tensorflow_model_analysis.evaluators import jackknife
from tensorflow_model_analysis.evaluators import keras_util
from tensorflow_model_analysis.evaluators import metrics_validator
from tensorflow_model_analysis.evaluators import poisson_bootstrap
from tensorflow_model_analysis.extractors import slice_key_extractor
from tensorflow_model_analysis.metrics import metric_specs
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.slicer import slicer_lib as slicer
from google.protobuf import message
from tensorflow_metadata.proto.v0 import schema_pb2

_COMBINER_INPUTS_KEY = '_combiner_inputs'
_DEFAULT_COMBINER_INPUT_KEY = '_default_combiner_input'
_DEFAULT_NUM_JACKKNIFE_BUCKETS = 20
_DEFAULT_NUM_BOOTSTRAP_SAMPLES = 20

# A fanout of 8 is used here to reduce stragglers that occur during the merger
# of large datasets such as histogram buckets. This has little effect on the
# msec profiles, but can impact the wall time and memory usage. If experiencing
# significantly extended run times due to stragglers, try bumping this to a
# larger number.
# TODO(b/151283457): replace hard-coded value with dynamic estimate.
_COMBINE_PER_SLICE_KEY_HOT_KEY_FANOUT = 8


def MetricsPlotsAndValidationsEvaluator(  # pylint: disable=invalid-name
    eval_config: config.EvalConfig,
    eval_shared_model: Optional[types.MaybeMultipleEvalSharedModels] = None,
    metrics_key: Text = constants.METRICS_KEY,
    plots_key: Text = constants.PLOTS_KEY,
    attributions_key: Text = constants.ATTRIBUTIONS_KEY,
    run_after: Text = slice_key_extractor.SLICE_KEY_EXTRACTOR_STAGE_NAME,
    schema: Optional[schema_pb2.Schema] = None,
    random_seed_for_testing: Optional[int] = None) -> evaluator.Evaluator:
  """Creates an Evaluator for evaluating metrics and plots.

  Args:
    eval_config: Eval config.
    eval_shared_model: Optional shared model (single-model evaluation) or list
      of shared models (multi-model evaluation). Only required if there are
      metrics to be computed in-graph using the model.
    metrics_key: Name to use for metrics key in Evaluation output.
    plots_key: Name to use for plots key in Evaluation output.
    attributions_key: Name to use for attributions key in Evaluation output.
    run_after: Extractor to run after (None means before any extractors).
    schema: A schema to use for customizing metrics and plots.
    random_seed_for_testing: Seed to use for unit testing.

  Returns:
    Evaluator for evaluating metrics and plots. The output will be stored under
    'metrics' and 'plots' keys.
  """
  eval_shared_models = model_util.verify_and_update_eval_shared_models(
      eval_shared_model)
  if eval_shared_models:
    eval_shared_models = {m.model_name: m for m in eval_shared_models}

  # pylint: disable=no-value-for-parameter
  return evaluator.Evaluator(
      stage_name='EvaluateMetricsAndPlots',
      run_after=run_after,
      ptransform=_EvaluateMetricsPlotsAndValidations(
          eval_config=eval_config,
          eval_shared_models=eval_shared_models,
          metrics_key=metrics_key,
          plots_key=plots_key,
          attributions_key=attributions_key,
          schema=schema,
          random_seed_for_testing=random_seed_for_testing))


def _filter_and_separate_computations(
    computations: metric_types.MetricComputations
) -> Tuple[List[metric_types.MetricComputation],
           List[metric_types.DerivedMetricComputation]]:
  """Filters duplicate computations and separates non-derived and derived.

  All metrics are based on either direct computations using combiners or are
  based on the results of one or more other computations. This code separates
  the two types of computations so that only the combiner based computations are
  passed to the main combiner call and the remainder are processed after those
  combiners have run. Filtering is required because DerivedMetricComputations
  typically include copies of the MetricComputations that they depend on in
  order to avoid having to pre-construct and pass around all the dependencies at
  the time the metrics are constructed. Instead, each derived metric creates a
  version of the metric it depends on and then this code de-dups metrics that
  are identical so only one gets computed.

  Args:
    computations: Computations.

  Returns:
    Tuple of (metric computations, derived metric computations).
  """
  non_derived_computations = []
  derived_computations = []
  types_and_keys = {}
  for c in computations:
    if isinstance(c, metric_types.MetricComputation):
      cls = c.__class__.__name__
      keys = sorted(c.keys)
      if cls in types_and_keys:
        # TODO(mdreves): This assumes the user used unique names for all the
        # keys and classes. This could mask a bug where the same name is
        # accidently used for different metric configurations. Add support for
        # creating a dict config for the computations (similar to keras) and
        # then comparing the configs to ensure the classes are identical.
        if keys == types_and_keys[cls]:
          continue
      types_and_keys[cls] = keys
      non_derived_computations.append(c)
    elif isinstance(c, metric_types.DerivedMetricComputation):
      derived_computations.append(c)
    else:
      raise TypeError('Unsupported metric computation type: {}'.format(c))
  return non_derived_computations, derived_computations


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(types.Extracts)
def _GroupByQueryKey(  # pylint: disable=invalid-name
    extracts: beam.pvalue.PCollection,
    query_key: Text,
) -> beam.pvalue.PCollection:
  """PTransform for grouping extracts by a query key.

  Args:
    extracts: Incoming PCollection consisting of extracts.
    query_key: Query key to group extracts by. Must be a member of the dict of
      features stored under tfma.FEATURES_KEY.

  Returns:
    PCollection of lists of extracts where each list is associated with same
    query key.
  """
  missing_query_key_counter = beam.metrics.Metrics.counter(
      constants.METRICS_NAMESPACE, 'missing_query_key')

  def key_by_query_key(extracts: types.Extracts,
                       query_key: Text) -> Tuple[Text, types.Extracts]:
    """Extract the query key from the extract and key by that."""
    value = metric_util.to_scalar(
        util.get_by_keys(
            extracts, [constants.FEATURES_KEY, query_key], optional=True),
        tensor_name=query_key)
    if value is None:
      missing_query_key_counter.inc()
      return ('', extracts)
    return ('{}'.format(value), extracts)

  # pylint: disable=no-value-for-parameter
  return (extracts
          | 'KeyByQueryId' >> beam.Map(key_by_query_key, query_key)
          | 'GroupByKey' >> beam.CombinePerKey(beam.combiners.ToListCombineFn())
          | 'DropQueryId' >> beam.Map(lambda kv: kv[1])
          | 'MergeExtracts' >> beam.Map(util.merge_extracts))


class _PreprocessorDoFn(beam.DoFn):
  """Do function that computes initial state from extracts.

  The outputs for each preprocessor are stored under the key '_combiner_inputs'
  in the overall extracts returned by this process call. These outputs are
  stored as a list in same order as the computations were passed as input so
  that the combiner can later access them by index. For computations that use
  the default labels, predictions, and example weights as their combiner inputs,
  the list entries will contain None values. A '_default_combiner_inputs'
  extract will also exist (if needed) containing StandardMetricInputs.

  If a FeaturePreprocessor is used the outputs of the preprocessor will be
  combined with the default labels, predictions, and example weights and stored
  in the StandardMetricInputs features value under the _default_combiner_inputs
  key.

  If the incoming data is a list of extracts (i.e. a query_key was used), the
  output will be a single extract with the keys within the extract representing
  the list as processed by the preprocessor. For example, the _slice_key_types
  will be a merger of all unique _slice key_types across the extracts list
  and the _default_combiner_inputs will be a list of StandardMetricInputs (one
  for each example matching the query_key).
  """

  def __init__(self, computations: List[metric_types.MetricComputation]):
    self._computations = computations
    self._evaluate_num_instances = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, 'evaluate_num_instances')
    self._timer = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE, '_PreprocessorDoFn_seconds')

  def setup(self):
    for computation in self._computations:
      if computation.preprocessor is not None:
        computation.preprocessor.setup()

  def start_bundle(self):
    for computation in self._computations:
      if computation.preprocessor is not None:
        computation.preprocessor.start_bundle()

  def finish_bundle(self):
    for computation in self._computations:
      if computation.preprocessor is not None:
        computation.preprocessor.finish_bundle()

  def teardown(self):
    for computation in self._computations:
      if computation.preprocessor is not None:
        computation.preprocessor.teardown()

  def process(self, extracts: types.Extracts) -> Iterable[Any]:
    start_time = datetime.datetime.now()
    self._evaluate_num_instances.inc(1)

    use_default_combiner_input = None
    features = None
    combiner_inputs = []
    for computation in self._computations:
      if computation.preprocessor is None:
        combiner_inputs.append(None)
        use_default_combiner_input = True
      elif isinstance(computation.preprocessor,
                      metric_types.FeaturePreprocessor):
        if features is None:
          features = {}
        for v in computation.preprocessor.process(extracts):
          features.update(v)
        combiner_inputs.append(None)
        use_default_combiner_input = True
      else:
        combiner_inputs.append(next(computation.preprocessor.process(extracts)))

    output = {
        constants.SLICE_KEY_TYPES_KEY: extracts[constants.SLICE_KEY_TYPES_KEY],
        _COMBINER_INPUTS_KEY: combiner_inputs
    }
    if use_default_combiner_input:
      default_combiner_input = []
      if features is not None:
        extracts = copy.copy(extracts)
        extracts.update({constants.FEATURES_KEY: features})
      default_combiner_input = metric_util.to_standard_metric_inputs(
          extracts, include_features=features is not None)
      output[_DEFAULT_COMBINER_INPUT_KEY] = default_combiner_input
    yield output

    self._timer.update(
        int((datetime.datetime.now() - start_time).total_seconds()))


@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(metric_types.MetricsDict)
class _ComputationsCombineFn(beam.combiners.SingleInputTupleCombineFn):
  """Combine function that computes metric using initial state from extracts."""

  def __init__(self,
               computations: List[metric_types.MetricComputation],
               compute_with_sampling: Optional[bool] = False,
               random_seed_for_testing: Optional[int] = None):
    """Init.

    If compute_with_sampling is true a bootstrap resample of the data will be
    performed where each input will be represented in the resample one or more
    times as drawn from Poisson(1). This technically works with small or empty
    batches, but as the technique is an approximation the approximation gets
    better as the number of examples gets larger. If the results themselves are
    empty TFMA will reject the sample. For any samples of a reasonable size, the
    chances of this are exponentially tiny. See "The mathematical fine print"
    section of the blog post linked below.

    See:
    http://www.unofficialgoogledatascience.com/2015/08/an-introduction-to-poisson-bootstrap26.html

    Args:
      computations: List of MetricComputations.
      compute_with_sampling: True to compute with sampling.
      random_seed_for_testing: Seed to use for unit testing.
    """
    super(_ComputationsCombineFn,
          self).__init__(*[c.combiner for c in computations])
    self._compute_with_sampling = compute_with_sampling
    self._random_state = np.random.RandomState(random_seed_for_testing)
    self._num_compacts = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, 'num_compacts')
    # This keeps track of the number of times the poisson bootstrap encounters
    # an empty set of elements for a slice sample. Should be extremely rare in
    # practice, keeping this counter will help us understand if something is
    # misbehaving.
    self._num_bootstrap_empties = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, 'num_bootstrap_empties')

  def add_input(self, accumulator: Any, element: types.Extracts):
    elements = [element]
    if self._compute_with_sampling:
      elements = [element] * int(self._random_state.poisson(1, 1))
    if not elements:
      return accumulator

    def get_combiner_input(element, i):
      item = element[_COMBINER_INPUTS_KEY][i]
      if item is None:
        item = element[_DEFAULT_COMBINER_INPUT_KEY]
      return item

    results = []
    for i, (c, a) in enumerate(zip(self._combiners, accumulator)):
      result = c.add_input(a, get_combiner_input(elements[0], i))
      for e in elements[1:]:
        result = c.add_input(result, get_combiner_input(e, i))
      results.append(result)
    return tuple(results)

  def compact(self, accumulator: Any) -> Any:
    self._num_compacts.inc(1)
    return super(_ComputationsCombineFn, self).compact(accumulator)

  def extract_output(self, accumulator: Any) -> Tuple[metric_types.MetricsDict]:
    result = []
    for c, a in zip(self._combiners, accumulator):
      output = c.extract_output(a)
      if not output:
        # Increase a counter for empty bootstrap samples. When sampling is not
        # enabled, this should never be exected. This should only occur when the
        # slice sizes are incredibly small, and seeing large values of this
        # counter is a sign that something has gone wrong.
        self._num_bootstrap_empties.inc(1)
      result.append(output)
    return tuple(result)


@beam.ptransform_fn
def _AddCrossSliceMetrics(  # pylint: disable=invalid-name
    sliced_combiner_outputs: beam.pvalue.PCollection,
    cross_slice_specs: Optional[Iterable[config.CrossSlicingSpec]]
) -> Tuple[slicer.SliceKeyOrCrossSliceKeyType, metric_types.MetricsDict]:
  """Generates CrossSlice metrics from SingleSlices."""

  def is_slice_applicable(
      sliced_combiner_output: Tuple[slicer.SliceKeyType,
                                    metric_types.MetricsDict],
      slicing_specs: Union[config.SlicingSpec, Iterable[config.SlicingSpec]]
  ) -> bool:
    slice_key, _ = sliced_combiner_output
    for slicing_spec in slicing_specs:
      if slicer.SingleSliceSpec(
          spec=slicing_spec).is_slice_applicable(slice_key):
        return True
    return False

  def compute_cross_slices(
      baseline_slice: Tuple[slicer.SliceKeyType, metric_types.MetricsDict],
      comparison_slices: Iterable[Tuple[slicer.SliceKeyType,
                                        Dict[metric_types.MetricKey, Any]]]
  ) -> Iterator[Tuple[slicer.CrossSliceKeyType, Dict[metric_types.MetricKey,
                                                     Any]]]:
    baseline_slice_key, baseline_metrics = baseline_slice
    for (comparison_slice_key, comparison_metrics) in comparison_slices:
      result = {}
      for (comparison_metric_key,
           comparison_metric_value) in comparison_metrics.items():
        if comparison_metric_key not in baseline_metrics:
          continue
        result[comparison_metric_key] = (
            baseline_metrics[comparison_metric_key] - comparison_metric_value)
      yield ((baseline_slice_key, comparison_slice_key), result)

  cross_slice_outputs = []
  for cross_slice_ind, cross_slice_spec in enumerate(cross_slice_specs):
    baseline_slices = (
        sliced_combiner_outputs
        | 'FilterBaselineSlices(%d)' % cross_slice_ind >> beam.Filter(
            is_slice_applicable, [cross_slice_spec.baseline_spec]))

    slicing_specs = list(cross_slice_spec.slicing_specs)
    comparison_slices = (
        sliced_combiner_outputs
        | 'FilterComparisonSlices(%d)' % cross_slice_ind >> beam.Filter(
            is_slice_applicable, slicing_specs))

    cross_slice_outputs.append(
        baseline_slices
        | 'GenerateCrossSlices(%d)' % cross_slice_ind >> beam.FlatMap(
            compute_cross_slices,
            comparison_slices=beam.pvalue.AsIter(comparison_slices)))

  if cross_slice_outputs:
    cross_slice_outputs = (
        cross_slice_outputs
        | 'FlattenCrossSliceResults' >> beam.Flatten())
    return ([sliced_combiner_outputs, cross_slice_outputs]
            | 'CombineSingleSlicesWithCrossSlice' >> beam.Flatten())
  else:
    return sliced_combiner_outputs


@beam.ptransform_fn
@beam.typehints.with_input_types(Tuple[slicer.SliceKeyType, types.Extracts])
@beam.typehints.with_output_types(Tuple[slicer.SliceKeyType,
                                        Dict[metric_types.MetricKey, Any]])
def _ComputePerSlice(  # pylint: disable=invalid-name
    sliced_extracts: beam.pvalue.PCollection,
    computations: List[metric_types.MetricComputation],
    derived_computations: List[metric_types.DerivedMetricComputation],
    cross_slice_specs: Optional[Iterable[config.CrossSlicingSpec]] = None,
    compute_with_sampling: Optional[bool] = False,
    num_jackknife_samples: int = 0,
    skip_ci_metric_keys: Set[metric_types.MetricKey] = frozenset(),
    random_seed_for_testing: Optional[int] = None,
    baseline_model_name: Optional[Text] = None) -> beam.pvalue.PCollection:
  """PTransform for computing, aggregating and combining metrics and plots.

  Args:
    sliced_extracts: Incoming PCollection consisting of slice key and extracts.
    computations: List of MetricComputations.
    derived_computations: List of DerivedMetricComputations.
    cross_slice_specs: List of CrossSlicingSpec.
    compute_with_sampling: True to compute with bootstrap sampling. This allows
      _ComputePerSlice to be used to generate unsampled values from the whole
      data set, as well as bootstrap resamples, in which each element is treated
      as if it showed up p ~ poission(1) times.
    num_jackknife_samples: number of delete-d jackknife estimates to use in
      computing standard errors on metrics.
    skip_ci_metric_keys: List of metric keys for which to skip confidence
      interval computation.
    random_seed_for_testing: Seed to use for unit testing.
    baseline_model_name: Name for baseline model.

  Returns:
    PCollection of (slice key, dict of metrics).
  """
  # TODO(b/123516222): Remove this workaround per discussions in CL/227944001
  sliced_extracts.element_type = beam.typehints.Any

  def convert_and_add_derived_values(
      sliced_results: Tuple[slicer.SliceKeyType, Tuple[metric_types.MetricsDict,
                                                       ...]],
      derived_computations: List[metric_types.DerivedMetricComputation],
  ) -> Tuple[slicer.SliceKeyType, metric_types.MetricsDict]:
    """Converts per slice tuple of dicts into single dict and adds derived."""
    result = {}
    for v in sliced_results[1]:
      result.update(v)
    for c in derived_computations:
      result.update(c.result(result))
    # Remove private metrics
    keys = list(result.keys())
    for k in keys:
      if k.name.startswith('_') and not k.name.startswith('__'):
        result.pop(k)
    return sliced_results[0], result

  def add_diff_metrics(
      sliced_metrics: Tuple[Union[slicer.SliceKeyType,
                                  slicer.CrossSliceKeyType],
                            Dict[metric_types.MetricKey, Any]],
      baseline_model_name: Optional[Text],
  ) -> Tuple[slicer.SliceKeyType, Dict[metric_types.MetricKey, Any]]:
    """Add diff metrics if there is a baseline model."""

    result = copy.copy(sliced_metrics[1])

    if baseline_model_name:
      diff_result = {}
      for k, v in result.items():
        if k.model_name != baseline_model_name and k.make_baseline_key(
            baseline_model_name) in result:
          # plots will not be diffed.
          if not isinstance(v, message.Message):
            diff_result[k.make_diff_key(
            )] = v - result[k.make_baseline_key(baseline_model_name)]
      result.update(diff_result)

    return (sliced_metrics[0], result)

  combiner = _ComputationsCombineFn(
      computations=computations,
      compute_with_sampling=compute_with_sampling,
      random_seed_for_testing=random_seed_for_testing)
  if num_jackknife_samples:
    # We do not use the hotkey fanout hint used by the non-jacknife path because
    # the random jackknife partitioning naturally mitigates hot keys.
    sliced_combiner_outputs = (
        sliced_extracts
        | 'JackknifeCombinePerSliceKey' >> jackknife.JackknifeCombinePerKey(
            combiner, num_jackknife_samples))
  else:
    sliced_combiner_outputs = (
        sliced_extracts
        | 'CombinePerSliceKey' >> beam.CombinePerKey(combiner)
        .with_hot_key_fanout(_COMBINE_PER_SLICE_KEY_HOT_KEY_FANOUT))

  sliced_derived_values_and_diffs = (
      sliced_combiner_outputs
      | 'ConvertAndAddDerivedValues' >> beam.Map(convert_and_add_derived_values,
                                                 derived_computations)
      | 'AddCrossSliceMetrics' >> _AddCrossSliceMetrics(cross_slice_specs)  # pylint: disable=no-value-for-parameter
      | 'AddDiffMetrics' >> beam.Map(add_diff_metrics, baseline_model_name))

  if num_jackknife_samples:
    return (sliced_derived_values_and_diffs
            | 'MergeJackknifeSamples' >> jackknife.MergeJackknifeSamples(
                num_jackknife_samples, skip_ci_metric_keys))
  else:
    return sliced_derived_values_and_diffs


def _filter_by_key_type(
    sliced_metrics_plots_attributions: Tuple[slicer.SliceKeyType,
                                             Dict[metric_types.MetricKey, Any]],
    key_type: Type[Union[metric_types.MetricKey, metric_types.PlotKey,
                         metric_types.AttributionsKey]]
) -> Tuple[slicer.SliceKeyType, Dict[metric_types.MetricKey, Any]]:
  """Filters metrics and plots by key type."""
  slice_value, metrics_plots_attributions = sliced_metrics_plots_attributions
  output = {}
  for k, v in metrics_plots_attributions.items():
    # PlotKey is a subclass of MetricKey so must check key_type based on PlotKey
    if key_type == metric_types.PlotKey:
      if isinstance(k, metric_types.PlotKey):
        output[k] = v
    # AttributionsKey is a also subclass of MetricKey
    elif key_type == metric_types.AttributionsKey:
      if isinstance(k, metric_types.AttributionsKey):
        output[k] = v
    else:
      if (not isinstance(k, metric_types.PlotKey) and
          not isinstance(k, metric_types.AttributionsKey)):
        output[k] = v
  return (slice_value, output)


_ConfidenceIntervalParams = NamedTuple(
    '_ConfidenceIntervalParams',
    [('num_jackknife_samples', int), ('num_bootstrap_samples', int),
     ('skip_ci_metric_keys', Iterable[metric_types.MetricKey])])


def _get_confidence_interval_params(
    eval_config: config.EvalConfig,
    metrics_specs: Iterable[config.MetricsSpec]) -> _ConfidenceIntervalParams:
  """Helper method for extracting confidence interval info from configs.

  Args:
    eval_config: The eval_config.
    metrics_specs: The metrics_specs containing either all metrics, or the ones
      which share a query key.

  Returns:
    A _ConfidenceIntervalParams object containing the number of jacknife samples
    to use for computing a jackknife confidence interval, the number of
    bootstrap samples to use for computing Poisson bootstrap confidence
    intervals, and the set of metric keys which should not have confidence
    intervals displayed in the output.
  """
  skip_ci_metric_keys = (
      metric_specs.metric_keys_to_skip_for_confidence_intervals(metrics_specs))
  num_jackknife_samples = 0
  num_bootstrap_samples = 1
  ci_method = eval_config.options.confidence_intervals.method
  if eval_config.options.compute_confidence_intervals.value:
    if ci_method == config.ConfidenceIntervalOptions.JACKKNIFE:
      num_jackknife_samples = _DEFAULT_NUM_JACKKNIFE_BUCKETS
    elif ci_method == config.ConfidenceIntervalOptions.POISSON_BOOTSTRAP:
      num_bootstrap_samples = _DEFAULT_NUM_BOOTSTRAP_SAMPLES
  return _ConfidenceIntervalParams(num_jackknife_samples, num_bootstrap_samples,
                                   skip_ci_metric_keys)


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(Any)
def _ComputeMetricsAndPlots(  # pylint: disable=invalid-name
    extracts: beam.pvalue.PCollection,
    eval_config: config.EvalConfig,
    metrics_specs: List[config.MetricsSpec],
    eval_shared_models: Optional[Dict[Text, types.EvalSharedModel]] = None,
    metrics_key: Text = constants.METRICS_KEY,
    plots_key: Text = constants.PLOTS_KEY,
    attributions_key: Text = constants.ATTRIBUTIONS_KEY,
    schema: Optional[schema_pb2.Schema] = None,
    random_seed_for_testing: Optional[int] = None) -> evaluator.Evaluation:
  """Computes metrics and plots.

  Args:
    extracts: PCollection of Extracts. If a query_key was used then the
      PCollection will contain a list of extracts.
    eval_config: Eval config.
    metrics_specs: Subset of the metric specs to compute metrics for. If a
      query_key was used all of the metric specs will be for the same query_key.
    eval_shared_models: Optional dict of shared models keyed by model name. Only
      required if there are metrics to be computed in-graph using the model.
    metrics_key: Name to use for metrics key in Evaluation output.
    plots_key: Name to use for plots key in Evaluation output.
    attributions_key: Name to use for attributions key in Evaluation output.
    schema: A schema to use for customizing metrics and plots.
    random_seed_for_testing: Seed to use for unit testing.

  Returns:
    Evaluation containing dict of PCollections of (slice_key, results_dict)
    tuples where the dict is keyed by either the metrics_key (e.g. 'metrics'),
    plots_key (e.g. 'plots'), or attributions_key (e.g. 'attributions')
    depending on what the results_dict contains.
  """
  computations = []
  # Add default metric computations
  if eval_shared_models:
    for model_name, eval_shared_model in eval_shared_models.items():
      if not eval_shared_model.include_default_metrics:
        continue
      if eval_shared_model.model_type == constants.TF_KERAS:
        keras_specs = keras_util.metrics_specs_from_keras(
            model_name, eval_shared_model.model_loader)
        metrics_specs = keras_specs + metrics_specs[:]
        # TODO(mdreves): Add support for calling keras.evaluate().
      elif (eval_shared_model.model_type == constants.TF_ESTIMATOR and
            eval_constants.EVAL_TAG in eval_shared_model.model_loader.tags):
        # Note that there is the possibility for metric naming collisions here
        # (e.g. 'auc' calculated within the EvalSavedModel as well as by AUC
        # metric computation performed outside the model). Currently all the
        # overlapping metrics such as AUC that are computed outside the model
        # are all derived metrics so they will override the metrics calculated
        # by the model which is the desired behavior.
        computations.extend(
            eval_saved_model_util.metric_computations_using_eval_saved_model(
                model_name, eval_shared_model.model_loader))
  # Add metric computations from specs
  computations_from_specs, derived_computations = (
      _filter_and_separate_computations(
          metric_specs.to_computations(
              metrics_specs, eval_config=eval_config, schema=schema)))
  computations.extend(computations_from_specs)

  # Find out which model is baseline.
  baseline_spec = model_util.get_baseline_model_spec(eval_config)
  baseline_model_name = baseline_spec.name if baseline_spec else None

  # pylint: disable=no-value-for-parameter

  # Input: Single extract per example (or list of extracts if query_key used)
  #        where each item contains slice keys and other extracts from upstream
  #        extractors (e.g. labels, predictions, etc).
  # Output: Single extract (per example) containing slice keys and initial
  #         combiner state returned from preprocessor. Note that even if a
  #         query_key was used the output is still only a single extract
  #         (though, that extract may contain lists of values (predictions,
  #         labels, etc) in its keys).
  #
  # Note that the output of this step is extracts instead of just a tuple of
  # computation outputs because FanoutSlices takes extracts as input (and in
  # many cases a subset of the extracts themselves are what is fanned out).
  extracts = (
      extracts
      | 'Preprocesss' >> beam.ParDo(_PreprocessorDoFn(computations)))

  # Input: Single extract containing slice keys and initial combiner inputs. If
  #        query_key is used the extract represents multiple examples with the
  #        same query_key, otherwise the extract represents a single example.
  # Output: Tuple (slice key, combiner inputs extracts). Notice that the per
  #         example (or list or examples if query_key used) input extract turns
  #         into n logical extracts, references to which are replicated once per
  #         applicable slice key.
  slices = extracts | 'FanoutSlices' >> slicer.FanoutSlices()

  slices_count = (
      slices
      | 'ExtractSliceKeys' >> beam.Keys()
      | 'CountPerSliceKey' >> beam.combiners.Count.PerElement())

  _ = (
      extracts.pipeline
      | 'IncrementMetricsSpecsCounters' >>
      counter_util.IncrementMetricsSpecsCounters(metrics_specs), slices_count
      |
      'IncrementSliceSpecCounters' >> counter_util.IncrementSliceSpecCounters())

  ci_params = _get_confidence_interval_params(eval_config, metrics_specs)

  cross_slice_specs = []
  if eval_config.cross_slicing_specs:
    cross_slice_specs = eval_config.cross_slicing_specs

  # TODO(b/151482616): Make bootstrap and jackknife confidence interval
  # implementations more parallel.

  # Input: Tuple of (slice key, combiner input extracts).
  # Output: Tuple of (slice key, dict of computed metrics/plots/attributions).
  #         The dicts will be keyed by MetricKey/PlotKey/AttributionsKey and the
  #         values will be the result of the associated computations. A given
  #         MetricComputation can perform computations for multiple keys, but
  #         the keys should be unique across computations.
  sliced_metrics_plots_and_attributions = (
      slices
      | 'ComputePerSlice' >> poisson_bootstrap.ComputeWithConfidenceIntervals(
          _ComputePerSlice,
          computations=computations,
          derived_computations=derived_computations,
          baseline_model_name=baseline_model_name,
          cross_slice_specs=cross_slice_specs,
          num_jackknife_samples=ci_params.num_jackknife_samples,
          num_bootstrap_samples=ci_params.num_bootstrap_samples,
          skip_ci_metric_keys=ci_params.skip_ci_metric_keys,
          random_seed_for_testing=random_seed_for_testing))

  if eval_config.options.min_slice_size.value > 1:
    sliced_metrics_plots_and_attributions = (
        sliced_metrics_plots_and_attributions
        | 'FilterSmallSlices' >> slicer.FilterOutSlices(
            slices_count, eval_config.options.min_slice_size.value))

  sliced_metrics = (
      sliced_metrics_plots_and_attributions
      | 'FilterByMetrics' >> beam.Map(_filter_by_key_type,
                                      metric_types.MetricKey))
  sliced_plots = (
      sliced_metrics_plots_and_attributions
      | 'FilterByPlots' >> beam.Map(_filter_by_key_type, metric_types.PlotKey))

  sliced_attributions = (
      sliced_metrics_plots_and_attributions
      | 'FilterByAttributions' >> beam.Map(_filter_by_key_type,
                                           metric_types.AttributionsKey))

  # pylint: enable=no-value-for-parameter

  return {
      metrics_key: sliced_metrics,
      plots_key: sliced_plots,
      attributions_key: sliced_attributions
  }


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(Any)
def _EvaluateMetricsPlotsAndValidations(  # pylint: disable=invalid-name
    extracts: beam.pvalue.PCollection,
    eval_config: config.EvalConfig,
    eval_shared_models: Optional[Dict[Text, types.EvalSharedModel]] = None,
    metrics_key: Text = constants.METRICS_KEY,
    plots_key: Text = constants.PLOTS_KEY,
    attributions_key: Text = constants.ATTRIBUTIONS_KEY,
    validations_key: Text = constants.VALIDATIONS_KEY,
    schema: Optional[schema_pb2.Schema] = None,
    random_seed_for_testing: Optional[int] = None) -> evaluator.Evaluation:
  """Evaluates metrics, plots, and validations.

  Args:
    extracts: PCollection of Extracts. The extracts must contain a list of
      slices of type SliceKeyType keyed by tfma.SLICE_KEY_TYPES_KEY as well as
      any extracts required by the metric implementations (typically this will
      include labels keyed by tfma.LABELS_KEY, predictions keyed by
      tfma.PREDICTIONS_KEY, and example weights keyed by
      tfma.EXAMPLE_WEIGHTS_KEY). Usually these will be added by calling the
      default_extractors function.
    eval_config: Eval config.
    eval_shared_models: Optional dict of shared models keyed by model name. Only
      required if there are metrics to be computed in-graph using the model.
    metrics_key: Name to use for metrics key in Evaluation output.
    plots_key: Name to use for plots key in Evaluation output.
    attributions_key: Name to use for attributions key in Evaluation output.
    validations_key: Name to use for validation key in Evaluation output.
    schema: A schema to use for customizing metrics and plots.
    random_seed_for_testing: Seed to use for unit testing.

  Returns:
    Evaluation containing dict of PCollections of (slice_key, results_dict)
    tuples where the dict is keyed by either the metrics_key (e.g. 'metrics'),
    plots_key (e.g. 'plots'), attributions_key (e.g. 'attributions'), or
    validation_key (e.g. 'validations') depending on what the results_dict
    contains.
  """
  # Separate metrics based on query_key (which may be None).
  metrics_specs_by_query_key = {}
  for spec in eval_config.metrics_specs:
    if spec.query_key not in metrics_specs_by_query_key:
      metrics_specs_by_query_key[spec.query_key] = []
    metrics_specs_by_query_key[spec.query_key].append(spec)

  # If there are no metrics specs then add an empty one (this is required for
  # cases where only the default metrics from the model are used).
  if not metrics_specs_by_query_key:
    metrics_specs_by_query_key[''] = [config.MetricsSpec()]

  # pylint: disable=no-value-for-parameter

  evaluations = {}
  for query_key, metrics_specs in metrics_specs_by_query_key.items():
    query_key_text = query_key if query_key else ''
    if query_key:
      extracts_for_evaluation = (
          extracts
          | 'GroupByQueryKey({})'.format(query_key_text) >>
          _GroupByQueryKey(query_key))
      include_default_metrics = False
    else:
      extracts_for_evaluation = extracts
      include_default_metrics = (
          eval_config and
          (not eval_config.options.HasField('include_default_metrics') or
           eval_config.options.include_default_metrics.value))
    evaluation = (
        extracts_for_evaluation
        | 'ComputeMetricsAndPlots({})'.format(query_key_text) >>
        _ComputeMetricsAndPlots(
            eval_config=eval_config,
            metrics_specs=metrics_specs,
            eval_shared_models=(eval_shared_models
                                if include_default_metrics else None),
            metrics_key=metrics_key,
            plots_key=plots_key,
            attributions_key=attributions_key,
            schema=schema,
            random_seed_for_testing=random_seed_for_testing))

    for k, v in evaluation.items():
      if k not in evaluations:
        evaluations[k] = []
      evaluations[k].append(v)
  evaluation_results = evaluator.combine_dict_based_evaluations(evaluations)

  validations = (
      evaluation_results[metrics_key]
      | 'ValidateMetrics' >> beam.Map(metrics_validator.validate_metrics,
                                      eval_config))
  evaluation_results[validations_key] = validations
  return evaluation_results
