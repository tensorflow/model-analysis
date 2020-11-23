# Lint as: python3
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

from typing import Any, Dict, Generator, Iterable, List, Optional, Text, Tuple

import apache_beam as beam
import numpy as np

from tensorflow_model_analysis import constants
from tensorflow_model_analysis import model_util
from tensorflow_model_analysis import types
from tensorflow_model_analysis import util
from tensorflow_model_analysis.eval_metrics_graph import eval_metrics_graph
from tensorflow_model_analysis.slicer import slicer_lib as slicer


@beam.ptransform_fn
@beam.typehints.with_input_types(Tuple[slicer.SliceKeyType, types.Extracts])
@beam.typehints.with_output_types(Tuple[slicer.SliceKeyType, Dict[Text, Any]])
def ComputePerSliceMetrics(  # pylint: disable=invalid-name
    slice_result: beam.pvalue.PCollection,
    eval_shared_model: types.EvalSharedModel,
    desired_batch_size: Optional[int] = None,
    compute_with_sampling: Optional[bool] = False,
    random_seed_for_testing: Optional[int] = None) -> beam.pvalue.PCollection:
  """PTransform for computing, aggregating and combining metrics.

  Args:
    slice_result: Incoming PCollection consisting of slice key and extracts.
    eval_shared_model: Shared model parameters for EvalSavedModel.
    desired_batch_size: Optional batch size for batching in Aggregate.
    compute_with_sampling: True to compute with sampling.
    random_seed_for_testing: Seed to use for unit testing.

  Returns:
    PCollection of (slice key, dict of metrics).
  """
  # TODO(b/123516222): Remove this workaround per discussions in CL/227944001
  slice_result.element_type = beam.typehints.Any

  return (
      slice_result
      # _ModelLoadingIdentityFn loads the EvalSavedModel into memory
      # under a shared handle that can be used by subsequent steps.
      # Combiner lifting and producer-consumer fusion should ensure
      # that these steps run in the same process and memory space.
      # TODO(b/69566045): Remove _ModelLoadingIdentityFn and move model
      # loading to CombineFn.setup after it is available in Beam.
      | 'LoadModel' >> beam.ParDo(
          _ModelLoadingIdentityFn(eval_shared_model=eval_shared_model))
      | 'CombinePerSlice' >> beam.CombinePerKey(
          _AggregateCombineFn(
              eval_shared_model=eval_shared_model,
              desired_batch_size=desired_batch_size,
              compute_with_sampling=compute_with_sampling,
              seed_for_testing=random_seed_for_testing))
      | 'InterpretOutput' >> beam.ParDo(
          _ExtractOutputDoFn(eval_shared_model=eval_shared_model)))


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
  and a list of FeaturesPredictionsLabels or other inputs. See
  _AggregateCombineFn for why we need this.
  """

  # We really want the batch size to be adaptive like it is in
  # beam.BatchElements(), but there isn't an easy way to make it so. For now
  # we will limit stored inputs to a max overall byte size.
  # TODO(b/73789023): Figure out how to make this batch size dynamic.
  _TOTAL_INPUT_BYTE_SIZE_THRESHOLD = 16 << 20  # 16MiB
  _DEFAULT_DESIRED_BATCH_SIZE = 1000

  __slots__ = [
      'metric_variables', 'inputs', 'size_estimator', '_desired_batch_size'
  ]

  def __init__(self, desired_batch_size: Optional[int] = None):
    self.metric_variables = None  # type: Optional[types.MetricVariablesType]
    self.inputs = []  # type: List[bytes]
    self.size_estimator = util.SizeEstimator(
        size_threshold=self._TOTAL_INPUT_BYTE_SIZE_THRESHOLD, size_fn=len)
    if desired_batch_size and desired_batch_size > 0:
      self._desired_batch_size = desired_batch_size
    else:
      self._desired_batch_size = self._DEFAULT_DESIRED_BATCH_SIZE

  def __iadd__(self, other: '_AggState') -> '_AggState':
    self.metric_variables = _add_metric_variables(self.metric_variables,
                                                  other.metric_variables)
    self.inputs.extend(other.inputs)
    self.size_estimator += other.size_estimator
    return self

  def add_input(self, new_input: bytes):
    self.inputs.append(new_input)
    self.size_estimator.update(new_input)

  def clear_inputs(self):
    del self.inputs[:]
    self.size_estimator.clear()

  def add_metrics_variables(self, metric_variables: types.MetricVariablesType):
    self.metric_variables = _add_metric_variables(self.metric_variables,
                                                  metric_variables)

  def should_flush(self) -> bool:
    return (len(self.inputs) >= self._desired_batch_size or
            self.size_estimator.should_flush())


@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(Optional[List[Any]])
class _AggregateCombineFn(model_util.CombineFnWithModels):
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

  # TODO(b/173811366): Consider removing the desired_batch_size knob and
  # only use input size.
  def __init__(self,
               eval_shared_model: types.EvalSharedModel,
               desired_batch_size: Optional[int] = None,
               compute_with_sampling: Optional[bool] = False,
               seed_for_testing: Optional[int] = None) -> None:
    super(_AggregateCombineFn,
          self).__init__({'': eval_shared_model.model_loader})
    self._seed_for_testing = seed_for_testing
    self._eval_metrics_graph = None  # type: eval_metrics_graph.EvalMetricsGraph
    self._desired_batch_size = desired_batch_size

    self._compute_with_sampling = compute_with_sampling
    self._random_state = np.random.RandomState(seed_for_testing)

    # Metrics.
    self._combine_batch_size = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE, 'combine_batch_size')
    self._combine_total_input_byte_size = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE, 'combine_batch_bytes_size')
    self._num_compacts = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, 'num_compacts')

  def _poissonify(self, accumulator: _AggState) -> List[bytes]:
    # pylint: disable=line-too-long
    """Creates a bootstrap resample of the data in an accumulator.

    Given a set of data, it will be represented in the resample set a number of
    times, that number of times is drawn from Poisson(1).
    See
    http://www.unofficialgoogledatascience.com/2015/08/an-introduction-to-poisson-bootstrap26.html
    for a detailed explanation of the technique. This will work technically with
    small or empty batches but as the technique is an approximation, the
    approximation gets better as the number of examples gets larger. If the
    results themselves are empty TFMA will reject the sample. For any samples of
    a reasonable size, the chances of this are exponentially tiny. See "The
    mathematical fine print" section of the blog post linked above.

    Args:
      accumulator: Accumulator containing FPLs from a sample

    Returns:
      A list of FPLs representing a bootstrap resample of the accumulator items.
    """
    result = []
    if accumulator.inputs:
      poisson_counts = self._random_state.poisson(1, len(accumulator.inputs))
      for i, input_item in enumerate(accumulator.inputs):
        result.extend([input_item] * poisson_counts[i])
    return result

  def _maybe_do_batch(self,
                      accumulator: _AggState,
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
      self._setup_if_needed()
      self._eval_metrics_graph = self._loaded_models['']
    if force or accumulator.should_flush():
      if accumulator.inputs:
        self._combine_batch_size.update(len(accumulator.inputs))
        self._combine_total_input_byte_size.update(
            accumulator.size_estimator.get_estimate())
        inputs_for_metrics = accumulator.inputs
        if self._compute_with_sampling:
          # If we are computing with multiple bootstrap replicates, use fpls
          # generated by the Poisson bootstrapping technique.
          inputs_for_metrics = self._poissonify(accumulator)
        if inputs_for_metrics:
          accumulator.add_metrics_variables(
              self._eval_metrics_graph.metrics_reset_update_get_list(
                  inputs_for_metrics))
        else:
          # Call to metrics_reset_update_get_list does a reset prior to the
          # metrics update, but does not handle empty updates. Explicitly
          # calling just reset here, to make the flow clear.
          self._eval_metrics_graph.reset_metric_variables()
        accumulator.clear_inputs()

  def create_accumulator(self) -> _AggState:
    return _AggState(desired_batch_size=self._desired_batch_size)

  def add_input(self, accumulator: _AggState,
                elem: types.Extracts) -> _AggState:
    accumulator.add_input(elem[constants.INPUT_KEY])
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


@beam.typehints.with_input_types(Tuple[slicer.SliceKeyType,
                                       Optional[List[Any]]])
# TODO(b/123516222): Add output typehints. Similarly elsewhere that it applies.
class _ExtractOutputDoFn(model_util.DoFnWithModels):
  """A DoFn that extracts the metrics output."""

  def __init__(self, eval_shared_model: types.EvalSharedModel) -> None:
    super(_ExtractOutputDoFn,
          self).__init__({'': eval_shared_model.model_loader})

    # This keeps track of the number of times the poisson bootstrap encounters
    # an empty set of elements for a slice sample. Should be extremely rare in
    # practice, keeping this counter will help us understand if something is
    # misbehaving.
    self._num_bootstrap_empties = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, 'num_bootstrap_empties')

  def process(
      self, element: Tuple[slicer.SliceKeyType, types.MetricVariablesType]
  ) -> Generator[Tuple[slicer.SliceKeyType, Dict[Text, Any]], None, None]:
    (slice_key, metric_variables) = element
    if metric_variables:
      eval_saved_model = self._loaded_models['']
      result = eval_saved_model.metrics_set_variables_and_get_values(
          metric_variables)
      yield (slice_key, result)
    else:
      # Increase a counter for empty bootstrap samples. When sampling is not
      # enabled, this should never be exected. This should only occur when the
      # slice sizes are incredibly small, and seeing large values of this
      # counter is a sign that something has gone wrong.
      self._num_bootstrap_empties.inc(1)


@beam.typehints.with_input_types(Tuple[slicer.SliceKeyType, types.Extracts])
@beam.typehints.with_output_types(Tuple[slicer.SliceKeyType, types.Extracts])
class _ModelLoadingIdentityFn(model_util.DoFnWithModels):
  """A DoFn that loads the EvalSavedModel and returns the input unchanged."""

  def __init__(self, eval_shared_model: types.EvalSharedModel) -> None:
    super(_ModelLoadingIdentityFn,
          self).__init__({'': eval_shared_model.model_loader})

  def process(
      self, element: Tuple[slicer.SliceKeyType, types.Extracts]
  ) -> List[Tuple[slicer.SliceKeyType, types.Extracts]]:
    return [element]
