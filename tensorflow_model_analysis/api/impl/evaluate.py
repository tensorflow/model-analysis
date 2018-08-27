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
"""Public API for performing evaluations using the EvalSavedModel."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function




import apache_beam as beam

from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.eval_saved_model import dofn
from tensorflow_model_analysis.eval_saved_model import load
from tensorflow_model_analysis.eval_saved_model.post_export_metrics import metric_keys
from tensorflow_model_analysis.extractors import feature_extractor
from tensorflow_model_analysis.extractors import predict_extractor
from tensorflow_model_analysis.slicer import slicer
from tensorflow_transform.beam import shared
from tensorflow_model_analysis.types_compat import Any, Callable, Dict, Generator, List, Optional, Tuple

MetricVariablesType = List[Any]  # pylint: disable=invalid-name

# For use in Beam type annotations, because Beam's support for Python types
# in Beam type annotations is not complete.
_BeamSliceKeyType = beam.typehints.Tuple[  # pylint: disable=invalid-name
    beam.typehints.Tuple[bytes, beam.typehints.Union[bytes, int, float]], Ellipsis]

_METRICS_NAMESPACE = 'tensorflow_model_analysis'


@beam.typehints.with_input_types(beam.typehints.Any)
@beam.typehints.with_output_types(
    beam.typehints.Tuple[_BeamSliceKeyType, beam.typehints.Any])
class _SliceDoFn(beam.DoFn):
  """A DoFn that performs slicing."""

  def __init__(self, slice_spec):
    self._slice_spec = slice_spec
    self._num_slices_generated_per_instance = beam.metrics.Metrics.distribution(
        _METRICS_NAMESPACE, 'num_slices_generated_per_instance')
    self._post_slice_num_instances = beam.metrics.Metrics.counter(
        _METRICS_NAMESPACE, 'post_slice_num_instances')

  def process(self, element):
    features = element.features
    slice_count = 0
    for slice_key in slicer.get_slices_for_features_dict(
        features, self._slice_spec):
      slice_count += 1
      yield (slice_key, element)

    self._num_slices_generated_per_instance.update(slice_count)
    self._post_slice_num_instances.inc(slice_count)


@beam.ptransform_fn
@beam.typehints.with_input_types(
    beam.typehints.Tuple[types.DictOfTensorType, beam.typehints.Any]
)
@beam.typehints.with_output_types(
    beam.typehints.Tuple[_BeamSliceKeyType, beam.typehints.Any])  # pylint: disable=invalid-name
def _Slice(intro_result,
           slice_spec):
  return intro_result | beam.ParDo(_SliceDoFn(slice_spec))


def _add_metric_variables(  # pylint: disable=invalid-name
    left,
    right):
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

  def __init__(self):
    self.metric_variables = None  # type: Optional[MetricVariablesType]
    self.fpls = []  # type: List[beam.typehints.Any]

  def copy_from(  # pylint: disable=invalid-name
      self, other):
    if other.metric_variables:
      self.metric_variables = other.metric_variables
    self.fpls = other.fpls

  def __iadd__(self, other):
    self.metric_variables = _add_metric_variables(self.metric_variables,
                                                  other.metric_variables)
    self.fpls.extend(other.fpls)
    return self

  def add_fpl(  # pylint: disable=invalid-name
      self, fpl):
    self.fpls.append(fpl)

  def add_metrics_variables(  # pylint: disable=invalid-name
      self, metric_variables):
    self.metric_variables = _add_metric_variables(self.metric_variables,
                                                  metric_variables)


@beam.typehints.with_input_types(beam.typehints.Any)
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
  BEAM-3737: Key-aware batching function.
  """

  _DEFAULT_BATCH_SIZE = 1000

  def __init__(self,
               eval_saved_model_path,
               add_metrics_callbacks,
               shared_handle,
               desired_batch_size = None):
    self._eval_saved_model_path = eval_saved_model_path
    self._add_metrics_callbacks = add_metrics_callbacks
    self._shared_handle = shared_handle
    self._eval_saved_model = None  # type: load.EvalSavedModel
    self._model_load_seconds = beam.metrics.Metrics.distribution(
        _METRICS_NAMESPACE, 'model_load_seconds')
    self._combine_batch_size = beam.metrics.Metrics.distribution(
        _METRICS_NAMESPACE, 'combine_batch_size')

    # We really want the batch size to be adaptive like it is in
    # beam.BatchElements(), but there isn't an easy way to make it so.
    if desired_batch_size > 0:
      self._desired_batch_size = desired_batch_size
    else:
      self._desired_batch_size = self._DEFAULT_BATCH_SIZE

  def _start_bundle(self):
    # There's no initialisation method for CombineFns.
    # See BEAM-3736: Add SetUp() and TearDown() for CombineFns.
    self._eval_saved_model = self._shared_handle.acquire(
        dofn.make_construct_fn(self._eval_saved_model_path,
                               self._add_metrics_callbacks,
                               self._model_load_seconds))

  def _maybe_do_batch(self, accumulator,
                      force = False):
    """Maybe intro metrics and update accumulator in place.

    Checks if accumulator has enough FPLs for a batch, and if so, does the
    intro metrics for the batch and updates accumulator in place.

    Args:
      accumulator: Accumulator. Will be updated in place.
      force: Force intro metrics even if accumulator has less FPLs than the
        batch size.
    """
    if self._eval_saved_model is None:
      self._start_bundle()
    batch_size = len(accumulator.fpls)
    if force or batch_size >= self._desired_batch_size:
      self._combine_batch_size.update(batch_size)
      if accumulator.fpls:  # Might be empty if force is True
        accumulator.add_metrics_variables(
            self._eval_saved_model.metrics_reset_update_get_list(
                accumulator.fpls))
      accumulator.fpls = []

  def create_accumulator(self):
    return _AggState()

  def add_input(self, accumulator,
                elem):
    # Note that we're mutating the accumulator in-place. Beam guarantees that
    # this is safe.
    accumulator.add_fpl(elem)
    self._maybe_do_batch(accumulator)
    return accumulator

  def merge_accumulators(self, accumulators):
    result = _AggState()
    for acc in accumulators:
      result += acc
      # Compact within the loop to avoid accumulating too much data.
      #
      # During the "map" side of combining combining happens per bundle,
      # but on the "reduce" side it's across all bundles (for a given key).
      #
      # So we could potentially accumulate get num_bundles * batch_size
      # elements if we don't process the batches within the loop, which
      # could cause OOM errors (b/77293756).
      self._maybe_do_batch(result)
    return result

  def extract_output(self,
                     accumulator):
    # Note that we're mutating the accumulator in-place. Beam guarantees that
    # this is safe.
    self._maybe_do_batch(accumulator, force=True)
    return accumulator.metric_variables


@beam.ptransform_fn
@beam.typehints.with_input_types(
    beam.typehints.Tuple[_BeamSliceKeyType, beam.typehints.Any])
@beam.typehints.with_output_types(beam.typehints.Tuple[
    _BeamSliceKeyType, beam.typehints.List[beam.typehints.Any]])
def _Aggregate(  # pylint: disable=invalid-name
    slice_result,
    eval_saved_model_path,
    add_metrics_callbacks,
    desired_batch_size = None,
):
  return (
      slice_result
      | 'CombinePerKey' >> beam.CombinePerKey(
          _AggregateCombineFn(
              eval_saved_model_path=eval_saved_model_path,
              add_metrics_callbacks=add_metrics_callbacks,
              shared_handle=shared.Shared(),
              desired_batch_size=desired_batch_size))
  )


@beam.typehints.with_input_types(beam.typehints.Tuple[
    _BeamSliceKeyType, beam.typehints.List[beam.typehints.Any]])
# No typehint for output type, since it's a multi-output DoFn result that
# Beam doesn't support typehints for yet (BEAM-3280).
class _ExtractOutputDoFn(dofn.EvalSavedModelDoFn):
  """A DoFn that extracts the metrics output."""

  OUTPUT_TAG_METRICS = 'tag_metrics'
  OUTPUT_TAG_PLOTS = 'tag_plots'

  def process(self, element
             ):
    (slice_key, metric_variables) = element
    self._eval_saved_model.set_metric_variables(metric_variables)
    result = self._eval_saved_model.get_metric_values()
    slicing_metrics = {}
    plots = {}
    for k, v in result.items():
      if k in metric_keys.PLOT_KEYS:
        plots[k] = v
      else:
        slicing_metrics[k] = v

    yield (slice_key, slicing_metrics)
    if plots:
      yield beam.pvalue.TaggedOutput(self.OUTPUT_TAG_PLOTS, (slice_key, plots))  # pytype: disable=bad-return-type


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.typehints.Tuple[
    _BeamSliceKeyType, beam.typehints.List[beam.typehints.Any]])
# No typehint for output type, since it's a multi-output DoFn result that
# Beam doesn't support typehints for yet (BEAM-3280).
def _ExtractOutput(  # pylint: disable=invalid-name
    aggregate_result, eval_saved_model_path,
    add_metrics_callbacks
):
  return aggregate_result | beam.ParDo(
      _ExtractOutputDoFn(
          eval_saved_model_path=eval_saved_model_path,
          add_metrics_callbacks=add_metrics_callbacks,
          shared_handle=shared.Shared())).with_outputs(
              _ExtractOutputDoFn.OUTPUT_TAG_PLOTS,
              main=_ExtractOutputDoFn.OUTPUT_TAG_METRICS)


@beam.ptransform_fn
# No typehint for output type, since it's a multi-output DoFn result that
# Beam doesn't support typehints for yet (BEAM-3280).
def Evaluate(
    # pylint: disable=invalid-name
    examples,
    eval_saved_model_path,
    add_metrics_callbacks = None,
    slice_spec = None,
    desired_batch_size = None,
):
  """Evaluate the given EvalSavedModel on the given examples.

  This is for TFMA use only. Users should call tfma.EvaluateAndWriteResults
  instead of this function.

  Args:
    examples: PCollection of input examples. Can be any format the model accepts
      (e.g. string containing CSV row, TensorFlow.Example, etc).
    eval_saved_model_path: Path to EvalSavedModel. This directory should contain
      the saved_model.pb file.
    add_metrics_callbacks: Optional list of callbacks for adding additional
      metrics to the graph. The names of the metrics added by the callbacks
      should not conflict with existing metrics, or metrics added by other
      callbacks. See below for more details about what each callback should do.
    slice_spec: Optional list of SingleSliceSpec specifying the slices to slice
      the data into. If None, defaults to the overall slice.
    desired_batch_size: Optional batch size for batching in Predict and
      Aggregate.

  More details on add_metrics_callbacks:

    Each add_metrics_callback should have the following prototype:
      def add_metrics_callback(features_dict, predictions_dict, labels_dict):

    Note that features_dict, predictions_dict and labels_dict are not
    necessarily dictionaries - they might also be Tensors, depending on what the
    model's eval_input_receiver_fn returns.

    It should create and return a metric_ops dictionary, such that
    metric_ops['metric_name'] = (value_op, update_op), just as in the Trainer.

    Short example:

    def add_metrics_callback(features_dict, predictions_dict, labels):
      metrics_ops = {}
      metric_ops['mean_label'] = tf.metrics.mean(labels)
      metric_ops['mean_probability'] = tf.metrics.mean(tf.slice(
        predictions_dict['probabilities'], [0, 1], [2, 1]))
      return metric_ops

  Returns:
    DoOutputsTuple. The tuple entries are
    PCollection of (slice key, metrics) and
    PCollection of (slice key, plot metrics).
  """
  if slice_spec is None:
    slice_spec = [slicer.SingleSliceSpec()]

  # pylint: disable=no-value-for-parameter
  return (
      examples
      # Our diagnostic outputs, pass types.ExampleAndExtracts throughout,
      # however our aggregating functions do not use this interface.
      | beam.Map(lambda x: types.ExampleAndExtracts(example=x, extracts={}))

      # Map function which loads and runs the eval_saved_model against every
      # example, yielding an types.ExampleAndExtracts containing a
      # FeaturesPredictionsLabels value (where key is 'fpl').
      | 'Predict' >> predict_extractor.TFMAPredict(
          eval_saved_model_path=eval_saved_model_path,
          desired_batch_size=desired_batch_size)

      # Unwrap the types.ExampleAndExtracts.
      # The rest of this pipeline expects FeaturesPredictionsLabels
      | beam.Map(lambda x:  # pylint: disable=g-long-lambda
                 x.extracts[constants.FEATURES_PREDICTIONS_LABELS_KEY])

      # Input: one example fpl at a time
      # Output: one fpl example per slice key (notice that the example turns
      #         into n, replicated once per applicable slice key)
      | 'Slice' >> _Slice(slice_spec)

      # Each slice key lands on one shard where metrics are computed for all
      # examples in that shard -- the "map" and "reduce" parts of the
      # computation happen within this shard.
      # Output: Tuple[slicer.SliceKeyType, MetricVariablesType]
      | 'Aggregate' >> _Aggregate(
          eval_saved_model_path=eval_saved_model_path,
          add_metrics_callbacks=add_metrics_callbacks,
          desired_batch_size=desired_batch_size)

      # Different metrics for a given slice key are brought together.
      | 'ExtractOutput' >> _ExtractOutput(eval_saved_model_path,
                                          add_metrics_callbacks))


@beam.ptransform_fn
@beam.typehints.with_input_types(bytes)
@beam.typehints.with_output_types(beam.typehints.Any)
def BuildDiagnosticTable(
    # pylint: disable=invalid-name
    examples,
    eval_saved_model_path,
    desired_batch_size = None):
  """Build diagnostics for the spacified EvalSavedModel and example collection.

  Args:
    examples: PCollection of input examples. Can be any format the model accepts
      (e.g. string containing CSV row, TensorFlow.Example, etc).
    eval_saved_model_path: Path to EvalSavedModel. This directory should contain
      the saved_model.pb file.
    desired_batch_size: Optional batch size for batching in Predict and
      Aggregate.

  Returns:
    PCollection of ExampleAndExtracts
  """
  return (examples
          | 'ToExampleAndExtracts' >>
          beam.Map(lambda x: types.ExampleAndExtracts(example=x, extracts={}))
          | 'Predict' >> predict_extractor.TFMAPredict(eval_saved_model_path,
                                                       desired_batch_size)
          | 'ExtractFeatures' >> feature_extractor.ExtractFeatures())
