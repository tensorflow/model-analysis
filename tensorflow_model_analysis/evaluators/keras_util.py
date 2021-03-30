# Lint as: python3
# Copyright 2020 Google LLC
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
"""Utils for evaluations using the keras."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import collections
from itertools import chain  # pylint: disable=g-importing-member

from typing import Dict, Iterable, List, Optional, Text

import apache_beam as beam
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import config
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import model_util
from tensorflow_model_analysis import types
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.metrics import tf_metric_accumulators
from tfx_bsl.coders import example_coder
from tfx_bsl.tfxio import tensor_adapter


def metric_computations_using_keras_saved_model(
    model_name: Text,
    model_loader: types.ModelLoader,
    eval_config: Optional[config.EvalConfig],
    tensor_adapter_config: Optional[tensor_adapter.TensorAdapterConfig] = None,
    batch_size: Optional[int] = None) -> metric_types.MetricComputations:
  """Returns computations for computing metrics natively using keras.

  Args:
    model_name: Name of model.
    model_loader: Loader for shared model containing keras saved model to use
      for metric computations.
    eval_config: Eval config.
    tensor_adapter_config: Tensor adapter config which specifies how to obtain
      tensors from the Arrow RecordBatch.
    batch_size: Batch size to use during evaluation (testing only).
  """
  model = model_loader.load()
  # If metrics were only added using model.compile then use
  # model.compiled_metrics and model.compiled_loss to compute the metrics,
  # otherwise custom metrics added via model.add_metric were also used and we
  # need to call model.evaluate.
  if not model.metrics:
    return []
  elif (hasattr(model, 'compiled_metrics') and model.compiled_metrics and
        hasattr(model, 'compiled_loss') and model.compiled_loss and
        len(model.compiled_metrics.metrics) + len(model.compiled_loss.metrics)
        == len(model.metrics)):
    output_names = model.output_names if hasattr(model, 'output_names') else []
    keys = _metric_keys(
        chain(model.compiled_metrics.metrics, model.compiled_loss.metrics),
        model_name, output_names)
    return [
        metric_types.MetricComputation(
            keys=keys,
            preprocessor=None,
            combiner=_KerasCompiledMetricsCombiner(keys, model_name,
                                                   model_loader, eval_config,
                                                   batch_size))
    ]
  else:
    output_names = model.output_names if hasattr(model, 'output_names') else []
    keys = _metric_keys(model.metrics, model_name, output_names)
    return [
        metric_types.MetricComputation(
            keys=keys,
            # TODO(b/178158073): By using inputs instead of batched features we
            # incur the cost of having to parse the inputs a second time. In
            # addition, transformed features (i.e. TFT, KPL) are not supported.
            preprocessor=metric_types.InputPreprocessor(),
            combiner=_KerasEvaluateCombiner(keys, model_name, model_loader,
                                            eval_config, tensor_adapter_config,
                                            batch_size))
    ]


def _metric_keys(metrics: Iterable[tf.keras.metrics.Metric], model_name: Text,
                 output_names: Iterable[Text]) -> List[metric_types.MetricKey]:
  """Returns metric keys for given metrics."""
  # We need to use the metric name to determine the associated output because
  # keras does not provide an API (see b/149780822). Keras names its metrics
  # using the following format:
  #   <output_name>_[weighted]_<metric_name>
  result = []
  for metric in metrics:
    sub_key = None
    if hasattr(metric, 'class_id') and metric.class_id is not None:
      sub_key = metric_types.SubKey(class_id=metric.class_id)
    elif hasattr(metric, 'top_k') and metric.top_k is not None:
      sub_key = metric_types.SubKey(top_k=metric.top_k)
    for output_name in output_names or []:
      if metric.name.startswith(output_name + '_'):
        # TODO(b/171559113): Output prefixes used to be added multiple times.
        # Remove this while loop after the last TF version with the issue is
        # no longer supported.
        name = metric.name
        while name.startswith(output_name + '_'):
          name = name[len(output_name) + 1:]
        result.append(
            metric_types.MetricKey(
                name=name,
                model_name=model_name,
                output_name=output_name,
                sub_key=sub_key))
        break
    else:
      result.append(
          metric_types.MetricKey(
              name=metric.name, model_name=model_name, sub_key=sub_key))
  return result


@beam.typehints.with_input_types(metric_types.StandardMetricInputs)
@beam.typehints.with_output_types(Dict[metric_types.MetricKey, np.ndarray])
class _KerasCombiner(model_util.CombineFnWithModels):
  """Base combiner for aggregating metrics for keras based models."""

  def __init__(self,
               keys: List[metric_types.MetricKey],
               model_name: Text,
               model_loader: types.ModelLoader,
               eval_config: Optional[config.EvalConfig],
               desired_batch_size: Optional[int] = None,
               beam_metrics_prefix: Text = ''):
    super(_KerasCombiner, self).__init__({model_name: model_loader})
    self._keys = keys
    self._model_name = model_name
    self._eval_config = eval_config
    self._desired_batch_size = desired_batch_size
    self._model = None
    # This combiner makes use of the TFMetricsAccumulator to track the inputs
    # and outputs. While the TFMetricsAccumulator is designed to store output
    # weights for each input, this doesn't work well with metrics from the keras
    # model because all the outputs get mixed together. So in this case the
    # _output_names will contain ['', output1, output2, ...] and _output_counts
    # will contain the corresponding counts of metrics for each output (with ''
    # including the counts of metrics that are aggregations over multiple
    # outputs - e.g. weighted loss). For the actual computations, we will store
    # the inputs under the respective output indices (with '' having no inputs),
    # but store all the metric weights under output index 0.
    self._output_names = sorted(set(key.output_name or '' for key in keys))
    counts = collections.defaultdict(int)
    for key in keys:
      if key.output_name:
        counts[key.output_name] += 1
    counts[''] = len(keys)
    self._output_counts = [counts[name] for name in self._output_names]
    self._batch_size_beam_metric_dist = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE,
        '{}_combine_batch_size'.format(beam_metrics_prefix))
    self._total_input_byte_size_beam_metric_dist = (
        beam.metrics.Metrics.distribution(
            constants.METRICS_NAMESPACE,
            '{}_combine_batch_bytes_size'.format(beam_metrics_prefix)))
    self._num_compacts = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, 'num_compacts')

  def setup(self):
    if self._model is None:
      # TODO(b/179500321): We are skipping the shared handle here to ensure that
      # we don't have issues with sharing the model between threads. This is
      # very inefficient, we should just clone the model but
      # tf.keras.models.clone_model removes compiled_metrics.
      self._model = self._model_loaders[  # pylint: disable=protected-access
          self._model_name]._construct_fn_with_load_time(
              self._set_model_load_seconds)()

  def _metrics(self) -> Iterable[tf.keras.metrics.Metric]:
    """Returns metrics used by combiner."""
    raise NotImplementedError('Subclasses are expected to override this.')

  def _create_accumulator(self) -> tf_metric_accumulators.TFMetricsAccumulator:
    """Returns a new accumulator."""
    raise NotImplementedError('Subclasses are expected to override this.')

  def _add_input(
      self, accumulator: tf_metric_accumulators.TFMetricsAccumulator,
      element: metric_types.StandardMetricInputs
  ) -> tf_metric_accumulators.TFMetricsAccumulator:
    """Add input to the accumulator."""
    raise NotImplementedError('Subclasses are expected to override this.')

  def _update_state(self,
                    accumulator: tf_metric_accumulators.TFMetricsAccumulator):
    """Updates state for metrics associated with model."""
    raise NotImplementedError('Subclasses are expected to override this.')

  def _process_batch(self,
                     accumulator: tf_metric_accumulators.TFMetricsAccumulator):
    if accumulator.len_inputs() == 0:
      return
    self._batch_size_beam_metric_dist.update(accumulator.len_inputs())
    self._total_input_byte_size_beam_metric_dist.update(
        accumulator.get_size_estimate())
    for metric_index, metric in enumerate(self._metrics()):
      metric.reset_states()
    self._update_state(accumulator)
    # For metrics stored with the model, the outputs get encoded in the
    # metric names so we will use a single output for the weights and parse the
    # names at the end to separate metrics by output.
    for metric_index, metric in enumerate(self._metrics()):
      accumulator.add_weights(0, metric_index, metric.get_weights())
    accumulator.clear_inputs()

  def create_accumulator(self) -> tf_metric_accumulators.TFMetricsAccumulator:
    return self._create_accumulator()

  def add_input(
      self, accumulator: tf_metric_accumulators.TFMetricsAccumulator,
      element: metric_types.StandardMetricInputs
  ) -> tf_metric_accumulators.TFMetricsAccumulator:
    accumulator = self._add_input(accumulator, element)
    if accumulator.should_flush():
      self._process_batch(accumulator)
    return accumulator

  def merge_accumulators(
      self, accumulators: Iterable[tf_metric_accumulators.TFMetricsAccumulator]
  ) -> tf_metric_accumulators.TFMetricsAccumulator:
    accumulators = iter(accumulators)
    result = next(accumulators)
    # Finish processing last batch
    self._process_batch(result)
    for accumulator in accumulators:
      # Finish processing last batch
      self._process_batch(accumulator)
      # Merge the weights
      for metric_index in range(len(self._keys)):
        weights = accumulator.get_weights(0, metric_index)
        if weights is None:
          # It is possible for beam to create an accumulator but pass no
          # inputs to it resulting in empty weights. In theory all weights
          # should be empty but we check on a per metric weights basis.
          continue
        result.add_weights(0, metric_index, weights)
    return result

  def compact(
      self, accumulator: tf_metric_accumulators.TFMetricsAccumulator
  ) -> tf_metric_accumulators.TFMetricsAccumulator:
    self._process_batch(accumulator)
    self._num_compacts.inc(1)
    return accumulator

  def extract_output(
      self, accumulator: tf_metric_accumulators.TFMetricsAccumulator
  ) -> Dict[metric_types.MetricKey, np.ndarray]:
    # Finish processing last batch
    self._process_batch(accumulator)
    result = {}
    for metric_index, metric in enumerate(self._metrics()):
      key = self._keys[metric_index]
      weights = accumulator.get_weights(0, metric_index)
      if weights is not None:
        metric.set_weights(weights)
      else:
        metric.reset_states()
      result[key] = metric.result().numpy()
    return result


@beam.typehints.with_input_types(metric_types.StandardMetricInputs)
@beam.typehints.with_output_types(Dict[metric_types.MetricKey, np.ndarray])
class _KerasCompiledMetricsCombiner(_KerasCombiner):
  """Aggregates metrics using keras compiled_metrics and compiled_loss."""

  def __init__(self,
               keys: List[metric_types.MetricKey],
               model_name: Text,
               model_loader: types.ModelLoader,
               eval_config: Optional[config.EvalConfig],
               desired_batch_size: Optional[int] = None):
    super(_KerasCompiledMetricsCombiner,
          self).__init__(keys, model_name, model_loader, eval_config,
                         desired_batch_size, 'keras_compiled_metrics_combine')

  def _metrics(self) -> Iterable[tf.keras.metrics.Metric]:
    return chain(self._model.compiled_metrics.metrics,
                 self._model.compiled_loss.metrics)

  def _create_accumulator(
      self) -> tf_metric_accumulators.TFCompilableMetricsAccumulator:
    padding_options = None
    if self._eval_config is not None:
      model_spec = model_util.get_model_spec(self._eval_config,
                                             self._model_name)
      if model_spec is not None and model_spec.HasField('padding_options'):
        padding_options = model_spec.padding_options
    return tf_metric_accumulators.TFCompilableMetricsAccumulator(
        padding_options,
        self._output_counts,
        desired_batch_size=self._desired_batch_size)

  def _add_input(
      self, accumulator: tf_metric_accumulators.TFCompilableMetricsAccumulator,
      element: metric_types.StandardMetricInputs
  ) -> tf_metric_accumulators.TFCompilableMetricsAccumulator:
    for i, output_name in enumerate(self._output_names):
      if not output_name and len(self._output_names) > 1:
        # The first output_name for multi-output models is '' and is used to
        # store combined metric weights for all outputs, but is not for inputs.
        labels, predictions, example_weights = None, None, None
      else:
        labels, predictions, example_weights = next(
            metric_util.to_label_prediction_example_weight(
                element,
                self._eval_config,
                self._model_name,
                output_name,
                flatten=False))
      accumulator.add_input(i, labels, predictions, example_weights)
    return accumulator

  def _update_state(
      self, accumulator: tf_metric_accumulators.TFCompilableMetricsAccumulator):
    if len(self._output_names) == 1:
      # Single-output models don't use dicts.
      l, p, w = accumulator.get_inputs(0)
      labels = tf.convert_to_tensor(l)
      predictions = tf.convert_to_tensor(p)
      example_weights = tf.convert_to_tensor(w)
    else:
      labels = {}
      predictions = {}
      example_weights = {}
      for i, output_name in enumerate(self._output_names):
        if not output_name:
          # The empty output_name for multi-output models is not used for inputs
          continue
        l, p, w = accumulator.get_inputs(i)
        labels[output_name] = tf.convert_to_tensor(l)
        predictions[output_name] = tf.convert_to_tensor(p)
        example_weights[output_name] = tf.convert_to_tensor(w)
    self._model.compiled_metrics.update_state(
        labels, predictions, sample_weight=example_weights)
    self._model.compiled_loss(
        labels, predictions, sample_weight=example_weights)


@beam.typehints.with_input_types(metric_types.StandardMetricInputs)
@beam.typehints.with_output_types(Dict[metric_types.MetricKey, np.ndarray])
class _KerasEvaluateCombiner(_KerasCombiner):
  """Aggregates metrics using keras model.evaluate method."""

  def __init__(self,
               keys: List[metric_types.MetricKey],
               model_name: Text,
               model_loader: types.ModelLoader,
               eval_config: Optional[config.EvalConfig],
               tensor_adapter_config: Optional[
                   tensor_adapter.TensorAdapterConfig] = None,
               desired_batch_size: Optional[int] = None):
    super(_KerasEvaluateCombiner,
          self).__init__(keys, model_name, model_loader, eval_config,
                         desired_batch_size, 'keras_evaluate_combine')
    self._tensor_adapter_config = tensor_adapter_config
    self._tensor_adapter = None
    self._decoder = None

  def setup(self):
    super(_KerasEvaluateCombiner, self).setup()
    # TODO(b/180125126): Re-enable use of passed in TensorAdapter after bug
    # requiring matching schema's is fixed.
    # if self._tensor_adapter is None and
    #    self._tensor_adapter_config is not None:
    #   self._tensor_adapter = tensor_adapter.TensorAdapter(
    #       self._tensor_adapter_config)
    if self._decoder is None:
      self._decoder = example_coder.ExamplesToRecordBatchDecoder()

  def _metrics(self) -> Iterable[tf.keras.metrics.Metric]:
    return self._model.metrics

  def _create_accumulator(self) -> tf_metric_accumulators.TFMetricsAccumulator:
    return tf_metric_accumulators.TFMetricsAccumulator(
        # Separate inputs are tracked for (inputs, labels, example_weights).
        # Since the inputs are the same for each output, only the first output
        # index will set the input data.
        input_counts=[3] * len(self._output_counts),
        metric_counts=self._output_counts,
        size_estimator_fn=len,
        desired_batch_size=self._desired_batch_size)

  def _add_input(
      self, accumulator: tf_metric_accumulators.TFMetricsAccumulator,
      element: metric_types.StandardMetricInputs
  ) -> tf_metric_accumulators.TFMetricsAccumulator:
    for i, output_name in enumerate(self._output_names):
      if not output_name and len(self._output_names) > 1:
        # The first output_name for multi-output models is '' and is used to
        # store combined metric weights for all outputs, but is not for labels
        # and example weights.
        labels, example_weights = None, None
      else:
        labels, _, example_weights = next(
            metric_util.to_label_prediction_example_weight(
                element,
                self._eval_config,
                self._model_name,
                output_name,
                flatten=False))
      accumulator.add_input(i, element.inputs if i == 0 else None, labels,
                            example_weights)

    return accumulator

  def _update_state(self,
                    accumulator: tf_metric_accumulators.TFMetricsAccumulator):
    serialized_examples = None
    labels = {}
    example_weights = {}
    for i, output_name in enumerate(self._output_names):
      e, l, w = accumulator.get_inputs(i)
      if i == 0:
        serialized_examples = e
      if not output_name and len(self._output_names) > 1:
        # The empty output_name for multi-output models is not used for inputs.
        continue
      labels[output_name] = np.array(l)
      weights = np.array(w)
      # TFv1 will not squeeze the weights, so must do manually
      if weights.shape[-1] == 1:
        weights = weights.squeeze(axis=-1)
      example_weights[output_name] = weights
    if len(self._output_names) == 1:
      # Single-output models don't use dicts.
      labels = next(iter(labels.values()))
      example_weights = next(iter(example_weights.values()))
    record_batch = self._decoder.DecodeBatch(serialized_examples)
    input_specs = model_util.get_input_specs(self._model, signature_name=None)
    inputs = model_util.get_inputs(record_batch, input_specs,
                                   self._tensor_adapter)
    if inputs is None:
      raise ValueError('unable to prepare inputs for evaluation: '
                       'input_specs={}, record_batch={}'.format(
                           input_specs, record_batch))
    self._model.evaluate(
        x=inputs,
        y=labels,
        batch_size=record_batch.num_rows,
        verbose=0,
        sample_weight=example_weights)
