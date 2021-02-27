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
from tensorflow_model_analysis.metrics import metric_specs
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.metrics import tf_metric_accumulators


def metrics_specs_from_keras(
    model_name: Text,
    model_loader: types.ModelLoader,
) -> List[config.MetricsSpec]:
  """Returns metrics specs for metrics and losses associated with the model."""
  model = model_loader.construct_fn()
  if model is None:
    return []

  metric_names = []
  metrics = []
  if hasattr(model, 'loss_functions'):
    # Legacy keras metrics separate the losses from the metrics and store them
    # under loss_functions. The first name in metric_names is always 'loss'
    # followed by the loss_function names (prefixed by output_name if multiple
    # outputs) and then followed by the metric names (also prefixed by output
    # name). Note that names in loss_functions will not have any output name
    # prefixes (if used) while the metrics will so we need to use the names in
    # metric_names for matching with outputs not the names in the functions.
    metric_names = model.metrics_names
    metrics.extend(model.loss_functions)
    metrics.extend(model.metrics)
    if len(metric_names) > len(metrics) and metric_names[0] == 'loss':
      metric_names = metric_names[1:]
  elif hasattr(model, 'compiled_loss') and hasattr(model, 'compiled_metrics'):
    # In the new keras metric setup the metrics include the losses (in the form
    # of a metric type not a loss type) and the metrics_names align with the
    # names in the metric classes. The metrics itself contains compiled_loss,
    # compiled_metrics, and custom metrics (added via add_metric). Since we only
    # care about compiled metrics we use these APIs instead. Note that the
    # overall loss metric is an average of the other losses which doesn't take
    # y_true, y_pred as inputs so it can't be calculated via standard inputs so
    # we remove it.
    for m in model.compiled_loss.metrics:
      # TODO(b/143228390): Pure Mean metrics cannot be calculated using labels,
      # predictions, and example weights.
      if type(m) in (tf.keras.metrics.Mean,):  # pylint: disable=unidiomatic-typecheck
        continue
      metrics.append(m)
    metrics.extend(model.compiled_metrics.metrics)
    metric_names = [m.name for m in metrics]

  specs = []

  # Need to check if model.output_names exists because the keras Sequential
  # model doesn't always contain output_names (b/150510258).
  if hasattr(model, 'output_names') and len(model.output_names) > 1:
    unmatched_metrics = {m for m in metrics}
    for output_name in model.output_names:
      per_output_metrics = []
      for (name, metric) in zip(metric_names, metrics):
        if name.startswith(output_name + '_'):
          per_output_metrics.append(metric)
          unmatched_metrics.remove(metric)
      if per_output_metrics:
        specs.extend(
            metric_specs.specs_from_metrics(
                metrics=per_output_metrics,
                model_names=[model_name],
                output_names=[output_name],
                include_example_count=False,
                include_weighted_example_count=False))
    metrics = list(unmatched_metrics)

  if metrics:
    specs.extend(
        metric_specs.specs_from_metrics(
            metrics=metrics,
            model_names=[model_name],
            include_example_count=False,
            include_weighted_example_count=False))

  return specs


def metric_computations_using_keras_saved_model(
    model_name: Text,
    model_loader: types.ModelLoader,
    eval_config: Optional[config.EvalConfig],
    batch_size: Optional[int] = None) -> metric_types.MetricComputations:
  """Returns computations for computing metrics natively using keras.

  Args:
    model_name: Name of model.
    model_loader: Loader for shared model containing keras saved model to use
      for metric computations.
    eval_config: Eval config.
    batch_size: Batch size to use during evaluation (testing only).
  """
  model = model_loader.load()
  if hasattr(model, 'compiled_metrics') and hasattr(model, 'compiled_loss'):
    # TODO(b/154395500): Add support for calling keras model.evaluate() when
    # custom metrics used.
    if (len(model.compiled_metrics.metrics) + len(model.compiled_loss.metrics)
        != len(model.metrics)):
      tf.compat.v1.logging.warning(
          'TFMA does not currently support custom metrics added by '
          'model.add_metric, silently ignoring custom metrics')
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
    raise NotImplementedError(
        'evaluation using model.evaluate is not yet supported')


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
    for output_name in output_names:
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

  def _setup_if_needed(self):
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
    self._setup_if_needed()
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
      self, accumulators: List[tf_metric_accumulators.TFMetricsAccumulator]
  ) -> tf_metric_accumulators.TFMetricsAccumulator:
    result = accumulators[0]
    # Finish processing last batch
    self._process_batch(result)
    for accumulator in accumulators[1:]:
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
