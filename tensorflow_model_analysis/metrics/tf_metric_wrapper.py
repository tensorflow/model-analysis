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
"""TF metric wrapper."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import collections
import importlib
import itertools

from typing import Any, Dict, List, Optional, Text, Type, Tuple, Union

import apache_beam as beam
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import config
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import util
from tensorflow_model_analysis.metrics import binary_confusion_matrices
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util

_CONFIG_KEY = 'config'
_NUM_THRESHOLDS_KEY = 'num_thresholds'
_THRESHOLDS_KEY = 'thresholds'
_CLASS_ID_KEY = 'class_id'
_TOP_K_KEY = 'top_k'
_DEFAULT_NUM_THRESHOLDS_IN_KERAS = 200

_TFMetricOrLoss = Union[tf.keras.metrics.Metric, tf.keras.losses.Loss]


def tf_metric_computations(
    metrics: Union[List[_TFMetricOrLoss], Dict[Text, List[_TFMetricOrLoss]]],
    eval_config: Optional[config.EvalConfig] = None,
    model_name: Text = '',
    sub_key: Optional[metric_types.SubKey] = None,
    aggregation_type: Optional[metric_types.AggregationType] = None,
    class_weights: Optional[Dict[int, float]] = None,
    desired_batch_size: Optional[int] = None
) -> metric_types.MetricComputations:
  """Returns metric computations for the given TF metrics.

  Note that there is no requirement that a one to one mapping exist between the
  input metrics and the output metric computations. The implementation may
  combine multiple metrics into a single computation for efficency.

  Args:
    metrics: Dict from metric name to tf.keras.metrics.Metric or
      tf.keras.metrics.Loss. For multi-output models a dict of dicts may be
      passed where the first dict is indexed by the output_name.
    eval_config: Eval config.
    model_name: Optional model name (if multi-model evaluation).
    sub_key: Optional sub key.
    aggregation_type: Optional aggregation type.
    class_weights: Optional class weights to apply to multi-class / multi-label
      labels and predictions. This should only be used when the aggregation_type
      is set.
    desired_batch_size: Batch size to use when calling TF metrics
      (testing only).

  Returns:
    Metric computations.
  """
  if not isinstance(metrics, dict):
    metrics = {'': metrics}

  if aggregation_type is not None:
    sparse_metrics = _sparse_metrics(metrics)
    if sparse_metrics:
      raise ValueError(
          'sparse metrics cannot be used with aggregation options. Either '
          'disable aggregation settings or replace the sparse metrics with'
          'non-sparse versions: {}'.format(sparse_metrics))

  metrics = _filter_duplicate_metrics(metrics, model_name, sub_key)

  computations = []

  # For efficency, metrics are separated into confusion matrix based vs
  # non-confusion matrix based metrics. Since the confusion matrix based metrics
  # can all be calculated from the calibration histogram, these metrics are
  # computed separately as derived metrics. The remaining non-confusion matrix
  # metrics are calculated using batches of predictions/labels in eager mode
  # (possibly with additional pre-processing of the values to perform
  # binarization, etc).
  #
  # Note that in theory if a model was provided, all the metrics could be
  # calculated by calling model.evaluate(). However, this call is inefficient
  # for confusion matrix based metrics given the large number of weights that
  # need to be calculated and the overlapping computations between the metrics.
  # In addition, some metrics and plots are only defined in TFMA so a separate
  # evaluation step would still be required. Lastly, if the metrics have any
  # binarization, etc applied the inputs and outputs will not match those
  # expected by the model. For these reasons, a separate implementation is used
  # for each specific use case. It also allows evaluations that are not
  # associated with a model (i.e. raw predictions are passed as input) to share
  # the same code path as model based evaluations where possible.
  confusion_matrix_metrics, non_confusion_matrix_metrics = (
      _separate_confusion_matrix_metrics(metrics))

  for output_name, metrics in confusion_matrix_metrics.items():
    for metric in metrics:
      computations.extend(
          _wrap_confusion_matrix_metric(metric, eval_config, model_name,
                                        output_name, sub_key, aggregation_type,
                                        class_weights))

  if non_confusion_matrix_metrics:
    custom_objects = _custom_objects(non_confusion_matrix_metrics)
    metric_keys, metric_configs, loss_configs = _metric_keys_and_configs(
        non_confusion_matrix_metrics, model_name, sub_key)
    for sub_key, keys in metric_keys.items():
      computations.append(
          metric_types.MetricComputation(
              keys=keys,
              preprocessor=None,
              combiner=_CompilableMetricsCombiner(
                  metric_configs[sub_key],
                  loss_configs[sub_key],
                  custom_objects,
                  eval_config,
                  model_name,
                  sub_key,
                  aggregation_type,
                  class_weights,
                  desired_batch_size,
              )))

  return computations


def _filter_duplicate_metrics(
    metrics: Dict[Text, List[tf.keras.metrics.Metric]],
    model_name: Text,
    sub_key: Optional[metric_types.SubKey] = None,
) -> Dict[Text, List[tf.keras.metrics.Metric]]:
  """Filters duplicate metrics from the metrics."""
  for output_name, metrics_list in metrics.items():
    unique_metrics = {}
    for metric in metrics_list:
      key = metric_types.MetricKey(
          name=metric.name,
          model_name=model_name,
          output_name=output_name,
          sub_key=_verify_and_update_sub_key(model_name, output_name, sub_key,
                                             metric))
      # Replace any previous metric (i.e. last added metric wins).
      unique_metrics[key] = metric
    metrics[output_name] = list(unique_metrics.values())
  return metrics


def _sparse_metrics(
    metrics: Dict[Text, List[tf.keras.metrics.Metric]]
) -> Dict[Text, List[tf.keras.metrics.Metric]]:
  """Returns input metrics filtered to contain only the sparse metrics."""
  results = {}
  for k, v in metrics.items():
    for m in v:
      if m.__class__.__name__.startswith('Sparse'):
        if k not in results:
          results[k] = []
        results[k].append(m)
  return results


def _separate_confusion_matrix_metrics(
    metrics: Dict[Optional[Text], List[_TFMetricOrLoss]]
) -> Tuple[Dict[Optional[Text], List[tf.keras.metrics.Metric]], Dict[
    Optional[Text], List[_TFMetricOrLoss]]]:
  """Separates the confusion matrix metrics from the other metrics."""
  confusion_matrix_metrics = {}
  non_confusion_matrix_metrics = {}
  for output_name, metrics in metrics.items():
    for metric in metrics:
      # We are using type instead of isinstance here because we only want to
      # match specific types and not their subclasses. Note that if the top_k
      # setting is specified as part of the keras metric directly, then we
      # compute the value directly in keras. Otherwise, if the top_k setting is
      # only provided via BinarizeOptions then we compute the value using the
      # the confusion matrix.
      if (type(metric) in (  # pylint: disable=unidiomatic-typecheck
          tf.keras.metrics.AUC, tf.keras.metrics.SpecificityAtSensitivity,
          tf.keras.metrics.SensitivityAtSpecificity,
          tf.keras.metrics.TruePositives, tf.keras.metrics.FalsePositives,
          tf.keras.metrics.TrueNegatives, tf.keras.metrics.FalseNegatives,
          tf.keras.metrics.Precision, tf.keras.metrics.Recall) and
          not (hasattr(metric, _TOP_K_KEY) and metric.top_k is not None)):
        if output_name not in confusion_matrix_metrics:
          confusion_matrix_metrics[output_name] = []
        confusion_matrix_metrics[output_name].append(metric)
      else:
        if output_name not in non_confusion_matrix_metrics:
          non_confusion_matrix_metrics[output_name] = []
        non_confusion_matrix_metrics[output_name].append(metric)
  return confusion_matrix_metrics, non_confusion_matrix_metrics


def _verify_and_update_sub_key(model_name: Text, output_name: Text,
                               sub_key: metric_types.SubKey,
                               metric: _TFMetricOrLoss):
  """Verifies the multi-class metric key matches settings used by the metric."""
  if hasattr(metric, _CLASS_ID_KEY) and metric.class_id is not None:
    if sub_key and sub_key.class_id != metric.class_id:
      raise ValueError(
          '{} tf.keras.metric has class_id = {}, but the metric is being added '
          'using sub_key = {}: model_name={}, output_name={}'.format(
              metric.name, metric.class_id, sub_key, model_name, output_name))
    return metric_types.SubKey(class_id=metric.class_id)
  elif hasattr(metric, _TOP_K_KEY) and metric.top_k is not None:
    if sub_key and sub_key.top_k != metric.top_k:
      raise ValueError(
          '{} tf.keras.metric has top_k = {}, but the metric is being added '
          'using sub_key = {}: model_name={}, output_name={}'.format(
              metric.name, metric.top_k, sub_key, model_name, output_name))
    return metric_types.SubKey(top_k=metric.top_k)
  else:
    return sub_key


_KeysBySubKey = Dict[Optional[metric_types.SubKey],
                     List[metric_types.MetricKey]]
_ConfigsBySubKey = Dict[Optional[metric_types.SubKey],
                        Dict[Text, List[Dict[Text, Any]]]]


def _metric_keys_and_configs(
    metrics: Dict[Text, List[_TFMetricOrLoss]], model_name: Text,
    sub_key: Optional[metric_types.SubKey]
) -> Tuple[_KeysBySubKey, _ConfigsBySubKey, _ConfigsBySubKey]:
  """Returns metric keys, metric configs, and loss configs by sub key."""
  metric_keys = collections.defaultdict(list)
  metric_configs = collections.defaultdict(dict)
  loss_configs = collections.defaultdict(dict)
  for output_name, metrics_list in metrics.items():
    for metric in metrics_list:
      updated_sub_key = _verify_and_update_sub_key(model_name, output_name,
                                                   sub_key, metric)
      if output_name not in metric_configs[updated_sub_key]:
        metric_configs[updated_sub_key][output_name] = []
      if output_name not in loss_configs[updated_sub_key]:
        loss_configs[updated_sub_key][output_name] = []
      metric_keys[updated_sub_key].append(
          metric_types.MetricKey(
              name=metric.name,
              model_name=model_name,
              output_name=output_name,
              sub_key=updated_sub_key))
      if isinstance(metric, tf.keras.metrics.Metric):
        metric_configs[updated_sub_key][output_name].append(
            metric_util.serialize_metric(metric))
      elif isinstance(metric, tf.keras.losses.Loss):
        loss_configs[updated_sub_key][output_name].append(
            metric_util.serialize_loss(metric))
  return metric_keys, metric_configs, loss_configs


def _deserialize_metrics(
    metric_configs: List[Dict[Text, Any]]) -> List[tf.keras.metrics.Metric]:
  return [tf.keras.metrics.deserialize(c) for c in metric_configs]


def _deserialize_losses(
    loss_configs: List[Dict[Text, Any]]) -> List[tf.keras.losses.Loss]:
  return [tf.keras.losses.deserialize(c) for c in loss_configs]


def _custom_objects(
    metrics: Dict[Text,
                  List[tf.keras.metrics.Metric]]) -> List[Tuple[Text, Text]]:
  """Returns list of (module, class_name) tuples for custom objects."""
  custom_objects = []
  for metric_list in metrics.values():
    for metric in metric_list:
      if (metric.__class__.__module__ != tf.keras.metrics.__name__ and
          metric.__class__.__module__ != tf.keras.losses.__name__):
        custom_objects.append(
            (metric.__class__.__module__, metric.__class__.__name__))
  return custom_objects


def _load_custom_objects(
    custom_objects: List[Tuple[Text, Text]]) -> Dict[Text, Type[Any]]:
  """Loads custom metric options."""
  loaded_custom_objects = {}
  for module_name, class_name in custom_objects:
    module = importlib.import_module(module_name)
    loaded_custom_objects[class_name] = getattr(module, class_name)
  return loaded_custom_objects


def _get_config_value(key: Text, metric_config: Dict[Text,
                                                     Any]) -> Optional[Any]:
  """Returns value for key within config or None."""
  if _CONFIG_KEY in metric_config and key in metric_config[_CONFIG_KEY]:
    return metric_config[_CONFIG_KEY][key]
  return None


def _wrap_confusion_matrix_metric(
    metric: tf.keras.metrics.Metric, eval_config: config.EvalConfig,
    model_name: Text, output_name: Text, sub_key: Optional[metric_types.SubKey],
    aggregation_type: Optional[metric_types.AggregationType],
    class_weights: Optional[Dict[int,
                                 float]]) -> metric_types.MetricComputations:
  """Returns confusion matrix metric wrapped in a more efficient computation."""

  # Special handling for AUC metric which supports aggregation inherently via
  # multi_label flag.
  if (isinstance(metric, tf.keras.metrics.AUC) and
      hasattr(metric, 'label_weights')):
    if metric.label_weights:
      if class_weights:
        raise ValueError(
            'class weights are configured in two different places: (1) via the '
            'tf.keras.metrics.AUC class (using "label_weights") and (2) via '
            'the MetricsSpecs (using "aggregate.class_weights"). Either remove '
            'the label_weights settings in the AUC class or remove the '
            'class_weights from the AggregationOptions: metric={}, '
            'class_weights={}'.format(metric, class_weights))
      class_weights = {i: v for i, v in enumerate(metric.label_weights)}
    if metric.multi_label:
      raise NotImplementedError('AUC.multi_label=True is not implemented yet.')

  sub_key = _verify_and_update_sub_key(model_name, output_name, sub_key, metric)
  key = metric_types.MetricKey(
      name=metric.name,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key)

  metric_config = tf.keras.metrics.serialize(metric)

  thresholds = None
  num_thresholds = None
  # The top_k metrics have special settings. If we are setting the top_k value
  # outside of keras (i.e. using BinarizeOptions), then we need to set the
  # special threshold ourselves otherwise the default threshold of 0.5 is used.
  if (sub_key and sub_key.top_k is not None and
      _get_config_value(_TOP_K_KEY, metric_config) is None and
      _get_config_value(_THRESHOLDS_KEY, metric_config) is None):
    thresholds = [float('-inf')]
  elif hasattr(metric, _THRESHOLDS_KEY):
    if (len(
        metric.thresholds) == binary_confusion_matrices.DEFAULT_NUM_THRESHOLDS):
      num_thresholds = binary_confusion_matrices.DEFAULT_NUM_THRESHOLDS
    else:
      thresholds = metric.thresholds
  # Only one of either thresholds or num_thresholds should be used. Keras AUC
  # allows both but thresholds has more precedence.
  if thresholds is None and hasattr(metric, _NUM_THRESHOLDS_KEY):
    num_thresholds = metric.num_thresholds

  # By default use separate compuations for the confusion matrices since the
  # metrics might be using different thresholds (note, the underlying histogram
  # the confusion matrices are based on will still only be calculated once).
  if (num_thresholds is not None and
      num_thresholds == binary_confusion_matrices.DEFAULT_NUM_THRESHOLDS):
    name = binary_confusion_matrices.BINARY_CONFUSION_MATRICES_NAME
  else:
    name = '{}{}'.format(
        metric.name, binary_confusion_matrices.BINARY_CONFUSION_MATRICES_NAME)
    name = name if name.startswith('_') else '_' + name

  # Make sure matrices are calculated.
  computations = binary_confusion_matrices.binary_confusion_matrices(
      num_thresholds=num_thresholds,
      thresholds=thresholds,
      name=name,
      eval_config=eval_config,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key,
      aggregation_type=aggregation_type,
      class_weights=class_weights)
  matrices_key = computations[-1].keys[-1]

  def result(
      metrics: Dict[metric_types.MetricKey, Any]
  ) -> Dict[metric_types.MetricKey, Any]:
    """Returns result derived from binary confustion matrices."""
    matrices = metrics[matrices_key]

    metric = tf.keras.metrics.deserialize(metric_config)
    if (isinstance(metric, tf.keras.metrics.AUC) or
        isinstance(metric, tf.keras.metrics.SpecificityAtSensitivity) or
        isinstance(metric, tf.keras.metrics.SensitivityAtSpecificity)):
      metric.true_positives.assign(np.array(matrices.tp))
      metric.true_negatives.assign(np.array(matrices.tn))
      metric.false_positives.assign(np.array(matrices.fp))
      metric.false_negatives.assign(np.array(matrices.fn))
    elif isinstance(metric, tf.keras.metrics.Precision):
      metric.true_positives.assign(np.array(matrices.tp))
      metric.false_positives.assign(np.array(matrices.fp))
    elif isinstance(metric, tf.keras.metrics.Recall):
      metric.true_positives.assign(np.array(matrices.tp))
      metric.false_negatives.assign(np.array(matrices.fn))
    elif isinstance(metric, tf.keras.metrics.TruePositives):
      metric.accumulator.assign(np.array(matrices.tp))
    elif isinstance(metric, tf.keras.metrics.FalsePositives):
      metric.accumulator.assign(np.array(matrices.fp))
    elif isinstance(metric, tf.keras.metrics.TrueNegatives):
      metric.accumulator.assign(np.array(matrices.tn))
    elif isinstance(metric, tf.keras.metrics.FalseNegatives):
      metric.accumulator.assign(np.array(matrices.fn))
    return {key: metric.result().numpy()}

  derived_computation = metric_types.DerivedMetricComputation(
      keys=[key], result=result)
  computations.append(derived_computation)
  return computations


class _LossMetric(tf.keras.metrics.Mean):
  """Converts a loss function into a metric."""

  def __init__(self, loss, name=None, dtype=None):
    if name is None:
      name = loss.name
    super(_LossMetric, self).__init__(name=name, dtype=dtype)
    self.loss = loss

  def update_state(self, y_true, y_pred, sample_weight):
    return super(_LossMetric, self).update_state(
        self.loss(y_true, y_pred), sample_weight=sample_weight)


def _numpy_array_size_fn(array: np.ndarray) -> int:
  """Size estimator for numpy arrays."""
  return array.nbytes


class _CompilableMetricsAccumulator(object):
  """Accumulator for compilable metrics.

  Attributes:
    inputs: Accumulated batch of inputs. The inputs are stored in a
      multi-dimensional list. The first dimension is used to index the
      associated output (for single-output models this will only have one item).
      The second dimension is used to store the args passed to update_state
      (i.e. (y_true, y_pred, example_weight)). Batching is done on the last
      dimension.
    weights: Accumulated weights. The weights are stored in a multi-dimensional
      list where the first dimension is used to index the associated output (for
      single-output models this will only have one item). The second dimension
      is used to store the accumulated weights for each metric associated with
      the output dimension.
    pad: True if padding needed.
    last_dim: Max size of the last dimension of labels or predictions (used with
      padding).
    size_estimator: Batch size estimator.
    desired_batch_size: Desired batch size.
  """

  # We really want the batch size to be adaptive like it is in
  # beam.BatchElements(), but there isn't an easy way to make it so. For now
  # we will limit stored inputs to a max overall byte size.
  # TODO(b/73789023): Figure out how to make this batch size dynamic.
  _TOTAL_INPUT_BYTE_SIZE_THRESHOLD = 16 << 20  # 16MiB
  _DEFAULT_DESIRED_BATCH_SIZE = 1000

  __slots__ = [
      '_inputs', '_weights', '_pad', '_last_dim', '_size_estimator',
      '_desired_batch_size'
  ]

  def __init__(self,
               metric_counts: List[int],
               desired_batch_size: Optional[int] = None):
    """Initializes accumulator using a list of metric counts per output."""
    # Inputs have shape (num_outputs, num_metrics, num_accumulated_inputs)
    self._inputs = [
    ]  # type: List[Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]]
    # Weights have shape (num_outputs, num_metrics)
    self._weights = []  # type: List[List[Optional[np.ndarray]]]
    for output_metric_count in metric_counts:
      self._inputs.append(([], [], []))
      self._weights.append([None] * output_metric_count)
    self._pad = False
    self._last_dim = 0
    self._size_estimator = util.SizeEstimator(
        size_threshold=self._TOTAL_INPUT_BYTE_SIZE_THRESHOLD,
        size_fn=_numpy_array_size_fn)
    if desired_batch_size and desired_batch_size > 0:
      self._desired_batch_size = desired_batch_size
    else:
      self._desired_batch_size = self._DEFAULT_DESIRED_BATCH_SIZE

  def len_inputs(self) -> int:
    return len(self._inputs[0][0])

  def add_input(self, output_index: int, label: np.ndarray,
                prediction: np.ndarray, example_weight: np.ndarray):
    for i, v in enumerate((label, prediction, example_weight)):
      self._inputs[output_index][i].append(v)
      self._size_estimator.update(v)
    last_dim = max(label.shape[-1], prediction.shape[-1])
    if self._last_dim and self._last_dim != last_dim:
      self._pad = True
    self._last_dim = max(self._last_dim, last_dim)

  def get_inputs(
      self, output_index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns labels, predictions, and weights for output at given offset."""
    labels, preds, example_weights = self._inputs[output_index]
    if self._pad:

      def pad_value(a: np.array) -> Union[int, float]:
        return -1 if a.dtype.kind == 'i' else -1.0

      labels = [
          metric_util.pad(l, self._last_dim, pad_value(l)) for l in labels
      ]
      preds = [metric_util.pad(p, self._last_dim, pad_value(p)) for p in preds]
    return (np.array(labels), np.array(preds), np.array(example_weights))

  def clear_inputs(self):
    for output_index in range(len(self._inputs)):
      for i in (0, 1, 2):
        del self._inputs[output_index][i][:]
    self._size_estimator.clear()
    self._pad = False
    self._last_dim = 0

  def add_weights(self, output_index: int, metric_index: int,
                  weights: np.ndarray):
    cur_weights = self._weights[output_index][metric_index]
    if cur_weights is None:
      self._weights[output_index][metric_index] = weights
    else:
      self._weights[output_index][metric_index] = np.add(cur_weights, weights)

  def get_weights(self, output_index: int,
                  metric_index: int) -> Optional[np.ndarray]:
    return self._weights[output_index][metric_index]

  def should_flush(self) -> bool:
    return (self.len_inputs() >= self._desired_batch_size or
            self._size_estimator.should_flush())

  def get_size_estimate(self) -> int:
    return self._size_estimator.get_estimate()


class _CompilableMetricsCombiner(beam.CombineFn):
  """Combines compilable metric weights and computes result."""

  # TODO(b/173811366): Consider removing the desired_batch_size knob and
  # only use input size.
  def __init__(self,
               metric_configs: Dict[Text, List[Dict[Text, Any]]],
               loss_configs: Dict[Text, List[Dict[Text, Any]]],
               custom_objects: List[Tuple[Text, Text]],
               eval_config: Optional[config.EvalConfig],
               model_name: Optional[Text],
               sub_key: Optional[metric_types.SubKey],
               aggregation_type: Optional[metric_types.AggregationType],
               class_weights: Dict[int, float],
               desired_batch_size: Optional[int] = None):
    # Use parallel lists to store output_names and configs to guarantee
    # consistent ordering and for natural alignment with the accumulator where
    # lists are used instead of dicts for efficency.
    self._eval_config = eval_config
    self._model_name = model_name
    self._output_names = sorted(metric_configs.keys())
    self._metric_configs = [metric_configs[n] for n in self._output_names]
    self._loss_configs = [loss_configs[n] for n in self._output_names]
    self._custom_objects = custom_objects
    self._sub_key = sub_key
    self._aggregation_type = aggregation_type
    self._class_weights = class_weights
    # True if the sub_key is part of the metric config already (i.e. top_k).
    self._sub_key_in_config = sub_key and sub_key.top_k is not None
    for cfg in itertools.chain.from_iterable(metric_configs.values()):
      if _get_config_value(_TOP_K_KEY, cfg) is None:
        self._sub_key_in_config = False
        break
    self._metrics = None  # type: Dict[Text, List[tf.keras.metrics.Metric]]
    self._desired_batch_size = desired_batch_size
    self._batch_size_beam_metric = (
        beam.metrics.Metrics.distribution(
            constants.METRICS_NAMESPACE,
            'keras_compilable_metrics_combine_batch_size'))
    self._total_input_byte_size_beam_metric = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE,
        'keras_compilable_metrics_combine_batch_bytes_size')
    self._num_compacts = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, 'num_compacts')

  def _setup_if_needed(self):
    if self._metrics is None:
      self._metrics = {}
      with tf.keras.utils.custom_object_scope(
          _load_custom_objects(self._custom_objects)):
        for i, output_name in enumerate(self._output_names):
          self._metrics[output_name] = (
              _deserialize_metrics(self._metric_configs[i]))
          for loss in _deserialize_losses(self._loss_configs[i]):
            self._metrics[output_name].append(_LossMetric(loss))

  def _process_batch(self, accumulator: _CompilableMetricsAccumulator):
    self._setup_if_needed()
    if accumulator.len_inputs() == 0:
      return
    self._batch_size_beam_metric.update(accumulator.len_inputs())
    self._total_input_byte_size_beam_metric.update(
        accumulator.get_size_estimate())
    for output_index, output_name in enumerate(self._output_names):
      inputs = accumulator.get_inputs(output_index)
      for metric_index, metric in enumerate(self._metrics[output_name]):
        metric.reset_states()
        metric.update_state(*inputs)
        accumulator.add_weights(output_index, metric_index,
                                metric.get_weights())
    accumulator.clear_inputs()

  def create_accumulator(self) -> _CompilableMetricsAccumulator:
    configs = zip(self._metric_configs, self._loss_configs)
    return _CompilableMetricsAccumulator(
        [len(m) + len(l) for m, l in configs],
        desired_batch_size=self._desired_batch_size)

  def add_input(
      self, accumulator: _CompilableMetricsAccumulator,
      element: metric_types.StandardMetricInputs
  ) -> _CompilableMetricsAccumulator:
    for i, output_name in enumerate(self._output_names):
      # When micro averaging is being used, flatten should be set to True so
      # that each class is treated as though it was an independent example.
      micro_average = (
          self._aggregation_type and self._aggregation_type.micro_average)
      for label, prediction, example_weight in (
          metric_util.to_label_prediction_example_weight(
              element,
              eval_config=self._eval_config,
              model_name=self._model_name,
              output_name=output_name,
              # Skip sub_key processing if part of the keras config
              sub_key=self._sub_key if not self._sub_key_in_config else None,
              aggregation_type=self._aggregation_type,
              class_weights=self._class_weights,
              flatten=micro_average)):
        # Keras requires non-sparse keys for its calcuations.
        if self._sub_key_in_config and label.shape != prediction.shape:
          label = metric_util.one_hot(label, prediction)
        accumulator.add_input(i, label, prediction, example_weight)
    if accumulator.should_flush():
      self._process_batch(accumulator)
    return accumulator

  def merge_accumulators(
      self, accumulators: List[_CompilableMetricsAccumulator]
  ) -> _CompilableMetricsAccumulator:
    result = self.create_accumulator()
    for accumulator in accumulators:
      # Finish processing last batch
      self._process_batch(accumulator)
      # Merge the weights
      for output_index in range(len(self._output_names)):
        for metric_index in range(len(self._metric_configs[output_index])):
          weights = accumulator.get_weights(output_index, metric_index)
          if weights is None:
            # It is possible for beam to create an accumulator but pass no
            # inputs to it resulting in in empty weights. In theory all weights
            # should be empty but we check on a per metric weights basis.
            continue
          result.add_weights(output_index, metric_index, weights)
    return result

  def compact(
      self, accumulator: _CompilableMetricsAccumulator
  ) -> _CompilableMetricsAccumulator:
    self._process_batch(accumulator)
    self._num_compacts.inc(1)
    return accumulator

  def extract_output(
      self, accumulator: _CompilableMetricsAccumulator
  ) -> Dict[metric_types.MetricKey, Any]:
    self._process_batch(accumulator)
    result = {}
    for output_index, output_name in enumerate(self._output_names):
      for metric_index, metric in enumerate(self._metrics[output_name]):
        key = metric_types.MetricKey(
            name=metric.name,
            model_name=self._model_name,
            output_name=output_name,
            sub_key=self._sub_key)
        weights = accumulator.get_weights(output_index, metric_index)
        if weights is not None:
          metric.set_weights(weights)
        else:
          metric.reset_states()
        result[key] = metric.result().numpy()
    return result
