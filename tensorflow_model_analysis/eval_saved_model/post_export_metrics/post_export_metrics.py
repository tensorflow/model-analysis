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
"""Library containing helpers for adding post export metrics for evaluation.

These post export metrics can be included in the add_post_export_metrics
parameter of Evaluate to compute them.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import abc

import tensorflow as tf
from tensorflow_model_analysis import types
from tensorflow_model_analysis.eval_saved_model.post_export_metrics import metric_keys
from tensorflow_model_analysis.eval_saved_model.post_export_metrics import metrics
from tensorflow_model_analysis.proto import metrics_for_slice_pb2
from tensorflow_model_analysis.types_compat import Any, Dict, Optional, Tuple, Type

from tensorflow.python.estimator.canned import prediction_keys
from tensorflow.python.ops import metrics_impl


def _export(name):
  """Decorator for exporting a post export metric class.

  The net effect of the decorator is to create a function with the given name
  that can be called to create a callback for use with add_metrics_callbacks.

  Args:
    name: Name of the exported function.

  Returns:
    Decorator for exporting a post export metric class.
  """

  def _actual_export(cls):
    """This is the actual decorator."""

    def fn(*args, **kwargs):
      """This is the function that the user calls."""

      def callback(features_dict,
                   predictions_dict,
                   labels_dict):
        """This actual callback that goes into add_metrics_callbacks."""
        metric = cls(*args, **kwargs)
        metric.check_compatibility(features_dict, predictions_dict, labels_dict)
        metric_ops = {}
        for key, value in (metric.get_metric_ops(
            features_dict, predictions_dict, labels_dict).iteritems()):
          metric_ops[key] = value
        return metric_ops

      # We store the metric's export name in the .name property of the callback.
      callback.name = name
      callback.populate_stats_and_pop = cls(*args,
                                            **kwargs).populate_stats_and_pop
      return callback

    globals()[name] = fn
    return cls

  return _actual_export


def _get_prediction_tensor(
    predictions_dict):
  """Returns prediction Tensor for a specific Estimators.

  Returns the prediction Tensor for some regression Estimators.

  Args:
    predictions_dict: Predictions dictionary.

  Returns:
    Predictions tensor.
  """
  if types.is_tensor(predictions_dict):
    ref_tensor = predictions_dict
  else:
    ref_tensor = predictions_dict.get(prediction_keys.PredictionKeys.LOGISTIC)
    if ref_tensor is None:
      ref_tensor = predictions_dict.get(
          prediction_keys.PredictionKeys.PREDICTIONS)
  return ref_tensor


def _check_labels_and_predictions(
    predictions_dict,
    labels_dict):
  """Raise TypeError if the predictions and labels cannot be understood."""
  if not (types.is_tensor(predictions_dict) or
          prediction_keys.PredictionKeys.LOGISTIC in predictions_dict or
          prediction_keys.PredictionKeys.PREDICTIONS in predictions_dict):
    raise TypeError(
        'cannot find predictions in %s. It is expected that either'
        'predictions_dict is a tensor or it contains PredictionKeys.LOGISTIC'
        'or PredictionKeys.PREDICTIONS.' % predictions_dict)

  if not types.is_tensor(labels_dict):
    raise TypeError('labels_dict is %s, which is not a tensor' % labels_dict)


def _check_weight_present(features_dict,
                          example_weight_key = None):
  """Raise ValueError if the example weight is not present."""
  if (example_weight_key is not None and
      example_weight_key not in features_dict):
    raise ValueError(
        'example weight key %s not found in features_dict. '
        'features were: %s' % (example_weight_key, features_dict.keys()))


def _populate_to_bounded_value_and_pop(
    combined_metrics,
    output_metrics,
    metric_key):
  """Converts the given metric to bounded_value type in dict `output_metrics`.

  The metric to be converted should be in the dict `combined_metrics` with key
  as `metric_key`. The `combined_metrics` should also contain
  metric_keys.lower_bound(metric_key) and metric_keys.upper_bound(metric_key)
  which store the lower_bound and upper_bound of that metric. The result will be
  stored as bounded_value type in dict `output_metrics`. After the conversion,
  the metric will be poped out from the `combined_metrics`.

  Args:
    combined_metrics: The dict containing raw TFMA metrics.
    output_metrics: The dict where we convert the metrics to.
    metric_key: The key in the dict `metircs` for extracting the metric value.
  """
  output_metrics[
      metric_key].bounded_value.lower_bound.value = combined_metrics.pop(
          metric_keys.lower_bound(metric_key))
  output_metrics[
      metric_key].bounded_value.upper_bound.value = combined_metrics.pop(
          metric_keys.upper_bound(metric_key))
  output_metrics[metric_key].bounded_value.value.value = combined_metrics.pop(
      metric_key)


class _PostExportMetric(object):
  """Abstract base class for post export metrics."""

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def check_compatibility(self, features_dict,
                          predictions_dict,
                          labels_dict):
    """Checks whether this metric is compatible with the model.

    This function should make this determination based on the features,
    predictions and labels dict. It should raise an Exception if the metric is
    not compatible with the model.

    Args:
      features_dict: Dictionary containing references to the features Tensors
        for the model.
      predictions_dict: Dictionary containing references to the predictions
        Tensors for the model.
      labels_dict: Dictionary containing references to the labels Tensors for
        the model.

    Raises:
      Exception if the metric is not compatible with the model.
    """
    raise NotImplementedError('not implemented')

  @abc.abstractmethod
  def get_metric_ops(self, features_dict,
                     predictions_dict,
                     labels_dict
                    ):
    """Returns the metric_ops entry for this metric.

    Note that the metric will be added to metric_ops via
    metric_ops.update(metric.get_metric_ops()).

    Args:
      features_dict: Dictionary containing references to the features Tensors
        for the model.
      predictions_dict: Dictionary containing references to the predictions
        Tensors for the model.
      labels_dict: Dictionary containing references to the labels Tensors for
        the model.

    Returns:
      A metric op dictionary,
      i.e. a dictionary[metric_name] = (value_op, update_op) containing all
      the metrics and ops for this metric.
    """
    raise NotImplementedError('not implemented')

  def populate_stats_and_pop(
      self, combined_metrics,
      output_metrics):
    """Converts the metric in `combined_metrics` to `output_metrics` and pops.

    Please override the method if the metric should be converted into non-float
    type. The metric should also be poped up from `combined_metrics` after
    conversion. By default, this method does nothing. The metric, along with the
    rest metrics in `combined_metrics` will be converted into float values
    afterwards.

    Args:
      combined_metrics: The dict containing raw TFMA metrics.
      output_metrics: The dict where we convert the metrics to.
    """
    pass


@_export('example_count')
class _ExampleCount(_PostExportMetric):
  """Metric that counts the number of examples processed."""

  def check_compatibility(self, features_dict,
                          predictions_dict,
                          labels_dict):
    _check_labels_and_predictions(predictions_dict, labels_dict)

  def get_metric_ops(self, features_dict,
                     predictions_dict,
                     labels_dict
                    ):
    ref_tensor = _get_prediction_tensor(predictions_dict)
    return {metric_keys.EXAMPLE_COUNT: tf.contrib.metrics.count(ref_tensor)}


@_export('example_weight')
class _ExampleWeight(_PostExportMetric):
  """Metric that computes the sum of example weights."""

  def __init__(self, example_weight_key):
    """Create a metric that computes the sum of example weights.

    Args:
      example_weight_key: The key of the example weight column in the
        features_dict.
    """
    self._example_weight_key = example_weight_key

  def check_compatibility(self, features_dict,
                          predictions_dict,
                          labels_dict):
    _check_weight_present(features_dict, self._example_weight_key)

  def get_metric_ops(self, features_dict,
                     predictions_dict,
                     labels_dict
                    ):
    value = features_dict[self._example_weight_key]
    return {metric_keys.EXAMPLE_WEIGHT: metrics.total(value)}


_DEFAULT_NUM_BUCKETS = 10000


@_export('calibration_plot_and_prediction_histogram')
class _CalibrationPlotAndPredictionHistogram(_PostExportMetric):
  """Plot metric for calibration plot and prediction histogram.

  Note that this metric is only applicable to models for which the predictions
  and labels are in [0, 1].

  The plot contains uniformly-sized buckets for predictions in [0, 1],
  and additional buckets for predictions less than 0 and greater than 1 at the
  ends.
  """

  def __init__(self,
               example_weight_key = None,
               num_buckets = _DEFAULT_NUM_BUCKETS):
    """Create a plot metric for calibration plot and prediction histogram.

    Predictions should be one of:
      (a) a single float in [0, 1]
      (b) a dict containing the LOGISTIC key
      (c) a dict containing the PREDICTIONS key, where the prediction is
          in [0, 1]

    Label should be a single float that is in [0, 1].

    Args:
      example_weight_key: The key of the example weight column in the
        features dict. If None, all predictions are given a weight of 1.0.
      num_buckets: The number of buckets used for the plot.
    """
    self._example_weight_key = example_weight_key
    self._num_buckets = num_buckets

  def check_compatibility(self, features_dict,
                          predictions_dict,
                          labels_dict):
    _check_weight_present(features_dict, self._example_weight_key)
    _check_labels_and_predictions(predictions_dict, labels_dict)

  def get_metric_ops(self, features_dict,
                     predictions_dict,
                     labels_dict
                    ):
    # Note that we have to squeeze predictions, labels, weights so they are all
    # N element vectors (otherwise some of them might be N x 1 tensors, and
    # multiplying a N element vector with a N x 1 tensor uses matrix
    # multiplication rather than element-wise multiplication).
    squeezed_weights = None
    if self._example_weight_key:
      squeezed_weights = tf.squeeze(features_dict[self._example_weight_key])
    prediction_tensor = _get_prediction_tensor(predictions_dict)
    return {
        metric_keys.CALIBRATION_PLOT_MATRICES:
            metrics.calibration_plot(
                predictions=tf.squeeze(prediction_tensor),
                labels=tf.squeeze(labels_dict),
                left=0.0,
                right=1.0,
                num_buckets=self._num_buckets,
                weights=squeezed_weights),
        metric_keys.CALIBRATION_PLOT_BOUNDARIES: (
            tf.range(0.0, self._num_buckets + 1) / self._num_buckets,
            tf.no_op()),
    }


@_export('auc_plots')
class _AucPlots(_PostExportMetric):
  """Plot metric for AUROC and AUPRC for predictions in [0, 1]."""

  def __init__(self,
               example_weight_key = None,
               num_buckets = _DEFAULT_NUM_BUCKETS):
    """Create a plot metric for AUROC and AUPRC.

    Predictions should be one of:
      (a) a single float in [0, 1]
      (b) a dict containing the LOGISTIC key
      (c) a dict containing the PREDICTIONS key, where the prediction is
          in [0, 1]

    Label should be a single float that is either exactly 0 or exactly 1
    (soft labels, i.e. labels between 0 and 1 are *not* supported).

    Args:
      example_weight_key: The key of the example weight column in the
        features dict. If None, all predictions are given a weight of 1.0.
      num_buckets: The number of buckets used for plot.
    """
    self._example_weight_key = example_weight_key
    self._num_buckets = num_buckets

  def check_compatibility(self, features_dict,
                          predictions_dict,
                          labels_dict):
    _check_weight_present(features_dict, self._example_weight_key)
    _check_labels_and_predictions(predictions_dict, labels_dict)

  def get_metric_ops(self, features_dict,
                     predictions_dict,
                     labels_dict
                    ):
    # Note that we have to squeeze predictions, labels, weights so they are all
    # N element vectors (otherwise some of them might be N x 1 tensors, and
    # multiplying a N element vector with a N x 1 tensor uses matrix
    # multiplication rather than element-wise multiplication).
    squeezed_weights = None
    if self._example_weight_key:
      squeezed_weights = tf.squeeze(features_dict[self._example_weight_key])
    thresholds = [
        i * 1.0 / self._num_buckets for i in range(0, self._num_buckets + 1)
    ]
    thresholds = [-1e-6] + thresholds
    prediction_tensor = tf.cast(
        _get_prediction_tensor(predictions_dict), tf.float64)
    values, update_ops = metrics_impl._confusion_matrix_at_thresholds(  # pylint: disable=protected-access
        tf.squeeze(labels_dict), tf.squeeze(prediction_tensor), thresholds,
        squeezed_weights)

    values['precision'] = values['tp'] / (values['tp'] + values['fp'])
    values['recall'] = values['tp'] / (values['tp'] + values['fn'])

    # The final matrix will look like the following:
    #
    # [ fn@threshold_0 tn@threshold_0 ... recall@threshold_0 ]
    # [ fn@threshold_1 tn@threshold_1 ... recall@threshold_1 ]
    # [       :              :        ...         :          ]
    # [       :              :        ...         :          ]
    # [ fn@threshold_k tn@threshold_k ... recall@threshold_k ]
    #
    value_op = tf.transpose(
        tf.stack([
            values['fn'], values['tn'], values['fp'], values['tp'],
            values['precision'], values['recall']
        ]))
    update_op = tf.group(update_ops['fn'], update_ops['tn'], update_ops['fp'],
                         update_ops['tp'])

    return {
        metric_keys.AUC_PLOTS_MATRICES: (value_op, update_op),
        metric_keys.AUC_PLOTS_THRESHOLDS: (tf.identity(thresholds), tf.no_op()),
    }


@_export('auc')
class _Auc(_PostExportMetric):
  """Metric that computes bounded AUC or AUPRC for predictions in [0, 1].

  This calls tf.metrics.auc to do the computation with 10000 buckets instead of
  the default (200) for more precision. We use 'careful_interpolation' summation
  for the metric value, and also 'minoring' and 'majoring' to generate the
  boundaries for the metric.
  """

  def __init__(self,
               example_weight_key = None,
               curve='ROC',
               num_buckets = _DEFAULT_NUM_BUCKETS):
    """Create a metric that computes bounded AUROC or AUPRC.

    Predictions should be one of:
      (a) a single float in [0, 1]
      (b) a dict containing the LOGISTIC key
      (c) a dict containing the PREDICTIONS key, where the prediction is
          in [0, 1]

    Label should be a single float that is either exactly 0 or exactly 1
    (soft labels, i.e. labels between 0 and 1 are *not* supported).

    Args:
      example_weight_key: The key of the example weight column in the features
        dict. If None, all predictions are given a weight of 1.0.
      curve: Specifies the name of the curve to be computed, 'ROC' [default] or
        'PR' for the Precision-Recall-curve. It will be passed to
        tf.metrics.auc() directly.
      num_buckets: The number of buckets used for the curve. (num_buckets + 1)
        is used as the num_thresholds in tf.metrics.auc().
    Raises:
      ValueError: if the curve is neither 'ROC' nor 'PR'.
    """
    self._example_weight_key = example_weight_key
    self._curve = curve
    self._num_buckets = num_buckets

    if curve == 'ROC':
      self._metric_name = metric_keys.AUC
    elif curve == 'PR':
      self._metric_name = metric_keys.AUPRC
    else:
      raise ValueError('got unsupported curve: %s' % curve)

  def check_compatibility(self, features_dict,
                          predictions_dict,
                          labels_dict):
    _check_weight_present(features_dict, self._example_weight_key)
    _check_labels_and_predictions(predictions_dict, labels_dict)

  def get_metric_ops(self, features_dict,
                     predictions_dict,
                     labels_dict
                    ):
    # Note that we have to squeeze predictions, labels, weights so they are all
    # N element vectors (otherwise some of them might be N x 1 tensors, and
    # multiplying a N element vector with a N x 1 tensor uses matrix
    # multiplication rather than element-wise multiplication).
    weights = None
    if self._example_weight_key:
      weights = tf.squeeze(features_dict[self._example_weight_key])
    predictions = tf.squeeze(
        tf.cast(_get_prediction_tensor(predictions_dict), tf.float64))
    labels = tf.squeeze(labels_dict)

    value_ops, value_update = tf.metrics.auc(
        labels=labels,
        predictions=predictions,
        weights=weights,
        num_thresholds=self._num_buckets + 1,
        curve=self._curve,
        summation_method='careful_interpolation')
    lower_bound_ops, lower_bound_update = tf.metrics.auc(
        labels=labels,
        predictions=predictions,
        weights=weights,
        num_thresholds=self._num_buckets + 1,
        curve=self._curve,
        summation_method='minoring')
    upper_bound_ops, upper_bound_update = tf.metrics.auc(
        labels=labels,
        predictions=predictions,
        weights=weights,
        num_thresholds=self._num_buckets + 1,
        curve=self._curve,
        summation_method='majoring')

    return {
        self._metric_name: (value_ops, value_update),
        metric_keys.lower_bound(self._metric_name): (lower_bound_ops,
                                                     lower_bound_update),
        metric_keys.upper_bound(self._metric_name): (upper_bound_ops,
                                                     upper_bound_update),
    }

  def populate_stats_and_pop(
      self, combine_metrics,
      output_metrics):
    _populate_to_bounded_value_and_pop(combine_metrics, output_metrics,
                                       self._metric_name)
