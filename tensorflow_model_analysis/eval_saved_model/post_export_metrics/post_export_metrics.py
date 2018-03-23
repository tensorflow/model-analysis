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
from tensorflow_model_analysis.types_compat import Dict, Optional, Tuple, Type

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
        if not metric.is_compatible(features_dict, predictions_dict,
                                    labels_dict):
          return {}
        metric_ops = {}
        for key, value in (metric.get_metric_ops(
            features_dict, predictions_dict, labels_dict).iteritems()):
          metric_ops[key] = value
        return metric_ops

      # We store the metric's export name in the .name property of the callback.
      callback.name = name
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


class _PostExportMetric(object):
  """Abstract base class for post export metrics."""

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def is_compatible(self, features_dict,
                    predictions_dict,
                    labels_dict):
    """Returns whether this metric is compatible with the model.

    This function should make this determination based on the features,
    predictions and labels dict.

    Args:
      features_dict: Dictionary containing references to the features Tensors
        for the model.
      predictions_dict: Dictionary containing references to the predictions
        Tensors for the model.
      labels_dict: Dictionary containing references to the labels Tensors for
        the model.

    Returns:
      True if the metric is compatible with the model, False otherwise.
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


@_export('example_count')
class _ExampleCount(_PostExportMetric):
  """Metric that counts the number of examples processed."""

  def is_compatible(self, features_dict,
                    predictions_dict,
                    labels_dict):
    if (types.is_tensor(predictions_dict) or
        prediction_keys.PredictionKeys.LOGISTIC in predictions_dict or
        prediction_keys.PredictionKeys.PREDICTIONS in predictions_dict):
      return True

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

  def is_compatible(self, features_dict,
                    predictions_dict,
                    labels_dict):
    if self._example_weight_key not in features_dict:
      raise ValueError('example weight key %s not found in features_dict. '
                       'features were: %s' % (self._example_weight_key,
                                              features_dict.keys()))
    return True

  def get_metric_ops(self, features_dict,
                     predictions_dict,
                     labels_dict
                    ):
    value = features_dict[self._example_weight_key]
    return {metric_keys.EXAMPLE_WEIGHT: metrics.total(value)}


@_export('calibration_plot_and_prediction_histogram')
class _CalibrationPlotAndPredictionHistogram(_PostExportMetric):
  """Plot metric for calibration plot and prediction histogram.

  Note that this metric is only applicable to models for which the predictions
  and labels are in [0, 1].

  The plot contains uniformly-sized buckets for predictions in [0, 1],
  and additional buckets for predictions less than 0 and greater than 1 at the
  ends.
  """

  _PLOT_NUM_BUCKETS = 10000

  def __init__(self, example_weight_key = None):
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
    """
    self._example_weight_key = example_weight_key

  def is_compatible(self, features_dict,
                    predictions_dict,
                    labels_dict):
    if (self._example_weight_key and
        self._example_weight_key not in features_dict):
      raise ValueError('example weight key %s not found in features_dict. '
                       'features were: %s' % (self._example_weight_key,
                                              features_dict.keys()))
    prediction_ok = False
    label_ok = False
    if (types.is_tensor(predictions_dict) or
        prediction_keys.PredictionKeys.LOGISTIC in predictions_dict or
        prediction_keys.PredictionKeys.PREDICTIONS in predictions_dict):
      prediction_ok = True
    if types.is_tensor(labels_dict):
      label_ok = True
    return prediction_ok and label_ok

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
                num_buckets=self._PLOT_NUM_BUCKETS,
                weights=squeezed_weights),
        metric_keys.CALIBRATION_PLOT_BOUNDARIES: (
            tf.range(0.0, self._PLOT_NUM_BUCKETS + 1) / self._PLOT_NUM_BUCKETS,
            tf.no_op()),
    }


@_export('auc_plots')
class _AucPlots(_PostExportMetric):
  """Plot metric for AUROC and AUPRC for predictions in [0, 1]."""

  _HISTOGRAM_NUM_BUCKETS = 10000

  def __init__(self, example_weight_key = None):
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
    """
    self._example_weight_key = example_weight_key

  def is_compatible(self, features_dict,
                    predictions_dict,
                    labels_dict):
    if (self._example_weight_key and
        self._example_weight_key not in features_dict):
      raise ValueError('example weight key %s not found in features_dict. '
                       'features were: %s' % (self._example_weight_key,
                                              features_dict.keys()))
    if (types.is_tensor(predictions_dict) or
        prediction_keys.PredictionKeys.LOGISTIC in predictions_dict or
        prediction_keys.PredictionKeys.PREDICTIONS in predictions_dict):
      if types.is_tensor(labels_dict):
        return True
    return False

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
        i * 1.0 / self._HISTOGRAM_NUM_BUCKETS
        for i in range(0, self._HISTOGRAM_NUM_BUCKETS + 1)
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
