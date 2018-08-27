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
from tensorflow_model_analysis.types_compat import Any, Dict, List, Optional, Tuple, Type

from tensorflow.python.estimator.canned import prediction_keys
from tensorflow.python.ops import metrics_impl


def _export(name):
  """Decorator for exporting a _PostExportMetric class.

  The net effect of the decorator is to create a function with the given name
  that can be called to create a callback for use with add_metrics_callbacks.

  Example usage:
    @_export('my_metric')
    class _MyMetric(_PostExportMetric):

      def check_compatibility(self, features_dict: types.TensorTypeMaybeDict,
                              predictions_dict: types.TensorTypeMaybeDict,
                              labels_dict: types.TensorTypeMaybeDict) -> None:
        # check compatibility needed for the metric here.

      def get_metric_ops(self, features_dict: types.TensorTypeMaybeDict,
                         predictions_dict: types.TensorTypeMaybeDict,
                         labels_dict: types.TensorTypeMaybeDict
                        ) -> Dict[bytes, Tuple[types.TensorType,
                        types.TensorType]]:
        # metric computation here
        return {'my_metric_key': (value_op, update_op)}

    where callers can call my_metric() as a post export metric passed to
    `add_metrics_callbacks` in tfma APIs when needed.

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
      callback.populate_plots_and_pop = cls(*args,
                                            **kwargs).populate_plots_and_pop
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
    Predictions tensor, or None if none of the expected keys are found in
    the predictions_dict.
  """
  if types.is_tensor(predictions_dict):
    return predictions_dict

  key_precedence = (prediction_keys.PredictionKeys.LOGISTIC,
                    prediction_keys.PredictionKeys.PREDICTIONS,
                    prediction_keys.PredictionKeys.PROBABILITIES,
                    prediction_keys.PredictionKeys.LOGITS)
  for key in key_precedence:
    ref_tensor = predictions_dict.get(key)
    if ref_tensor is not None:
      return ref_tensor

  return None


def _check_labels(labels_dict):
  """Raise TypeError if the labels cannot be understood."""
  if not types.is_tensor(labels_dict):
    raise TypeError('labels_dict is %s, which is not a tensor' % labels_dict)


def _check_predictions(predictions_dict):
  """Raise KeyError if the predictions cannot be understood."""
  if _get_prediction_tensor(predictions_dict) is None:
    raise KeyError('cannot find any of the standard keysin predictions_dict %s.'
                   % (predictions_dict))


def _check_labels_and_predictions(
    predictions_dict,
    labels_dict):
  """Raise TypeError if the predictions and labels cannot be understood."""
  _check_predictions(predictions_dict)
  _check_labels(labels_dict)


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

    Please override the method if the metric is NOT plot type and should be
    converted into non-float type. The metric should also be popped out of
    `combined_metrics` after conversion. By default, this method does nothing.
    The metric, along with the rest metrics in `combined_metrics` will be
    converted into float values afterwards.

    Args:
      combined_metrics: The dict containing raw TFMA metrics.
      output_metrics: The dict where we convert the metrics to.
    """
    pass

  def populate_plots_and_pop(
      self, plots,
      output_plots):
    """Converts the metric in `plots` to `output_plots` and pops.

    Please override the method if the metric is plot type. The plot should also
    be popped out of `plots` after conversion.

    Args:
      plots: The dict containing raw TFMA plots.
      output_plots: The PlotData where we convert the plots to.
    """
    pass


@_export('example_count')
class _ExampleCount(_PostExportMetric):
  """Metric that counts the number of examples processed.

  We get the example count by looking at the predictions dictionary and picking
  a reference Tensor. If we can find a standard key (e.g.
  PredictionKeys.LOGISTIC, etc), we use that as the reference Tensor. Otherwise,
  we just use the first key in sorted order from one of the dictionaries
  (predictions, labels) as the reference Tensor.

  We assume the first dimension is the batch size, and take that to be the
  number of examples in the batch.
  """

  def check_compatibility(self, features_dict,
                          predictions_dict,
                          labels_dict):
    pass

  def get_metric_ops(self, features_dict,
                     predictions_dict,
                     labels_dict
                    ):
    ref_tensor = _get_prediction_tensor(predictions_dict)
    if ref_tensor is None:
      # Note that if predictions_dict is a Tensor and not a dict,
      # get_predictions_tensor will return predictions_dict, so if we get
      # here, if means that predictions_dict is a dict without any of the
      # standard keys.
      #
      # If we can't get any of standard keys, then pick the first key
      # in alphabetical order if the predictions dict is non-empty.
      # If the predictions dict is empty, try the labels dict.
      # If that is empty too, default to the empty Tensor.
      tf.logging.info(
          'ExampleCount post export metric: could not find any of '
          'the standard keys in predictions_dict (keys were: %s)',
          predictions_dict.keys())
      if predictions_dict is not None and predictions_dict.keys():
        first_key = sorted(predictions_dict.keys())[0]
        ref_tensor = predictions_dict[first_key]
        tf.logging.info('Using the first key from predictions_dict: %s',
                        first_key)
      elif labels_dict is not None:
        if types.is_tensor(labels_dict):
          ref_tensor = labels_dict
          tf.logging.info('Using the labels Tensor')
        elif labels_dict.keys():
          first_key = sorted(labels_dict.keys())[0]
          ref_tensor = labels_dict[first_key]
          tf.logging.info('Using the first key from labels_dict: %s', first_key)

      if ref_tensor is None:
        tf.logging.info('Could not find a reference Tensor for example count. '
                        'Defaulting to the empty Tensor.')
        ref_tensor = tf.constant([])

    return {metric_keys.EXAMPLE_COUNT: metrics.total(tf.shape(ref_tensor)[0])}


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

  def populate_plots_and_pop(
      self, plots,
      output_plots):
    matrices = plots.pop(metric_keys.CALIBRATION_PLOT_MATRICES)
    boundaries = plots.pop(metric_keys.CALIBRATION_PLOT_BOUNDARIES)
    if len(matrices) != len(boundaries) + 1:
      raise ValueError(
          'len(matrices) should be equal to len(boundaries) + 1, but lengths '
          'were len(matrices)=%d and len(boundaries)=%d instead' %
          (len(matrices), len(boundaries)))

    for matrix_row, lower_threshold, upper_threshold in zip(
        matrices, [float('-inf')] + list(boundaries),
        list(boundaries) + [float('inf')]):
      total_pred, total_label, total_weight = matrix_row
      output_plots.calibration_histogram_buckets.buckets.add(
          lower_threshold_inclusive=lower_threshold,
          upper_threshold_exclusive=upper_threshold,
          total_weighted_refined_prediction={
              'value': total_pred,
          },
          total_weighted_label={
              'value': total_label,
          },
          num_weighted_examples={
              'value': total_weight,
          },
      )


def _confusion_matrix_metric_ops(
    features_dict,
    predictions_dict,
    labels_dict,
    example_weight_key,
    thresholds,
):
  """Metric ops for computing confusion matrix at the given thresholds.

  This is factored out because it's common to AucPlots and
  ConfusionMatrixAtThresholds.

  Args:
    features_dict: Features dict.
    predictions_dict: Predictions dict.
    labels_dict: Labels dict.
    example_weight_key: Example weight key (into features_dict).
    thresholds: List of thresholds to compute the confusion matrix at.

  Returns:
    (value_op, update_op) for the metric. Note that the value_op produces a
    matrix as described in the comments below.
  """
  # Note that we have to squeeze predictions, labels, weights so they are all
  # N element vectors (otherwise some of them might be N x 1 tensors, and
  # multiplying a N element vector with a N x 1 tensor uses matrix
  # multiplication rather than element-wise multiplication).
  squeezed_weights = None
  if example_weight_key:
    squeezed_weights = tf.squeeze(features_dict[example_weight_key])
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

  return (value_op, update_op)


@_export('confusion_matrix_at_thresholds')
class _ConfusionMatrixAtThresholds(_PostExportMetric):
  """Confusion matrix at threhsolds."""

  def __init__(self,
               thresholds,
               example_weight_key = None):
    """Create a metric that computes the confusion matrix at given thresholds.

    Predictions should be one of:
      (a) a single float in [0, 1]
      (b) a dict containing the LOGISTIC key
      (c) a dict containing the PREDICTIONS key, where the prediction is
          in [0, 1]

    Label should be a single float that is either exactly 0 or exactly 1
    (soft labels, i.e. labels between 0 and 1 are *not* supported).

    Args:
      thresholds: List of thresholds to compute the confusion matrix at.
      example_weight_key: The key of the example weight column in the
        features dict. If None, all predictions are given a weight of 1.0.
    """
    self._example_weight_key = example_weight_key
    self._thresholds = sorted(thresholds)

  def check_compatibility(self, features_dict,
                          predictions_dict,
                          labels_dict):
    _check_weight_present(features_dict, self._example_weight_key)
    _check_labels_and_predictions(predictions_dict, labels_dict)

  def get_metric_ops(self, features_dict,
                     predictions_dict,
                     labels_dict
                    ):
    value_op, update_op = _confusion_matrix_metric_ops(
        features_dict, predictions_dict, labels_dict, self._example_weight_key,
        self._thresholds)
    return {
        metric_keys.CONFUSION_MATRIX_AT_THRESHOLDS_MATRICES: (value_op,
                                                              update_op),
        metric_keys.CONFUSION_MATRIX_AT_THRESHOLDS_THRESHOLDS: (tf.identity(
            self._thresholds), tf.no_op()),
    }

  def populate_stats_and_pop(
      self, combine_metrics,
      output_metrics):
    matrices = combine_metrics.pop(
        metric_keys.CONFUSION_MATRIX_AT_THRESHOLDS_MATRICES)
    thresholds = combine_metrics.pop(
        metric_keys.CONFUSION_MATRIX_AT_THRESHOLDS_THRESHOLDS)

    # We assume that thresholds are already sorted.
    if len(matrices) != len(thresholds):
      raise ValueError(
          'matrices should have the same length as thresholds, but lengths '
          'were: matrices: %d, thresholds: %d' % (len(matrices),
                                                  len(thresholds)))

    for threshold, matrix in zip(thresholds, matrices):
      output_matrix = (
          output_metrics[metric_keys.CONFUSION_MATRIX_AT_THRESHOLDS]
          .confusion_matrix_at_thresholds.matrices.add())
      output_matrix.threshold = threshold
      output_matrix.false_negatives = matrix[0]
      output_matrix.true_negatives = matrix[1]
      output_matrix.false_positives = matrix[2]
      output_matrix.true_positives = matrix[3]
      # +inf, -inf, and NaNs will all get stored correctly.
      output_matrix.precision = matrix[4]
      output_matrix.recall = matrix[5]


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
    thresholds = [
        i * 1.0 / self._num_buckets for i in range(0, self._num_buckets + 1)
    ]
    thresholds = [-1e-6] + thresholds
    value_op, update_op = _confusion_matrix_metric_ops(
        features_dict, predictions_dict, labels_dict, self._example_weight_key,
        thresholds)
    return {
        metric_keys.AUC_PLOTS_MATRICES: (value_op, update_op),
        metric_keys.AUC_PLOTS_THRESHOLDS: (tf.identity(thresholds), tf.no_op()),
    }

  def populate_plots_and_pop(
      self, plots,
      output_plots):
    matrices = plots.pop(metric_keys.AUC_PLOTS_MATRICES)
    thresholds = plots.pop(metric_keys.AUC_PLOTS_THRESHOLDS)
    if len(matrices) != len(thresholds):
      raise ValueError(
          'len(matrices) should be equal to len(thresholds), but lengths were '
          'len(matrices)=%d and len(thresholds)=%d instead' % (len(matrices),
                                                               len(thresholds)))
    for matrix_row, threshold in zip(matrices, list(thresholds)):
      matrix = output_plots.confusion_matrix_at_thresholds.matrices.add()
      matrix.threshold = threshold
      # The column indices need to match _confusion_matrix_metric_ops output.
      (matrix.false_negatives, matrix.true_negatives, matrix.false_positives,
       matrix.true_positives, matrix.precision, matrix.recall) = matrix_row


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


@_export('precision_recall_at_k')
class _PrecisionRecallAtK(_PostExportMetric):
  """Metric that computes precision and recall at K for classification models.

  Create a metric that computes precision and recall at K.

  Predictions should be a dict containing the CLASSES key and PROBABILITIES
  keys. Predictions should have the same size for all examples. The model
  should NOT, for instance, produce 2 classes on one example and 4 classes on
  another example.

  Labels should be a string Tensor, or a SparseTensor whose dense form is
  a string Tensor whose entries are the corresponding labels. Note that the
  values of the CLASSES in the predictions and that of labels will be compared
  directly, so they should come from the "same vocabulary", so if predictions
  are class IDs, then labels should be class IDs, and so on.
  """

  def __init__(self,
               cutoffs,
               example_weight_key = None):
    """Creates a metric that computes the precision and recall at `k`.

    Args:
      cutoffs: List of `k` values at which to compute the precision and recall.
        Use a value of `k` = 0 to indicate that all predictions should be
        considered.
      example_weight_key: The optional key of the example weight column in the
        features_dict. If not given, all examples will be assumed to have
        a weight of 1.0.
    """
    self._cutoffs = cutoffs
    self._example_weight_key = example_weight_key

  def check_compatibility(self, features_dict,
                          predictions_dict,
                          labels_dict):
    if not isinstance(predictions_dict, dict):
      raise TypeError('predictions_dict should be a dict. predictions_dict '
                      'was: %s' % predictions_dict)
    if prediction_keys.PredictionKeys.CLASSES not in predictions_dict:
      raise KeyError('predictions_dict should contain PredictionKeys.CLASSES. '
                     'predictions_dict was: %s' % predictions_dict)
    if prediction_keys.PredictionKeys.PROBABILITIES not in predictions_dict:
      raise KeyError('predictions_dict should contain '
                     'PredictionKeys.PROBABILITIES. predictions_dict was: %s' %
                     predictions_dict)
    if not types.is_tensor(labels_dict):
      raise TypeError(
          'labels_dict should be a tensor. labels_dict was: %s' % labels_dict)
    _check_weight_present(features_dict, self._example_weight_key)

  def get_metric_ops(self, features_dict,
                     predictions_dict,
                     labels_dict
                    ):

    squeezed_weights = None
    if self._example_weight_key:
      squeezed_weights = tf.squeeze(features_dict[self._example_weight_key])

    labels = labels_dict
    if isinstance(labels_dict, tf.SparseTensor):
      labels = tf.sparse_tensor_to_dense(labels_dict, default_value='')

    classes = predictions_dict[prediction_keys.PredictionKeys.CLASSES]
    scores = predictions_dict[prediction_keys.PredictionKeys.PROBABILITIES]

    return {
        metric_keys.PRECISION_RECALL_AT_K:
            metrics.precision_recall_at_k(classes, scores, labels,
                                          self._cutoffs, squeezed_weights)
    }

  def populate_stats_and_pop(
      self, combine_metrics,
      output_metrics):
    table = combine_metrics.pop(metric_keys.PRECISION_RECALL_AT_K)
    cutoff_column = table[:, 0]
    precision_column = table[:, 1]
    recall_column = table[:, 2]
    for cutoff, precision, recall in zip(cutoff_column, precision_column,
                                         recall_column):
      row = output_metrics[
          metric_keys.PRECISION_AT_K].value_at_cutoffs.values.add()
      row.cutoff = int(cutoff)
      row.value = precision

      row = output_metrics[
          metric_keys.RECALL_AT_K].value_at_cutoffs.values.add()
      row.cutoff = int(cutoff)
      row.value = recall
