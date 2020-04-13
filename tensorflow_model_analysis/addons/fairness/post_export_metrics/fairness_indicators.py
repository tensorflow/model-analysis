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
# Standard __future__ imports
from __future__ import print_function

# Standard Imports
import tensorflow as tf
from tensorflow_model_analysis import types
from tensorflow_model_analysis.post_export_metrics import metric_keys
from tensorflow_model_analysis.post_export_metrics import post_export_metrics
from tensorflow_model_analysis.proto import metrics_for_slice_pb2 as metrics_pb2
from tensorflow_model_analysis.slicer import slicer_lib as slicer
from typing import Any, Dict, List, Optional, Text, Tuple


# pylint: disable=protected-access
@post_export_metrics._export('fairness_indicators')
class _FairnessIndicators(post_export_metrics._ConfusionMatrixBasedMetric):
  """Metrics that can be used to evaluate the following fairness metrics.

    * Demographic Parity or Equality of Outcomes.
      For each slice measure the Positive* Rate, or the percentage of all
      examples receiving positive scores.
    * Equality of Opportunity
      Equality of Opportunity attempts to match the True Positive* rate
      (aka recall) of different data slices.
    * Equality of Odds
      In addition to looking at Equality of Opportunity, looks at equalizing the
      False Positive* rates of slices as well.

  The choice to focus on these metrics as a starting point is based primarily on
  the paper Equality of Opportunity in Supervised Learning and the excellent
  visualization created as a companion to the paper.

  https://arxiv.org/abs/1610.02413
  http://research.google.com/bigpicture/attacking-discrimination-in-ml/

  * Note that these fairness formulations assume that a positive prediction is
  associated with a positive outcome for the user--in certain contexts such as
  abuse, positive predictions translate to non-opportunity. You may want to use
  the provided negative rates for comparison instead.
  """

  _thresholds = ...  # type: List[float]
  _example_weight_key = ...  # type: Text
  _labels_key = ...  # type: Text
  _metric_tag = None  # type: Text

  # We could use the same keys as the ConfusionMatrix metrics, but with the way
  # that post_export_metrics are currently implemented, if both
  # post_export_metrics were specified we would pop the matrices/thresholds in
  # the first call, and have issues with the second.
  thresholds_key = metric_keys.FAIRNESS_CONFUSION_MATRIX_THESHOLDS
  matrices_key = metric_keys.FAIRNESS_CONFUSION_MATRIX_MATRICES

  def __init__(self,
               thresholds: Optional[List[float]] = None,
               example_weight_key: Optional[Text] = None,
               target_prediction_keys: Optional[List[Text]] = None,
               labels_key: Optional[Text] = None,
               metric_tag: Optional[Text] = None,
               tensor_index: Optional[int] = None) -> None:
    if not thresholds:
      thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Determine the number of threshold digits to display as part of the metric
    # key. We want lower numbers for readability, but allow differentiation
    # between close thresholds.
    self._key_digits = 2
    for t in thresholds:
      if len(str(t)) - 2 > self._key_digits:
        self._key_digits = len(str(t)) - 2

    super(_FairnessIndicators, self).__init__(
        thresholds,
        example_weight_key,
        target_prediction_keys,
        labels_key,
        metric_tag,
        tensor_index=tensor_index)

  def get_metric_ops(self, features_dict: types.TensorTypeMaybeDict,
                     predictions_dict: types.TensorTypeMaybeDict,
                     labels_dict: types.TensorTypeMaybeDict
                    ) -> Dict[Text, Tuple[types.TensorType, types.TensorType]]:

    values, update_ops = self.confusion_matrix_metric_ops(
        features_dict, predictions_dict, labels_dict)
    # True positive rate is computed by confusion_matrix_metric_ops as 'recall'.
    # pytype: disable=unsupported-operands
    values['tnr'] = tf.math.divide_no_nan(values['tn'],
                                          values['tn'] + values['fp'])
    values['fpr'] = tf.math.divide_no_nan(values['fp'],
                                          values['fp'] + values['tn'])
    values['positive_rate'] = tf.math.divide_no_nan(
        values['tp'] + values['fp'],
        values['tp'] + values['fp'] + values['tn'] + values['fn'])
    values['fnr'] = tf.math.divide_no_nan(values['fn'],
                                          values['fn'] + values['tp'])
    values['negative_rate'] = tf.math.divide_no_nan(
        values['tn'] + values['fn'],
        values['tp'] + values['fp'] + values['tn'] + values['fn'])

    values['false_discovery_rate'] = tf.math.divide_no_nan(
        values['fp'], values['fp'] + values['tp'])
    values['false_omission_rate'] = tf.math.divide_no_nan(
        values['fn'], values['fn'] + values['tn'])

    # pytype: enable=unsupported-operands

    update_op = tf.group(update_ops['fn'], update_ops['tn'], update_ops['fp'],
                         update_ops['tp'])
    value_op = tf.transpose(
        a=tf.stack([
            values['fn'], values['tn'], values['fp'], values['tp'],
            values['precision'], values['recall']
        ]))

    output_dict = {
        self._metric_key(self.matrices_key): (value_op, update_op),
        self._metric_key(self.thresholds_key): (tf.identity(self._thresholds),
                                                tf.no_op()),
    }
    for i, threshold in enumerate(self._thresholds):
      output_dict[self._metric_key(
          metric_keys.base_key(
              'positive_rate@%.*f' %
              (self._key_digits, threshold)))] = (values['positive_rate'][i],
                                                  update_op)
      output_dict[self._metric_key(
          metric_keys.base_key(
              'true_positive_rate@%.*f' %
              (self._key_digits, threshold)))] = (values['recall'][i],
                                                  update_op)
      output_dict[self._metric_key(
          metric_keys.base_key(
              'false_positive_rate@%.*f' %
              (self._key_digits, threshold)))] = (values['fpr'][i], update_op)
      output_dict[self._metric_key(
          metric_keys.base_key(
              'negative_rate@%.*f' %
              (self._key_digits, threshold)))] = (values['negative_rate'][i],
                                                  update_op)
      output_dict[self._metric_key(
          metric_keys.base_key(
              'true_negative_rate@%.*f' %
              (self._key_digits, threshold)))] = (values['tnr'][i], update_op)
      output_dict[self._metric_key(
          metric_keys.base_key(
              'false_negative_rate@%.*f' %
              (self._key_digits, threshold)))] = (values['fnr'][i], update_op)
      output_dict[self._metric_key(
          metric_keys.base_key('false_discovery_rate@%.*f' %
                               (self._key_digits, threshold)))] = (
                                   values['false_discovery_rate'][i], update_op)
      output_dict[self._metric_key(
          metric_keys.base_key('false_omission_rate@%.*f' %
                               (self._key_digits, threshold)))] = (
                                   values['false_omission_rate'][i], update_op)
    return output_dict  # pytype: disable=bad-return-type

  def populate_stats_and_pop(
      self, unused_slice_key: slicer.SliceKeyType, combine_metrics: Dict[Text,
                                                                         Any],
      output_metrics: Dict[Text, metrics_pb2.MetricValue]) -> None:
    matrices = combine_metrics.pop(self._metric_key(self.matrices_key))
    thresholds = combine_metrics.pop(self._metric_key(self.thresholds_key))

    # We assume that thresholds are already sorted.
    if len(matrices) != len(thresholds):
      raise ValueError(
          'matrices should have the same length as thresholds, but lengths '
          'were: matrices: %d, thresholds: %d' %
          (len(matrices), len(thresholds)))
    for threshold, raw_matrix in zip(thresholds, matrices):
      # Adds confusion matrix table as well as ratios used for fairness metrics.
      if isinstance(threshold, types.ValueWithTDistribution):
        threshold = threshold.unsampled_value
      output_matrix = post_export_metrics._create_confusion_matrix_proto(
          raw_matrix, threshold)
      (output_metrics[self._metric_key(metric_keys.FAIRNESS_CONFUSION_MATRIX)]
       .confusion_matrix_at_thresholds.matrices.add().CopyFrom(output_matrix))


# If the fairness_indicator in enabled, the slicing inside the tfx evaluator
# config will also be added into this metrics as a subgroup key.
# However, handling the subgroup metrics with slices is still TBD.
@post_export_metrics._export('fairness_auc')
class _FairnessAuc(post_export_metrics._PostExportMetric):
  """Metric that computes bounded AUC for predictions in [0, 1].

  This metrics calculates the subgroup auc, the background positive subgroup
  negative auc and background negative subgroup positive auc. For more
  explanation about the concepts of these auc metrics, please refer to paper
  [Measuring and Mitigating Unintended Bias in Text
  Classification](https://ai.google/research/pubs/pub46743)
  """

  _target_prediction_keys = ...  # type: List[Text]
  _labels_key = ...  # type: Text
  _metric_tag = None  # type: Text
  _tensor_index = ...  # type: int

  def __init__(self,
               subgroup_key: Text,
               example_weight_key: Optional[Text] = None,
               num_buckets: int = post_export_metrics._DEFAULT_NUM_BUCKETS,
               target_prediction_keys: Optional[List[Text]] = None,
               labels_key: Optional[Text] = None,
               metric_tag: Optional[Text] = None,
               tensor_index: Optional[int] = None) -> None:
    """Create a metric that computes fairness auc.

    Predictions should be one of:
      (a) a single float in [0, 1]
      (b) a dict containing the LOGISTIC key
      (c) a dict containing the PREDICTIONS key, where the prediction is
          in [0, 1]

    Label should be a single float that is either exactly 0 or exactly 1
    (soft labels, i.e. labels between 0 and 1 are *not* supported).

    Args:
      subgroup_key: The key inside the feature column to indicate where this
        example belongs to the subgroup or not. The expected mapping tensor of
        this key should contain an integer/float value that's either 1 or 0.
      example_weight_key: The key of the example weight column in the features
        dict. If None, all predictions are given a weight of 1.0.
      num_buckets: The number of buckets used for the curve. (num_buckets + 1)
        is used as the num_thresholds in tf.metrics.auc().
      target_prediction_keys: If provided, the prediction keys to look for in
        order.
      labels_key: If provided, a custom label key.
      metric_tag: If provided, a custom metric tag. Only necessary to
        disambiguate instances of the same metric on different predictions.
      tensor_index: Optional index to specify class predictions to calculate
        metrics on in the case of multi-class models.
    """

    self._subgroup_key = subgroup_key
    self._example_weight_key = example_weight_key
    self._curve = 'ROC'
    self._num_buckets = num_buckets
    self._metric_name = metric_keys.FAIRNESS_AUC
    self._subgroup_auc_metric = self._metric_key(self._metric_name +
                                                 '/subgroup_auc/' +
                                                 self._subgroup_key)
    self._bpsn_auc_metric = self._metric_key(self._metric_name + '/bpsn_auc/' +
                                             self._subgroup_key)
    self._bnsp_auc_metric = self._metric_key(self._metric_name + '/bnsp_auc/' +
                                             self._subgroup_key)

    super(_FairnessAuc, self).__init__(
        target_prediction_keys=target_prediction_keys,
        labels_key=labels_key,
        metric_tag=metric_tag,
        tensor_index=tensor_index)

  def check_compatibility(self, features_dict: types.TensorTypeMaybeDict,
                          predictions_dict: types.TensorTypeMaybeDict,
                          labels_dict: types.TensorTypeMaybeDict) -> None:
    post_export_metrics._check_feature_present(features_dict,
                                               self._example_weight_key)
    post_export_metrics._check_feature_present(features_dict,
                                               self._subgroup_key)
    self._get_labels_and_predictions(predictions_dict, labels_dict)

  def get_metric_ops(self, features_dict: types.TensorTypeMaybeDict,
                     predictions_dict: types.TensorTypeMaybeDict,
                     labels_dict: types.TensorTypeMaybeDict
                    ) -> Dict[Text, Tuple[types.TensorType, types.TensorType]]:
    # Note that we have to squeeze predictions, labels, weights so they are all
    # N element vectors (otherwise some of them might be N x 1 tensors, and
    # multiplying a N element vector with a N x 1 tensor uses matrix
    # multiplication rather than element-wise multiplication).
    predictions, labels = self._get_labels_and_predictions(
        predictions_dict, labels_dict)
    predictions = post_export_metrics._flatten_to_one_dim(
        tf.cast(predictions, tf.float64))
    labels = post_export_metrics._flatten_to_one_dim(
        tf.cast(labels, tf.float64))
    weights = tf.ones_like(predictions)
    subgroup = post_export_metrics._flatten_to_one_dim(
        tf.cast(features_dict[self._subgroup_key], tf.bool))
    if self._example_weight_key:
      weights = post_export_metrics._flatten_to_one_dim(
          tf.cast(features_dict[self._example_weight_key], tf.float64))

    predictions, labels, weights = (
        post_export_metrics
        ._create_predictions_labels_weights_for_fractional_labels(
            predictions, labels, weights))
    # To let subgroup tensor match the size with prediction, labels and weights
    # above.
    subgroup = tf.concat([subgroup, subgroup], axis=0)

    labels_bool = tf.cast(labels, tf.bool)
    pos_subgroup = tf.math.logical_and(labels_bool, subgroup)
    neg_subgroup = tf.math.logical_and(
        tf.math.logical_not(labels_bool), subgroup)
    pos_background = tf.math.logical_and(labels_bool,
                                         tf.math.logical_not(subgroup))
    neg_background = tf.math.logical_and(
        tf.math.logical_not(labels_bool), tf.math.logical_not(subgroup))
    bnsp = tf.math.logical_or(pos_subgroup, neg_background)
    bpsn = tf.math.logical_or(neg_subgroup, pos_background)

    ops_dict = {}
    # Add subgroup auc.
    ops_dict.update(
        post_export_metrics._build_auc_metrics_ops(
            self._subgroup_auc_metric, labels, predictions,
            tf.multiply(weights, tf.cast(subgroup, tf.float64)),
            self._num_buckets + 1, self._curve))
    # Add backgroup positive subgroup negative auc.
    ops_dict.update(
        post_export_metrics._build_auc_metrics_ops(
            self._bpsn_auc_metric, labels, predictions,
            tf.multiply(weights, tf.cast(bpsn, tf.float64)),
            self._num_buckets + 1, self._curve))
    # Add backgroup negative subgroup positive auc.
    ops_dict.update(
        post_export_metrics._build_auc_metrics_ops(
            self._bnsp_auc_metric, labels, predictions,
            tf.multiply(weights, tf.cast(bnsp, tf.float64)),
            self._num_buckets + 1, self._curve))

    return ops_dict

  def populate_stats_and_pop(
      self, slice_key: slicer.SliceKeyType, combine_metrics: Dict[Text, Any],
      output_metrics: Dict[Text, metrics_pb2.MetricValue]) -> None:
    # Remove metrics if it's not Overall slice. This post export metrics
    # calculate subgroup_auc, bpsn_auc, bnsp_auc. All of these are based on
    # all examples. That's why only the overall slice makes sence and the rest
    # will be removed.
    if slice_key:
      for metrics_key in (self._subgroup_auc_metric, self._bpsn_auc_metric,
                          self._bnsp_auc_metric):
        combine_metrics.pop(metric_keys.lower_bound_key(metrics_key))
        combine_metrics.pop(metric_keys.upper_bound_key(metrics_key))
        combine_metrics.pop(metrics_key)
    else:
      for metrics_key in (self._subgroup_auc_metric, self._bpsn_auc_metric,
                          self._bnsp_auc_metric):
        post_export_metrics._populate_to_auc_bounded_value_and_pop(
            combine_metrics, output_metrics, metrics_key)


# pylint: enable=protected-access
