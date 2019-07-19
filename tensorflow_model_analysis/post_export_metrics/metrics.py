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
"""Library for useful metrics not provided by tf.metrics."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

# Standard Imports
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import types
from typing import List, Optional, Tuple


def total(values: types.TensorType
         ) -> Tuple[types.TensorType, types.TensorType]:
  """Metric to compute the running total of a value."""

  with tf.compat.v1.variable_scope('total', values):
    values = tf.cast(values, tf.float64)
    total_value = tf.compat.v1.Variable(
        initial_value=0.0,
        dtype=tf.float64,
        trainable=False,
        collections=[
            tf.compat.v1.GraphKeys.METRIC_VARIABLES,
            tf.compat.v1.GraphKeys.LOCAL_VARIABLES
        ],
        validate_shape=True,
        name='total')
    update_op = tf.compat.v1.assign_add(total_value,
                                        tf.reduce_sum(input_tensor=values))
    value_op = tf.identity(total_value)
    return value_op, update_op


def squared_pearson_correlation(predictions: types.TensorType,
                                labels: types.TensorType,
                                weights: Optional[types.TensorType] = None
                               ) -> Tuple[types.TensorType, types.TensorType]:
  """Metric to compute the squared pearson correlation (R squared)."""
  with tf.compat.v1.variable_scope('squared_pearson_correlation',
                                   [predictions, labels, weights]):
    weighted_predictions = tf.multiply(predictions, weights)
    weighted_labels = tf.multiply(labels, weights)

    regression_error = tf.compat.v1.Variable(
        initial_value=0.0,
        dtype=tf.float64,
        trainable=False,
        collections=[
            tf.compat.v1.GraphKeys.METRIC_VARIABLES,
            tf.compat.v1.GraphKeys.LOCAL_VARIABLES
        ],
        validate_shape=True,
        name='regression_error')

    update_regression_error_op = tf.compat.v1.assign_add(
        regression_error,
        tf.reduce_sum(
            input_tensor=tf.square(
                tf.subtract(weighted_labels, weighted_predictions))))

    total_error = tf.compat.v1.Variable(
        initial_value=0.0,
        dtype=tf.float64,
        trainable=False,
        collections=[
            tf.compat.v1.GraphKeys.METRIC_VARIABLES,
            tf.compat.v1.GraphKeys.LOCAL_VARIABLES
        ],
        validate_shape=True,
        name='total_error')

    update_total_error_op = tf.compat.v1.assign_add(
        total_error,
        tf.reduce_sum(
            input_tensor=tf.square(
                tf.subtract(weighted_labels,
                            tf.reduce_mean(input_tensor=weighted_labels)))))

    div_op = tf.math.divide
    if hasattr(tf.math, 'divide_no_nan'):
      div_op = tf.math.divide_no_nan

    def r_squared(_, regression_error, total_error):
      return tf.subtract(
          tf.cast(1.0, tf.float64), div_op(regression_error, total_error))

    value_op = tf.distribute.get_replica_context().merge_call(
        r_squared, args=(regression_error, total_error))
    update_op = tf.subtract(
        tf.cast(1.0, tf.float64),
        div_op(update_regression_error_op, update_total_error_op))
    return value_op, update_op


def calibration_plot(predictions: types.TensorType,
                     labels: types.TensorType,
                     left: float,
                     right: float,
                     num_buckets: int,
                     weights: Optional[types.TensorType] = None
                    ) -> Tuple[types.TensorType, types.TensorType]:
  # pyformat: disable
  """Calibration plot for predictions in [left, right].

  A calibration plot contains multiple buckets, based on the prediction.
  Each bucket contains:
    (i) the weighted sum of predictions that fall within that bucket
   (ii) the weighted sum of labels associated with those predictions
  (iii) the sum of weights of the associated examples

  Note that the calibration plot also contains enough information to build
  to prediction histogram (which doesn't need the information about the labels).

  Args:
    predictions: Predictions to compute calibration plot for.
    labels: Labels associated with the corresponding predictions.
    left: Left-most bucket boundary.
    right: Right-most bucket boundary.
    num_buckets: Number of buckets to divide [left, right] into.
    weights: Optional weights for each of the predictions/labels. If None,
      each of the predictions/labels will be assumed to have a weight of 1.0.

  left=1.0, right=2.0, num_buckets=2 yields buckets:
    bucket 0: (-inf, 1.0)
    bucket 1: [1.0, 1.5)
    bucket 2: [1.5, 2.0)
    bucket 3: [2.0, inf)

  The value_op will return a matrix with num_buckets + 2 rows and 3 columns:
  [ bucket 0 weighted prediction sum, weighted label sum, sum of weights ]
  [ bucket 1 weighted prediction sum, weighted label sum, sum of weights ]
  [               :                            :               :         ]
  [               :                            :               :         ]
  [ bucket k weighted prediction sum, weighted label sum, sum of weights ]
  where k = num_buckets + 1

  Returns:
    (value_op, update_op) for the calibration plot.
  """
  # pyformat: enable

  with tf.compat.v1.variable_scope('calibration_plot', [predictions, labels]):
    predictions_f64 = tf.cast(predictions, tf.float64)
    labels_f64 = tf.cast(labels, tf.float64)
    # Ensure that we don't mistakenly use the non-casted versions.
    del predictions, labels

    prediction_bucket_counts = tf.compat.v1.Variable(
        initial_value=[0.0] * (num_buckets + 2),
        dtype=tf.float64,
        trainable=False,
        collections=[
            tf.compat.v1.GraphKeys.METRIC_VARIABLES,
            tf.compat.v1.GraphKeys.LOCAL_VARIABLES
        ],
        validate_shape=True,
        name='prediction_bucket_counts')
    label_bucket_counts = tf.compat.v1.Variable(
        initial_value=[0.0] * (num_buckets + 2),
        dtype=tf.float64,
        trainable=False,
        collections=[
            tf.compat.v1.GraphKeys.METRIC_VARIABLES,
            tf.compat.v1.GraphKeys.LOCAL_VARIABLES
        ],
        validate_shape=True,
        name='label_bucket_counts')
    weight_bucket_counts = tf.compat.v1.Variable(
        initial_value=[0.0] * (num_buckets + 2),
        dtype=tf.float64,
        trainable=False,
        collections=[
            tf.compat.v1.GraphKeys.METRIC_VARIABLES,
            tf.compat.v1.GraphKeys.LOCAL_VARIABLES
        ],
        validate_shape=True,
        name='weight_bucket_counts')

    bucket_width = (right - left) / num_buckets
    indices = tf.cast(
        tf.clip_by_value(
            tf.floor((predictions_f64 - left) / bucket_width),
            clip_value_min=-1,
            clip_value_max=num_buckets) + 1, tf.int32)

    if weights is not None:
      weights_f64 = tf.cast(weights, tf.float64)
    else:
      weights_f64 = tf.ones_like(indices, dtype=tf.float64)

    update_prediction_buckets_op = tf.compat.v1.scatter_add(
        prediction_bucket_counts, indices, predictions_f64 * weights_f64)
    update_label_buckets_op = tf.compat.v1.scatter_add(label_bucket_counts,
                                                       indices,
                                                       labels_f64 * weights_f64)
    update_weight_buckets_op = tf.compat.v1.scatter_add(weight_bucket_counts,
                                                        indices, weights_f64)
    update_op = tf.group(update_prediction_buckets_op, update_label_buckets_op,
                         update_weight_buckets_op)
    value_op = tf.transpose(
        a=tf.stack([
            prediction_bucket_counts, label_bucket_counts, weight_bucket_counts
        ]))

  return value_op, update_op


def _precision_recall_at_k(classes: types.TensorType,
                           scores: types.TensorType,
                           labels: types.TensorType,
                           cutoffs: List[int],
                           weights: Optional[types.TensorType] = None,
                           precision: Optional[bool] = True,
                           recall: Optional[bool] = True
                          ) -> Tuple[types.TensorType, types.TensorType]:
  # pyformat: disable
  """Precision and recall at `k`.

  Args:
    classes: Tensor containing class names. Should be a BATCH_SIZE x NUM_CLASSES
      Tensor.
    scores: Tensor containing the associated scores. Should be a
      BATCH_SIZE x NUM_CLASSES Tensor.
    labels: Tensor containing the true labels. Should be a rank-2 Tensor where
      the first dimension is BATCH_SIZE. The second dimension can be anything.
    cutoffs: List containing the values for the `k` at which to compute the
      precision and recall for. Use a value of `k` = 0 to indicate that all
      predictions should be considered.
    weights: Optional weights for each of the examples. If None,
      each of the predictions/labels will be assumed to have a weight of 1.0.
      If present, should be a BATCH_SIZE Tensor.
    precision: True to compute precision.
    recall: True to compute recall.

  The value_op will return a matrix with len(cutoffs) rows and 3 columns:
  [ cutoff 0, precision at cutoff 0, recall at cutoff 0 ]
  [ cutoff 1, precision at cutoff 1, recall at cutoff 1 ]
  [     :                :                  :           ]
  [ cutoff n, precision at cutoff n, recall at cutoff n ]

  If only one of precision or recall is True then the value_op will return only
  2 columns (cutoff and ether precision or recall depending on which is True).

  Returns:
    (value_op, update_op) for the precision/recall at K metric.
  """
  # pyformat: enable
  if not precision and not recall:
    raise ValueError('one of either precision or recall must be set')

  num_cutoffs = len(cutoffs)

  if precision and recall:
    scope = 'precision_recall_at_k'
  elif precision:
    scope = 'precision_at_k'
  else:
    scope = 'recall_at_k'

  with tf.compat.v1.variable_scope(scope, [classes, scores, labels]):

    # Predicted positive.
    predicted_positives = tf.compat.v1.Variable(
        initial_value=[0.0] * num_cutoffs,
        dtype=tf.float64,
        trainable=False,
        collections=[
            tf.compat.v1.GraphKeys.METRIC_VARIABLES,
            tf.compat.v1.GraphKeys.LOCAL_VARIABLES
        ],
        validate_shape=True,
        name='predicted_positives')

    # Predicted positive, label positive.
    true_positives = tf.compat.v1.Variable(
        initial_value=[0.0] * num_cutoffs,
        dtype=tf.float64,
        trainable=False,
        collections=[
            tf.compat.v1.GraphKeys.METRIC_VARIABLES,
            tf.compat.v1.GraphKeys.LOCAL_VARIABLES
        ],
        validate_shape=True,
        name='true_positives')

    # Label positive.
    actual_positives = tf.compat.v1.Variable(
        initial_value=0.0,
        dtype=tf.float64,
        trainable=False,
        collections=[
            tf.compat.v1.GraphKeys.METRIC_VARIABLES,
            tf.compat.v1.GraphKeys.LOCAL_VARIABLES
        ],
        validate_shape=True,
        name='actual_positives')

    if weights is not None:
      weights_f64 = tf.cast(weights, tf.float64)
    else:
      weights_f64 = tf.ones(tf.shape(input=labels)[0], tf.float64)

  def compute_batch_stats(classes: np.ndarray, scores: np.ndarray,
                          labels: np.ndarray,
                          weights: np.ndarray) -> np.ndarray:
    """Compute precision/recall intermediate stats for a batch.

    Args:
      classes: Tensor containing class names. Should be a BATCH_SIZE x
        NUM_CLASSES Tensor.
      scores: Tensor containing the associated scores. Should be a BATCH_SIZE x
        NUM_CLASSES Tensor.
      labels: Tensor containing the true labels. Should be a rank-2 Tensor where
        the first dimension is BATCH_SIZE. The second dimension can be anything.
      weights: Weights for the associated exmaples. Should be a BATCH_SIZE
        Tesnor.

    Returns:
      True positives, predicted positives, actual positives computed for the
      batch of examples.

    Raises:
      ValueError: classes and scores have different shapes; or labels has
       a different batch size from classes and scores
    """

    if classes.shape != scores.shape:
      raise ValueError('classes and scores should have same shape, but got '
                       '%s and %s' % (classes.shape, scores.shape))

    batch_size = classes.shape[0]
    num_classes = classes.shape[1]
    if labels.shape[0] != batch_size:
      raise ValueError('labels should have the same batch size of %d, but got '
                       '%d instead' % (batch_size, labels.shape[0]))

    # Sort classes, by row, by their associated scores, in descending order of
    # score.
    sorted_classes = np.flip(
        classes[np.arange(batch_size)[:, None],
                np.argsort(scores)], axis=1)

    true_positives = np.zeros(num_cutoffs, dtype=np.float64)
    predicted_positives = np.zeros(num_cutoffs, dtype=np.float64)
    actual_positives = 0.0

    for predicted_row, label_row, weight in zip(sorted_classes, labels,
                                                weights):

      label_set = set(label_row)
      label_set.discard(b'')  # Remove filler elements.

      for i, cutoff in enumerate(cutoffs):
        cutoff_to_use = cutoff if cutoff > 0 else num_classes
        cut_predicted_row = predicted_row[:cutoff_to_use]
        true_pos = set(cut_predicted_row) & label_set
        true_positives[i] += len(true_pos) * weight
        predicted_positives[i] += len(cut_predicted_row) * weight

      actual_positives += len(label_set) * weight

    return true_positives, predicted_positives, actual_positives  # pytype: disable=bad-return-type

  # Value op returns
  # [ K | precision at K | recall at K ]
  # PyType doesn't like TF operator overloads: b/92797687
  # pytype: disable=unsupported-operands
  precision_op = true_positives / predicted_positives
  recall_op = true_positives / actual_positives
  # pytype: enable=unsupported-operands
  if precision and recall:
    value_op = tf.transpose(
        a=tf.stack([cutoffs, precision_op, recall_op], axis=0))
  elif precision:
    value_op = tf.transpose(a=tf.stack([cutoffs, precision_op], axis=0))
  else:
    value_op = tf.transpose(a=tf.stack([cutoffs, recall_op], axis=0))

  true_positives_update, predicted_positives_update, actual_positives_update = (
      tf.compat.v1.py_func(compute_batch_stats,
                           [classes, scores, labels, weights_f64],
                           [tf.float64, tf.float64, tf.float64]))

  update_op = tf.group(
      tf.compat.v1.assign_add(true_positives, true_positives_update),
      tf.compat.v1.assign_add(predicted_positives, predicted_positives_update),
      tf.compat.v1.assign_add(actual_positives, actual_positives_update))

  return value_op, update_op


def precision_at_k(classes: types.TensorType,
                   scores: types.TensorType,
                   labels: types.TensorType,
                   cutoffs: List[int],
                   weights: Optional[types.TensorType] = None
                  ) -> Tuple[types.TensorType, types.TensorType]:
  # pyformat: disable
  """Precision at `k`.

  Args:
    classes: Tensor containing class names. Should be a BATCH_SIZE x NUM_CLASSES
      Tensor.
    scores: Tensor containing the associated scores. Should be a
      BATCH_SIZE x NUM_CLASSES Tensor.
    labels: Tensor containing the true labels. Should be a rank-2 Tensor where
      the first dimension is BATCH_SIZE. The second dimension can be anything.
    cutoffs: List containing the values for the `k` at which to compute the
      precision for. Use a value of `k` = 0 to indicate that all predictions
      should be considered.
    weights: Optional weights for each of the examples. If None,
      each of the predictions/labels will be assumed to have a weight of 1.0.
      If present, should be a BATCH_SIZE Tensor.

  The value_op will return a matrix with len(cutoffs) rows and 2 columns:
  [ cutoff 0, precision at cutoff 0 ]
  [ cutoff 1, precision at cutoff 1 ]
  [     :                :          ]
  [ cutoff n, precision at cutoff n ]

  Returns:
    (value_op, update_op) for the precision at K metric.
  """
  # pyformat: enable
  return _precision_recall_at_k(classes, scores, labels, cutoffs, weights, True,
                                False)


def recall_at_k(classes: types.TensorType,
                scores: types.TensorType,
                labels: types.TensorType,
                cutoffs: List[int],
                weights: Optional[types.TensorType] = None
               ) -> Tuple[types.TensorType, types.TensorType]:
  # pyformat: disable
  """Recall at `k`.

  Args:
    classes: Tensor containing class names. Should be a BATCH_SIZE x NUM_CLASSES
      Tensor.
    scores: Tensor containing the associated scores. Should be a
      BATCH_SIZE x NUM_CLASSES Tensor.
    labels: Tensor containing the true labels. Should be a rank-2 Tensor where
      the first dimension is BATCH_SIZE. The second dimension can be anything.
    cutoffs: List containing the values for the `k` at which to compute the
      recall for. Use a value of `k` = 0 to indicate that all predictions should
      be considered.
    weights: Optional weights for each of the examples. If None,
      each of the predictions/labels will be assumed to have a weight of 1.0.
      If present, should be a BATCH_SIZE Tensor.

  The value_op will return a matrix with len(cutoffs) rows and 2 columns:
  [ cutoff 0, recall at cutoff 0 ]
  [ cutoff 1, recall at cutoff 1 ]
  [     :                :       ]
  [ cutoff n, recall at cutoff n ]

  Returns:
    (value_op, update_op) for the recall at K metric.
  """
  # pyformat: enable
  return _precision_recall_at_k(classes, scores, labels, cutoffs, weights,
                                False, True)
