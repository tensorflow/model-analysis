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
"""Exports a simple estimator with control dependencies using tf.Learn.

This is the fixed prediction estimator with extra fields, but it creates
metrics with control dependencies on the features, predictions and labels.
This is for use in tests to verify that TFMA correctly works around the
TensorFlow issue #17568.

This model always predicts the value of the "prediction" feature.

The eval_input_receiver_fn also parses the "fixed_float", "fixed_string",
"fixed_int", and "var_float", "var_string", "var_int" features.
"""
from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

# Standard Imports

import tensorflow as tf
from tensorflow_model_analysis.eval_saved_model import export
from tensorflow_model_analysis.eval_saved_model.example_trainers import util

from tensorflow.python.estimator.canned import metric_keys
from tensorflow.python.estimator.canned import prediction_keys


def simple_control_dependency_estimator(export_path, eval_export_path):
  """Exports a simple estimator with control dependencies."""

  def control_dependency_metric(increment, target):
    """Metric that introduces a control dependency on target.

    The value is incremented by increment each time the metric is called
    (so the value can vary depending on how things are batched). This is mainly
    to verify that the metric was called.

    Args:
      increment: Amount to increment the value by each time the metric is
        called.
      target: Tensor to introduce the control dependency on.

    Returns:
      value_op, update_op for the metric.
    """

    total_value = tf.compat.v1.Variable(
        initial_value=0.0,
        dtype=tf.float64,
        trainable=False,
        collections=[
            tf.compat.v1.GraphKeys.METRIC_VARIABLES,
            tf.compat.v1.GraphKeys.LOCAL_VARIABLES
        ],
        validate_shape=True)

    with tf.control_dependencies([target]):
      update_op = tf.identity(tf.compat.v1.assign_add(total_value, increment))
    value_op = tf.identity(total_value)
    return value_op, update_op

  def model_fn(features, labels, mode, config):
    """Model function for custom estimator."""
    del config
    predictions = features['prediction']
    predictions_dict = {
        prediction_keys.PredictionKeys.PREDICTIONS: predictions,
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=predictions_dict,
          export_outputs={
              tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                  tf.estimator.export.RegressionOutput(predictions)
          })

    loss = tf.compat.v1.losses.mean_squared_error(predictions,
                                                  labels['actual_label'])
    train_op = tf.compat.v1.assign_add(tf.compat.v1.train.get_global_step(), 1)

    eval_metric_ops = {}
    if mode == tf.estimator.ModeKeys.EVAL:
      eval_metric_ops = {
          metric_keys.MetricKeys.LOSS_MEAN:
              tf.compat.v1.metrics.mean(loss),
          'control_dependency_on_fixed_float':
              control_dependency_metric(1.0, features['fixed_float']),
          # Introduce a direct dependency on the values Tensor. If we
          # introduce another intervening op like sparse_tensor_to_dense then
          # regardless of whether TFMA correctly wrap SparseTensors we will not
          # encounter the TF bug.
          'control_dependency_on_var_float':
              control_dependency_metric(10.0, features['var_float'].values),
          'control_dependency_on_actual_label':
              control_dependency_metric(100.0, labels['actual_label']),
          'control_dependency_on_var_int_label':
              control_dependency_metric(1000.0, labels['var_int'].values),
          # Note that TFMA does *not* wrap predictions, so in most cases
          # if there's a control dependency on predictions they will be
          # recomputed.
          'control_dependency_on_prediction':
              control_dependency_metric(10000.0, predictions),
      }

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        predictions=predictions_dict,
        eval_metric_ops=eval_metric_ops)

  def train_input_fn():
    """Train input function."""
    return {
        'prediction': tf.constant([[1.0], [2.0], [3.0], [4.0]]),
    }, {
        'actual_label': tf.constant([[1.0], [2.0], [3.0], [4.0]])
    }

  feature_spec = {'prediction': tf.io.FixedLenFeature([1], dtype=tf.float32)}
  eval_feature_spec = {
      'prediction': tf.io.FixedLenFeature([1], dtype=tf.float32),
      'label': tf.io.FixedLenFeature([1], dtype=tf.float32),
      'fixed_float': tf.io.FixedLenFeature([1], dtype=tf.float32),
      'fixed_string': tf.io.FixedLenFeature([1], dtype=tf.string),
      'fixed_int': tf.io.FixedLenFeature([1], dtype=tf.int64),
      'var_float': tf.io.VarLenFeature(dtype=tf.float32),
      'var_string': tf.io.VarLenFeature(dtype=tf.string),
      'var_int': tf.io.VarLenFeature(dtype=tf.int64),
  }

  estimator = tf.estimator.Estimator(model_fn=model_fn)
  estimator.train(input_fn=train_input_fn, steps=1)

  def eval_input_receiver_fn():
    """An input_fn that expects a serialized tf.Example."""
    serialized_tf_example = tf.compat.v1.placeholder(
        dtype=tf.string, shape=[None], name='input_example_tensor')
    features = tf.io.parse_example(
        serialized=serialized_tf_example, features=eval_feature_spec)
    labels = {'actual_label': features['label'], 'var_int': features['var_int']}
    return export.EvalInputReceiver(
        features=features,
        labels=labels,
        receiver_tensors={'examples': serialized_tf_example})

  return util.export_model_and_eval_model(
      estimator=estimator,
      serving_input_receiver_fn=(
          tf.estimator.export.build_parsing_serving_input_receiver_fn(
              feature_spec)),
      eval_input_receiver_fn=eval_input_receiver_fn,
      export_path=export_path,
      eval_export_path=eval_export_path)
