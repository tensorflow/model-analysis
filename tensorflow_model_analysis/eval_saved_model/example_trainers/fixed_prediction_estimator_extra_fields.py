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
"""Exports a simple "fixed prediction" estimator using tf.Learn.

This model always predicts the value of the "prediction" feature.

The eval_input_receiver_fn also parses the "fixed_float", "fixed_string",
"fixed_int", and "var_float", "var_string", "var_int" features.
"""
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function



import tensorflow as tf
from tensorflow_model_analysis.eval_saved_model import export
from tensorflow_model_analysis.eval_saved_model.example_trainers import util

from tensorflow.python.estimator.canned import metric_keys
from tensorflow.python.estimator.canned import prediction_keys


def simple_fixed_prediction_estimator_extra_fields(export_path,
                                                   eval_export_path):
  """Exports a simple fixed prediction estimator that parses extra fields."""

  def model_fn(features, labels, mode, params):
    """Model function for custom estimator."""
    del params
    predictions = features['prediction']
    predictions_dict = {
        prediction_keys.PredictionKeys.PREDICTIONS: predictions,
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=predictions_dict,
          export_outputs={
              tf.saved_model.signature_constants.
              DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                  tf.estimator.export.RegressionOutput(predictions)
          })

    loss = tf.losses.mean_squared_error(predictions, labels)
    train_op = tf.assign_add(tf.train.get_global_step(), 1)
    eval_metric_ops = {
        metric_keys.MetricKeys.LOSS_MEAN: tf.metrics.mean(loss),
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
    }, tf.constant([[1.0], [2.0], [3.0], [4.0]]),

  feature_spec = {'prediction': tf.FixedLenFeature([1], dtype=tf.float32)}
  eval_feature_spec = {
      'prediction': tf.FixedLenFeature([1], dtype=tf.float32),
      'label': tf.FixedLenFeature([1], dtype=tf.float32),
      'fixed_float': tf.FixedLenFeature([1], dtype=tf.float32),
      'fixed_string': tf.FixedLenFeature([1], dtype=tf.string),
      'fixed_int': tf.FixedLenFeature([1], dtype=tf.int64),
      'var_float': tf.VarLenFeature(dtype=tf.float32),
      'var_string': tf.VarLenFeature(dtype=tf.string),
      'var_int': tf.VarLenFeature(dtype=tf.int64),
  }

  estimator = tf.estimator.Estimator(model_fn=model_fn)
  estimator.train(input_fn=train_input_fn, steps=1)

  return util.export_model_and_eval_model(
      estimator=estimator,
      serving_input_receiver_fn=(
          tf.estimator.export.build_parsing_serving_input_receiver_fn(
              feature_spec)),
      eval_input_receiver_fn=export.build_parsing_eval_input_receiver_fn(
          eval_feature_spec, label_key='label'),
      export_path=export_path,
      eval_export_path=eval_export_path)
