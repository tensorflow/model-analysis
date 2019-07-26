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


def get_simple_fixed_prediction_estimator_and_metadata(
    output_prediction_key=prediction_keys.PredictionKeys.PREDICTIONS):
  """Returns a simple fixed prediction estimator and metadata.

  Exposed for use with ExporterTest.

  Args:
    output_prediction_key: Output prediction key.

  Returns:
    Dictionary containing estimator, eval_input_receiver_fn,
    serving_input_receiver_fn, train_input_fn and model_fn.
  """

  def model_fn(features, labels, mode, config):
    """Model function for custom estimator."""
    del config
    predictions = features['prediction']

    if output_prediction_key is not None:
      predictions_dict = {
          output_prediction_key: predictions,
      }
    else:
      # For simulating Estimators which don't return a predictions dict in
      # EVAL mode.
      predictions_dict = {}

    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=predictions_dict,
          export_outputs={
              tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                  tf.estimator.export.RegressionOutput(predictions)
          })

    loss = tf.compat.v1.losses.mean_squared_error(predictions, labels)
    train_op = tf.compat.v1.assign_add(tf.compat.v1.train.get_global_step(), 1)
    eval_metric_ops = {
        metric_keys.MetricKeys.LOSS_MEAN: tf.compat.v1.metrics.mean(loss),
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

  estimator = tf.estimator.Estimator(model_fn=model_fn)

  feature_spec = {'prediction': tf.io.FixedLenFeature([1], dtype=tf.float32)}
  eval_feature_spec = {
      'prediction': tf.io.FixedLenFeature([1], dtype=tf.float32),
      'label': tf.io.FixedLenFeature([1], dtype=tf.float32),
  }

  return {
      'estimator':
          estimator,
      'serving_input_receiver_fn':
          (tf.estimator.export.build_parsing_serving_input_receiver_fn(
              feature_spec)),
      'eval_input_receiver_fn':
          export.build_parsing_eval_input_receiver_fn(
              eval_feature_spec, label_key='label'),
      'train_input_fn':
          train_input_fn,
      'model_fn':
          model_fn,
  }


def simple_fixed_prediction_estimator(
    export_path,
    eval_export_path,
    output_prediction_key=prediction_keys.PredictionKeys.PREDICTIONS):
  estimator_metadata = get_simple_fixed_prediction_estimator_and_metadata(
      output_prediction_key)
  estimator_metadata['estimator'].train(
      input_fn=estimator_metadata['train_input_fn'], steps=1)
  return util.export_model_and_eval_model(
      estimator=estimator_metadata['estimator'],
      serving_input_receiver_fn=estimator_metadata['serving_input_receiver_fn'],
      eval_input_receiver_fn=estimator_metadata['eval_input_receiver_fn'],
      export_path=export_path,
      eval_export_path=eval_export_path)
