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
"""Trains and exports a simple custom estimator using tf.Learn.

The true model is age * 3 + 1.

This is a custom estimator with a custom model fn that defines its own
eval_metric_ops which passes "transformed" predictions and labels to the
metrics.
"""
from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

# Standard Imports

import tensorflow as tf
from tensorflow_model_analysis.eval_saved_model import export
from tensorflow_model_analysis.eval_saved_model.example_trainers import util


def simple_custom_estimator(export_path, eval_export_path):
  """Trains and exports a simple custom estimator."""

  def model_fn(features, labels, mode, config):
    """Model function for custom estimator."""
    del config
    m = tf.Variable(0.0, dtype=tf.float32, name='m')
    c = tf.Variable(0.0, dtype=tf.float32, name='c')
    predictions = m * features['age'] + c

    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions={'score': predictions},
          export_outputs={
              'score': tf.estimator.export.RegressionOutput(predictions)
          })

    loss = tf.compat.v1.losses.mean_squared_error(labels, predictions)
    eval_metric_ops = {
        'mean_absolute_error':
            tf.compat.v1.metrics.mean_absolute_error(
                tf.cast(labels, tf.float64), tf.cast(predictions, tf.float64)),
        'mean_prediction':
            tf.compat.v1.metrics.mean(predictions),
        'mean_label':
            tf.compat.v1.metrics.mean(labels),
    }

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.compat.v1.train.get_global_step())

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        predictions=predictions)

  def train_input_fn():
    """Train input function."""
    return {
        'age': tf.constant([[1.0], [2.0], [3.0], [4.0]]),
    }, tf.constant([[4.0], [7.0], [10.0], [13.0]]),

  def serving_input_receiver_fn():
    """Serving input receiver function."""
    serialized_tf_example = tf.compat.v1.placeholder(
        dtype=tf.string, shape=[None], name='input_example_tensor')
    feature_spec = {'age': tf.io.FixedLenFeature([1], dtype=tf.float32)}
    receiver_tensors = {'examples': serialized_tf_example}
    features = tf.io.parse_example(
        serialized=serialized_tf_example, features=feature_spec)
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

  def eval_input_receiver_fn():
    """Eval input receiver function."""
    serialized_tf_example = tf.compat.v1.placeholder(
        dtype=tf.string, shape=[None], name='input_example_tensor')
    feature_spec = {
        'age': tf.io.FixedLenFeature([1], dtype=tf.float32),
        'label': tf.io.FixedLenFeature([1], dtype=tf.float32)
    }
    receiver_tensors = {'examples': serialized_tf_example}
    features = tf.io.parse_example(
        serialized=serialized_tf_example, features=feature_spec)

    return export.EvalInputReceiver(
        features=features,
        labels=features['label'],
        receiver_tensors=receiver_tensors)

  estimator = tf.estimator.Estimator(model_fn=model_fn)
  estimator.train(input_fn=train_input_fn, steps=1000)

  return util.export_model_and_eval_model(
      estimator=estimator,
      serving_input_receiver_fn=serving_input_receiver_fn,
      eval_input_receiver_fn=eval_input_receiver_fn,
      export_path=export_path,
      eval_export_path=eval_export_path)
