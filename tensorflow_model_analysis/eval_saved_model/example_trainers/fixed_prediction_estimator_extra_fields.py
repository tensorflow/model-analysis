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

The eval_input_receiver_fn also parses the "fixed_float", "fixed_string" and
"var_float", "var_string" features.
"""
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function



import tensorflow as tf
from tensorflow_model_analysis.eval_saved_model import export


def simple_fixed_prediction_estimator_extra_fields(export_path,
                                                   eval_export_path):
  """Exports a simple fixed prediction estimator that parses extra fields."""

  def model_fn(features, labels, mode, params):
    """Model function for custom estimator."""
    del params
    predictions = features['prediction']

    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions={'score': predictions},
          export_outputs={
              'score': tf.estimator.export.RegressionOutput(predictions)
          })

    loss = tf.losses.mean_squared_error(predictions, labels)
    train_op = tf.assign_add(tf.train.get_global_step(), 1)

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, train_op=train_op, predictions=predictions)

  def train_input_fn():
    """Train input function."""
    return {
        'prediction': tf.constant([[1.0], [2.0], [3.0], [4.0]]),
    }, tf.constant([[1.0], [2.0], [3.0], [4.0]]),

  def serving_input_receiver_fn():
    """Serving input receiver function."""
    serialized_tf_example = tf.placeholder(
        dtype=tf.string, shape=[None], name='input_example_tensor')
    feature_spec = {'prediction': tf.FixedLenFeature([1], dtype=tf.float32)}
    receiver_tensors = {'examples': serialized_tf_example}
    features = tf.parse_example(serialized_tf_example, feature_spec)
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

  def eval_input_receiver_fn():
    """Eval input receiver function."""
    serialized_tf_example = tf.placeholder(
        dtype=tf.string, shape=[None], name='input_example_tensor')
    feature_spec = {
        'prediction': tf.FixedLenFeature([1], dtype=tf.float32),
        'label': tf.FixedLenFeature([1], dtype=tf.float32),
        'fixed_float': tf.FixedLenFeature([1], dtype=tf.float32),
        'fixed_string': tf.FixedLenFeature([1], dtype=tf.string),
        'var_float': tf.VarLenFeature(dtype=tf.float32),
        'var_string': tf.VarLenFeature(dtype=tf.string)
    }
    receiver_tensors = {'examples': serialized_tf_example}
    features = tf.parse_example(serialized_tf_example, feature_spec)

    return export.EvalInputReceiver(
        features=features,
        receiver_tensors=receiver_tensors,
        labels=features['label'])

  estimator = tf.estimator.Estimator(model_fn=model_fn)
  estimator.train(input_fn=train_input_fn, steps=1)

  export_dir = None
  eval_export_dir = None
  if export_path:
    export_dir = estimator.export_savedmodel(
        export_dir_base=export_path,
        serving_input_receiver_fn=serving_input_receiver_fn)

  if eval_export_path:
    eval_export_dir = export.export_eval_savedmodel(
        estimator=estimator,
        export_dir_base=eval_export_path,
        eval_input_receiver_fn=eval_input_receiver_fn)

  return export_dir, eval_export_dir
