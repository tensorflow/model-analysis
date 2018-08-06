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
"""Exports a simple "fixed prediction" classifier using tf.Learn.

This model generates (class, score) pairs from zipping the "classes" and
"scores" features.
"""
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function



import tensorflow as tf
from tensorflow_model_analysis.eval_saved_model import export
from tensorflow_model_analysis.eval_saved_model.example_trainers import util

from tensorflow.python.estimator.canned import metric_keys
from tensorflow.python.estimator.canned import prediction_keys


def simple_fixed_prediction_classifier(export_path, eval_export_path):
  """Exports a simple fixed prediction classifier."""

  def model_fn(features, labels, mode, params):
    """Model function for custom estimator."""
    del labels
    del params
    classes = tf.sparse_tensor_to_dense(features['classes'], default_value='?')
    scores = tf.sparse_tensor_to_dense(features['scores'], default_value=0.0)

    predictions = {
        prediction_keys.PredictionKeys.LOGITS: scores,
        prediction_keys.PredictionKeys.PROBABILITIES: scores,
        prediction_keys.PredictionKeys.CLASSES: classes,
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=predictions,
          export_outputs={
              tf.saved_model.signature_constants.
              DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                  tf.estimator.export.ClassificationOutput(
                      scores=scores, classes=classes),
          })

    # Note that this is always going to be 0.
    loss = tf.losses.mean_squared_error(scores, tf.ones_like(scores))
    train_op = tf.assign_add(tf.train.get_global_step(), 1)
    eval_metric_ops = {
        metric_keys.MetricKeys.LOSS_MEAN: tf.metrics.mean(loss),
    }

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        predictions=predictions,
        eval_metric_ops=eval_metric_ops)

  def train_input_fn():
    """Train input function."""
    classes = tf.SparseTensor(
        values=[
            'first1',
            'first2',
            'second1',
            'third1',
            'third2',
            'third3',
            'fourth1',
        ],
        indices=[[0, 0], [0, 1], [1, 0], [2, 0], [2, 1], [2, 2], [3, 0]],
        dense_shape=[4, 3])
    scores = tf.SparseTensor(
        values=[0.9, 0.9, 0.9, 0.9, 0.8, 0.7, 0.9],
        indices=[[0, 0], [0, 1], [1, 0], [2, 0], [2, 1], [2, 2], [3, 0]],
        dense_shape=[4, 3])
    return {
        'classes': classes,
        'scores': scores,
    }, classes

  estimator = tf.estimator.Estimator(model_fn=model_fn)
  estimator.train(input_fn=train_input_fn, steps=1)

  serving_input_receiver_fn = (
      tf.estimator.export.build_parsing_serving_input_receiver_fn(
          feature_spec={
              'classes': tf.VarLenFeature(dtype=tf.string),
              'scores': tf.VarLenFeature(dtype=tf.float32)
          }))
  eval_input_receiver_fn = export.build_parsing_eval_input_receiver_fn(
      feature_spec={
          'classes': tf.VarLenFeature(dtype=tf.string),
          'scores': tf.VarLenFeature(dtype=tf.float32),
          'labels': tf.VarLenFeature(dtype=tf.string),
      },
      label_key='labels')

  return util.export_model_and_eval_model(
      estimator=estimator,
      serving_input_receiver_fn=serving_input_receiver_fn,
      eval_input_receiver_fn=eval_input_receiver_fn,
      export_path=export_path,
      eval_export_path=eval_export_path)
