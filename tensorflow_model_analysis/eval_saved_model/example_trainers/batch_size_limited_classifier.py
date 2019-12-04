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
"""Exports a simple model which explicitly limits batch size for testing."""
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


def model_fn(features, labels, mode, config):
  """Model function for custom estimator."""
  del labels
  del config
  classes = features['classes']
  scores = features['scores']

  with tf.control_dependencies(
      [tf.assert_less(tf.shape(classes)[0], tf.constant(2))]):
    scores = tf.identity(scores)

  predictions = {
      prediction_keys.PredictionKeys.LOGITS: scores,
      prediction_keys.PredictionKeys.PROBABILITIES: scores,
      prediction_keys.PredictionKeys.PREDICTIONS: scores,
      prediction_keys.PredictionKeys.CLASSES: classes,
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                tf.estimator.export.ClassificationOutput(
                    scores=scores, classes=classes),
        })

  loss = tf.constant(0.0)
  train_op = tf.compat.v1.assign_add(tf.compat.v1.train.get_global_step(), 1)
  eval_metric_ops = {
      metric_keys.MetricKeys.LOSS_MEAN: tf.compat.v1.metrics.mean(loss),
  }

  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      predictions=predictions,
      eval_metric_ops=eval_metric_ops)


def train_input_fn():
  """Train input function."""
  classes = tf.constant('first', shape=[1])
  scores = tf.constant(1.0, shape=[1])
  return {
      'classes': classes,
      'scores': scores,
  }, classes


def simple_batch_size_limited_classifier(export_path, eval_export_path):
  """Exports a simple fixed prediction classifier."""

  estimator = tf.estimator.Estimator(model_fn=model_fn)
  estimator.train(input_fn=train_input_fn, steps=1)

  serving_input_receiver_fn = (
      tf.estimator.export.build_parsing_serving_input_receiver_fn(
          feature_spec={
              'classes': tf.io.FixedLenFeature([], dtype=tf.string),
              'scores': tf.io.FixedLenFeature([], dtype=tf.float32)
          }))
  eval_input_receiver_fn = export.build_parsing_eval_input_receiver_fn(
      feature_spec={
          'classes': tf.io.FixedLenFeature([], dtype=tf.string),
          'scores': tf.io.FixedLenFeature([], dtype=tf.float32),
          'labels': tf.io.FixedLenFeature([], dtype=tf.string),
      },
      label_key='labels')

  return util.export_model_and_eval_model(
      estimator=estimator,
      serving_input_receiver_fn=serving_input_receiver_fn,
      eval_input_receiver_fn=eval_input_receiver_fn,
      export_path=export_path,
      eval_export_path=eval_export_path)
