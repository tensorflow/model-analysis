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
"""Trains and exports a simple DNN classifier.

The true model is language == 'english'.

The model has the standard metrics added by DNNClassifier, plus additional
metrics added using tf.contrib.estimator.
"""
from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

# Standard Imports

import tensorflow as tf
from tensorflow_model_analysis.eval_saved_model import export
from tensorflow_model_analysis.eval_saved_model.example_trainers import util


def build_parsing_eval_input_receiver_fn(feature_spec, label_key):
  """Builds parsing eval receiver fn that handles sparse labels."""

  def eval_input_receiver_fn():
    """An input_fn that expects a serialized tf.Example."""
    # Note it's *required* that the batch size should be variable for TFMA.
    serialized_tf_example = tf.placeholder(
        dtype=tf.string, shape=[None], name='input_example_tensor')
    features = tf.parse_example(serialized_tf_example, feature_spec)
    labels = None if label_key is None else features[label_key]

    if isinstance(labels, tf.SparseTensor):
      # This bit here is why a custom eval_input_receiver_fn is specified.
      labels = tf.sparse_tensor_to_dense(labels, default_value=-1)

    return export.EvalInputReceiver(
        features=features,
        labels=labels,
        receiver_tensors={'examples': serialized_tf_example})

  return eval_input_receiver_fn


def get_simple_dnn_classifier_and_metadata(n_classes=2):
  feature_spec = tf.feature_column.make_parse_example_spec(
      feature_columns=util.dnn_columns(True, n_classes=n_classes))
  classifier = tf.estimator.DNNClassifier(
      hidden_units=[4],
      feature_columns=util.dnn_columns(False),
      n_classes=n_classes)
  classifier = tf.contrib.estimator.add_metrics(classifier,
                                                util.classifier_extra_metrics)
  return {
      'estimator':
          classifier,
      'serving_input_receiver_fn': (
          tf.estimator.export.build_parsing_serving_input_receiver_fn(
              tf.feature_column.make_parse_example_spec(
                  util.dnn_columns(False)))),
      'eval_input_receiver_fn':
          build_parsing_eval_input_receiver_fn(
              tf.feature_column.make_parse_example_spec(
                  util.dnn_columns(True, n_classes=n_classes)),
              label_key='label'),
      'train_input_fn':
          util.make_classifier_input_fn(feature_spec, n_classes),
  }


def simple_dnn_classifier(export_path, eval_export_path, n_classes=2):
  """Trains and exports a simple DNN classifier."""

  estimator_metadata = get_simple_dnn_classifier_and_metadata(
      n_classes=n_classes)
  estimator_metadata['estimator'].train(
      input_fn=estimator_metadata['train_input_fn'], steps=1000)

  return util.export_model_and_eval_model(
      estimator=estimator_metadata['estimator'],
      serving_input_receiver_fn=estimator_metadata['serving_input_receiver_fn'],
      eval_input_receiver_fn=estimator_metadata['eval_input_receiver_fn'],
      export_path=export_path,
      eval_export_path=eval_export_path)
