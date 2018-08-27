# Copyright 2018 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Trains and exports a simple linear classifier with multivalent features.

The true model is animals CONTAINS 'cat' and 'dog'.
"""
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function



import tensorflow as tf
from tensorflow_model_analysis.eval_saved_model import export
from tensorflow_model_analysis.eval_saved_model.example_trainers import util


def simple_linear_classifier_multivalent(export_path, eval_export_path):
  """Trains and exports a simple linear classifier with multivalent features."""

  def input_fn():
    """Train input function."""
    return {
        'animals':
            tf.SparseTensor(
                values=[
                    'cat', 'dog', 'bird', 'cat', 'dog', 'cat', 'bird', 'dog',
                    'bird', 'cat', 'dog', 'bird'
                ],
                indices=[[0, 0], [1, 0], [2, 0], [4, 0], [4, 1], [5, 0], [5, 1],
                         [6, 0], [6, 1], [7, 0], [7, 1], [7, 2]],
                dense_shape=[8, 3])
    }, tf.constant([[0], [0], [0], [0], [1], [0], [0], [1]])

  animals = tf.contrib.layers.sparse_column_with_keys('animals',
                                                      ['bird', 'cat', 'dog'])
  label = tf.contrib.layers.real_valued_column('label')

  all_features = [animals]
  feature_spec = tf.contrib.layers.create_feature_spec_for_parsing(all_features)
  eval_feature_spec = tf.contrib.layers.create_feature_spec_for_parsing(
      all_features + [label])

  classifier = tf.estimator.LinearClassifier(feature_columns=all_features)
  classifier.train(input_fn=input_fn, steps=5000)

  return util.export_model_and_eval_model(
      estimator=classifier,
      serving_input_receiver_fn=(
          tf.estimator.export.build_parsing_serving_input_receiver_fn(
              feature_spec)),
      eval_input_receiver_fn=export.build_parsing_eval_input_receiver_fn(
          eval_feature_spec, label_key='label'),
      export_path=export_path,
      eval_export_path=eval_export_path)
