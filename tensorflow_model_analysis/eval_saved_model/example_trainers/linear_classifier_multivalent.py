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


def simple_linear_classifier_multivalent(export_path, eval_export_path):
  """Trains and exports a simple linear classifier with multivalent features."""

  def eval_input_receiver_fn():
    """Eval input receiver function."""
    serialized_tf_example = tf.placeholder(
        dtype=tf.string, shape=[None], name='input_example_tensor')

    animals = tf.contrib.layers.sparse_column_with_keys('animals',
                                                        ['bird', 'cat', 'dog'])
    label = tf.contrib.layers.real_valued_column('label')
    all_features = [animals, label]
    feature_spec = tf.contrib.layers.create_feature_spec_for_parsing(
        all_features)
    receiver_tensors = {'examples': serialized_tf_example}
    features = tf.parse_example(serialized_tf_example, feature_spec)

    return export.EvalInputReceiver(
        features=features,
        receiver_tensors=receiver_tensors,
        labels=features['label'])

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
  all_features = [animals]
  feature_spec = tf.contrib.layers.create_feature_spec_for_parsing(all_features)

  classifier = tf.estimator.LinearClassifier(feature_columns=all_features)
  classifier.train(input_fn=input_fn, steps=5000)

  export_dir = None
  eval_export_dir = None
  if export_path:
    export_dir = classifier.export_savedmodel(
        export_dir_base=export_path,
        serving_input_receiver_fn=tf.estimator.export.
        build_parsing_serving_input_receiver_fn(feature_spec))

  if eval_export_path:
    eval_export_dir = export.export_eval_savedmodel(
        estimator=classifier,
        export_dir_base=eval_export_path,
        eval_input_receiver_fn=eval_input_receiver_fn)

  return export_dir, eval_export_dir
