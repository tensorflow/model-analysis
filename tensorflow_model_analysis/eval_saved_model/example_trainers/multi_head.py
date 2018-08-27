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
"""Trains and exports a simple multi-headed model.

Note that this model uses the CONTRIB estimators (not the CORE estimators).

The true model for the English head is language == 'english'.
The true model for the Chinese head is language == 'chinese'.
The true model for the Other head is language == 'other'.
"""
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function



import tensorflow as tf
from tensorflow_model_analysis.eval_saved_model import export
from tensorflow_model_analysis.eval_saved_model.example_trainers import util


def simple_multi_head(export_path, eval_export_path):
  """Trains and exports a simple multi-headed model."""

  def eval_input_receiver_fn():
    """Eval input receiver function."""
    serialized_tf_example = tf.placeholder(
        dtype=tf.string, shape=[None], name='input_example_tensor')

    language = tf.contrib.layers.sparse_column_with_keys(
        'language', ['english', 'chinese', 'other'])
    age = tf.contrib.layers.real_valued_column('age')
    english_label = tf.contrib.layers.real_valued_column('english_label')
    chinese_label = tf.contrib.layers.real_valued_column('chinese_label')
    other_label = tf.contrib.layers.real_valued_column('other_label')
    all_features = [age, language, english_label, chinese_label, other_label]
    feature_spec = tf.contrib.layers.create_feature_spec_for_parsing(
        all_features)
    receiver_tensors = {'examples': serialized_tf_example}
    features = tf.parse_example(serialized_tf_example, feature_spec)

    labels = {
        'english_head': features['english_label'],
        'chinese_head': features['chinese_label'],
        'other_head': features['other_label'],
    }

    return export.EvalInputReceiver(
        features=features, receiver_tensors=receiver_tensors, labels=labels)

  def input_fn():
    """Train input function."""
    labels = {
        'english_head': tf.constant([[1], [1], [0], [0], [0], [0]]),
        'chinese_head': tf.constant([[0], [0], [1], [1], [0], [0]]),
        'other_head': tf.constant([[0], [0], [0], [0], [1], [1]])
    }
    features = {
        'age':
            tf.constant([[1], [2], [3], [4], [5], [6]]),
        'language':
            tf.SparseTensor(
                values=[
                    'english', 'english', 'chinese', 'chinese', 'other', 'other'
                ],
                indices=[[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]],
                dense_shape=[6, 1]),
    }
    return features, labels

  language = tf.contrib.layers.sparse_column_with_keys(
      'language', ['english', 'chinese', 'other'])
  age = tf.contrib.layers.real_valued_column('age')
  all_features = [age, language]
  feature_spec = tf.contrib.layers.create_feature_spec_for_parsing(all_features)

  english_head = tf.contrib.estimator.binary_classification_head(
      name='english_head')
  chinese_head = tf.contrib.estimator.binary_classification_head(
      name='chinese_head')
  other_head = tf.contrib.estimator.binary_classification_head(
      name='other_head')
  combined_head = tf.contrib.estimator.multi_head(
      [english_head, chinese_head, other_head])

  estimator = tf.contrib.estimator.DNNLinearCombinedEstimator(
      head=combined_head,
      dnn_feature_columns=[],
      dnn_optimizer=tf.train.AdagradOptimizer(learning_rate=0.01),
      dnn_hidden_units=[],
      linear_feature_columns=[language, age],
      linear_optimizer=tf.train.FtrlOptimizer(learning_rate=0.05))
  estimator.train(input_fn=input_fn, steps=1000)

  return util.export_model_and_eval_model(
      estimator=estimator,
      serving_input_receiver_fn=(
          tf.estimator.export.build_parsing_serving_input_receiver_fn(
              feature_spec)),
      eval_input_receiver_fn=eval_input_receiver_fn,
      export_path=export_path,
      eval_export_path=eval_export_path)
