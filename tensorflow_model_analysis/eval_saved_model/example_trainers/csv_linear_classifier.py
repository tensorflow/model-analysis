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
"""Trains and exports a simple linear classifier that reads from a CSV file.

The true model is language == 'english'.

Note that this model only has standard metrics (no custom metrics), in contrast
to linear_classifier.py.

"""
from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

# Standard Imports

import tensorflow as tf
from tensorflow_model_analysis.eval_saved_model import export
from tensorflow_model_analysis.eval_saved_model.example_trainers import util


def simple_csv_linear_classifier(export_path, eval_export_path):
  """Trains and exports a simple linear classifier."""

  def parse_csv(rows_string_tensor):
    """Takes the string input tensor and returns a dict of rank-2 tensors."""

    csv_columns = ['age', 'language', 'label']
    csv_column_defaults = [[0.0], ['unknown'], [0.0]]

    # Takes a rank-1 tensor and converts it into rank-2 tensor
    # Example if the data is ['csv,line,1', 'csv,line,2', ..] to
    # [['csv,line,1'], ['csv,line,2']] which after parsing will result in a
    # tuple of tensors: [['csv'], ['csv']], [['line'], ['line']], [[1], [2]]
    row_columns = tf.expand_dims(rows_string_tensor, -1)
    columns = tf.io.decode_csv(
        records=row_columns, record_defaults=csv_column_defaults)
    features = dict(zip(csv_columns, columns))
    return features

  def eval_input_receiver_fn():
    """Eval input receiver function."""
    csv_row = tf.compat.v1.placeholder(
        dtype=tf.string, shape=[None], name='input_csv_row')
    features = parse_csv(csv_row)
    receiver_tensors = {'examples': csv_row}

    return export.EvalInputReceiver(
        features=features,
        labels=features['label'],
        receiver_tensors=receiver_tensors)

  def input_fn():
    """Train input function."""
    return {
        'age':
            tf.constant([[1], [2], [3], [4]]),
        'language':
            tf.SparseTensor(
                values=['english', 'english', 'chinese', 'chinese'],
                indices=[[0, 0], [1, 0], [2, 0], [3, 0]],
                dense_shape=[4, 1])
    }, tf.constant([[1], [1], [0], [0]])

  language = tf.feature_column.categorical_column_with_vocabulary_list(
      'language', ['english', 'chinese'])
  age = tf.feature_column.numeric_column('age')
  all_features = [age, language]
  feature_spec = tf.feature_column.make_parse_example_spec(all_features)

  classifier = tf.estimator.LinearClassifier(
      feature_columns=all_features, loss_reduction=tf.losses.Reduction.SUM)
  classifier.train(input_fn=input_fn, steps=1000)

  return util.export_model_and_eval_model(
      estimator=classifier,
      serving_input_receiver_fn=(
          tf.estimator.export.build_parsing_serving_input_receiver_fn(
              feature_spec)),
      eval_input_receiver_fn=eval_input_receiver_fn,
      export_path=export_path,
      eval_export_path=eval_export_path)
