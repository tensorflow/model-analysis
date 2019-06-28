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
"""Helper functions for building example regressor Estimator models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Standard Imports

import tensorflow as tf
from tensorflow_model_analysis.eval_saved_model import export
from tensorflow_model_analysis.eval_saved_model import util

from tensorflow.core.example import example_pb2


def make_regressor_input_fn(feature_spec):
  """Train input function.

  Args:
    feature_spec: a dictionary mapping feature_name to Tensor or SparseTensor.

  Returns:
    A function.
  """

  def _input_fn():
    """Example-based input function."""

    serialized_examples = [
        x.SerializeToString() for x in [
            util.make_example(age=1.0, language='english', label=4.0),
            util.make_example(age=2.0, language='english', label=7.0),
            util.make_example(age=3.0, language='english', label=10.0),
            util.make_example(age=4.0, language='english', label=13.0),
            util.make_example(age=1.0, language='chinese', label=3.0),
            util.make_example(age=2.0, language='chinese', label=6.0),
            util.make_example(age=3.0, language='chinese', label=9.0),
            util.make_example(age=4.0, language='chinese', label=12.0),
            util.make_example(age=10.0, language='english', label=31.0),
            util.make_example(age=20.0, language='english', label=61.0),
            util.make_example(age=30.0, language='english', label=91.0),
            util.make_example(age=40.0, language='english', label=121.0),
            util.make_example(age=10.0, language='chinese', label=30.0),
            util.make_example(age=20.0, language='chinese', label=60.0),
            util.make_example(age=30.0, language='chinese', label=90.0),
            util.make_example(age=40.0, language='chinese', label=120.0)
        ]
    ]
    features = tf.io.parse_example(
        serialized=serialized_examples, features=feature_spec)
    labels = features.pop('label')
    return features, labels

  return _input_fn


def make_classifier_input_fn(feature_spec, n_classes=2, label_vocabulary=None):
  """Train input function.

  Args:
    feature_spec: a dictionary mapping feature_name to Tensor or SparseTensor.
    n_classes: set for multiclass.
    label_vocabulary: (Optional) Label vocabulary to use for labels.

  Returns:
    A function.
  """

  def _input_fn():
    """Example-based input function."""

    english_label = label_vocabulary[1] if label_vocabulary else 1.0
    chinese_label = label_vocabulary[0] if label_vocabulary else 0.0
    if n_classes > 2:
      # For multi-class labels, English is class 0, Chinese is class 1.
      chinese_label = label_vocabulary[1] if label_vocabulary else 1
      english_label = label_vocabulary[0] if label_vocabulary else 0

    serialized_examples = [
        x.SerializeToString() for x in [
            util.make_example(age=1.0, language='english', label=english_label),
            util.make_example(age=2.0, language='english', label=english_label),
            util.make_example(age=3.0, language='chinese', label=chinese_label),
            util.make_example(age=4.0, language='chinese', label=chinese_label)
        ]
    ]
    features = tf.io.parse_example(
        serialized=serialized_examples, features=feature_spec)
    labels = features.pop('label')
    if n_classes > 2 and not label_vocabulary:
      labels = tf.sparse.to_dense(labels, default_value=-1)

    return features, labels

  return _input_fn


def make_example(age, language, label=None):
  example = example_pb2.Example()
  example.features.feature['age'].float_list.value.append(age)
  example.features.feature['language'].bytes_list.value.append(language)
  if label:
    if isinstance(label, list):
      example.features.feature['label'].int64_list.value.extend(label)
    else:
      example.features.feature['label'].float_list.value.append(label)
  return example


def linear_columns(include_label_column=False):
  """Return feature_columns for linear model."""
  language = tf.feature_column.indicator_column(
      tf.feature_column.categorical_column_with_vocabulary_list(
          key='language', vocabulary_list=('english', 'chinese')))
  age = tf.feature_column.numeric_column(key='age', default_value=0.0)
  features = [age, language]
  if include_label_column:
    label = tf.feature_column.numeric_column(key='label', default_value=0.0)
    features.append(label)
  return features


def dnn_columns(include_label_column=False, n_classes=2):
  """Return feature_columns for DNN model."""
  language = tf.feature_column.embedding_column(
      tf.feature_column.categorical_column_with_vocabulary_list(
          key='language', vocabulary_list=('english', 'chinese')),
      dimension=1)
  age = tf.feature_column.numeric_column(key='age', default_value=0.0)
  features = [age, language]
  if include_label_column:
    label = tf.feature_column.numeric_column(key='label', default_value=0.0)
    if n_classes > 2:
      label = tf.feature_column.categorical_column_with_identity(
          key='label', num_buckets=n_classes)
    features.append(label)
  return features


def regressor_extra_metrics(features, labels, predictions):
  return {
      'my_mean_prediction':
          tf.compat.v1.metrics.mean(predictions['predictions']),
      'my_mean_age':
          tf.compat.v1.metrics.mean(features['age']),
      'my_mean_label':
          tf.compat.v1.metrics.mean(labels),
      'my_mean_age_times_label':
          tf.compat.v1.metrics.mean(labels * features['age']),
  }


def classifier_extra_metrics(features, labels, predictions):
  """Returns extra metrics to use with classifier."""
  if 'logistic' in predictions:
    metrics = {
        'my_mean_prediction':
            tf.compat.v1.metrics.mean(predictions['logistic']),
        'my_mean_age':
            tf.compat.v1.metrics.mean(features['age']),
    }
    if labels.dtype != tf.string:
      metrics.update({
          'my_mean_label':
              tf.compat.v1.metrics.mean(labels),
          'my_mean_age_times_label':
              tf.compat.v1.metrics.mean(labels * features['age']),
      })
    return metrics
  # Logistic won't be present in multiclass cases.
  return {
      'mean_english_prediction':
          tf.compat.v1.metrics.mean(predictions['probabilities'][0]),
      'my_mean_age':
          tf.compat.v1.metrics.mean(features['age']),
  }


def export_model_and_eval_model(estimator,
                                serving_input_receiver_fn=None,
                                eval_input_receiver_fn=None,
                                export_path=None,
                                eval_export_path=None):
  """Export SavedModel and EvalSavedModel.

  Args:
    estimator: Estimator to export.
    serving_input_receiver_fn: Serving input receiver function.
    eval_input_receiver_fn: Eval input receiver function.
    export_path: Export path. If None, inference model is not exported.
    eval_export_path: Eval export path. If None, EvalSavedModel is not exported.

  Returns:
    Tuple of (path to the export directory, path to eval export directory).
  """
  export_path_result = None
  eval_export_path_result = None

  if export_path and serving_input_receiver_fn:
    export_path_result = estimator.export_saved_model(
        export_dir_base=export_path,
        serving_input_receiver_fn=serving_input_receiver_fn)
  if eval_export_path and eval_input_receiver_fn:
    eval_export_path_result = export.export_eval_savedmodel(
        estimator=estimator,
        export_dir_base=eval_export_path,
        eval_input_receiver_fn=eval_input_receiver_fn,
        serving_input_receiver_fn=serving_input_receiver_fn)

  return export_path_result, eval_export_path_result
