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
"""Trains and exports a simple linear classifier.

The true model is language == 'english'.

The model has the standard metrics added by LinearClassifier, plus additional
metrics added using tf.estimator.

This model also extracts an additional slice_key feature for evaluation
(this feature is not used in training).
"""
from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

# Standard Imports

import tensorflow as tf
from tensorflow_model_analysis.eval_saved_model import export
from tensorflow_model_analysis.eval_saved_model.example_trainers import util


def simple_linear_classifier(export_path, eval_export_path):
  """Trains and exports a simple linear classifier."""

  feature_spec = tf.feature_column.make_parse_example_spec(
      util.linear_columns(False))
  eval_feature_spec = tf.feature_column.make_parse_example_spec(
      util.linear_columns(True) +
      [tf.feature_column.categorical_column_with_hash_bucket('slice_key', 100)])

  classifier = tf.estimator.LinearClassifier(
      feature_columns=util.linear_columns(),
      loss_reduction=tf.losses.Reduction.SUM)
  classifier = tf.estimator.add_metrics(classifier,
                                        util.classifier_extra_metrics)
  classifier.train(
      input_fn=util.make_classifier_input_fn(
          tf.feature_column.make_parse_example_spec(util.linear_columns(True))),
      steps=1000)

  return util.export_model_and_eval_model(
      estimator=classifier,
      serving_input_receiver_fn=(
          tf.estimator.export.build_parsing_serving_input_receiver_fn(
              feature_spec)),
      eval_input_receiver_fn=export.build_parsing_eval_input_receiver_fn(
          eval_feature_spec, label_key='label'),
      export_path=export_path,
      eval_export_path=eval_export_path)
