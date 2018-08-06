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

from __future__ import print_function



import tensorflow as tf
from tensorflow_model_analysis.eval_saved_model import export
from tensorflow_model_analysis.eval_saved_model.example_trainers import util


def simple_dnn_classifier(export_path, eval_export_path):
  """Trains and exports a simple DNN classifier."""

  feature_spec = tf.feature_column.make_parse_example_spec(
      feature_columns=util.dnn_columns(True))
  classifier = tf.estimator.DNNClassifier(
      hidden_units=[4], feature_columns=util.dnn_columns(False))
  classifier = tf.contrib.estimator.add_metrics(classifier,
                                                util.classifier_extra_metrics)
  classifier.train(
      input_fn=util.make_classifier_input_fn(feature_spec), steps=1000)

  return util.export_model_and_eval_model(
      estimator=classifier,
      serving_input_receiver_fn=(
          tf.estimator.export.build_parsing_serving_input_receiver_fn(
              tf.feature_column.make_parse_example_spec(
                  util.dnn_columns(False)))),
      eval_input_receiver_fn=export.build_parsing_eval_input_receiver_fn(
          tf.feature_column.make_parse_example_spec(util.dnn_columns(True)),
          label_key='label'),
      export_path=export_path,
      eval_export_path=eval_export_path)
