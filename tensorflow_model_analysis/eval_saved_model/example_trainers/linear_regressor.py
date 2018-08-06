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
"""Trains and exports a simple linear regressor.

The true model is age * 3 + (language == 'english')

The model has the standard metrics added by LinearRegressor, plus additional
metrics added using tf.contrib.estimator.

This model also extracts an additional slice_key feature for evaluation
(this feature is not used in training).
"""
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function



import tensorflow as tf
from tensorflow_model_analysis.eval_saved_model import export
from tensorflow_model_analysis.eval_saved_model.example_trainers import util


def simple_linear_regressor(export_path, eval_export_path):
  """Trains and exports a simple linear regressor."""

  feature_spec = tf.feature_column.make_parse_example_spec(
      util.linear_columns(False))
  eval_feature_spec = tf.feature_column.make_parse_example_spec(
      util.linear_columns(True) +
      [tf.feature_column.categorical_column_with_hash_bucket('slice_key', 100)])

  regressor = tf.estimator.LinearRegressor(
      feature_columns=util.linear_columns())
  regressor = tf.contrib.estimator.add_metrics(regressor,
                                               util.regressor_extra_metrics)
  regressor.train(
      input_fn=util.make_regressor_input_fn(
          tf.feature_column.make_parse_example_spec(util.linear_columns(True))),
      steps=3000)

  return util.export_model_and_eval_model(
      estimator=regressor,
      serving_input_receiver_fn=(
          tf.estimator.export.build_parsing_serving_input_receiver_fn(
              feature_spec)),
      eval_input_receiver_fn=export.build_parsing_eval_input_receiver_fn(
          eval_feature_spec, label_key='label'),
      export_path=export_path,
      eval_export_path=eval_export_path)
