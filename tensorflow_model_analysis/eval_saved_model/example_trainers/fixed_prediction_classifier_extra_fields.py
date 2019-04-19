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
"""Exports a simple "fixed prediction" classifier using tf.Learn.

This model generates (class, score) pairs from zipping the "classes" and
"scores" features.

The eval_input_receiver_fn also parses the "fixed_float", "fixed_string",
"fixed_int", and "var_float", "var_string", "var_int" features.
"""
from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

# Standard Imports

import tensorflow as tf
from tensorflow_model_analysis.eval_saved_model import export
from tensorflow_model_analysis.eval_saved_model.example_trainers import fixed_prediction_classifier
from tensorflow_model_analysis.eval_saved_model.example_trainers import util


def simple_fixed_prediction_classifier_extra_fields(export_path,
                                                    eval_export_path):
  """Exports a simple fixed prediction classifier that parses extra fields."""

  estimator = tf.estimator.Estimator(
      model_fn=fixed_prediction_classifier.model_fn)
  estimator.train(input_fn=fixed_prediction_classifier.train_input_fn, steps=1)

  serving_input_receiver_fn = (
      tf.estimator.export.build_parsing_serving_input_receiver_fn(
          feature_spec={
              'classes': tf.io.VarLenFeature(dtype=tf.string),
              'scores': tf.io.VarLenFeature(dtype=tf.float32)
          }))
  eval_input_receiver_fn = export.build_parsing_eval_input_receiver_fn(
      feature_spec={
          'classes': tf.io.VarLenFeature(dtype=tf.string),
          'scores': tf.io.VarLenFeature(dtype=tf.float32),
          'labels': tf.io.VarLenFeature(dtype=tf.string),
          'fixed_float': tf.io.FixedLenFeature([1], dtype=tf.float32),
          'fixed_string': tf.io.FixedLenFeature([1], dtype=tf.string),
          'fixed_int': tf.io.FixedLenFeature([1], dtype=tf.int64),
          'var_float': tf.io.VarLenFeature(dtype=tf.float32),
          'var_string': tf.io.VarLenFeature(dtype=tf.string),
          'var_int': tf.io.VarLenFeature(dtype=tf.int64),
      },
      label_key='labels')

  return util.export_model_and_eval_model(
      estimator=estimator,
      serving_input_receiver_fn=serving_input_receiver_fn,
      eval_input_receiver_fn=eval_input_receiver_fn,
      export_path=export_path,
      eval_export_path=eval_export_path)
