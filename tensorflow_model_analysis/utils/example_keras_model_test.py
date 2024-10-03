# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Tests for example_keras_model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import tempfile

import numpy as np
import six
from tensorflow import keras
import tensorflow.compat.v1 as tf
import tensorflow_model_analysis as tfma
from tensorflow_model_analysis.utils import example_keras_model

from google.protobuf import text_format


tf.compat.v1.enable_eager_execution()


class ExampleModelTest(tf.test.TestCase):

  def setUp(self):
    super(ExampleModelTest, self).setUp()
    self._base_dir = tempfile.gettempdir()

    self._model_dir = os.path.join(
        self._base_dir,
        'train',
        datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
    )

  def _create_example(self, language, label, age, slice_value):
    example = tf.train.Example()
    example.features.feature[example_keras_model.LANGUAGE].bytes_list.value[
        :
    ] = [six.ensure_binary(language, 'utf8')]
    example.features.feature[example_keras_model.SLICE].bytes_list.value[:] = [
        six.ensure_binary(slice_value, 'utf8')
    ]
    example.features.feature[example_keras_model.LABEL].float_list.value[:] = [
        label
    ]
    example.features.feature[example_keras_model.AGE].float_list.value[:] = [
        age
    ]
    return example

  def _create_data(self):
    examples = []
    examples.append(self._create_example('nontoxic', 0.0, 3, 'slice1'))
    examples.append(self._create_example('toxic comment', 0.0, 4, 'slice1'))
    examples.append(self._create_example('japanese', 0.0, 1, 'slice1'))
    examples.append(self._create_example('hindi', 0.0, 10, 'slice2'))
    examples.append(self._create_example('chinese', 0.0, 90, 'slice2'))
    examples.append(self._create_example('english', 1.0, 45, 'slice3'))
    examples.append(self._create_example('english', 1.0, 15, 'slice3'))
    examples.append(self._create_example('abcdef', 0.0, 10, 'slice3'))
    examples.append(self._create_example('english', 1.0, 24, 'slice3'))
    examples.append(self._create_example('abc', 0.0, 43, 'slice1'))
    examples.append(self._create_example('abcdef', 0.0, 75, 'slice3'))
    examples.append(self._create_example('random', 0.0, 39, 'slice1'))
    return examples

  def _write_tf_records(self, examples):
    data_location = os.path.join(self._base_dir, 'input_data.rio')
    with tf.io.TFRecordWriter(data_location) as writer:
      for example in examples:
        writer.write(example.SerializeToString())
    return data_location

  def test_example_keras_model(self):
    data = self._create_data()
    classifier = example_keras_model.ExampleClassifierModel(
        example_keras_model.LANGUAGE
    )
    classifier.compile(optimizer=keras.optimizers.Adam(), loss='mse')
    classifier.fit(
        tf.constant([e.SerializeToString() for e in data]),
        np.array([
            e.features.feature[example_keras_model.LABEL].float_list.value[:][0]
            for e in data
        ]),
    )
    classifier.save(self._model_dir, save_format='tf')

    eval_config = text_format.Parse(
        """
        model_specs {
          signature_name: "serving_default"
          prediction_key: "predictions" # placeholder
          label_key: "label" # placeholder
        }
        slicing_specs {}
        slicing_specs {
          feature_keys: ["my_slice"]
        }
        metrics_specs {
          metrics {
            class_name: "ExampleCount"
          }
          metrics {
            class_name: "Accuracy"
          }
        }
  """,
        tfma.EvalConfig(),
    )

    validate_tf_file_path = self._write_tf_records(data)
    tfma_eval_result_path = os.path.join(self._model_dir, 'tfma_eval_result')
    example_keras_model.evaluate_model(
        self._model_dir,
        validate_tf_file_path,
        tfma_eval_result_path,
        eval_config,
    )

    evaluation_results = tfma.load_eval_result(tfma_eval_result_path)

    expected_slice_keys = [
        (),
        (('my_slice', 'slice1'),),
        (('my_slice', 'slice2'),),
        (('my_slice', 'slice3'),),
    ]
    slice_keys = [
        slice_key for slice_key, _ in evaluation_results.slicing_metrics
    ]
    self.assertEqual(set(expected_slice_keys), set(slice_keys))
    metric_values = dict(evaluation_results.slicing_metrics)[(
        ('my_slice', 'slice1'),
    )]['']['']
    self.assertEqual(metric_values['example_count'], {'doubleValue': 5.0})

    self.assertEqual(
        metric_values['accuracy'],
        {'doubleValue': 0.0},
    )


