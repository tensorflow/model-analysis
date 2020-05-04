# Lint as: python3
# Copyright 2019 Google LLC
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
"""Tests for model_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import config
from tensorflow_model_analysis import model_util


class ModelUtilTest(tf.test.TestCase, parameterized.TestCase):

  def testRebatchByInputNames(self):
    extracts = [{
        'features': {
            'a': np.array([1.1]),
            'b': np.array([1.2])
        }
    }, {
        'features': {
            'a': np.array([2.1]),
            'b': np.array([2.2])
        }
    }]
    expected = {
        'a': [np.array([1.1]), np.array([2.1])],
        'b': [np.array([1.2]), np.array([2.2])]
    }
    got = model_util.rebatch_by_input_names(extracts, input_names=['a', 'b'])
    self.assertEqual(expected, got)

  def testRebatchByInputNamesSingleDimInput(self):
    extracts = [{
        'features': {
            'a': np.array([1.1]),
            'b': np.array([1.2])
        }
    }, {
        'features': {
            'a': np.array([2.1]),
            'b': np.array([2.2])
        }
    }]
    expected = {'a': [1.1, 2.1], 'b': [1.2, 2.2]}
    input_specs = {
        'a': tf.TensorSpec(shape=(2,)),
        'b': tf.TensorSpec(shape=(2,))
    }
    got = model_util.rebatch_by_input_names(
        extracts, input_names=['a', 'b'], input_specs=input_specs)
    self.assertEqual(expected, got)
    self.assertNotIsInstance(got['a'][0], np.ndarray)

  def testFilterTensorsByInputNames(self):
    tensors = {
        'f1': tf.constant([[1.1], [2.1]], dtype=tf.float32),
        'f2': tf.constant([[1], [2]], dtype=tf.int64),
        'f3': tf.constant([['hello'], ['world']], dtype=tf.string)
    }
    filtered_tensors = model_util.filter_tensors_by_input_names(
        tensors, ['f1', 'f3'])
    self.assertLen(filtered_tensors, 2)
    self.assertAllEqual(
        tf.constant([[1.1], [2.1]], dtype=tf.float32), filtered_tensors['f1'])
    self.assertAllEqual(
        tf.constant([['hello'], ['world']], dtype=tf.string),
        filtered_tensors['f3'])

  def testFilterTensorsByInputNamesKeras(self):
    tensors = {
        'f1': tf.constant([[1.1], [2.1]], dtype=tf.float32),
        'f2': tf.constant([[1], [2]], dtype=tf.int64),
        'f3': tf.constant([['hello'], ['world']], dtype=tf.string)
    }
    filtered_tensors = model_util.filter_tensors_by_input_names(
        tensors, [
            'f1' + model_util.KERAS_INPUT_SUFFIX,
            'f3' + model_util.KERAS_INPUT_SUFFIX
        ])
    self.assertLen(filtered_tensors, 2)
    self.assertAllEqual(
        tf.constant([[1.1], [2.1]], dtype=tf.float32),
        filtered_tensors['f1' + model_util.KERAS_INPUT_SUFFIX])
    self.assertAllEqual(
        tf.constant([['hello'], ['world']], dtype=tf.string),
        filtered_tensors['f3' + model_util.KERAS_INPUT_SUFFIX])

  @parameterized.named_parameters(
      ('output_name_and_label_key', config.ModelSpec(label_key='label'),
       'output', 'label'),
      ('output_name_and_label_keys',
       config.ModelSpec(label_keys={'output': 'label'}), 'output', 'label'),
      ('output_name_and_no_label_keys', config.ModelSpec(), 'output', None),
      ('no_output_name_and_label_key', config.ModelSpec(label_key='label'), '',
       'label'),
      ('no_output_name_and_no_label_keys', config.ModelSpec(), '', None))
  def test_get_label_key(self, model_spec, output_name, expected_label_key):
    self.assertEqual(expected_label_key,
                     model_util.get_label_key(model_spec, output_name))

  def test_get_label_key_no_output_and_label_keys(self):
    with self.assertRaises(ValueError):
      model_util.get_label_key(
          config.ModelSpec(label_keys={'output1': 'label'}), '')


if __name__ == '__main__':
  tf.test.main()
