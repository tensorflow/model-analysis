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

import tempfile
import unittest

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import config
from tensorflow_model_analysis import model_util

_TF_MAJOR_VERSION = int(tf.version.VERSION.split('.')[0])


class ModelUtilTest(tf.test.TestCase, parameterized.TestCase):

  def createModelWithSingleInput(self, save_as_keras):
    input_layer = tf.keras.layers.Input(shape=(1,), name='input')
    output_layer = tf.keras.layers.Dense(
        1, activation=tf.nn.sigmoid)(
            input_layer)
    model = tf.keras.models.Model(input_layer, output_layer)

    @tf.function
    def serving_default(s):
      return model(s)

    input_spec = {
        'input': tf.TensorSpec(shape=(None, 1), dtype=tf.string, name='input'),
    }
    signatures = {
        'serving_default': serving_default.get_concrete_function(input_spec),
        'custom_signature': serving_default.get_concrete_function(input_spec),
    }

    export_path = tempfile.mkdtemp()
    if save_as_keras:
      model.save(export_path, save_format='tf', signatures=signatures)
      return tf.keras.models.load_model(export_path)
    else:
      tf.saved_model.save(model, export_path, signatures=signatures)
      return tf.compat.v1.saved_model.load_v2(export_path)

  def createModelWithMultipleInputs(self, save_as_keras):
    dense_input = tf.keras.layers.Input(
        shape=(2,), name='input_1', dtype=tf.int64)
    dense_float_input = tf.cast(dense_input, tf.float32)
    sparse_input = tf.keras.layers.Input(
        shape=(1,), name='input_2', sparse=True)
    dense_sparse_input = tf.keras.layers.Dense(
        1, name='dense_input2')(
            sparse_input)
    ragged_input = tf.keras.layers.Input(
        shape=(None,), name='input_3', ragged=True)
    dense_ragged_input = tf.keras.layers.Lambda(lambda x: x.to_tensor())(
        ragged_input)
    dense_ragged_input.set_shape((None, 1))
    inputs = [dense_input, sparse_input, ragged_input]
    input_layer = tf.keras.layers.concatenate(
        [dense_float_input, dense_sparse_input, dense_ragged_input])
    output_layer = tf.keras.layers.Dense(
        1, activation=tf.nn.sigmoid)(
            input_layer)
    model = tf.keras.models.Model(inputs, output_layer)

    @tf.function
    def serving_default(features):
      return model(features)

    input_spec = {
        'input_1':
            tf.TensorSpec(shape=(None, 2), dtype=tf.int64, name='input_1'),
        'input_2':
            tf.SparseTensorSpec(shape=(None, 1), dtype=tf.float32),
        'input_3':
            tf.RaggedTensorSpec(shape=(None, 1), dtype=tf.float32)
    }
    signatures = {
        'serving_default': serving_default.get_concrete_function(input_spec),
        'custom_signature': serving_default.get_concrete_function(input_spec),
    }

    export_path = tempfile.mkdtemp()
    if save_as_keras:
      model.save(export_path, save_format='tf', signatures=signatures)
      return tf.keras.models.load_model(export_path)
    else:
      tf.saved_model.save(model, export_path, signatures=signatures)
      return tf.compat.v1.saved_model.load_v2(export_path)

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

  @parameterized.named_parameters(
      ('keras_serving_default', True, 'serving_default'),
      ('keras_custom_signature', True, 'custom_signature'),
      ('tf2_serving_default', False, 'serving_default'),
      ('tf2_custom_signature', False, 'custom_signature'))
  def testGetCallableWithSignatures(self, save_as_keras, signature_name):
    model = self.createModelWithSingleInput(save_as_keras)
    self.assertIsNotNone(model_util.get_callable(model, signature_name))

  @parameterized.named_parameters(('keras', True), ('tf2', False))
  def testGetCallableWithMissingSignatures(self, save_as_keras):
    model = self.createModelWithSingleInput(save_as_keras)
    with self.assertRaises(ValueError):
      model_util.get_callable(model, 'non_existent')

  @unittest.skipIf(_TF_MAJOR_VERSION < 2,
                   'not all input types supported for TF1')
  def testGetCallableWithKerasModel(self):
    model = self.createModelWithMultipleInputs(True)
    self.assertEqual(model, model_util.get_callable(model))

  @parameterized.named_parameters(
      ('keras_serving_default', True, 'serving_default'),
      ('keras_custom_signature', True, 'custom_signature'),
      ('tf2_serving_default', False, None),
      ('tf2_custom_signature', False, 'custom_signature'))
  def testGetInputSpecsWithSignatures(self, save_as_keras, signature_name):
    model = self.createModelWithSingleInput(save_as_keras)
    self.assertEqual(
        {
            'input':
                tf.TensorSpec(name='input', shape=(None, 1), dtype=tf.string),
        }, model_util.get_input_specs(model, signature_name))

  @parameterized.named_parameters(('keras', True), ('tf2', False))
  def testGetInputSpecsWithMissingSignatures(self, save_as_keras):
    model = self.createModelWithSingleInput(save_as_keras)
    with self.assertRaises(ValueError):
      model_util.get_callable(model, 'non_existent')

  @unittest.skipIf(_TF_MAJOR_VERSION < 2,
                   'not all input types supported for TF1')
  def testGetInputSpecsWithKerasModel(self):
    model = self.createModelWithMultipleInputs(True)
    # Some versions of TF set the TensorSpec.name and others do not. Since we
    # don't care about the name, clear it from the output for testing purposes
    specs = model_util.get_input_specs(model)
    for k, v in specs.items():
      if isinstance(v, tf.TensorSpec):
        specs[k] = tf.TensorSpec(shape=v.shape, dtype=v.dtype)
    self.assertEqual(
        {
            'input_1':
                tf.TensorSpec(shape=(None, 2), dtype=tf.int64),
            'input_2':
                tf.SparseTensorSpec(shape=(None, 1), dtype=tf.float32),
            'input_3':
                tf.RaggedTensorSpec(shape=(None, None), dtype=tf.float32),
        }, specs)


if __name__ == '__main__':
  tf.test.main()
