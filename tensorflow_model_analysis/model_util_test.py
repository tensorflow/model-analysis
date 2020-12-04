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
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import pyarrow as pa
import tensorflow as tf
from tensorflow_model_analysis import config
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import model_util
from tensorflow_model_analysis.eval_saved_model import testutil
from tfx_bsl.tfxio import tensor_adapter
from tfx_bsl.tfxio import tf_example_record

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2

_TF_MAJOR_VERSION = int(tf.version.VERSION.split('.')[0])


def _record_batch_to_extracts(record_batch):
  input_index = record_batch.schema.names.index(constants.ARROW_INPUT_COLUMN)
  return {
      constants.ARROW_RECORD_BATCH_KEY:
          record_batch,
      constants.INPUT_KEY:
          np.asarray(record_batch.columns[input_index].flatten())
  }


class ModelUtilTest(testutil.TensorflowModelAnalysisTest,
                    parameterized.TestCase):

  def createDenseInputsSchema(self):
    return text_format.Parse(
        """
        tensor_representation_group {
          key: ""
          value {
            tensor_representation {
              key: "input_1"
              value {
                dense_tensor {
                  column_name: "input_1"
                  shape { dim { size: 1 } }
                }
              }
            }
            tensor_representation {
              key: "input_2"
              value {
                dense_tensor {
                  column_name: "input_2"
                  shape { dim { size: 1 } }
                }
              }
            }
          }
        }
        feature {
          name: "input_1"
          type: FLOAT
        }
        feature {
          name: "input_2"
          type: FLOAT
        }
        feature {
          name: "non_model_feature"
          type: INT
        }
        """, schema_pb2.Schema())

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
    else:
      tf.saved_model.save(model, export_path, signatures=signatures)
    return export_path

  def createModelWithMultipleDenseInputs(self, save_as_keras):
    input1 = tf.keras.layers.Input(shape=(1,), name='input_1')
    input2 = tf.keras.layers.Input(shape=(1,), name='input_2')
    inputs = [input1, input2]
    input_layer = tf.keras.layers.concatenate(inputs)
    output_layer = tf.keras.layers.Dense(
        1, activation=tf.nn.sigmoid, name='output')(
            input_layer)
    model = tf.keras.models.Model(inputs, output_layer)

    # Add custom attribute to model to test callables stored as attributes
    model.custom_attribute = tf.keras.models.Model(inputs, output_layer)

    @tf.function
    def serving_default(serialized_tf_examples):
      parsed_features = tf.io.parse_example(
          serialized_tf_examples, {
              'input_1': tf.io.FixedLenFeature([1], dtype=tf.float32),
              'input_2': tf.io.FixedLenFeature([1], dtype=tf.float32)
          })
      return model(parsed_features)

    @tf.function
    def custom_single_output(features):
      return model(features)

    @tf.function
    def custom_multi_output(features):
      return {'output1': model(features), 'output2': model(features)}

    input_spec = tf.TensorSpec(shape=(None,), dtype=tf.string, name='examples')
    custom_input_spec = {
        'input_1':
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32, name='input_1'),
        'input_2':
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32, name='input_2')
    }
    signatures = {
        'serving_default':
            serving_default.get_concrete_function(input_spec),
        'custom_single_output':
            custom_single_output.get_concrete_function(custom_input_spec),
        'custom_multi_output':
            custom_multi_output.get_concrete_function(custom_input_spec)
    }

    export_path = tempfile.mkdtemp()
    if save_as_keras:
      model.save(export_path, save_format='tf', signatures=signatures)
    else:
      tf.saved_model.save(model, export_path, signatures=signatures)
    return export_path

  def createModelWithMultipleMixedInputs(self, save_as_keras):
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
    else:
      tf.saved_model.save(model, export_path, signatures=signatures)
    return export_path

  def testFilterByInputNames(self):
    tensors = {
        'f1': tf.constant([[1.1], [2.1]], dtype=tf.float32),
        'f2': tf.constant([[1], [2]], dtype=tf.int64),
        'f3': tf.constant([['hello'], ['world']], dtype=tf.string)
    }
    filtered_tensors = model_util.filter_by_input_names(tensors, ['f1', 'f3'])
    self.assertLen(filtered_tensors, 2)
    self.assertAllEqual(
        tf.constant([[1.1], [2.1]], dtype=tf.float32), filtered_tensors['f1'])
    self.assertAllEqual(
        tf.constant([['hello'], ['world']], dtype=tf.string),
        filtered_tensors['f3'])

  def testFilterByInputNamesKeras(self):
    tensors = {
        'f1': tf.constant([[1.1], [2.1]], dtype=tf.float32),
        'f2': tf.constant([[1], [2]], dtype=tf.int64),
        'f3': tf.constant([['hello'], ['world']], dtype=tf.string)
    }
    filtered_tensors = model_util.filter_by_input_names(tensors, [
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
  def testGetLabelKey(self, model_spec, output_name, expected_label_key):
    self.assertEqual(expected_label_key,
                     model_util.get_label_key(model_spec, output_name))

  def testGetLabelKeyNoOutputAndLabelKeys(self):
    with self.assertRaises(ValueError):
      model_util.get_label_key(
          config.ModelSpec(label_keys={'output1': 'label'}), '')

  @parameterized.named_parameters(
      {
          'testcase_name': 'single_model_single_key',
          'model_specs': [config.ModelSpec(label_key='feature1')],
          'field': 'label_key',
          'multi_output_field': 'label_keys',
          'expected_values': [[1.0, 1.1, 1.2],]
      },
      {
          'testcase_name':
              'single_model_multi_key',
          'model_specs': [
              config.ModelSpec(label_keys={
                  'output1': 'feature1',
                  'output2': 'feature2'
              })
          ],
          'field':
              'label_key',
          'multi_output_field':
              'label_keys',
          'expected_values': [{
              'output1': [1.0, 1.1, 1.2],
              'output2': [2.0, 2.1, 2.2]
          },]
      },
      {
          'testcase_name':
              'multi_model_single_key',
          'model_specs': [
              config.ModelSpec(name='model1', example_weight_key='feature2'),
              config.ModelSpec(name='model2', example_weight_key='feature3')
          ],
          'field':
              'example_weight_key',
          'multi_output_field':
              'example_weight_keys',
          'expected_values': [{
              'model1': [2.0, 2.1, 2.2],
              'model2': [3.0, 3.1, 3.2]
          },]
      },
      {
          'testcase_name':
              'multi_model_multi_key',
          'model_specs': [
              config.ModelSpec(
                  name='model1',
                  prediction_keys={
                      'output1': 'feature1',
                      'output2': 'feature2'
                  }),
              config.ModelSpec(
                  name='model2',
                  prediction_keys={
                      'output1': 'feature1',
                      'output3': 'feature3'
                  })
          ],
          'field':
              'prediction_key',
          'multi_output_field':
              'prediction_keys',
          'expected_values': [{
              'model1': {
                  'output1': [1.0, 1.1, 1.2],
                  'output2': [2.0, 2.1, 2.2]
              },
              'model2': {
                  'output1': [1.0, 1.1, 1.2],
                  'output3': [3.0, 3.1, 3.2]
              }
          },]
      },
  )
  def testGetFeatureValuesForModelSpecField(self, model_specs, field,
                                            multi_output_field,
                                            expected_values):
    extracts = {
        # Only need the num_rows from RecordBatch so use fake array of same len
        # as features.
        constants.ARROW_RECORD_BATCH_KEY:
            pa.RecordBatch.from_arrays([pa.array([1])], ['dummy']),
        constants.FEATURES_KEY: [{
            'feature1': [1.0, 1.1, 1.2],
            'feature2': [2.0, 2.1, 2.2],
            'feature3': [3.0, 3.1, 3.2],
        },]
    }
    got = model_util.get_feature_values_for_model_spec_field(
        model_specs, field, multi_output_field, extracts)
    self.assertAlmostEqual(expected_values, got)

  @parameterized.named_parameters(
      {
          'testcase_name': 'single_model_single_key',
          'model_specs': [config.ModelSpec(label_key='feature2')],
          'field': 'label_key',
          'multi_output_field': 'label_keys',
          'expected_values': [[4.0, 4.1, 4.2],]
      },
      {
          'testcase_name':
              'single_model_multi_key',
          'model_specs': [
              config.ModelSpec(label_keys={
                  'output1': 'feature1',
                  'output2': 'feature2'
              })
          ],
          'field':
              'label_key',
          'multi_output_field':
              'label_keys',
          'expected_values': [{
              'output1': [1.0, 1.1, 1.2],
              'output2': [4.0, 4.1, 4.2]
          },]
      },
  )
  def testGetFeatureValuesForModelSpecFieldWithSingleModelTransforedFeatures(
      self, model_specs, field, multi_output_field, expected_values):
    extracts = {
        # Only need the num_rows from RecordBatch so use fake array of same len
        # as features.
        constants.ARROW_RECORD_BATCH_KEY:
            pa.RecordBatch.from_arrays([pa.array([1])], ['dummy']),
        constants.FEATURES_KEY: [{
            'feature1': [1.0, 1.1, 1.2],
            'feature2': [2.0, 2.1, 2.2],
        },],
        constants.TRANSFORMED_FEATURES_KEY: [{
            'feature2': [4.0, 4.1, 4.2],
        },]
    }
    got = model_util.get_feature_values_for_model_spec_field(
        model_specs, field, multi_output_field, extracts)
    self.assertAlmostEqual(expected_values, got)

  @parameterized.named_parameters(
      {
          'testcase_name':
              'multi_model_single_key',
          'model_specs': [
              config.ModelSpec(name='model1', example_weight_key='feature2'),
              config.ModelSpec(name='model2', example_weight_key='feature3')
          ],
          'field':
              'example_weight_key',
          'multi_output_field':
              'example_weight_keys',
          'expected_values': [{
              'model1': [4.0, 4.1, 4.2],
              'model2': [7.0, 7.1, 7.2]
          },]
      },
      {
          'testcase_name':
              'multi_model_multi_key',
          'model_specs': [
              config.ModelSpec(
                  name='model1',
                  example_weight_keys={
                      'output1': 'feature1',
                      'output2': 'feature2'
                  }),
              config.ModelSpec(
                  name='model2',
                  example_weight_keys={
                      'output1': 'feature1',
                      'output3': 'feature3'
                  })
          ],
          'field':
              'example_weight_key',
          'multi_output_field':
              'example_weight_keys',
          'expected_values': [{
              'model1': {
                  'output1': [1.0, 1.1, 1.2],
                  'output2': [4.0, 4.1, 4.2]
              },
              'model2': {
                  'output1': [1.0, 1.1, 1.2],
                  'output3': [7.0, 7.1, 7.2]
              }
          },]
      },
  )
  def testGetFeatureValuesForModelSpecFieldWithMultiModelTransforedFeatures(
      self, model_specs, field, multi_output_field, expected_values):
    extracts = {
        # Only need the num_rows from RecordBatch so use fake array of same len
        # as features.
        constants.ARROW_RECORD_BATCH_KEY:
            pa.RecordBatch.from_arrays([pa.array([1])], ['dummy']),
        constants.FEATURES_KEY: [{
            'feature1': [1.0, 1.1, 1.2],
            'feature2': [2.0, 2.1, 2.2],
        },],
        constants.TRANSFORMED_FEATURES_KEY: [{
            'model1': {
                'feature2': [4.0, 4.1, 4.2],
                'feature3': [5.0, 5.1, 5.2]
            },
            'model2': {
                'feature2': [6.0, 6.1, 6.2],
                'feature3': [7.0, 7.1, 7.2]
            }
        },]
    }
    got = model_util.get_feature_values_for_model_spec_field(
        model_specs, field, multi_output_field, extracts)
    self.assertAlmostEqual(expected_values, got)

  def testGetFeatureValuesForModelSpecFieldNoValues(self):
    model_spec = config.ModelSpec(name='model1', example_weight_key='feature2')
    extracts = {
        constants.ARROW_RECORD_BATCH_KEY:
            pa.RecordBatch.from_arrays([pa.array([1])], ['dummy']),
    }
    got = model_util.get_feature_values_for_model_spec_field([model_spec],
                                                             'example_weight',
                                                             'example_weights',
                                                             extracts)
    self.assertIsNone(got)

  @parameterized.named_parameters(
      ('keras_serving_default', True, 'serving_default'),
      ('keras_custom_signature', True, 'custom_signature'),
      ('tf2_serving_default', False, 'serving_default'),
      ('tf2_custom_signature', False, 'custom_signature'))
  def testGetCallableWithSignatures(self, save_as_keras, signature_name):
    export_path = self.createModelWithSingleInput(save_as_keras)
    if save_as_keras:
      model = tf.keras.models.load_model(export_path)
    else:
      model = tf.compat.v1.saved_model.load_v2(export_path)
    self.assertIsNotNone(model_util.get_callable(model, signature_name))

  @parameterized.named_parameters(('keras', True), ('tf2', False))
  def testGetCallableWithMissingSignatures(self, save_as_keras):
    export_path = self.createModelWithSingleInput(save_as_keras)
    if save_as_keras:
      model = tf.keras.models.load_model(export_path)
    else:
      model = tf.compat.v1.saved_model.load_v2(export_path)
    with self.assertRaises(ValueError):
      model_util.get_callable(model, 'non_existent')

  @unittest.skipIf(_TF_MAJOR_VERSION < 2,
                   'not all input types supported for TF1')
  def testGetCallableWithKerasModel(self):
    export_path = self.createModelWithMultipleMixedInputs(True)
    model = tf.keras.models.load_model(export_path)
    self.assertEqual(model, model_util.get_callable(model))

  @parameterized.named_parameters(
      ('keras_serving_default', True, 'serving_default'),
      ('keras_custom_signature', True, 'custom_signature'),
      ('tf2_serving_default', False, None),
      ('tf2_custom_signature', False, 'custom_signature'))
  def testGetInputSpecsWithSignatures(self, save_as_keras, signature_name):
    export_path = self.createModelWithSingleInput(save_as_keras)
    if save_as_keras:
      model = tf.keras.models.load_model(export_path)
    else:
      model = tf.compat.v1.saved_model.load_v2(export_path)
    self.assertEqual(
        {
            'input':
                tf.TensorSpec(name='input', shape=(None, 1), dtype=tf.string),
        }, model_util.get_input_specs(model, signature_name))

  @parameterized.named_parameters(('keras', True), ('tf2', False))
  def testGetInputSpecsWithMissingSignatures(self, save_as_keras):
    export_path = self.createModelWithSingleInput(save_as_keras)
    if save_as_keras:
      model = tf.keras.models.load_model(export_path)
    else:
      model = tf.compat.v1.saved_model.load_v2(export_path)
    with self.assertRaises(ValueError):
      model_util.get_callable(model, 'non_existent')

  @unittest.skipIf(_TF_MAJOR_VERSION < 2,
                   'not all input types supported for TF1')
  def testGetInputSpecsWithKerasModel(self):
    export_path = self.createModelWithMultipleMixedInputs(True)
    model = tf.keras.models.load_model(export_path)

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

  def testInputSpecsToTensorRepresentations(self):
    tensor_representations = model_util.input_specs_to_tensor_representations({
        'input_1': tf.TensorSpec(shape=(None, 2), dtype=tf.int64),
        'input_2': tf.SparseTensorSpec(shape=(None, 1), dtype=tf.float32),
        'input_3': tf.RaggedTensorSpec(shape=(None, None), dtype=tf.float32),
    })
    dense_tensor_representation = text_format.Parse(
        """
        dense_tensor {
          column_name: "input_1"
          shape { dim { size: 2 } }
        }
        """, schema_pb2.TensorRepresentation())
    sparse_tensor_representation = text_format.Parse(
        """
        varlen_sparse_tensor {
          column_name: "input_2"
        }
        """, schema_pb2.TensorRepresentation())
    ragged_tensor_representation = text_format.Parse(
        """
        ragged_tensor {
          feature_path {
            step: "input_3"
          }
        }
        """, schema_pb2.TensorRepresentation())
    self.assertEqual(
        {
            'input_1': dense_tensor_representation,
            'input_2': sparse_tensor_representation,
            'input_3': ragged_tensor_representation
        }, tensor_representations)

  def testInputSpecsToTensorRepresentationsRaisesWithUnknownDims(self):
    with self.assertRaises(ValueError):
      model_util.input_specs_to_tensor_representations({
          'input_1': tf.TensorSpec(shape=(None, None), dtype=tf.int64),
      })

  @parameterized.named_parameters(
      ('keras_default', True, {
          constants.PREDICTIONS_KEY: {
              '': [None]
          }
      }, None, False, True, 1),
      ('tf_default', False, {
          constants.PREDICTIONS_KEY: {
              '': [None]
          }
      }, None, False, True, 1),
      ('keras_serving_default', True, {
          constants.PREDICTIONS_KEY: {
              '': ['serving_default']
          }
      }, None, False, True, 1),
      ('tf_serving_default', False, {
          constants.PREDICTIONS_KEY: {
              '': ['serving_default']
          }
      }, None, False, True, 1),
      ('keras_custom_single_output', True, {
          constants.PREDICTIONS_KEY: {
              '': ['custom_single_output']
          }
      }, None, False, True, 1),
      ('tf_custom_single_output', False, {
          constants.PREDICTIONS_KEY: {
              '': ['custom_single_output']
          }
      }, None, False, True, 1),
      ('keras_custom_multi_output', True, {
          constants.PREDICTIONS_KEY: {
              '': ['custom_multi_output']
          }
      }, None, False, True, 2),
      ('tf_custom_multi_output', False, {
          constants.PREDICTIONS_KEY: {
              '': ['custom_multi_output']
          }
      }, None, False, True, 2),
      ('multi_model', True, {
          constants.PREDICTIONS_KEY: {
              'model1': ['custom_multi_output'],
              'model2': ['custom_multi_output']
          }
      }, None, False, True, 2),
      ('default_signatures', True, {
          constants.PREDICTIONS_KEY: {
              '': [],
          }
      }, ['unknown', 'custom_single_output'], False, True, 1),
      ('keras_prefer_dict_outputs', True, {
          constants.FEATURES_KEY: {
              '': [],
          }
      }, ['unknown', 'custom_single_output', 'custom_multi_output'
         ], True, True, 3),
      ('tf_prefer_dict_outputs', False, {
          constants.FEATURES_KEY: {
              '': [],
          }
      }, ['unknown', 'custom_single_output', 'custom_multi_output'
         ], True, True, 3),
      ('custom_attribute', True, {
          constants.FEATURES_KEY: {
              '': ['custom_attribute'],
          }
      }, None, True, True, 1),
      ('keras_no_schema', True, {
          constants.PREDICTIONS_KEY: {
              '': [None]
          }
      }, None, False, False, 1),
      ('tf_no_schema', False, {
          constants.PREDICTIONS_KEY: {
              '': [None]
          }
      }, None, False, False, 1),
  )
  @unittest.skipIf(_TF_MAJOR_VERSION < 2,
                   'not all signatures supported for TF1')
  def testModelSignaturesDoFn(self, save_as_keras, signature_names,
                              default_signature_names, prefer_dict_outputs,
                              use_schema, expected_num_outputs):
    export_path = self.createModelWithMultipleDenseInputs(save_as_keras)
    eval_shared_models = {}
    model_specs = []
    for sigs in signature_names.values():
      for model_name in sigs:
        if model_name not in eval_shared_models:
          eval_shared_models[model_name] = self.createTestEvalSharedModel(
              eval_saved_model_path=export_path,
              model_name=model_name,
              tags=[tf.saved_model.SERVING])
          model_specs.append(config.ModelSpec(name=model_name))
    eval_config = config.EvalConfig(model_specs=model_specs)
    schema = self.createDenseInputsSchema() if use_schema else None
    tfx_io = tf_example_record.TFExampleBeamRecord(
        physical_format='text',
        schema=schema,
        raw_record_column_name=constants.ARROW_INPUT_COLUMN)
    tensor_adapter_config = None
    if use_schema:
      tensor_adapter_config = tensor_adapter.TensorAdapterConfig(
          arrow_schema=tfx_io.ArrowSchema(),
          tensor_representations=tfx_io.TensorRepresentations())

    examples = [
        self._makeExample(input_1=1.0, input_2=2.0),
        self._makeExample(input_1=3.0, input_2=4.0),
        self._makeExample(input_1=5.0, input_2=6.0),
    ]

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([e.SerializeToString() for e in examples])
          | 'BatchExamples' >> tfx_io.BeamSource(batch_size=3)
          | 'ToExtracts' >> beam.Map(_record_batch_to_extracts)
          | 'ModelSignatures' >> beam.ParDo(
              model_util.ModelSignaturesDoFn(
                  eval_config=eval_config,
                  eval_shared_models=eval_shared_models,
                  signature_names=signature_names,
                  default_signature_names=default_signature_names,
                  prefer_dict_outputs=prefer_dict_outputs,
                  tensor_adapter_config=tensor_adapter_config)))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          for key in signature_names:
            self.assertIn(key, got[0])
            if prefer_dict_outputs:
              for entry in got[0][key]:
                self.assertIsInstance(entry, dict)
                self.assertLen(entry, expected_num_outputs)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testHasRubberStamp(self):
    # Model agnostic.
    self.assertFalse(model_util.has_rubber_stamp(None))

    # All non baseline models has rubber stamp.
    baseline = self.createTestEvalSharedModel(
        model_name=constants.BASELINE_KEY, is_baseline=True)
    candidate = self.createTestEvalSharedModel(
        model_name=constants.CANDIDATE_KEY, rubber_stamp=True)
    self.assertTrue(model_util.has_rubber_stamp([baseline, candidate]))

    # Not all non baseline has rubber stamp.
    candidate_nr = self.createTestEvalSharedModel(
        model_name=constants.CANDIDATE_KEY)
    self.assertFalse(model_util.has_rubber_stamp([candidate_nr]))
    self.assertFalse(
        model_util.has_rubber_stamp([baseline, candidate, candidate_nr]))


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
