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
"""Simple tests for util."""

from absl.testing import parameterized
import numpy as np
import pyarrow as pa
import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import types
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.slicer import slicer_lib
from tensorflow_model_analysis.utils import util

from tensorflow_metadata.proto.v0 import schema_pb2


class UtilTest(tf.test.TestCase, parameterized.TestCase):

  def testToTensorValueFromTFSparseTensor(self):
    original = tf.SparseTensor(
        values=[0.5, -1., 0.5, -1.],
        indices=[[0, 3, 1], [0, 20, 0], [1, 3, 1], [1, 20, 0]],
        dense_shape=[2, 100, 3])
    sparse_value = util.to_tensor_value(original)
    self.assertAllClose(sparse_value.values, original.values.numpy())
    self.assertAllClose(sparse_value.indices, original.indices.numpy())
    self.assertAllClose(sparse_value.dense_shape, original.dense_shape.numpy())

  def testToTensorValueFromTFV1SparseTensorValue(self):
    original = tf.compat.v1.SparseTensorValue(
        values=np.array([0.5, -1., 0.5, -1.]),
        indices=np.array([[0, 3, 1], [0, 20, 0], [1, 3, 1], [1, 20, 0]]),
        dense_shape=np.array([2, 100, 3]))
    sparse_value = util.to_tensor_value(original)
    self.assertAllClose(sparse_value.values, original.values)
    self.assertAllClose(sparse_value.indices, original.indices)
    self.assertAllClose(sparse_value.dense_shape, original.dense_shape)

  def testToTensorValueFromTFRaggedTensor(self):
    original = tf.RaggedTensor.from_nested_row_splits(
        [3, 1, 4, 1, 5, 9, 2, 7, 1, 8, 8, 2, 1],
        [[0, 3, 6], [0, 2, 3, 4, 5, 5, 8], [0, 2, 3, 3, 6, 9, 10, 11, 13]])
    ragged_value = util.to_tensor_value(original)
    self.assertAllClose(ragged_value.values, original.flat_values.numpy())
    self.assertLen(ragged_value.nested_row_splits, 3)
    original_nested_row_splits = original.nested_row_splits
    self.assertAllClose(ragged_value.nested_row_splits[0],
                        original_nested_row_splits[0].numpy())
    self.assertAllClose(ragged_value.nested_row_splits[1],
                        original_nested_row_splits[1].numpy())
    self.assertAllClose(ragged_value.nested_row_splits[2],
                        original_nested_row_splits[2].numpy())

  def testToTensorValueFromTFRaggedTensorUsingRowLengths(self):
    original = tf.RaggedTensor.from_nested_row_lengths(
        [3, 1, 4, 1, 5, 9, 2, 7, 1, 8, 8, 2, 1],
        [[3, 3], [2, 1, 1, 1, 0, 3], [2, 1, 0, 3, 3, 1, 1, 2]])
    ragged_value = util.to_tensor_value(original)
    self.assertAllClose(ragged_value.values, original.flat_values.numpy())
    self.assertLen(ragged_value.nested_row_splits, 3)
    original_nested_row_splits = original.nested_row_splits
    self.assertAllClose(ragged_value.nested_row_splits[0],
                        original_nested_row_splits[0].numpy())
    self.assertAllClose(ragged_value.nested_row_splits[1],
                        original_nested_row_splits[1].numpy())
    self.assertAllClose(ragged_value.nested_row_splits[2],
                        original_nested_row_splits[2].numpy())

  def testToTensorValueFromTFV1RaggedTensorValue(self):
    ragged_value = util.to_tensor_value(
        tf.compat.v1.ragged.RaggedTensorValue(
            values=tf.compat.v1.ragged.RaggedTensorValue(
                values=tf.compat.v1.ragged.RaggedTensorValue(
                    values=np.array([3, 1, 4, 1, 5, 9, 2, 7, 1, 8, 8, 2, 1]),
                    row_splits=np.array([0, 2, 3, 3, 6, 9, 10, 11, 13])),
                row_splits=np.array([0, 2, 3, 4, 5, 5, 8])),
            row_splits=np.array([0, 3, 6])))
    self.assertAllClose(ragged_value.values,
                        np.array([3, 1, 4, 1, 5, 9, 2, 7, 1, 8, 8, 2, 1]))
    self.assertLen(ragged_value.nested_row_splits, 3)
    self.assertAllClose(ragged_value.nested_row_splits[0], np.array([0, 3, 6]))
    self.assertAllClose(ragged_value.nested_row_splits[1],
                        np.array([0, 2, 3, 4, 5, 5, 8]))
    self.assertAllClose(ragged_value.nested_row_splits[2],
                        np.array([0, 2, 3, 3, 6, 9, 10, 11, 13]))

  def testToTensorValueFromNumpy(self):
    self.assertAllClose(util.to_tensor_value([1, 2, 3]), np.array([1, 2, 3]))
    self.assertAllClose(
        util.to_tensor_value(np.array([1, 2, 3])), np.array([1, 2, 3]))

  def testToTensorValueFromTFTensor(self):
    self.assertAllClose(
        util.to_tensor_value(tf.constant([1, 2, 3])), np.array([1, 2, 3]))

  def testToTFSparseTensorFromSparseTensorValue(self):
    original = types.SparseTensorValue(
        values=np.array([0.5, -1., 0.5, -1.]),
        indices=np.array([[0, 3, 1], [0, 20, 0], [1, 3, 1], [1, 20, 0]]),
        dense_shape=np.array([2, 100, 3]))
    sparse_tensor = util.to_tensorflow_tensor(original)
    self.assertAllClose(sparse_tensor.values.numpy(), original.values)
    self.assertAllClose(sparse_tensor.indices.numpy(), original.indices)
    self.assertAllClose(sparse_tensor.dense_shape.numpy(), original.dense_shape)

  def testToTFRaggedTensorFromRaggedTensorValue(self):
    original = types.RaggedTensorValue(
        values=np.array([3, 1, 4, 1, 5, 9, 2, 7, 1, 8, 8, 2, 1]),
        nested_row_splits=[
            np.array([0, 3, 6]),
            np.array([0, 2, 3, 4, 5, 5, 8]),
            np.array([0, 2, 3, 3, 6, 9, 10, 11, 13])
        ])
    ragged_tensor = util.to_tensorflow_tensor(original)
    self.assertAllClose(ragged_tensor.flat_values.numpy(), original.values)
    self.assertLen(ragged_tensor.nested_row_splits, 3)
    self.assertAllClose(ragged_tensor.nested_row_splits[0].numpy(),
                        original.nested_row_splits[0])
    self.assertAllClose(ragged_tensor.nested_row_splits[1].numpy(),
                        original.nested_row_splits[1])
    self.assertAllClose(ragged_tensor.nested_row_splits[2].numpy(),
                        original.nested_row_splits[2])

  def testToTFTensorFromNumpy(self):
    self.assertAllClose(
        util.to_tensorflow_tensor(np.array([1, 2, 3])).numpy(),
        np.array([1, 2, 3]))

  def testToFromTensorValues(self):
    tensor_values = {
        'features': {
            'feature_1':
                np.array([1, 2, 3]),
            'feature_2':
                types.SparseTensorValue(
                    values=np.array([0.5, -1., 0.5, -1.]),
                    indices=np.array([[0, 3, 1], [0, 20, 0], [1, 3, 1],
                                      [1, 20, 0]]),
                    dense_shape=np.array([2, 100, 3])),
            'feature_3':
                types.RaggedTensorValue(
                    values=np.array([3, 1, 4, 1, 5, 9, 2, 7, 1, 8, 8, 2, 1]),
                    nested_row_splits=[
                        np.array([0, 3, 6]),
                        np.array([0, 2, 3, 4, 5, 5, 8]),
                        np.array([0, 2, 3, 3, 6, 9, 10, 11, 13])
                    ]),
            'feature_4':
                types.VarLenTensorValue(
                    values=np.array([1, 2, 3]),
                    indices=np.array([[0, 0], [0, 1], [1, 0]]),
                    dense_shape=np.array([2, 2]))
        },
        'labels': np.array([1])
    }
    actual = util.to_tensor_values(util.to_tensorflow_tensors(tensor_values))
    self.assertAllClose(actual, tensor_values)

  def testToFromTensorValuesWithSpecs(self):
    sparse_value = types.SparseTensorValue(
        values=np.array([0.5, -1., 0.5, -1.], dtype=np.float32),
        indices=np.array([[0, 3, 1], [0, 20, 0], [1, 3, 1], [1, 20, 0]]),
        dense_shape=np.array([2, 100, 3]))
    ragged_value = types.RaggedTensorValue(
        values=np.array([3, 1, 4, 1, 5, 9, 2, 7, 1, 8, 8, 2, 1],
                        dtype=np.float32),
        nested_row_splits=[
            np.array([0, 3, 6]),
            np.array([0, 2, 3, 4, 5, 5, 8]),
            np.array([0, 2, 3, 3, 6, 9, 10, 11, 13])
        ])
    tensor_values = {
        'features': {
            'feature_1': np.array([1, 2, 3], dtype=np.float32),
            'feature_2': sparse_value,
            'feature_3': ragged_value,
            'ignored_feature': np.array([1, 2, 3])
        },
        'labels': np.array([1], dtype=np.float32),
        'ignored': np.array([2])
    }
    specs = {
        'features': {
            'feature_1':
                tf.TensorSpec([3], dtype=tf.float32),
            'feature_2':
                tf.SparseTensorSpec(shape=[2, 100, 3], dtype=tf.float32),
            'feature_3':
                tf.RaggedTensorSpec(
                    shape=[2, None, None, None], dtype=tf.float32)
        },
        'labels': tf.TensorSpec([1], dtype=tf.float32)
    }
    actual = util.to_tensor_values(
        util.to_tensorflow_tensors(tensor_values, specs))
    expected = {
        'features': {
            'feature_1': np.array([1, 2, 3], dtype=np.float32),
            'feature_2': sparse_value,
            'feature_3': ragged_value
        },
        'labels': np.array([1], dtype=np.float32)
    }
    self.assertAllClose(actual, expected)

  def testToTensorflowTensorsRaisesUnknownKeyError(self):
    with self.assertRaisesRegex(ValueError, '.* not found in .*'):
      util.to_tensorflow_tensors(
          {'features': {
              'feature_1': np.array([1, 2, 3], dtype=np.float32)
          }}, {
              'features': {
                  'missing_feature': tf.TensorSpec([1], dtype=tf.float32)
              }
          })

  def testInferTensorSpecs(self):
    sparse_value = types.SparseTensorValue(
        values=np.array([0.5, -1., 0.5, -1.], dtype=np.float32),
        indices=np.array([[0, 3, 1], [0, 20, 0], [1, 3, 1], [1, 20, 0]]),
        dense_shape=np.array([2, 100, 3]))
    ragged_value = types.RaggedTensorValue(
        values=np.array([3, 1, 4, 1, 5, 9, 2, 7, 1, 8, 8, 2, 1],
                        dtype=np.float32),
        nested_row_splits=[
            np.array([0, 3, 6]),
            np.array([0, 2, 3, 4, 5, 5, 8]),
            np.array([0, 2, 3, 3, 6, 9, 10, 11, 13])
        ])
    tensor_values = {
        'features': {
            'feature_1': np.array([1, 2, 3], dtype=np.float32),
            'feature_2': sparse_value,
            'feature_3': ragged_value,
        },
        'labels': np.array([1], dtype=np.float32),
    }
    expected_specs = {
        'features': {
            'feature_1':
                tf.TensorSpec([None], dtype=tf.float32),
            'feature_2':
                tf.SparseTensorSpec(shape=[None, 100, 3], dtype=tf.float32),
            'feature_3':
                tf.RaggedTensorSpec(
                    shape=[None, None, None, None], dtype=tf.float32)
        },
        'labels': tf.TensorSpec([None], dtype=tf.float32)
    }
    got_specs = util.infer_tensor_specs(
        util.to_tensorflow_tensors(tensor_values))
    self.assertDictEqual(expected_specs, got_specs)

  def testRecordBatchToTensorValues(self):
    record_batch = pa.record_batch(
        [pa.array([[1], [2], [3]]),
         pa.array([[0], [1], [1]])], ['feature_1', 'feature_2'])
    actual = util.record_batch_to_tensor_values(record_batch)
    expected = {
        'feature_1': np.array([[1], [2], [3]]),
        'feature_2': np.array([[0], [1], [1]])
    }
    self.assertAllClose(actual, expected)

  def testRecordBatchToTensorValuesWithTensorRepresentation(self):
    record_batch = pa.record_batch(
        [pa.array([[1, 2], [2, 3], [3, 4]]),
         pa.array([[0], [1], [1]])], ['feature_1', 'feature_2'])
    tensor_representation = schema_pb2.TensorRepresentation()
    tensor_representation.dense_tensor.column_name = 'feature_1'
    tensor_representation.dense_tensor.shape.dim.append(
        schema_pb2.FixedShape.Dim(size=2))
    actual = util.record_batch_to_tensor_values(
        record_batch, {'feature_1': tensor_representation})
    expected = {
        'feature_1': np.array([[1, 2], [2, 3], [3, 4]]),
        'feature_2': np.array([[0], [1], [1]])
    }
    self.assertAllClose(actual, expected)

  def testRecordBatchToTensorValuesWithSparseTensorRepresentation(self):
    record_batch = pa.record_batch([
        pa.array([[0], [2, 3], [9, 10]]),
        pa.array([[0], [0.2, 0.3], [0.5, 0.6]])
    ], ['sparse_indices', 'sparse_values'])
    tensor_representation = schema_pb2.TensorRepresentation()
    tensor_representation.sparse_tensor.dense_shape.dim.append(
        schema_pb2.FixedShape.Dim(size=10))
    tensor_representation.sparse_tensor.index_column_names.append(
        'sparse_indices')
    tensor_representation.sparse_tensor.value_column_name = 'sparse_values'
    actual = util.record_batch_to_tensor_values(
        record_batch, {'sparse_feature': tensor_representation})
    expected = {
        'sparse_indices':
            types.VarLenTensorValue(
                values=np.array([0, 2, 3, 9, 10]),
                indices=np.array([[0, 0], [1, 0], [1, 1], [2, 0], [2, 1]]),
                dense_shape=np.array([3, 2])),
        'sparse_values':
            types.VarLenTensorValue(
                values=np.array([0, 0.2, 0.3, 0.5, 0.6]),
                indices=np.array([[0, 0], [1, 0], [1, 1], [2, 0], [2, 1]]),
                dense_shape=np.array([3, 2])),
        'sparse_feature':
            types.SparseTensorValue(
                indices=np.array([[0, 0], [1, 2], [1, 3], [2, 9], [2, 10]]),
                values=np.array([0, 0.2, 0.3, 0.5, 0.6]),
                dense_shape=np.array([3, 10]))
    }
    np.testing.assert_equal(
        actual, expected, err_msg=f'actual:{actual}\nexpected:{expected}')

  def testBatchSizeWithTensorValues(self):
    tensor_values = {
        'feature_1':
            np.array([1, 2], dtype=np.float32),
        'feature_2':
            types.SparseTensorValue(
                values=np.array([0.5, -1., 0.5, -1.], dtype=np.float32),
                indices=np.array([[0, 3, 1], [0, 20, 0], [1, 3, 1], [1, 20,
                                                                     0]]),
                dense_shape=np.array([2, 100, 3])),
        'feature_3':
            types.RaggedTensorValue(
                values=np.array([3, 1, 4, 1, 5, 9, 2, 7, 1, 8, 8, 2, 1],
                                dtype=np.float32),
                nested_row_splits=[
                    np.array([0, 3, 6]),
                    np.array([0, 2, 3, 4, 5, 5, 8]),
                    np.array([0, 2, 3, 3, 6, 9, 10, 11, 13])
                ]),
    }
    self.assertEqual(util.batch_size(tensor_values), 2)

  def testBatchSizeWithTFTensors(self):
    tensor_values = {
        'feature_1':
            tf.constant([1, 2]),
        'feature_2':
            tf.SparseTensor(
                values=[0.5, -1., 0.5, -1.],
                indices=[[0, 3, 1], [0, 20, 0], [1, 3, 1], [1, 20, 0]],
                dense_shape=[2, 100, 3]),
        'feature_3':
            tf.RaggedTensor.from_nested_row_lengths(
                [3, 1, 4, 1, 5, 9, 2, 7, 1, 8, 8, 2, 1],
                [[3, 3], [2, 1, 1, 1, 0, 3], [2, 1, 0, 3, 3, 1, 1, 2]]),
    }
    self.assertEqual(util.batch_size(tensor_values), 2)

  def testBatchSizeError(self):
    with self.assertRaisesRegex(ValueError,
                                'Batch sizes have differing values.*'):
      util.batch_size({
          'feature_1': np.array([1, 2, 3], dtype=np.int64),
          'feature_2': np.array([1, 2], dtype=np.int64)
      })

  def testUniqueKey(self):
    self.assertEqual('key', util.unique_key('key', ['key1', 'key2']))
    self.assertEqual('key1_1', util.unique_key('key1', ['key1', 'key2']))
    self.assertEqual('key1_2', util.unique_key('key1', ['key1', 'key1_1']))

  def testUniqueKeyWithUpdateKeys(self):
    keys = ['key1', 'key2']
    util.unique_key('key1', keys, update_keys=True)
    self.assertEqual(['key1', 'key2', 'key1_1'], keys)

  def testCompoundKey(self):
    self.assertEqual('a_b', util.compound_key(['a_b']))
    self.assertEqual('a__b', util.compound_key(['a', 'b']))
    self.assertEqual('a__b____c__d', util.compound_key(['a', 'b__c', 'd']))

  def testGetByKeys(self):
    self.assertEqual([1], util.get_by_keys({'labels': [1]}, ['labels']))

  def testGetByKeysMissingAndDefault(self):
    self.assertEqual('a', util.get_by_keys({}, ['labels'], default_value='a'))
    self.assertEqual(
        'a', util.get_by_keys({'labels': {}}, ['labels'], default_value='a'))

  def testGetByKeysMissingAndOptional(self):
    self.assertIsNone(util.get_by_keys({}, ['labels'], optional=True))
    self.assertIsNone(
        util.get_by_keys({'labels': {}}, ['labels'], optional=True))

  def testGetByKeysMissingAndNonOptional(self):
    with self.assertRaisesRegex(ValueError, 'not found'):
      util.get_by_keys({}, ['labels'])
    with self.assertRaisesRegex(ValueError, 'not found'):
      util.get_by_keys({'labels': {}}, ['labels'])

  def testGetByKeysWitMultiLevel(self):
    self.assertEqual([1],
                     util.get_by_keys({'predictions': {
                         'output': [1]
                     }}, ['predictions', 'output']))

    self.assertEqual([1],
                     util.get_by_keys(
                         {'predictions': {
                             'model': {
                                 'output': [1],
                             },
                         }}, ['predictions', 'model', 'output']))

  def testGetByKeysWithPrefix(self):
    self.assertEqual({
        'all_classes': ['a', 'b'],
        'probabilities': [1]
    },
                     util.get_by_keys(
                         {
                             'predictions': {
                                 'output/all_classes': ['a', 'b'],
                                 'output/probabilities': [1],
                             },
                         }, ['predictions', 'output']))
    self.assertEqual({
        'all_classes': ['a', 'b'],
        'probabilities': [1]
    },
                     util.get_by_keys(
                         {
                             'predictions': {
                                 'model': {
                                     'output/all_classes': ['a', 'b'],
                                     'output/probabilities': [1],
                                 },
                             },
                         }, ['predictions', 'model', 'output']))

  def testSetdefaultByKeypath_SingleKey(self):
    data = {}
    data = util.copy_and_set_by_keys(data, ['labels'], 'a')
    self.assertEqual({'labels': 'a'}, data)

  def testSetdefaultByKeypath_SingleKeyCollision(self):
    data = {'labels': 'b'}
    data = util.copy_and_set_by_keys(data, ['labels'], 'a')
    self.assertEqual({'labels': 'a'}, data)

  def testSetdefaultByKeypath_MultiKeys(self):
    data = {'labels': {'model0': 0}}
    data = util.copy_and_set_by_keys(data, ['labels', 'model1'], 'a')
    self.assertEqual({'labels': {'model1': 'a', 'model0': 0}}, data)

  def testSetdefaultByKeypath_MultiKeysCollision(self):
    data = {'predictions': {'modela': 1}}
    data = util.copy_and_set_by_keys(data, ['predictions', 'modela'], 'a')
    self.assertEqual({'predictions': {'modela': 'a'}}, data)

  def testSetdefaultByKeypath_KeyMismatchLevel(self):
    data = {'labels': 'modela'}
    with self.assertRaisesRegex(ValueError, 'a non-mapping object'):
      util.copy_and_set_by_keys(data, ['labels', 'model1'], 'a')

  def testSetdefaultByKeypath_EmptyKeyWithValue(self):
    data = {}
    with self.assertRaisesRegex(
        ValueError, 'Cannot set value when keypath is empty'
    ):
      util.copy_and_set_by_keys(data, [], 'a')

  def testGetModelAndOutputNamesEmptyExtracts(self):
    eval_config = config_pb2.EvalConfig(model_specs=[config_pb2.ModelSpec()])
    self.assertEmpty(
        util.StandardExtracts({}).get_model_and_output_names(eval_config))

  def testGetModelAndOutputNamesEmptyPredictions(self):
    eval_config = config_pb2.EvalConfig(model_specs=[config_pb2.ModelSpec()])
    self.assertEmpty(
        util.StandardExtracts({
            constants.PREDICTIONS_KEY: {}
        }).get_model_and_output_names(eval_config))

  def testGetModelAndOutputNamesMultiModel(self):
    eval_config = config_pb2.EvalConfig(model_specs=[
        config_pb2.ModelSpec(name=constants.BASELINE_KEY),
        config_pb2.ModelSpec(name=constants.CANDIDATE_KEY)
    ])
    self.assertEqual([(constants.BASELINE_KEY, None),
                      (constants.CANDIDATE_KEY, None)],
                     util.StandardExtracts({
                         constants.PREDICTIONS_KEY: {
                             constants.BASELINE_KEY: np.array([]),
                             constants.CANDIDATE_KEY: np.array([])
                         }
                     }).get_model_and_output_names(eval_config))

  def testGetModelAndOutputNamesMultiOutput(self):
    eval_config = config_pb2.EvalConfig(model_specs=[config_pb2.ModelSpec()])
    self.assertEqual([(None, 'output1'), (None, 'output2')],
                     util.StandardExtracts({
                         constants.PREDICTIONS_KEY: {
                             'output1': np.array([]),
                             'output2': np.array([])
                         }
                     }).get_model_and_output_names(eval_config))

  def testGetModelAndOutputNamesMultiModelMultiOutput(self):
    eval_config = config_pb2.EvalConfig(model_specs=[
        config_pb2.ModelSpec(name=constants.BASELINE_KEY),
        config_pb2.ModelSpec(name=constants.CANDIDATE_KEY)
    ])
    self.assertEqual([(constants.BASELINE_KEY, 'output1'),
                      (constants.BASELINE_KEY, 'output2'),
                      (constants.CANDIDATE_KEY, 'output1'),
                      (constants.CANDIDATE_KEY, 'output2')],
                     util.StandardExtracts({
                         constants.PREDICTIONS_KEY: {
                             constants.BASELINE_KEY: {
                                 'output1': np.array([]),
                                 'output2': np.array([])
                             },
                             constants.CANDIDATE_KEY: {
                                 'output1': np.array([]),
                                 'output2': np.array([])
                             }
                         },
                     }).get_model_and_output_names(eval_config))

  def testGetLabels(self):
    self.assertEqual(7, util.StandardExtracts({'labels': 7}).labels)

  def testGetLabelsMultiOutput(self):
    self.assertEqual(
        7,
        util.StandardExtracts({
            'labels': {
                'output1': 7
            },
        }).get_labels(output_name='output1'))

  def testGetLabelsMultiModel(self):
    self.assertEqual(
        7,
        util.StandardExtracts({
            'labels': {
                'candidate': 7,
                'baseline': 8
            },
        }).get_labels('candidate'))

  def testGetLabelsMultiModelMultiOutput(self):
    self.assertEqual(
        7,
        util.StandardExtracts({
            'labels': {
                'candidate': {
                    'output1': 7
                },
                'baseline': {
                    'output1': 8
                },
            },
        }).get_labels('candidate', 'output1'))

  def testGetLabelsNone(self):
    self.assertIsNone(
        util.StandardExtracts({
            'labels': {
                'candidate': None,
                'baseline': None
            },
        }).get_labels('candidate'))

  def testSetLabelsModelNameEmptyExtracts(self):
    extracts = util.StandardExtracts({})
    extracts.set_labels(7, model_name='candidate')
    self.assertEqual(7, extracts.get_labels(model_name='candidate'))

  def testSetLabelsModelNameOutputNameEmptyExtracts(self):
    extracts = util.StandardExtracts({})
    extracts.set_labels(7, model_name='candidate', output_name='output1')
    self.assertEqual(
        7, extracts.get_labels(model_name='candidate', output_name='output1'))

  def testSetLabelsModelNameOutputNameOverwrite(self):
    extracts = util.StandardExtracts(
        {'labels': {
            'baseline': {
                'output1': 3
            },
            'candidate': {
                'output1': 3
            }
        }})
    extracts.set_labels(7, model_name='candidate', output_name='output1')
    self.assertEqual(
        7, extracts.get_labels(model_name='candidate', output_name='output1'))
    self.assertEqual(
        {'labels': {
            'baseline': {
                'output1': 3
            },
            'candidate': {
                'output1': 7
            }
        }}, dict(extracts))

  def testSetLabelsEmptyLabels(self):
    extracts = util.StandardExtracts({'labels': {}})
    extracts.set_labels(7, model_name='candidate')
    self.assertEqual(7, extracts.get_labels(model_name='candidate'))

  def testSetLabelsNonMappingRaisesException(self):
    with self.assertRaisesRegex(RuntimeError,
                                'set_by_keys failed with arguments:') as ctx:
      util.StandardExtracts({
          'labels': np.array([7]),
      }).set_labels(
          7, model_name='candidate')
    self.assertIsInstance(ctx.exception.__cause__, ValueError)
    self.assertRegex(
        str(ctx.exception.__cause__),
        r'Cannot set keys \(\[\'candidate\'\]\) on a non-mapping root.*')

  def testGetByKeysMissingSecondaryKey(self):
    with self.assertRaisesRegex(ValueError, 'not found'):
      util.get_by_keys({'predictions': {
          'missing': [1]
      }}, ['predictions', 'output'])

  def testIncludeFilter(self):
    got = util.include_filter(
        include={
            'b': {},
            'c': {
                'c2': {
                    'c21': {}
                }
            },
            'e': {
                'e2': {
                    'e21': {}
                }
            }
        },
        target={
            'a': 1,
            'b': {
                'b2': 2
            },
            'c': {
                'c2': {
                    'c21': 3,
                    'c22': 4
                }
            },
            'd': {
                'd2': 4
            },
            'e': {
                'e2': {
                    'e22': {}
                }
            }
        })
    self.assertEqual(got, {
        'b': {
            'b2': 2
        },
        'c': {
            'c2': {
                'c21': 3
            }
        },
        'e': {
            'e2': {}
        }
    })

  def testExcludeFilter(self):
    got = util.exclude_filter(
        exclude={
            'b': {},
            'c': {
                'c2': {
                    'c21': {}
                }
            }
        },
        target={
            'a': 1,
            'b': {
                'b2': 2
            },
            'c': {
                'c2': {
                    'c21': 3,
                    'c22': 4
                }
            },
            'd': {
                'd2': 4
            }
        })
    self.assertEqual(got, {'a': 1, 'c': {'c2': {'c22': 4}}, 'd': {'d2': 4}})

  def testMergeFilters(self):
    filter1 = {
        'features': {
            'feature_1': {},
            'feature_2': {},
        },
        'labels': {},
        'example_weights': {
            'model1': {},
        },
        'predictions': {
            'model1': {
                'output1': {},
            },
            'model2': {
                'output1': {}
            }
        },
        'attributions': {
            'model1': {}
        },
    }
    filter2 = {
        'features': {
            'feature_2': {},
            'feature_3': {},
        },
        'labels': {
            'model1': {},
            'model2': {},
        },
        'example_weights': {
            'model2': {},
        },
        'predictions': {
            'model1': {
                'output2': {},
            },
            'model2': {
                'output1': {},
                'output2': {},
            }
        },
        'attributions': {
            'model1': {
                'output1': {
                    'feature1': {}
                },
            },
        },
    }
    merged = util.merge_filters(filter1, filter2)
    self.assertEqual(
        merged, {
            'features': {
                'feature_1': {},
                'feature_2': {},
                'feature_3': {},
            },
            'labels': {},
            'example_weights': {
                'model1': {},
                'model2': {},
            },
            'predictions': {
                'model1': {
                    'output1': {},
                    'output2': {},
                },
                'model2': {
                    'output1': {},
                    'output2': {},
                }
            },
            'attributions': {
                'model1': {},
            },
        })

  def testKwargsOnly(self):

    @util.kwargs_only
    def fn(a, b, c, d=None, e=5):
      if d is None:
        d = 100
      if e is None:
        e = 1000
      return a + b + c + d + e

    self.assertEqual(1 + 2 + 3 + 100 + 5, fn(a=1, b=2, c=3))
    self.assertEqual(1 + 2 + 3 + 100 + 1000, fn(a=1, b=2, c=3, e=None))
    with self.assertRaisesRegex(TypeError, 'keyword-arguments only'):
      fn(1, 2, 3)
    with self.assertRaisesRegex(TypeError, 'with c specified'):
      fn(a=1, b=2, e=5)  # pylint: disable=no-value-for-parameter
    with self.assertRaisesRegex(TypeError, 'with extraneous kwargs'):
      fn(a=1, b=2, c=3, f=11)  # pylint: disable=unexpected-keyword-arg

  def testGetFeaturesFromExtracts(self):
    self.assertEqual(
        {'a': np.array([1])},
        util.get_features_from_extracts({
            constants.FEATURES_PREDICTIONS_LABELS_KEY:
                types.FeaturesPredictionsLabels(
                    input_ref=0,
                    features={'a': np.array([1])},
                    predictions={},
                    labels={})
        }),
    )
    self.assertEqual(
        {'a': np.array([1])},
        util.get_features_from_extracts(
            {constants.FEATURES_KEY: {
                'a': np.array([1])
            }}),
    )
    self.assertEqual({}, util.get_features_from_extracts({}))

  def testMergeExtracts(self):
    extracts = [
        {
            'features': {
                'feature_1': np.array([1.0, 2.0]),
                'feature_2': np.array([1.0, 2.0]),
                'feature_3': types.SparseTensorValue(
                    values=np.array([1]),
                    indices=np.array([[0, 1]]),
                    dense_shape=np.array([1, 3]),
                ),
                'feature_4': types.RaggedTensorValue(
                    values=np.array([3, 1, 4, 1, 5, 9, 2, 6]),
                    nested_row_splits=[np.array([0, 4, 4, 7, 8, 8])],
                ),
                'feature_5': types.SparseTensorValue(
                    values=np.array([1]),
                    indices=np.array([[0, 1]]),
                    dense_shape=np.array([1, 3]),
                ),
                'feature_6': np.array([]),
                'feature_7': np.array([]),
            },
            'transformed_features': None,
            'labels': np.array([1.0]),
            'example_weights': np.array(0.0),
            'predictions': {
                'model1': np.array([0.1, 0.2]),
                'model2': np.array([0.1, 0.2]),
            },
            '_slice_key_types': slicer_lib.slice_keys_to_numpy_array(
                [('gender', 'm'), ()]
            ),
        },
        {
            'features': {
                'feature_1': np.array([3.0, 4.0]),
                'feature_2': np.array([3.0, 4.0]),
                'feature_3': types.SparseTensorValue(
                    values=np.array([2]),
                    indices=np.array([[0, 2]]),
                    dense_shape=np.array([1, 3]),
                ),
                'feature_4': types.RaggedTensorValue(
                    values=np.array([3, 1, 4, 1, 5, 9, 2, 6]),
                    nested_row_splits=[np.array([0, 4, 4, 7, 8, 8])],
                ),
                'feature_5': types.SparseTensorValue(
                    values=np.array([2]),
                    indices=np.array([[0, 2]]),
                    dense_shape=np.array([1, 4]),
                ),
                'feature_6': np.array([]),
                'feature_7': np.array(None, dtype=object),
            },
            'transformed_features': None,
            'labels': np.array([0.0]),
            'example_weights': np.array(0.5),
            'predictions': {
                'model1': np.array([0.3, 0.4]),
                'model2': np.array([0.3, 0.4]),
            },
            '_slice_key_types': slicer_lib.slice_keys_to_numpy_array([()]),
        },
        {
            'features': {
                'feature_1': np.array([5.0, 6.0]),
                'feature_2': np.array([5.0, 6.0]),
                'feature_3': types.SparseTensorValue(
                    values=np.array([3]),
                    indices=np.array([[0, 0]]),
                    dense_shape=np.array([1, 3]),
                ),
                'feature_4': types.RaggedTensorValue(
                    values=np.array([3, 1, 4, 1, 5, 9, 2, 6]),
                    nested_row_splits=[np.array([0, 4, 4, 7, 8, 8])],
                ),
                'feature_5': types.SparseTensorValue(
                    values=np.array([3]),
                    indices=np.array([[0, 3]]),
                    dense_shape=np.array([1, 5]),
                ),
                'feature_6': np.array([2.0, 3.0]),
                'feature_7': np.array(None, dtype=object),
            },
            'transformed_features': None,
            'labels': np.array([1.0]),
            'example_weights': np.array(1.0),
            'predictions': {
                'model1': np.array([0.5, 0.6]),
                'model2': np.array([0.5, 0.6]),
            },
            '_slice_key_types': slicer_lib.slice_keys_to_numpy_array([()]),
        },
    ]

    expected = {
        'features': {
            'feature_1': np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            'feature_2': np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            'feature_3': types.SparseTensorValue(
                values=np.array([1, 2, 3]),
                indices=np.array([[0, 0, 1], [1, 0, 2], [2, 0, 0]]),
                dense_shape=np.array([3, 1, 3]),
            ),
            'feature_4': types.RaggedTensorValue(
                values=np.array([
                    3,
                    1,
                    4,
                    1,
                    5,
                    9,
                    2,
                    6,
                    3,
                    1,
                    4,
                    1,
                    5,
                    9,
                    2,
                    6,
                    3,
                    1,
                    4,
                    1,
                    5,
                    9,
                    2,
                    6,
                ]),
                nested_row_splits=[
                    np.array([0, 5, 10, 15]),
                    np.array([
                        0,
                        4,
                        4,
                        7,
                        8,
                        8,
                        12,
                        12,
                        15,
                        16,
                        16,
                        20,
                        20,
                        23,
                        24,
                        24,
                    ]),
                ],
            ),
            'feature_5': types.SparseTensorValue(
                values=np.array([1, 2, 3]),
                indices=np.array([[0, 0, 1], [1, 0, 2], [2, 0, 3]]),
                dense_shape=np.array([3, 1, 5]),
            ),
            'feature_6': types.VarLenTensorValue(
                values=np.array([2.0, 3.0]),
                indices=np.array([[2, 0], [2, 1]]),
                dense_shape=np.array([3, 2]),
            ),
            'feature_7': types.VarLenTensorValue(
                values=np.array([None, None]),
                indices=np.array([[1, 0], [2, 0]]),
                dense_shape=np.array([3, 1]),
            ),
        },
        'transformed_features': None,
        'labels': np.array([1.0, 0.0, 1.0]),
        'example_weights': np.array([0.0, 0.5, 1.0]),
        'predictions': {
            'model1': np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            'model2': np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
        },
        '_slice_key_types': types.VarLenTensorValue(
            values=slicer_lib.slice_keys_to_numpy_array(
                [('gender', 'm'), (), (), ()]
            ),
            indices=np.array([[0, 0], [0, 1], [1, 0], [2, 0]]),
            dense_shape=np.array([3, 2]),
        ),
    }
    merged_extracts = util.merge_extracts(extracts)
    np.testing.assert_equal(merged_extracts, expected)

  def testMergeExtractsFlattensTwoDimmVector(self):
    extracts = [{
        'predictions': np.array([0.1], dtype=np.float64),
    }, {
        'predictions': np.array([0.3], dtype=np.float64),
    }, {
        'predictions': np.array([0.5], dtype=np.float64),
    }]

    expected = {
        'predictions': np.array([0.1, 0.3, 0.5], dtype=np.float64),
    }
    np.testing.assert_equal(util.merge_extracts(extracts), expected)

  def testMergeExtractsMaintainsTwoDimmVectorResult(self):
    extracts = [
        {
            'predictions': np.array([0.1], dtype=np.float64),
        },
        {
            'predictions': np.array([0.3], dtype=np.float64),
        },
        {
            'predictions': np.array([0.5], dtype=np.float64),
        },
    ]

    expected = {
        'predictions': np.array([[0.1], [0.3], [0.5]], dtype=np.float64),
    }
    np.testing.assert_equal(
        util.merge_extracts(extracts, squeeze_two_dim_vector=False), expected
    )

  def testMergeExtractsStackedArraysPredictions(self):
    extracts = [
        {
            'predictions': {
                'topk_ids': [
                    np.array([[1, 2, 3]], dtype=np.int32),
                    np.array([[4, 5, 6]], dtype=np.int32),
                ],
            },
        },
    ]

    expected = {
        'predictions': {
            'topk_ids': [
                np.array([
                    np.array([[1, 2, 3]], dtype=np.int32),
                    np.array([[4, 5, 6]], dtype=np.int32),
                ])
            ]
        },
    }
    np.testing.assert_equal(
        util.merge_extracts(extracts, squeeze_two_dim_vector=False), expected
    )

  def testMergeExtractsWithUnsizedNumpyArray(self):
    extracts = [
        {
            'predictions': np.array(0.1, dtype=np.float64),
        },
        {
            'predictions': np.array([0.3], dtype=np.float64),
        },
        {
            'predictions': np.array(None, dtype=object),
        },
    ]

    expected = {
        'predictions': types.VarLenTensorValue(
            values=np.array([0.1, 0.3, None]),
            indices=np.array([[0, 0], [1, 0], [2, 0]]),
            dense_shape=np.array([3, 1]),
        )
    }
    merged_extracts = util.merge_extracts(extracts)
    np.testing.assert_equal(merged_extracts, expected)

  def testSplitExtracts(self):
    extracts = {
        'features': {
            'feature_1': np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            'feature_2': np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            'feature_3': types.SparseTensorValue(
                values=np.array([1, 2, 3]),
                indices=np.array([[0, 0, 1], [1, 0, 2], [2, 0, 0]]),
                dense_shape=np.array([3, 1, 3]),
            ),
            'feature_4': types.RaggedTensorValue(
                values=np.array([
                    3,
                    1,
                    4,
                    1,
                    5,
                    9,
                    2,
                    6,
                    3,
                    1,
                    4,
                    1,
                    5,
                    9,
                    2,
                    6,
                    3,
                    1,
                    4,
                    1,
                    5,
                    9,
                    2,
                    6,
                ]),
                nested_row_splits=[
                    np.array([0, 5, 10, 15]),
                    np.array([
                        0,
                        4,
                        4,
                        7,
                        8,
                        8,
                        12,
                        12,
                        15,
                        16,
                        16,
                        20,
                        20,
                        23,
                        24,
                        24,
                    ]),
                ],
            ),
            'feature_5': types.VarLenTensorValue(
                values=np.array([1, 2, 3, 3, 3]),
                indices=np.array([[0, 0], [1, 0], [2, 0], [2, 1], [2, 2]]),
                dense_shape=np.array([3, 3]),
            ),
            'feature_6': types.VarLenTensorValue(
                values=np.array([1, 3, 3, 3]),
                indices=np.array([[0, 0], [2, 0], [2, 1], [2, 2]]),
                dense_shape=np.array([3, 3]),
            ),
        },
        'transformed_features': None,
        'labels': np.array([1.0, 0.0, 1.0]),
        'example_weights': np.array([0.0, 0.5, 1.0]),
        'predictions': {
            'model1': np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            'model2': np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
        },
        'empty': None,
        'multi_level_empty': {
            'empty': None,
            'next_level': {'empty': None},
        },
        '_slice_key_types': types.VarLenTensorValue.from_dense_rows([
            slicer_lib.slice_keys_to_numpy_array([(('gender', 'm'),), ()]),
            slicer_lib.slice_keys_to_numpy_array([()]),
            slicer_lib.slice_keys_to_numpy_array([()]),
        ]),
    }

    expected = [
        {
            'features': {
                'feature_1': np.array([1.0, 2.0]),
                'feature_2': np.array([1.0, 2.0]),
                'feature_3': types.SparseTensorValue(
                    values=np.array([1]),
                    indices=np.array([[0, 1]]),
                    dense_shape=np.array([1, 3]),
                ),
                'feature_4': types.RaggedTensorValue(
                    values=np.array([3, 1, 4, 1, 5, 9, 2, 6]),
                    nested_row_splits=[np.array([0, 4, 4, 7, 8, 8])],
                ),
                'feature_5': np.array([1.0]),
                'feature_6': np.array([1.0]),
            },
            'transformed_features': None,
            'labels': np.array([1.0]),
            'example_weights': np.array([0.0]),
            'predictions': {
                'model1': np.array([0.1, 0.2]),
                'model2': np.array([0.1, 0.2]),
            },
            'empty': None,
            'multi_level_empty': {
                'empty': None,
                'next_level': {'empty': None},
            },
            '_slice_key_types': np.array(
                [(('gender', 'm'),), ()], dtype=object
            ),
        },
        {
            'features': {
                'feature_1': np.array([3.0, 4.0]),
                'feature_2': np.array([3.0, 4.0]),
                'feature_3': types.SparseTensorValue(
                    values=np.array([2]),
                    indices=np.array([[0, 2]]),
                    dense_shape=np.array([1, 3]),
                ),
                'feature_4': types.RaggedTensorValue(
                    values=np.array([3, 1, 4, 1, 5, 9, 2, 6]),
                    nested_row_splits=[np.array([0, 4, 4, 7, 8, 8])],
                ),
                'feature_5': np.array([2.0]),
                'feature_6': np.array([]),
            },
            'transformed_features': None,
            'labels': np.array([0.0]),
            'example_weights': np.array([0.5]),
            'predictions': {
                'model1': np.array([0.3, 0.4]),
                'model2': np.array([0.3, 0.4]),
            },
            'empty': None,
            'multi_level_empty': {
                'empty': None,
                'next_level': {'empty': None},
            },
            '_slice_key_types': slicer_lib.slice_keys_to_numpy_array([()]),
        },
        {
            'features': {
                'feature_1': np.array([5.0, 6.0]),
                'feature_2': np.array([5.0, 6.0]),
                'feature_3': types.SparseTensorValue(
                    values=np.array([3]),
                    indices=np.array([[0, 0]]),
                    dense_shape=np.array([1, 3]),
                ),
                'feature_4': types.RaggedTensorValue(
                    values=np.array([3, 1, 4, 1, 5, 9, 2, 6]),
                    nested_row_splits=[np.array([0, 4, 4, 7, 8, 8])],
                ),
                'feature_5': np.array([3.0, 3.0, 3.0]),
                'feature_6': np.array([3.0, 3.0, 3.0]),
            },
            'transformed_features': None,
            'labels': np.array([1.0]),
            'example_weights': np.array([1.0]),
            'predictions': {
                'model1': np.array([0.5, 0.6]),
                'model2': np.array([0.5, 0.6]),
            },
            'empty': None,
            'multi_level_empty': {
                'empty': None,
                'next_level': {'empty': None},
            },
            '_slice_key_types': slicer_lib.slice_keys_to_numpy_array([()]),
        },
    ]
    np.testing.assert_equal(util.split_extracts(extracts), expected)

  def testSplitExtractsProducesNumpyScalars(self):
    extract = {
        'predictions': np.array([0.1, 0.3, 0.5], dtype=np.float64),
    }
    expected = [{
        'predictions': np.array(0.1, dtype=np.float64),
    }, {
        'predictions': np.array(0.3, dtype=np.float64),
    }, {
        'predictions': np.array(0.5, dtype=np.float64),
    }]
    got = util.split_extracts(extract, expand_zero_dims=False)
    np.testing.assert_equal(got, expected)
    for extract in got:
      # Assert we are receiving scalars.
      self.assertEqual(extract['predictions'].shape, ())
      # Assert we maintain dtype.
      self.assertEqual(extract['predictions'].dtype, np.float64)

  def testSplitExtractsProducesVectorLenOne(self):
    extract = {
        'predictions': np.array([0.1, 0.3, 0.5], dtype=np.float64),
    }
    expected = [{
        'predictions': np.array([0.1], dtype=np.float64),
    }, {
        'predictions': np.array([0.3], dtype=np.float64),
    }, {
        'predictions': np.array([0.5], dtype=np.float64),
    }]
    got = util.split_extracts(extract)
    np.testing.assert_equal(got, expected)
    for extract in got:
      # Assert we are receiving 1d vectors.
      self.assertEqual(extract['predictions'].shape, (1,))
      # Assert we maintain dtype.
      self.assertEqual(extract['predictions'].dtype, np.float64)

  @parameterized.named_parameters(
      dict(
          testcase_name='_list',
          extracts=[
              {'predictions': np.array([0.1, 0.3, 0.5], dtype=np.float64)}
          ],
          expected_extract={
              'predictions': np.array([0.1, 0.3, 0.5], dtype=np.float64)
          },
          key='predictions',
          shape=(3,),
          dtype=np.float64,
      ),
      dict(
          testcase_name='_2d_list',
          extracts=[
              {'labels': np.array([[0.1], [0.3], [0.5]], dtype=np.float64)}
          ],
          expected_extract={
              'labels': np.array([[0.1], [0.3], [0.5]], dtype=np.float64)
          },
          key='labels',
          shape=(3, 1),
          dtype=np.float64,
      ),
      dict(
          testcase_name='_array_scalar',
          extracts=[{'array_scalar': np.array([0.1], dtype=np.float64)}],
          expected_extract={'array_scalar': np.array([0.1], dtype=np.float64)},
          key='array_scalar',
          shape=(1,),
          dtype=np.float64,
      ),
      dict(
          testcase_name='_objects',
          extracts=[
              {'objects': np.array([b'sun', b'moon', b'call'], dtype=object)}
          ],
          expected_extract={
              'objects': np.array([b'sun', b'moon', b'call'], dtype=object)
          },
          key='objects',
          shape=(3,),
          dtype=object,
      ),
      dict(
          testcase_name='_batch_extracts',
          extracts=[
              {'objects': np.array([b'sun', b'moon'], dtype=object)},
              {'objects': np.array([b'call'], dtype=object)},
          ],
          expected_extract={
              'objects': np.array([b'sun', b'moon', b'call'], dtype=object)
          },
          key='objects',
          shape=(3,),
          dtype=object,
      ),
      dict(
          testcase_name='_batch_extracts_with_empty_extract',
          extracts=[{}, {'objects': np.array([b'sun', b'moon'], dtype=object)}],
          expected_extract={
              'objects': np.array([b'sun', b'moon'], dtype=object)
          },
          key='objects',
          shape=(2,),
          dtype=object,
      ),
  )
  def testSplitThenMergeAllowingScalarNumpyArray(self, extracts,
                                                 expected_extract, key, shape,
                                                 dtype):
    split_got = []
    for extract in extracts:
      split_got.extend(util.split_extracts(extract, expand_zero_dims=False))
    remerged_got = util.merge_extracts(split_got, squeeze_two_dim_vector=False)
    np.testing.assert_equal(remerged_got, expected_extract)
    # Assert vector shape. Numpy assertions will succeed for different shapes as
    # long as the arrays have equal values (e.g.
    # np.testing.asser_equal(1, np.array([1,1]))). Numpy will also attempt to
    # compare different numeric types as equal, so we directly assert the dtype
    # (e.g. np.testing.assert_equal(1, np.array([1,1], dtype=np.float64))).
    self.assertEqual(remerged_got[key].shape, shape)
    # Assert we maintain dtype.
    self.assertEqual(remerged_got[key].dtype, dtype)

  def testMergeBatchExtractsWithMissingKeys(self):
    extracts = [
        {
            'labels': np.array([0.1, 0.3], dtype=np.float64)
        },
        {},
    ]
    expected_extract = {
        'labels':
            types.VarLenTensorValue(
                values=np.array([0.1, 0.3]),
                indices=np.array([[0, 0], [0, 1]]),
                dense_shape=np.array([2, 2]))
    }
    merged_got = util.merge_extracts(extracts, squeeze_two_dim_vector=False)
    np.testing.assert_equal(merged_got, expected_extract)

  @parameterized.named_parameters(
      dict(
          testcase_name='_predictions',
          extract={'predictions': np.array([0.1, 0.3, 0.5], dtype=np.float64)},
          expected_extract={
              'predictions': np.array([0.1, 0.3, 0.5], dtype=np.float64)
          }),
      dict(
          testcase_name='_labels',
          extract={'labels': np.array([[0.1], [0.3], [0.5]], dtype=np.float64)},
          expected_extract={
              'labels': np.array([0.1, 0.3, 0.5], dtype=np.float64)
          }),
      dict(
          testcase_name='_array_scalar',
          extract={'array_scalar': np.array([0.1], dtype=np.float64)},
          expected_extract={'array_scalar': np.array([0.1], dtype=np.float64)}))
  def testSplitThenMergeDisallowingScalars(self, extract, expected_extract):
    split_got = util.split_extracts(extract)
    remerged_got = util.merge_extracts(split_got)
    # Assert that the remerged extracts are the same as the original.
    np.testing.assert_equal(remerged_got, expected_extract)


