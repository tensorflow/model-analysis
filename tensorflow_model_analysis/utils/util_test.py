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

import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.utils import util


class UtilTest(tf.test.TestCase):

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
                    ])
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

  def testToTensorflowTensorsRaisesIncompatibleSpecError(self):
    with self.assertRaisesRegex(ValueError, '.* is not compatible with .*'):
      util.to_tensorflow_tensors(
          {'features': {
              'feature_1': np.array([1, 2, 3], dtype=np.int64)
          }}, {'features': {
              'feature_1': tf.TensorSpec([1], dtype=tf.float32)
          }})

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
                'feature_1':
                    np.array([1.0, 2.0]),
                'feature_2':
                    np.array([1.0, 2.0]),
                'feature_3':
                    types.SparseTensorValue(
                        values=np.array([1]),
                        indices=np.array([[0, 1]]),
                        dense_shape=np.array([1, 3])),
                'feature_4':
                    types.RaggedTensorValue(
                        values=np.array([3, 1, 4, 1, 5, 9, 2, 6]),
                        nested_row_splits=[np.array([0, 4, 4, 7, 8, 8])])
            },
            'labels': np.array([1.0]),
            'example_weights': np.array(0.0),
            'predictions': {
                'model1': np.array([0.1, 0.2]),
                'model2': np.array([0.1, 0.2])
            },
            '_slice_key_types': [()]
        },
        {
            'features': {
                'feature_1':
                    np.array([3.0, 4.0]),
                'feature_2':
                    np.array([3.0, 4.0]),
                'feature_3':
                    types.SparseTensorValue(
                        values=np.array([2]),
                        indices=np.array([[0, 2]]),
                        dense_shape=np.array([1, 3])),
                'feature_4':
                    types.RaggedTensorValue(
                        values=np.array([3, 1, 4, 1, 5, 9, 2, 6]),
                        nested_row_splits=[np.array([0, 4, 4, 7, 8, 8])])
            },
            'labels': np.array([0.0]),
            'example_weights': np.array(0.5),
            'predictions': {
                'model1': np.array([0.3, 0.4]),
                'model2': np.array([0.3, 0.4])
            },
            '_slice_key_types': [()]
        },
        {
            'features': {
                'feature_1':
                    np.array([5.0, 6.0]),
                'feature_2':
                    np.array([5.0, 6.0]),
                'feature_3':
                    types.SparseTensorValue(
                        values=np.array([3]),
                        indices=np.array([[0, 0]]),
                        dense_shape=np.array([1, 3])),
                'feature_4':
                    types.RaggedTensorValue(
                        values=np.array([3, 1, 4, 1, 5, 9, 2, 6]),
                        nested_row_splits=[np.array([0, 4, 4, 7, 8, 8])])
            },
            'labels': np.array([1.0]),
            'example_weights': np.array(1.0),
            'predictions': {
                'model1': np.array([0.5, 0.6]),
                'model2': np.array([0.5, 0.6])
            },
            '_slice_key_types': [()]
        },
    ]

    expected = {
        'features': {
            'feature_1':
                np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            'feature_2':
                np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            'feature_3':
                types.SparseTensorValue(
                    values=np.array([1, 2, 3]),
                    indices=np.array([[0, 0, 1], [1, 0, 2], [2, 0, 0]]),
                    dense_shape=np.array([3, 1, 3])),
            'feature_4':
                types.RaggedTensorValue(
                    values=np.array([
                        3, 1, 4, 1, 5, 9, 2, 6, 3, 1, 4, 1, 5, 9, 2, 6, 3, 1, 4,
                        1, 5, 9, 2, 6
                    ]),
                    nested_row_splits=[
                        np.array([0, 5, 10, 15]),
                        np.array([
                            0, 4, 4, 7, 8, 8, 12, 12, 15, 16, 16, 20, 20, 23,
                            24, 24
                        ])
                    ])
        },
        'labels': np.array([1.0, 0.0, 1.0]),
        'example_weights': np.array([0.0, 0.5, 1.0]),
        'predictions': {
            'model1': np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            'model2': np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        },
        '_slice_key_types': np.array([(), (), ()])
    }

    self.assertAllClose(util.merge_extracts(extracts), expected)

  def testMergeExtractsRaisesException(self):
    extracts = [
        {
            'features': {
                'feature_3':
                    types.SparseTensorValue(
                        values=np.array([1]),
                        indices=np.array([[0, 1]]),
                        dense_shape=np.array([1, 2]))
            },
        },
        {
            'features': {
                'feature_3':
                    types.SparseTensorValue(
                        values=np.array([2]),
                        indices=np.array([[0, 2]]),
                        dense_shape=np.array([1, 3]))
            },
        },
    ]

    with self.assertRaisesWithPredicateMatch(
        RuntimeError, lambda exc: isinstance(exc.__cause__, RuntimeError)):
      util.merge_extracts(extracts)

  def testSplitExtracts(self):
    extracts = {
        'features': {
            'feature_1':
                np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            'feature_2':
                np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            'feature_3':
                types.SparseTensorValue(
                    values=np.array([1, 2, 3]),
                    indices=np.array([[0, 0, 1], [1, 0, 2], [2, 0, 0]]),
                    dense_shape=np.array([3, 1, 3])),
            'feature_4':
                types.RaggedTensorValue(
                    values=np.array([
                        3, 1, 4, 1, 5, 9, 2, 6, 3, 1, 4, 1, 5, 9, 2, 6, 3, 1, 4,
                        1, 5, 9, 2, 6
                    ]),
                    nested_row_splits=[
                        np.array([0, 5, 10, 15]),
                        np.array([
                            0, 4, 4, 7, 8, 8, 12, 12, 15, 16, 16, 20, 20, 23,
                            24, 24
                        ])
                    ])
        },
        'labels': np.array([1.0, 0.0, 1.0]),
        'example_weights': np.array([0.0, 0.5, 1.0]),
        'predictions': {
            'model1': np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            'model2': np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        },
        '_slice_key_types': np.array([(), (), ()])
    }

    expected = [
        {
            'features': {
                'feature_1':
                    np.array([1.0, 2.0]),
                'feature_2':
                    np.array([1.0, 2.0]),
                'feature_3':
                    types.SparseTensorValue(
                        values=np.array([1]),
                        indices=np.array([[0, 1]]),
                        dense_shape=np.array([1, 3])),
                'feature_4':
                    types.RaggedTensorValue(
                        values=np.array([3, 1, 4, 1, 5, 9, 2, 6]),
                        nested_row_splits=[np.array([0, 4, 4, 7, 8, 8])])
            },
            'labels': np.array([1.0]),
            'example_weights': np.array([0.0]),
            'predictions': {
                'model1': np.array([0.1, 0.2]),
                'model2': np.array([0.1, 0.2])
            },
            '_slice_key_types': np.array([()])
        },
        {
            'features': {
                'feature_1':
                    np.array([3.0, 4.0]),
                'feature_2':
                    np.array([3.0, 4.0]),
                'feature_3':
                    types.SparseTensorValue(
                        values=np.array([2]),
                        indices=np.array([[0, 2]]),
                        dense_shape=np.array([1, 3])),
                'feature_4':
                    types.RaggedTensorValue(
                        values=np.array([3, 1, 4, 1, 5, 9, 2, 6]),
                        nested_row_splits=[np.array([0, 4, 4, 7, 8, 8])])
            },
            'labels': np.array([0.0]),
            'example_weights': np.array([0.5]),
            'predictions': {
                'model1': np.array([0.3, 0.4]),
                'model2': np.array([0.3, 0.4])
            },
            '_slice_key_types': np.array([()])
        },
        {
            'features': {
                'feature_1':
                    np.array([5.0, 6.0]),
                'feature_2':
                    np.array([5.0, 6.0]),
                'feature_3':
                    types.SparseTensorValue(
                        values=np.array([3]),
                        indices=np.array([[0, 0]]),
                        dense_shape=np.array([1, 3])),
                'feature_4':
                    types.RaggedTensorValue(
                        values=np.array([3, 1, 4, 1, 5, 9, 2, 6]),
                        nested_row_splits=[np.array([0, 4, 4, 7, 8, 8])])
            },
            'labels': np.array([1.0]),
            'example_weights': np.array([1.0]),
            'predictions': {
                'model1': np.array([0.5, 0.6]),
                'model2': np.array([0.5, 0.6])
            },
            '_slice_key_types': np.array([()])
        },
    ]

    self.assertAllClose(util.split_extracts(extracts), expected)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
