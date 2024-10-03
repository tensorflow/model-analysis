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
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.eval_saved_model import util
from tensorflow.core.example import example_pb2


class UtilTest(testutil.TensorflowModelAnalysisTest):

  def testMakeExample(self):
    expected = example_pb2.Example()
    expected.features.feature['single_float'].float_list.value[:] = [1.0]
    expected.features.feature['single_int'].int64_list.value[:] = [2]
    expected.features.feature['single_str'].bytes_list.value[:] = [b'apple']
    expected.features.feature['multi_float'].float_list.value[:] = [4.0, 5.0]
    expected.features.feature['multi_int'].int64_list.value[:] = [6, 7]
    expected.features.feature['multi_str'].bytes_list.value[:] = [
        b'orange', b'banana'
    ]
    self.assertEqual(
        expected,
        util.make_example(
            single_float=1.0,
            single_int=2,
            single_str='apple',
            multi_float=[4.0, 5.0],
            multi_int=[6, 7],
            multi_str=['orange', 'banana']))

  def testSplitTensorValueDense(self):
    split_tensor_values = util.split_tensor_value(
        np.ndarray(shape=(3, 2), buffer=np.array([2, 4, 6, 8, 10, 12])))
    self.assertAllEqual([
        np.ndarray(shape=(1, 2), buffer=np.array([2, 4])),
        np.ndarray(shape=(1, 2), buffer=np.array([6, 8])),
        np.ndarray(shape=(1, 2), buffer=np.array([10, 12])),
    ], split_tensor_values)

  def testSplitTensorValueSparse(self):
    split_tensor_values = util.split_tensor_value(
        tf.compat.v1.SparseTensorValue(
            indices=np.array([[0, 0], [0, 1], [1, 0], [1, 1], [3, 0], [3, 1]]),
            values=np.array([1, 3, 5, 7, 9, 11]),
            dense_shape=np.array([4, 2])))
    expected_sparse_tensor_values = [
        tf.compat.v1.SparseTensorValue(
            indices=np.array([[0, 0], [0, 1]]),
            values=np.array([1, 3]),
            dense_shape=np.array([1, 2])),
        tf.compat.v1.SparseTensorValue(
            indices=np.array([[0, 0], [0, 1]]),
            values=np.array([5, 7]),
            dense_shape=np.array([1, 2])),
        tf.compat.v1.SparseTensorValue(
            indices=np.zeros([0, 2], dtype=np.int64),
            values=np.zeros([0], dtype=np.int64),
            dense_shape=np.array([1, 0])),
        tf.compat.v1.SparseTensorValue(
            indices=np.array([[0, 0], [0, 1]]),
            values=np.array([9, 11]),
            dense_shape=np.array([1, 2])),
    ]
    for expected_sparse_tensor_value, got_sparse_tensor_value in zip(
        expected_sparse_tensor_values, split_tensor_values):
      self.assertSparseTensorValueEqual(expected_sparse_tensor_value,
                                        got_sparse_tensor_value)

  def testSplitTensorValueSparseVarLen(self):
    split_tensor_values = util.split_tensor_value(
        tf.compat.v1.SparseTensorValue(
            indices=np.array([[0, 0], [1, 0], [1, 1], [1, 2], [1, 3], [2, 0],
                              [2, 1]]),
            values=np.array([1, 2, 3, 4, 5, 6, 7]),
            dense_shape=np.array([3, 4])))
    expected_sparse_tensor_values = [
        tf.compat.v1.SparseTensorValue(
            indices=np.array([[0, 0]]),
            values=np.array([1]),
            dense_shape=np.array([1, 1])),
        tf.compat.v1.SparseTensorValue(
            indices=np.array([[0, 0], [0, 1], [0, 2], [0, 3]]),
            values=np.array([2, 3, 4, 5]),
            dense_shape=np.array([1, 4])),
        tf.compat.v1.SparseTensorValue(
            indices=np.array([[0, 0], [0, 1]]),
            values=np.array([6, 7]),
            dense_shape=np.array([1, 2])),
    ]
    for expected_sparse_tensor_value, got_sparse_tensor_value in zip(
        expected_sparse_tensor_values, split_tensor_values):
      self.assertSparseTensorValueEqual(expected_sparse_tensor_value,
                                        got_sparse_tensor_value)

  def testSplitTensorValueSparseVarLenMultiDim(self):
    split_tensor_values = util.split_tensor_value(
        tf.compat.v1.SparseTensorValue(
            indices=np.array([[0, 0, 0], [0, 0, 1], [1, 1, 2], [1, 3, 4],
                              [3, 0, 3], [3, 2, 1], [3, 3, 0]],
                             dtype=np.int64),
            values=np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int64),
            dense_shape=np.array([4, 4, 5])))
    expected_sparse_tensor_values = [
        tf.compat.v1.SparseTensorValue(
            indices=np.array([[0, 0, 0], [0, 0, 1]]),
            values=np.array([1, 2]),
            dense_shape=np.array([1, 1, 2])),
        tf.compat.v1.SparseTensorValue(
            indices=np.array([[0, 1, 2], [0, 3, 4]]),
            values=np.array([3, 4]),
            dense_shape=np.array([1, 4, 5])),
        tf.compat.v1.SparseTensorValue(
            indices=np.zeros([0, 3], dtype=np.int64),
            values=np.zeros([0], dtype=np.int64),
            dense_shape=np.array([1, 0, 0])),
        tf.compat.v1.SparseTensorValue(
            indices=np.array([[0, 0, 3], [0, 2, 1], [0, 3, 0]]),
            values=np.array([5, 6, 7]),
            dense_shape=np.array([1, 4, 4])),
    ]
    for expected_sparse_tensor_value, got_sparse_tensor_value in zip(
        expected_sparse_tensor_values, split_tensor_values):
      self.assertSparseTensorValueEqual(expected_sparse_tensor_value,
                                        got_sparse_tensor_value)

  def testSplitTensorValueSparseTypesPreserved(self):
    split_tensor_values = util.split_tensor_value(
        tf.compat.v1.SparseTensorValue(
            indices=np.array([[0, 0], [0, 1], [2, 0], [3, 1]]),
            values=np.array(['zero0', 'zero1', 'two0', 'three1'], dtype=object),
            dense_shape=np.array([4, 3])))
    expected_sparse_tensor_values = [
        tf.compat.v1.SparseTensorValue(
            indices=np.array([[0, 0], [0, 1]]),
            values=np.array(['zero0', 'zero1'], dtype=object),
            dense_shape=np.array([1, 2])),
        tf.compat.v1.SparseTensorValue(
            indices=np.zeros([0, 2], dtype=np.int64),
            values=np.zeros([0], dtype=object),
            dense_shape=np.array([1, 0])),
        tf.compat.v1.SparseTensorValue(
            indices=np.array([[0, 0]]),
            values=np.array(['two0'], dtype=object),
            dense_shape=np.array([1, 1])),
        tf.compat.v1.SparseTensorValue(
            indices=np.array([[0, 1]]),
            values=np.array(['three1'], dtype=object),
            dense_shape=np.array([1, 2])),
    ]
    for expected_sparse_tensor_value, got_sparse_tensor_value in zip(
        expected_sparse_tensor_values, split_tensor_values):
      self.assertSparseTensorValueEqual(expected_sparse_tensor_value,
                                        got_sparse_tensor_value)

  def testMergeTensorValueDense(self):
    merged_tensor_values = util.merge_tensor_values(tensor_values=[
        np.ndarray(shape=(1, 2), buffer=np.array([1, 2])),
        np.ndarray(shape=(1, 2), buffer=np.array([3, 4])),
        np.ndarray(shape=(1, 2), buffer=np.array([5, 6])),
    ])
    self.assertAllEqual(
        np.ndarray(shape=(3, 2), buffer=np.array([1, 2, 3, 4, 5, 6])),
        merged_tensor_values)

  def testMergeTensorValueDenseDifferentShapesInts(self):
    merged_tensor_values = util.merge_tensor_values(tensor_values=[
        np.array([[[10], [11]]]),
        np.array([[[20, 21, 22]]]),
        np.array([[[30, 31], [32, 33], [34, 35]]]),
        np.array([[[40, 41]]]),
    ])
    self.assertAllEqual(
        np.array([
            # Row 0
            [[10, 0, 0], [11, 0, 0], [0, 0, 0]],
            # Row 1
            [[20, 21, 22], [0, 0, 0], [0, 0, 0]],
            # Row 2
            [[30, 31, 0], [32, 33, 0], [34, 35, 0]],
            # Row 3
            [[40, 41, 0], [0, 0, 0], [0, 0, 0]],
        ]),
        merged_tensor_values)

  def testMergeTensorValueDenseDifferentShapesStrings(self):
    merged_tensor_values = util.merge_tensor_values(tensor_values=[
        np.array([[['apple'], ['banana']]]),
        np.array([[['cherry', 'date', 'elderberry']]]),
        np.array([[['fig', 'guava'], ['honeydew', 'imbe'],
                   ['jackfruit', 'kiwi']]])
    ])
    self.assertAllEqual(
        np.array([
            # Row 0
            [['apple', '', ''], ['banana', '', ''], ['', '', '']],
            # Row 1
            [['cherry', 'date', 'elderberry'], ['', '', ''], ['', '', '']],
            # Row 2
            [['fig', 'guava', ''], ['honeydew', 'imbe', ''],
             ['jackfruit', 'kiwi', '']]
        ]),
        merged_tensor_values)

  def testMergeTensorValueSparse(self):
    merged_tensor_values = util.merge_tensor_values(tensor_values=[
        tf.compat.v1.SparseTensorValue(
            indices=np.array([[0, 0], [0, 1]]),
            values=np.array([1, 2]),
            dense_shape=np.array([1, 2])),
        tf.compat.v1.SparseTensorValue(
            indices=np.array([[0, 0], [0, 1]]),
            values=np.array([3, 4]),
            dense_shape=np.array([1, 2])),
        tf.compat.v1.SparseTensorValue(
            indices=np.array([[0, 0], [0, 1]]),
            values=np.array([5, 6]),
            dense_shape=np.array([1, 2])),
    ])
    self.assertSparseTensorValueEqual(
        tf.compat.v1.SparseTensorValue(
            indices=np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]),
            values=np.array([1, 2, 3, 4, 5, 6]),
            dense_shape=np.array([3, 2])), merged_tensor_values)

  def testMergeTensorValuesSparseOriginalsUnmodified(self):
    value1 = tf.compat.v1.SparseTensorValue(
        indices=np.array([]).reshape([0, 2]),
        values=np.array([]).reshape([0, 1]),
        dense_shape=np.array([1, 4]))
    value2 = tf.compat.v1.SparseTensorValue(
        indices=np.array([]).reshape([0, 2]),
        values=np.array([]).reshape([0, 1]),
        dense_shape=np.array([1, 4]))
    merged_tensor_values = util.merge_tensor_values(
        tensor_values=[value1, value2])

    # Check that the original SparseTensorValues were not mutated.
    self.assertSparseTensorValueEqual(
        tf.compat.v1.SparseTensorValue(
            indices=np.array([]).reshape([0, 2]),
            values=np.array([]).reshape([0, 1]),
            dense_shape=np.array([1, 4])), value1)
    self.assertSparseTensorValueEqual(
        tf.compat.v1.SparseTensorValue(
            indices=np.array([]).reshape([0, 2]),
            values=np.array([]).reshape([0, 1]),
            dense_shape=np.array([1, 4])), value2)

    # Check the merged SparseTensorValue.
    self.assertSparseTensorValueEqual(
        tf.compat.v1.SparseTensorValue(
            indices=np.array([]).reshape([0, 2]),
            values=np.array([]).reshape([0, 1]),
            dense_shape=np.array([2, 4])), merged_tensor_values)

  def testMergeTensorValueSparseDifferentShapes(self):
    merged_tensor_values = util.merge_tensor_values(tensor_values=[
        tf.compat.v1.SparseTensorValue(
            indices=np.array([[0, 0, 0], [0, 1, 1]]),
            values=np.array([10, 12]),
            dense_shape=np.array([1, 2, 2])),
        tf.compat.v1.SparseTensorValue(
            indices=np.array([[0, 2, 2]]),
            values=np.array([22]),
            dense_shape=np.array([1, 3, 3])),
        tf.compat.v1.SparseTensorValue(
            indices=np.array([[0, 0, 4]]),
            values=np.array([33]),
            dense_shape=np.array([1, 1, 5]))
    ])

    self.assertSparseTensorValueEqual(
        tf.compat.v1.SparseTensorValue(
            indices=np.array([[0, 0, 0], [0, 1, 1], [1, 2, 2], [2, 0, 4]]),
            values=np.array([10, 12, 22, 33]),
            dense_shape=np.array([3, 3, 5])), merged_tensor_values)


