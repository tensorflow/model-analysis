# Lint as: python3
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
"""Slice accessor test."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import pyarrow as pa
import tensorflow as tf
from tensorflow_model_analysis.eval_saved_model import encoding
from tensorflow_model_analysis.slicer import slice_accessor


class SliceAccessorTest(tf.test.TestCase, parameterized.TestCase):

  def testRaisesKeyError(self):
    accessor = slice_accessor.SliceAccessor({})
    with self.assertRaises(KeyError):
      accessor.get('no_such_key')

  @parameterized.named_parameters(
      ('sparse_tensor_value',
       tf.compat.v1.SparseTensorValue(
           indices=np.array([[0, 0], [1, 1]]),
           values=np.array(['apple', 'banana']),
           dense_shape=np.array([2, 2])), ['apple', 'banana']),
      ('ragged_tensor_value',
       tf.compat.v1.ragged.RaggedTensorValue(
           values=np.array([1, 2, 3]), row_splits=np.array([0, 0, 1])),
       [1, 2, 3]), ('dense', np.array([1.0, 2.0]), [1.0, 2.0]),
      ('dense_single', np.array([7.0]), [7.0]),
      ('dense_multidim', np.array([[1.0, 2.0], [3.0, 4.0]]),
       [1.0, 2.0, 3.0, 4.0]), ('squeeze_needed', np.array([[2.0]]), [2.0]),
      ('list', [1, 2, 3], [1, 2, 3]),
      ('pyarrow', pa.array([1, 2, 3]), [1, 2, 3]),
      ('pyarrow_ragged', pa.array([[1, 2], [3]]), [1, 2, 3]))
  def testAccessFeaturesDict(self, feature_value, slice_value):
    accessor = slice_accessor.SliceAccessor([{'feature': feature_value}])
    self.assertEqual(slice_value, accessor.get('feature'))
    # Test with multiple dicts and duplicate values
    accessor = slice_accessor.SliceAccessor([{
        'feature': feature_value
    }, {
        'feature': feature_value
    }])
    self.assertEqual(slice_value, accessor.get('feature'))
    # Test with default features dict
    accessor = slice_accessor.SliceAccessor(
        [{
            'unmatched_feature': feature_value
        }],
        default_features_dict={'feature': feature_value})
    self.assertEqual(slice_value, accessor.get('feature'))

  def testLegacyAccessFeaturesDict(self):
    with tf.compat.v1.Session() as sess:
      sparse = tf.SparseTensor(
          indices=[[0, 0], [1, 1]],
          values=['apple', 'banana'],
          dense_shape=[2, 2])
      dense = tf.constant([1.0, 2.0])
      dense_single = tf.constant([7.0])
      dense_multidim = tf.constant([[1.0, 2.0], [3.0, 4.0]])
      squeeze_needed = tf.constant([[2.0]])
      (sparse_value, dense_value, dense_single_value, dense_multidim_value,
       squeeze_needed_value
      ) = sess.run(
          fetches=[sparse, dense, dense_single, dense_multidim, squeeze_needed])
      features_dict = {
          'sparse': {
              encoding.NODE_SUFFIX: sparse_value
          },
          'dense': {
              encoding.NODE_SUFFIX: dense_value
          },
          'dense_single': {
              encoding.NODE_SUFFIX: dense_single_value
          },
          'squeeze_needed': {
              encoding.NODE_SUFFIX: squeeze_needed_value
          },
          'dense_multidim': {
              encoding.NODE_SUFFIX: dense_multidim_value
          },
      }
      accessor = slice_accessor.SliceAccessor([features_dict])
      self.assertEqual([b'apple', b'banana'], accessor.get('sparse'))
      self.assertEqual([1.0, 2.0], accessor.get('dense'))
      self.assertEqual([7.0], accessor.get('dense_single'))
      self.assertEqual([1.0, 2.0, 3.0, 4.0], accessor.get('dense_multidim'))
      self.assertEqual([2.0], accessor.get('squeeze_needed'))


if __name__ == '__main__':
  tf.test.main()
