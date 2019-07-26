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

# Standard Imports

import tensorflow as tf
from tensorflow_model_analysis.eval_saved_model import encoding
from tensorflow_model_analysis.slicer import slice_accessor


class SliceAccessorTest(tf.test.TestCase):

  def testAccessFeaturesDict(self):
    with tf.compat.v1.Session() as sess:
      sparse = tf.SparseTensor(
          indices=[[0, 0], [1, 1]],
          values=['apple', 'banana'],
          dense_shape=[2, 2])
      dense = tf.constant([1.0, 2.0])
      dense_single = tf.constant([7.0])
      bad_dense = tf.constant([[1.0, 2.0], [3.0, 4.0]])
      squeeze_needed = tf.constant([[2.0]])
      (sparse_value, dense_value, dense_single_value, bad_dense_value,
       squeeze_needed_value) = sess.run(
           fetches=[sparse, dense, dense_single, bad_dense, squeeze_needed])
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
          'bad_dense': {
              encoding.NODE_SUFFIX: bad_dense_value
          },
      }
      accessor = slice_accessor.SliceAccessor(features_dict)
      self.assertEqual([b'apple', b'banana'], accessor.get('sparse'))
      self.assertEqual([1.0, 2.0], accessor.get('dense'))
      self.assertEqual([7.0], accessor.get('dense_single'))
      self.assertEqual([2.0], accessor.get('squeeze_needed'))
      with self.assertRaises(ValueError):
        accessor.get('bad_dense')
      with self.assertRaises(KeyError):
        accessor.get('no_such_key')


if __name__ == '__main__':
  tf.test.main()
