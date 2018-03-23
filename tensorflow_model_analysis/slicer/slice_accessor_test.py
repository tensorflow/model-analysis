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



import tensorflow as tf
from tensorflow_model_analysis.eval_saved_model import encoding
from tensorflow_model_analysis.slicer import slice_accessor


class SliceAccessorTest(tf.test.TestCase):

  def testAccessFeaturesDict(self):
    sparse = tf.SparseTensor(
        indices=[[0, 0], [1, 1]],
        values=['apple', 'banana'],
        dense_shape=[2, 2])
    dense = tf.constant([1.0, 2.0])
    bad_dense = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    sess = tf.Session()
    sparse_value, dense_value, bad_dense_value = sess.run(
        fetches=[sparse, dense, bad_dense])
    features_dict = {
        'sparse': {
            encoding.NODE_SUFFIX: sparse_value
        },
        'dense': {
            encoding.NODE_SUFFIX: dense_value
        },
        'bad_dense': {
            encoding.NODE_SUFFIX: bad_dense_value
        },
    }
    accessor = slice_accessor.SliceAccessor(features_dict)
    self.assertEqual(['apple', 'banana'], list(accessor.get('sparse')))
    self.assertEqual([1.0, 2.0], list(accessor.get('dense')))
    with self.assertRaises(ValueError):
      accessor.get('bad_dense')
    with self.assertRaises(KeyError):
      accessor.get('no_such_key')


if __name__ == '__main__':
  tf.test.main()
