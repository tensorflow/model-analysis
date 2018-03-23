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
"""Unit test for encoding / decoding functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_model_analysis.eval_saved_model import encoding


class EncodingTest(tf.test.TestCase):

  def setUp(self):
    self.longMessage = True

  def testEncodeDecodeKey(self):
    test_cases = [
        'a', 'simple', 'dollar$', '$dollar', '$do$ll$ar$', ('a'),
        ('a', 'simple'), ('dollar$', 'simple'), ('do$llar', 'sim$ple', 'str$'),
        ('many', 'many', 'elements', 'in', 'the', 'tuple'), u'unicode\u1234',
        u'uni\u1234code\u2345', ('mixed', u'uni\u1234',
                                 u'\u2345\u1234'), (u'\u1234\u2345',
                                                    u'\u3456\u2345')
    ]
    for key in test_cases:
      self.assertEqual(key, encoding.decode_key(encoding.encode_key(key)))

  def testEncodeDecodeTensorNode(self):
    g = tf.Graph()
    with g.as_default():
      example = tf.placeholder(tf.string, name='example')
      features = tf.parse_example(
          example, {
              'age': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
              'gender': tf.FixedLenFeature([], dtype=tf.string),
              'varstr': tf.VarLenFeature(tf.string),
              'varint': tf.VarLenFeature(tf.int64),
              'varfloat': tf.VarLenFeature(tf.float32),
              u'unicode\u1234': tf.FixedLenFeature([], dtype=tf.string),
          })
      constant = tf.constant(1.0)
      sparse = tf.SparseTensor(
          indices=tf.placeholder(tf.int64),
          values=tf.placeholder(tf.int64),
          dense_shape=tf.placeholder(tf.int64))

    test_cases = [
        example, features['age'], features['gender'], features['varstr'],
        features['varint'], features['varfloat'], features[u'unicode\u1234'],
        constant, sparse
    ]
    for tensor in test_cases:
      got_tensor = encoding.decode_tensor_node(
          g, encoding.encode_tensor_node(tensor))
      if isinstance(tensor, tf.SparseTensor):
        self.assertEqual(tensor.indices, got_tensor.indices)
        self.assertEqual(tensor.values, got_tensor.values)
        self.assertEqual(tensor.dense_shape, got_tensor.dense_shape)
      else:
        self.assertEqual(tensor, got_tensor)


if __name__ == '__main__':
  tf.test.main()
