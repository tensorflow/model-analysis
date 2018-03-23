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
"""Unit test for graph_ref module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_model_analysis.eval_saved_model import encoding
from tensorflow_model_analysis.eval_saved_model import graph_ref

from tensorflow.core.protobuf import meta_graph_pb2


class GraphRefTest(tf.test.TestCase):

  def setUp(self):
    self.longMessage = True

  def testGetNodeMapBasic(self):
    meta_graph_def = meta_graph_pb2.MetaGraphDef()
    meta_graph_def.collection_def[
        'my_collection/%s' % encoding.KEY_SUFFIX].bytes_list.value[:] = map(
            encoding.encode_key, ['alpha', 'bravo', 'charlie'])
    meta_graph_def.collection_def[
        'my_collection/fruits'].bytes_list.value[:] = [
            'apple', 'banana', 'cherry'
        ]
    expected = {
        'alpha': {
            'fruits': 'apple'
        },
        'bravo': {
            'fruits': 'banana'
        },
        'charlie': {
            'fruits': 'cherry'
        }
    }
    self.assertDictEqual(expected,
                         graph_ref.get_node_map(meta_graph_def, 'my_collection',
                                                ['fruits']))

  def testGetNodeMapEmpty(self):
    meta_graph_def = meta_graph_pb2.MetaGraphDef()
    self.assertDictEqual({},
                         graph_ref.get_node_map(meta_graph_def, 'my_collection',
                                                ['fruits']))

  def testGetNodeMapMultiple(self):
    meta_graph_def = meta_graph_pb2.MetaGraphDef()
    meta_graph_def.collection_def[
        'my_collection/%s' % encoding.KEY_SUFFIX].bytes_list.value[:] = map(
            encoding.encode_key, ['alpha', 'bravo', 'charlie'])
    meta_graph_def.collection_def[
        'my_collection/fruits'].bytes_list.value[:] = [
            'apple', 'banana', 'cherry'
        ]
    meta_graph_def.collection_def[
        'my_collection/animals'].bytes_list.value[:] = [
            'aardvark', 'badger', 'camel'
        ]
    expected = {
        'alpha': {
            'fruits': 'apple',
            'animals': 'aardvark'
        },
        'bravo': {
            'fruits': 'banana',
            'animals': 'badger'
        },
        'charlie': {
            'fruits': 'cherry',
            'animals': 'camel'
        }
    }
    self.assertDictEqual(expected,
                         graph_ref.get_node_map(meta_graph_def, 'my_collection',
                                                ['fruits', 'animals']))

  def testGetNodeMapInGraph(self):
    g = tf.Graph()
    with g.as_default():
      apple = tf.constant(1.0)
      banana = tf.constant(2.0)
      cherry = tf.constant(3.0)
      aardvark = tf.constant('a')
      badger = tf.constant('b')
      camel = tf.constant('c')

    meta_graph_def = meta_graph_pb2.MetaGraphDef()
    meta_graph_def.collection_def[
        'my_collection/%s' % encoding.KEY_SUFFIX].bytes_list.value[:] = map(
            encoding.encode_key, ['alpha', 'bravo', 'charlie'])

    meta_graph_def.collection_def['my_collection/fruits'].any_list.value.extend(
        map(encoding.encode_tensor_node, [apple, banana, cherry]))
    meta_graph_def.collection_def[
        'my_collection/animals'].any_list.value.extend(
            map(encoding.encode_tensor_node, [aardvark, badger, camel]))
    expected = {
        'alpha': {
            'fruits': apple,
            'animals': aardvark,
        },
        'bravo': {
            'fruits': banana,
            'animals': badger,
        },
        'charlie': {
            'fruits': cherry,
            'animals': camel,
        }
    }
    self.assertDictEqual(expected,
                         graph_ref.get_node_map_in_graph(
                             meta_graph_def, 'my_collection',
                             ['fruits', 'animals'], g))

  def testGetNodeInGraph(self):
    g = tf.Graph()
    with g.as_default():
      apple = tf.constant(1.0)

    meta_graph_def = meta_graph_pb2.MetaGraphDef()
    meta_graph_def.collection_def['fruit_node'].any_list.value.extend(
        [encoding.encode_tensor_node(apple)])

    self.assertEqual(apple,
                     graph_ref.get_node_in_graph(meta_graph_def, 'fruit_node',
                                                 g))


if __name__ == '__main__':
  tf.test.main()
