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
    self.longMessage = True  # pylint: disable=invalid-name

  def testExtractSignatureOutputsWithPrefix(self):
    signature_def = meta_graph_pb2.SignatureDef()

    with tf.Graph().as_default():  # Needed to disable eager mode.

      def make_tensor_info(name):
        return tf.compat.v1.saved_model.utils.build_tensor_info(
            tf.constant(0.0, name=name))

      # Test for single entry (non-dict) tensors.
      signature_def.inputs['labels'].CopyFrom(make_tensor_info('labels'))

      signature_def.outputs['predictions'].CopyFrom(
          make_tensor_info('predictions'))
      signature_def.outputs['metrics/mean/value'].CopyFrom(
          make_tensor_info('mean_value'))
      signature_def.outputs['metrics/mean/update_op'].CopyFrom(
          make_tensor_info('mean_update'))

      # This case is to check that things like
      # predictions/predictions are okay.
      signature_def.outputs['prefix'].CopyFrom(make_tensor_info('prefix'))
      signature_def.outputs['prefix1'].CopyFrom(make_tensor_info('prefix1'))
      signature_def.outputs['prefix2'].CopyFrom(make_tensor_info('prefix2'))
      signature_def.outputs['prefix/stuff'].CopyFrom(
          make_tensor_info('prefix/stuff'))
      signature_def.outputs['prefix/sub/more'].CopyFrom(
          make_tensor_info('prefix/sub/more'))

      self.assertDictEqual(
          {'__labels': signature_def.inputs['labels']},
          graph_ref.extract_signature_inputs_or_outputs_with_prefix(
              'labels', signature_def.inputs, '__labels'))

      self.assertDictEqual(
          {'predictions': signature_def.outputs['predictions']},
          graph_ref.extract_signature_inputs_or_outputs_with_prefix(
              'predictions', signature_def.outputs))

      self.assertDictEqual(
          {
              'mean/value': signature_def.outputs['metrics/mean/value'],
              'mean/update_op': signature_def.outputs['metrics/mean/update_op']
          },
          graph_ref.extract_signature_inputs_or_outputs_with_prefix(
              'metrics', signature_def.outputs))

      self.assertDictEqual(
          {
              'prefix': signature_def.outputs['prefix'],
              'prefix1': signature_def.outputs['prefix1'],
              'prefix2': signature_def.outputs['prefix2'],
              'stuff': signature_def.outputs['prefix/stuff'],
              'sub/more': signature_def.outputs['prefix/sub/more'],
          },
          graph_ref.extract_signature_inputs_or_outputs_with_prefix(
              'prefix', signature_def.outputs))

  def testGetNodeMapBasic(self):
    meta_graph_def = meta_graph_pb2.MetaGraphDef()
    meta_graph_def.collection_def[
        'my_collection/%s' % encoding.KEY_SUFFIX].bytes_list.value[:] = map(
            encoding.encode_key, ['alpha', 'bravo', 'charlie'])
    meta_graph_def.collection_def[
        'my_collection/fruits'].bytes_list.value[:] = [
            b'apple', b'banana', b'cherry'
        ]
    expected = {
        'alpha': {
            'fruits': b'apple'
        },
        'bravo': {
            'fruits': b'banana'
        },
        'charlie': {
            'fruits': b'cherry'
        }
    }
    self.assertDictEqual(
        expected,
        graph_ref.get_node_map(meta_graph_def, 'my_collection', ['fruits']))

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
            b'apple', b'banana', b'cherry'
        ]
    meta_graph_def.collection_def[
        'my_collection/animals'].bytes_list.value[:] = [
            b'aardvark', b'badger', b'camel'
        ]
    expected = {
        'alpha': {
            'fruits': b'apple',
            'animals': b'aardvark'
        },
        'bravo': {
            'fruits': b'banana',
            'animals': b'badger'
        },
        'charlie': {
            'fruits': b'cherry',
            'animals': b'camel'
        }
    }
    self.assertDictEqual(
        expected,
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

      meta_graph_def.collection_def[
          'my_collection/fruits'].any_list.value.extend(
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
      self.assertDictEqual(
          expected,
          graph_ref.get_node_map_in_graph(meta_graph_def, 'my_collection',
                                          ['fruits', 'animals'], g))

  def testGetNodeInGraph(self):
    g = tf.Graph()
    with g.as_default():
      apple = tf.constant(1.0)

      meta_graph_def = meta_graph_pb2.MetaGraphDef()
      meta_graph_def.collection_def['fruit_node'].any_list.value.extend(
          [encoding.encode_tensor_node(apple)])

      self.assertEqual(
          apple, graph_ref.get_node_in_graph(meta_graph_def, 'fruit_node', g))


if __name__ == '__main__':
  tf.test.main()
