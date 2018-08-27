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
"""Library for finding nodes in a graph based on metadata in meta_graph_def.

This is an internal library for use only by load.py.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function


import tensorflow as tf
from tensorflow_model_analysis import types
from tensorflow_model_analysis.eval_saved_model import encoding
from tensorflow_model_analysis.types_compat import Dict, List, Union

from google.protobuf import any_pb2
from tensorflow.core.protobuf import meta_graph_pb2

CollectionDefValueType = Union[float, int, bytes, any_pb2.Any]  # pylint: disable=invalid-name


def extract_signature_outputs_with_prefix(
    prefix,
    # signature_outputs is not actually a Dict, but behaves like one
    signature_outputs
):
  """Extracts signature outputs with the given prefix.

  This is the reverse of _wrap_and_check_metrics / _wrap_and_check_outputs and
  _prefix_output_keys in tf.estimator.export.ExportOutput.

  This is designed to extract structures from the  SignatureDef outputs map.

   Structures of the following form:
      <prefix>/key1
      <prefix>/key2
  will map to dictionary elements like so:
      {key1: value1, key2: value2}

  Structures of the following form:
      <prefix>
      <prefix>_extrastuff
      <prefix>morestuff
  will map to dictionary elements like so:
      {<prefix: value1, <prefix>_extrastuff: value2, <prefix>morestuff: value3}

  Args:
    prefix: Prefix to extract
    signature_outputs: Signature outputs to extract from

  Returns:
    Dictionary extracted as described above. The values will be the TensorInfo
    associated with the keys.

  Raises:
    ValueError: There were duplicate keys.
  """
  result = {}
  for k, v in signature_outputs.items():
    if k.startswith(prefix + '/'):
      key = k[len(prefix) + 1:]
    elif k.startswith(prefix):
      key = k
    else:
      continue

    if key in result:
      raise ValueError(
          'key "%s" already in dictionary. you might have repeated keys. '
          'prefix was "%s", signature_outputs were: %s' % (prefix, key,
                                                           signature_outputs))
    result[key] = v
  return result


def get_node_map(meta_graph_def, prefix,
                 node_suffixes
                ):
  """Get node map from meta_graph_def.

  This is designed to extract structures of the following form from the
  meta_graph_def collection_def:
    prefix/key
      key1
      key2
      key3
    prefix/suffix_a
      node1
      node2
      node3
    prefix/suffix_b
      node4
      node5
      node6

   which will become a dictionary:
   {
     key1 : {suffix_a: node1, suffix_b: node4}
     key2 : {suffix_a: node2, suffix_b: node5}
     key3 : {suffix_a: node3, suffix_b: node6}
   }.

  Keys must always be bytes. Values can be any supported CollectionDef type
  (bytes_list, any_list, etc)

  Args:
     meta_graph_def: MetaGraphDef containing the CollectionDefs to extract the
       structure from.
     prefix: Prefix for the CollectionDef names.
     node_suffixes: The suffixes to the prefix to form the names of the
       CollectionDefs to extract the nodes from, e.g. in the example described
       above, node_suffixes would be ['suffix_a', 'suffix_b'].

  Returns:
    A dictionary of dictionaries, as described in the example above.

  Raises:
    ValueError: The length of some node list did not match length of the key
    list.
  """
  node_lists = []
  for node_suffix in node_suffixes:
    collection_def_name = '%s/%s' % (prefix, node_suffix)
    collection_def = meta_graph_def.collection_def.get(collection_def_name)
    if collection_def is None:
      # If we can't find the CollectionDef, append an empty list.
      #
      # Either all the CollectionDefs are missing, in which case we correctly
      # return an empty dict, or some of the CollectionDefs are non-empty,
      # in which case we raise an exception below.
      node_lists.append([])
    else:
      node_lists.append(
          getattr(collection_def, collection_def.WhichOneof('kind')).value)
  keys = meta_graph_def.collection_def['%s/%s' %
                                       (prefix,
                                        encoding.KEY_SUFFIX)].bytes_list.value
  if not all([len(node_list) == len(keys) for node_list in node_lists]):
    raise ValueError('length of each node_list should match length of keys. '
                     'prefix was %s, node_lists were %s, keys was %s' %
                     (prefix, node_lists, keys))
  result = {}
  for key, elems in zip(keys, zip(*node_lists)):
    result[encoding.decode_key(key)] = dict(zip(node_suffixes, elems))
  return result


def get_node_map_in_graph(
    meta_graph_def, prefix,
    node_suffixes,
    graph):
  """Like get_node_map, but looks up the nodes in the given graph.

  Args:
     meta_graph_def: MetaGraphDef containing the CollectionDefs to extract the
       structure from.
     prefix: Prefix for the CollectionDef names.
     node_suffixes: The suffixes to the prefix to form the names of the
       CollectionDefs to extract the nodes from, e.g. in the example described
       above, node_suffixes would be ['suffix_a', 'suffix_b'].
     graph: TensorFlow graph to lookup the nodes in.

  Returns:
    A dictionary of dictionaries like get_node_map, except the values are
    the actual nodes in the graph.
  """
  node_map = get_node_map(meta_graph_def, prefix, node_suffixes)
  result = {}
  for key, elems in node_map.items():
    result[key] = {
        k: encoding.decode_tensor_node(graph, n)
        for k, n in elems.items()
    }
  return result


def get_node_wrapped_tensor_info(meta_graph_def,
                                 path):
  """Get the Any-wrapped TensorInfo for the node from the meta_graph_def.

  Args:
     meta_graph_def: MetaGraphDef containing the CollectionDefs to extract the
       node name from.
     path: Name of the collection containing the node name.

  Returns:
    The Any-wrapped TensorInfo for the node retrieved from the CollectionDef.

  Raises:
    KeyError: There was no CollectionDef with the given name (path).
    ValueError: The any_list in the CollectionDef with the given name did
      not have length 1.
  """
  if path not in meta_graph_def.collection_def:
    raise KeyError('could not find path %s in collection defs. meta_graph_def '
                   'was %s' % (path, meta_graph_def))
  if len(meta_graph_def.collection_def[path].any_list.value) != 1:
    raise ValueError(
        'any_list should be of length 1. path was %s, any_list was: %s.' %
        (path, meta_graph_def.collection_def[path].any_list.value))
  return meta_graph_def.collection_def[path].any_list.value[0]


def get_node_in_graph(meta_graph_def, path,
                      graph):
  """Like get_node_wrapped_tensor_info, but looks up the node in the graph.

  Args:
     meta_graph_def: MetaGraphDef containing the CollectionDefs to extract the
       node name from.
     path: Name of the collection containing the node name.
     graph: TensorFlow graph to lookup the nodes in.

  Returns:
    The node in the graph with the name returned by
    get_node_wrapped_tensor_info.
  """
  return encoding.decode_tensor_node(graph,
                                     get_node_wrapped_tensor_info(
                                         meta_graph_def, path))
