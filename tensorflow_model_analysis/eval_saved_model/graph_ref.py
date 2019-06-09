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
# Standard __future__ imports
from __future__ import print_function

import collections
# Standard Imports
import tensorflow as tf
from tensorflow_model_analysis import types
from tensorflow_model_analysis.eval_saved_model import constants
from tensorflow_model_analysis.eval_saved_model import encoding
from tensorflow_model_analysis.eval_saved_model import util
from typing import Dict, List, Optional, Text, Tuple, Union

from google.protobuf import any_pb2
from tensorflow.core.protobuf import meta_graph_pb2

CollectionDefValueType = Union[float, int, bytes, any_pb2.Any]  # pylint: disable=invalid-name


def extract_signature_inputs_or_outputs_with_prefix(
    prefix: Text,
    # Inputs and outputs are not actually Dicts, but behave like them
    signature_inputs_or_outputs: Dict[Text, meta_graph_pb2.TensorInfo],
    key_if_single_element: Optional[Text] = None
) -> Dict[Text, meta_graph_pb2.TensorInfo]:
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
      {<prefix>: value1, <prefix>_extrastuff: value2, <prefix>morestuff: value3}

  Args:
    prefix: Prefix to extract
    signature_inputs_or_outputs: Signature inputs or outputs to extract from
    key_if_single_element: Key to use in the dictionary if the SignatureDef map
      had only one entry with key <prefix> representing a single tensor.

  Returns:
    Dictionary extracted as described above. The values will be the TensorInfo
    associated with the keys.

  Raises:
    ValueError: There were duplicate keys.
  """
  matched_prefix = False
  result = {}
  for k, v in signature_inputs_or_outputs.items():
    if k.startswith(prefix + '/'):
      key = k[len(prefix) + 1:]
    elif k.startswith(prefix):
      if k == prefix:
        matched_prefix = True
      key = k
    else:
      continue

    if key in result:
      raise ValueError(
          'key "%s" already in dictionary. you might have repeated keys. '
          'prefix was "%s", signature_def values were: %s' %
          (prefix, key, signature_inputs_or_outputs))
    result[key] = v

  if key_if_single_element and matched_prefix and len(result) == 1:
    return {key_if_single_element: result[prefix]}

  return result


# TODO(b/119308261): Remove once all exported EvalSavedModels are updated.
def load_legacy_inputs(
    meta_graph_def: tf.compat.v1.MetaGraphDef,
    signature_def: tf.compat.v1.MetaGraphDef.SignatureDefEntry,
    graph: tf.Graph) -> Tuple[Dict[Text, types.TensorType], types.TensorType]:
  """Loads legacy inputs.

  Args:
    meta_graph_def: MetaGraphDef to lookup nodes in.
    signature_def: SignatureDef to lookup nodes in.
    graph: TensorFlow graph to lookup the nodes in.

  Returns:
    Tuple of (inputs_map, input_refs_node)
  """
  input_node = tf.compat.v1.saved_model.utils.get_tensor_from_tensor_info(
      list(signature_def.inputs.values())[0], graph)
  try:
    input_refs_node = get_node_in_graph(meta_graph_def,
                                        encoding.EXAMPLE_REF_COLLECTION, graph)
  except KeyError:
    # If we can't find the ExampleRef collection, then this is probably a model
    # created before we introduced the ExampleRef parameter to
    # EvalInputReceiver. In that case, we default to a tensor of range(0,
    # len(input_example)).
    # TODO(b/117519999): Remove this backwards-compatibility shim once all
    # exported EvalSavedModels have ExampleRef.
    input_refs_node = tf.range(tf.size(input=input_node))
  inputs_map = collections.OrderedDict(
      {list(signature_def.inputs.keys())[0]: input_node})
  return (inputs_map, input_refs_node)


# TODO(b/119308261): Remove once all exported EvalSavedModels are updated.
def load_legacy_features_and_labels(
    meta_graph_def: tf.compat.v1.MetaGraphDef, graph: tf.Graph
) -> Tuple[Dict[Text, types.TensorType], Dict[Text, types.TensorType]]:
  """Loads legacy features and labels nodes.

  Args:
    meta_graph_def: MetaGraphDef to lookup nodes in.
    graph: TensorFlow graph to lookup the nodes in.

  Returns:
    Tuple of (features_map, labels_map)
  """
  encoded_features_map = collections.OrderedDict(
      get_node_map_in_graph(meta_graph_def, encoding.FEATURES_COLLECTION,
                            [encoding.NODE_SUFFIX], graph))
  features_map = collections.OrderedDict()
  for key in encoded_features_map:
    features_map[key] = encoded_features_map[key][encoding.NODE_SUFFIX]

  encoded_labels_map = collections.OrderedDict(
      get_node_map_in_graph(meta_graph_def, encoding.LABELS_COLLECTION,
                            [encoding.NODE_SUFFIX], graph))
  labels_map = collections.OrderedDict()
  for key in encoded_labels_map:
    labels_map[key] = encoded_labels_map[key][encoding.NODE_SUFFIX]

  # Assume that KeyType is only Text
  # pytype: disable=bad-return-type
  return (features_map, labels_map)
  # pytype: enable=bad-return-type


def load_tfma_version(
    signature_def: tf.compat.v1.MetaGraphDef.SignatureDefEntry,
    graph: tf.Graph,
) -> types.TensorType:
  """Loads TFMA version information from signature_def.inputs.

  Args:
    signature_def: SignatureDef to lookup node in.
    graph: TensorFlow graph to lookup the node in.

  Returns:
    TFMA version tensor.

  Raises:
    ValueError: If version not found signature_def.inputs.
  """
  if constants.SIGNATURE_DEF_TFMA_VERSION_KEY not in signature_def.inputs:
    raise ValueError('tfma version not found in signature_def: %s' %
                     signature_def)
  return tf.compat.v1.saved_model.utils.get_tensor_from_tensor_info(
      signature_def.inputs[constants.SIGNATURE_DEF_TFMA_VERSION_KEY], graph)


def load_inputs(
    signature_def: tf.compat.v1.MetaGraphDef.SignatureDefEntry,
    graph: tf.Graph,
) -> Tuple[Dict[Text, types.TensorType], types.TensorType]:
  """Loads input nodes from signature_def.inputs.

  Args:
    signature_def: SignatureDef to lookup nodes in.
    graph: TensorFlow graph to lookup the nodes in.

  Returns:
    Tuple of (inputs_map, input_refs_node) where inputs_map is an OrderedDict.

  Raises:
    ValueError: If inputs or input_refs not found signature_def.inputs.
  """
  inputs = extract_signature_inputs_or_outputs_with_prefix(
      constants.SIGNATURE_DEF_INPUTS_PREFIX, signature_def.inputs)
  if not inputs:
    raise ValueError('no inputs found in signature_def: %s' % signature_def)
  inputs_map = collections.OrderedDict()
  # Sort by key name so stable ordering is used when passing to feed_list.
  for k in sorted(inputs.keys()):
    inputs_map[k] = tf.compat.v1.saved_model.utils.get_tensor_from_tensor_info(
        inputs[k], graph)

  if constants.SIGNATURE_DEF_INPUT_REFS_KEY not in signature_def.inputs:
    raise ValueError('no input_refs found in signature_def: %s' % signature_def)
  input_refs_node = tf.compat.v1.saved_model.utils.get_tensor_from_tensor_info(
      signature_def.inputs[constants.SIGNATURE_DEF_INPUT_REFS_KEY], graph)
  return (inputs_map, input_refs_node)


def load_iterator_initializer_name(
    signature_def: tf.compat.v1.MetaGraphDef.SignatureDefEntry,
    graph: tf.Graph,
) -> Optional[types.TensorType]:
  """Loads iterator initializer name tensor from signature_def.inputs.

  Args:
    signature_def: SignatureDef to lookup initializer in.
    graph: TensorFlow graph to lookup the initializer in.

  Returns:
    Tensor containing iterator initializer op name or None if not used.
  """
  if constants.SIGNATURE_DEF_ITERATOR_INITIALIZER_KEY in signature_def.inputs:
    return tf.compat.v1.saved_model.utils.get_tensor_from_tensor_info(
        signature_def.inputs[constants.SIGNATURE_DEF_ITERATOR_INITIALIZER_KEY],
        graph)
  return None


def load_additional_inputs(
    prefix: Text,
    signature_def: tf.compat.v1.MetaGraphDef.SignatureDefEntry,
    graph: tf.Graph,
) -> Dict[Text, types.TensorType]:
  """Loads additional input tensors from signature_def.inputs.

  Args:
    prefix: Prefix used for tensors in signature_def.inputs (e.g. features,
      labels, etc)
    signature_def: SignatureDef to lookup nodes in.
    graph: TensorFlow graph to lookup the nodes in.

  Returns:
    OrderedDict of tensors.
  """
  tensors = collections.OrderedDict()
  for k, v in extract_signature_inputs_or_outputs_with_prefix(
      prefix, signature_def.inputs, util.default_dict_key(prefix)).items():
    tensors[k] = tf.compat.v1.saved_model.utils.get_tensor_from_tensor_info(
        v, graph)
  return tensors


def load_predictions(signature_def: tf.compat.v1.MetaGraphDef.SignatureDefEntry,
                     graph: tf.Graph) -> Dict[Text, types.TensorType]:
  """Loads prediction nodes from signature_def.outputs.

  Args:
    signature_def: SignatureDef to lookup nodes in.
    graph: TensorFlow graph to lookup the nodes in.

  Returns:
    Predictions map as an OrderedDict.
  """
  # The canonical ordering we use here is simply the ordering we get
  # from the predictions collection.
  predictions = extract_signature_inputs_or_outputs_with_prefix(
      constants.PREDICTIONS_NAME, signature_def.outputs,
      util.default_dict_key(constants.PREDICTIONS_NAME))
  predictions_map = collections.OrderedDict()
  for k, v in predictions.items():
    # Extract to dictionary with a single key for consistency with
    # how features and labels are extracted.
    predictions_map[
        k] = tf.compat.v1.saved_model.utils.get_tensor_from_tensor_info(
            v, graph)
  return predictions_map


def load_metrics(signature_def: tf.compat.v1.MetaGraphDef.SignatureDefEntry,
                 graph: tf.Graph
                ) -> Dict[types.FPLKeyType, Dict[Text, types.TensorType]]:
  """Loads metric nodes from signature_def.outputs.

  Args:
    signature_def: SignatureDef to lookup nodes in.
    graph: TensorFlow graph to lookup the nodes in.

  Returns:
    Metrics map as an OrderedDict.
  """
  metrics = extract_signature_inputs_or_outputs_with_prefix(
      constants.METRICS_NAME, signature_def.outputs)
  metrics_map = collections.defaultdict(dict)
  for k, v in metrics.items():
    node = tf.compat.v1.saved_model.utils.get_tensor_from_tensor_info(v, graph)

    if k.endswith('/' + constants.METRIC_VALUE_SUFFIX):
      key = k[:-len(constants.METRIC_VALUE_SUFFIX) - 1]
      metrics_map[key][encoding.VALUE_OP_SUFFIX] = node
    elif k.endswith('/' + constants.METRIC_UPDATE_SUFFIX):
      key = k[:-len(constants.METRIC_UPDATE_SUFFIX) - 1]
      metrics_map[key][encoding.UPDATE_OP_SUFFIX] = node
    else:
      raise ValueError('unrecognised suffix for metric. key was: %s' % k)
  return metrics_map


def get_node_map(meta_graph_def: meta_graph_pb2.MetaGraphDef, prefix: Text,
                 node_suffixes: List[Text]
                ) -> Dict[types.FPLKeyType, Dict[Text, CollectionDefValueType]]:
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
    collection_def_name = encoding.with_suffix(prefix, node_suffix)
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
  keys = meta_graph_def.collection_def[encoding.with_suffix(
      prefix, encoding.KEY_SUFFIX)].bytes_list.value
  if not all([len(node_list) == len(keys) for node_list in node_lists]):
    raise ValueError('length of each node_list should match length of keys. '
                     'prefix was %s, node_lists were %s, keys was %s' %
                     (prefix, node_lists, keys))
  result = {}
  for key, elems in zip(keys, zip(*node_lists)):
    result[encoding.decode_key(key)] = dict(zip(node_suffixes, elems))
  return result


def get_node_map_in_graph(
    meta_graph_def: meta_graph_pb2.MetaGraphDef, prefix: Text,
    node_suffixes: List[Text],
    graph: tf.Graph) -> Dict[types.FPLKeyType, Dict[Text, types.TensorType]]:
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
        k: encoding.decode_tensor_node(graph, n) for k, n in elems.items()
    }
  return result


def get_node_wrapped_tensor_info(meta_graph_def: meta_graph_pb2.MetaGraphDef,
                                 path: Text) -> any_pb2.Any:
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


def get_node_in_graph(meta_graph_def: meta_graph_pb2.MetaGraphDef, path: Text,
                      graph: tf.Graph) -> types.TensorType:
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
  return encoding.decode_tensor_node(
      graph, get_node_wrapped_tensor_info(meta_graph_def, path))
