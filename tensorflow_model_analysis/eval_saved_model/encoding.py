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
"""Library for encoding and decoding keys, Tensors, etc in EvalSavedModel."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function


import tensorflow as tf
from tensorflow_model_analysis import types

from google.protobuf import any_pb2
from tensorflow.core.protobuf import meta_graph_pb2

# Names for the various collections
TFMA_VERSION_COLLECTION = 'evaluation_only/metadata/tfma_version'
METRICS_COLLECTION = 'evaluation_only/metrics'
PREDICTIONS_COLLECTION = 'evaluation_only/predictions'
INPUT_EXAMPLE_COLLECTION = 'evaluation_only/label_graph/input_example'
LABELS_COLLECTION = 'evaluation_only/label_graph/labels'
FEATURES_COLLECTION = 'evaluation_only/label_graph/features'

# Suffixes for the collection names
KEY_SUFFIX = 'key'
NODE_SUFFIX = 'node'
VALUE_OP_SUFFIX = 'value_op'
UPDATE_OP_SUFFIX = 'update_op'

# If predictions/labels was not a dictionary, we internally wrap them
# in a dictionary with the respective default keys.
#
# Note that the key names start with two underscores to avoid collisions
# in the rare case that there are actually keys named 'predictions' or 'labels'
# in the respective dictionaries.
DEFAULT_PREDICTIONS_DICT_KEY = '__predictions'
DEFAULT_LABELS_DICT_KEY = '__labels'

# Encoding prefixes for keys
_TUPLE_KEY_PREFIX = '$Tuple$'
_BYTES_KEY_PREFIX = '$Bytes$'




def encode_key(key):
  """Encode a dictionary key as a string.

  For encoding dictionary keys in the prediction, label and feature
  dictionaries. We assume that they are either Tuples of bytes, or bytes.

  Implementation details:
    Strings are encoded as $Bytes$<String>
    Tuples of strings are encoded as:
      $Tuple$<len(tuple)>$len(tuple[0])$...$len(tuple[n])$tuple[0]$...$tuple[n]
      e.g. ('apple', 'banana', 'cherry') gets encoded as
      $Tuple$3$5$6$6$apple$banana$cherry

  Args:
    key: Dictionary key to encode.

  Returns:
    Encoded dictionary key.

  Raises:
    TypeError: Dictionary key is not either a Tuple of bytes/unicode,
      or bytes/unicode.
  """

  if isinstance(key, tuple):
    if not all(
        isinstance(elem, str) or isinstance(elem, unicode) for elem in key):
      raise TypeError('if key is tuple, all elements should be strings. '
                      'key was: %s' % key)
    utf8_keys = [elem.encode('utf8') for elem in key]
    lengths = map(len, utf8_keys)
    return '%s%d$%s$%s' % (_TUPLE_KEY_PREFIX, len(lengths),
                           '$'.join(map(str, lengths)), '$'.join(utf8_keys))
  elif isinstance(key, str) or isinstance(key, unicode):
    return '$Bytes$' + key.encode('utf8')
  else:
    raise TypeError('key has unrecognised type: type: %s, value %s' %
                    (type(key), key))


def decode_key(encoded_key):
  """Decode an encoded dictionary key encoded with encode_key.

  Args:
    encoded_key: Dictionary key, encoded with encode_key.

  Returns:
    Decoded dictionary key.

  Raises:
    ValueError: We couldn't decode the encoded key.
  """
  if encoded_key.startswith(_TUPLE_KEY_PREFIX):
    parts = encoded_key[len(_TUPLE_KEY_PREFIX):].split('$', 1)
    if len(parts) != 2:
      raise ValueError('invalid encoding: %s' % encoded_key)
    elem_count = int(parts[0])
    parts = parts[1].split('$', elem_count)
    if len(parts) != elem_count + 1:
      raise ValueError('invalid encoding: %s' % encoded_key)
    lengths = map(int, parts[:elem_count])
    parts = parts[elem_count]
    elems = []
    for length in lengths:
      elems.append(parts[:length].decode('utf8'))
      parts = parts[length + 1:]  # Add one for the $ delimiter
    return tuple(elems)
  elif encoded_key.startswith(_BYTES_KEY_PREFIX):
    return encoded_key[len(_BYTES_KEY_PREFIX):].decode('utf8')
  else:
    raise ValueError('invalid encoding: %s' % encoded_key)


def encode_tensor_node(node):
  """Encode a "reference" to a Tensor/SparseTensor as a TensorInfo in an Any.

  We put the Tensor / SparseTensor in a TensorInfo, which we then wrap in an
  Any so that it can be added to the CollectionDef.

  Args:
    node: Tensor node.

  Returns:
    Any proto wrapping a TensorInfo.
  """
  any_buf = any_pb2.Any()
  tensor_info = tf.saved_model.utils.build_tensor_info(node)
  any_buf.Pack(tensor_info)
  return any_buf


def decode_tensor_node(graph,
                       encoded_tensor_node):
  """Decode an encoded Tensor node encoded with encode_tensor_node.

  Decodes the encoded Tensor "reference", and returns the node in the given
  graph corresponding to that Tensor.

  Args:
    graph: Graph the Tensor
    encoded_tensor_node: Encoded Tensor.

  Returns:
    Decoded Tensor.
  """
  tensor_info = meta_graph_pb2.TensorInfo()
  encoded_tensor_node.Unpack(tensor_info)
  return tf.saved_model.utils.get_tensor_from_tensor_info(tensor_info, graph)
