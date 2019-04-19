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
# Standard __future__ imports
from __future__ import print_function

# Standard Imports
import six
import tensorflow as tf
from tensorflow_model_analysis import types
from typing import Text

from google.protobuf import any_pb2
from tensorflow.core.protobuf import meta_graph_pb2

# Names for the various collections
TFMA_VERSION_COLLECTION = 'evaluation_only/metadata/tfma_version'
METRICS_COLLECTION = 'evaluation_only/metrics'
PREDICTIONS_COLLECTION = 'evaluation_only/predictions'
INPUT_EXAMPLE_COLLECTION = 'evaluation_only/label_graph/input_example'
LABELS_COLLECTION = 'evaluation_only/label_graph/labels'
FEATURES_COLLECTION = 'evaluation_only/label_graph/features'
EXAMPLE_REF_COLLECTION = 'evaluation_only/label_graph/example_ref'

# Suffixes for the collection names
KEY_SUFFIX = 'key'
NODE_SUFFIX = 'node'
VALUE_OP_SUFFIX = 'value_op'
UPDATE_OP_SUFFIX = 'update_op'

# Encoding prefixes for keys
_TUPLE_KEY_PREFIX = b'$Tuple$'
_BYTES_KEY_PREFIX = b'$Bytes$'


def with_suffix(name: Text, suffix: Text) -> Text:
  return '%s/%s' % (name, suffix)  # pytype: disable=bad-return-type


def encode_key(key: types.FPLKeyType) -> bytes:
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
        isinstance(elem, six.binary_type) or isinstance(elem, six.text_type)
        for elem in key):
      raise TypeError('if key is tuple, all elements should be strings. '
                      'key was: %s' % key)
    utf8_keys = [tf.compat.as_bytes(elem) for elem in key]
    length_strs = [tf.compat.as_bytes('%d' % len(key)) for key in utf8_keys]
    return (_TUPLE_KEY_PREFIX + tf.compat.as_bytes('%d' % len(length_strs)) +
            b'$' + b'$'.join(length_strs) + b'$' + b'$'.join(utf8_keys))
  elif isinstance(key, six.binary_type) or isinstance(key, six.text_type):
    return b'$Bytes$' + tf.compat.as_bytes(key)
  else:
    raise TypeError('key has unrecognised type: type: %s, value %s' %
                    (type(key), key))


def decode_key(encoded_key: bytes) -> types.FPLKeyType:
  """Decode an encoded dictionary key encoded with encode_key.

  Args:
    encoded_key: Dictionary key, encoded with encode_key.

  Returns:
    Decoded dictionary key.

  Raises:
    ValueError: We couldn't decode the encoded key.
  """
  if encoded_key.startswith(_TUPLE_KEY_PREFIX):
    parts = encoded_key[len(_TUPLE_KEY_PREFIX):].split(b'$', 1)
    if len(parts) != 2:
      raise ValueError('invalid encoding: %s' % encoded_key)
    elem_count = int(parts[0])
    parts = parts[1].split(b'$', elem_count)
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


def encode_tensor_node(node: types.TensorType) -> any_pb2.Any:
  """Encode a "reference" to a Tensor/SparseTensor as a TensorInfo in an Any.

  We put the Tensor / SparseTensor in a TensorInfo, which we then wrap in an
  Any so that it can be added to the CollectionDef.

  Args:
    node: Tensor node.

  Returns:
    Any proto wrapping a TensorInfo.
  """
  any_buf = any_pb2.Any()
  tensor_info = tf.compat.v1.saved_model.utils.build_tensor_info(node)
  any_buf.Pack(tensor_info)
  return any_buf


def decode_tensor_node(graph: tf.Graph,
                       encoded_tensor_node: any_pb2.Any) -> types.TensorType:
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
  return tf.compat.v1.saved_model.utils.get_tensor_from_tensor_info(
      tensor_info, graph)
