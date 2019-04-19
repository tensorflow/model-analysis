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
"""Utility functions for EvalSavedModel."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import numpy as np
import six
import tensorflow as tf
from tensorflow_model_analysis import types
from tensorflow_model_analysis import util
from typing import List, Optional, Text, Tuple

from tensorflow.core.example import example_pb2


def default_dict_key(prefix: Text) -> Text:
  """Returns the default key to use with a dict associated with given prefix."""
  return util.KEY_SEPARATOR + prefix


def extract_tensor_maybe_dict(prefix: Text,
                              dict_of_tensors: types.DictOfTensorType
                             ) -> types.TensorTypeMaybeDict:
  """Returns tensor if single entry under default key else returns dict."""
  default_key = default_dict_key(prefix)
  if list(dict_of_tensors.keys()) == [default_key]:
    return dict_of_tensors[default_key]
  return dict_of_tensors


def wrap_tensor_or_dict_of_tensors_in_identity(
    tensor_or_dict_of_tensors: types.TensorTypeMaybeDict
) -> types.TensorTypeMaybeDict:
  # pyformat: disable
  """Wrap the given Tensor / dict of Tensors in tf.identity.

  Args:
    tensor_or_dict_of_tensors: Tensor or dict of Tensors to wrap around.

  Workaround for TensorFlow issue #17568 (b/71769512).

  Returns:
    Tensor or dict of Tensors wrapped with tf.identity.

  Raises:
    ValueError: We could not wrap the given Tensor / dict of Tensors in
      tf.identity.
  """
  # pyformat: enable

  def _wrap_tensor_in_identity(tensor: types.TensorType) -> types.TensorType:
    if isinstance(tensor, tf.Tensor):
      return tf.identity(tensor)
    elif isinstance(tensor, tf.SparseTensor):
      return tf.SparseTensor(
          indices=tf.identity(tensor.indices),
          values=tf.identity(tensor.values),
          dense_shape=tf.identity(tensor.dense_shape))
    else:
      raise ValueError('could not wrap Tensor %s in identity' % str(tensor))

  if isinstance(tensor_or_dict_of_tensors, dict):
    result = {}
    for k, v in tensor_or_dict_of_tensors.items():
      # Dictionary elements should only be Tensors (and not dictionaries).
      result[k] = _wrap_tensor_in_identity(v)
    return result
  else:
    return _wrap_tensor_in_identity(tensor_or_dict_of_tensors)


def make_example(**kwargs) -> example_pb2.Example:
  """Make a TensorFlow Example with the given fields.

  The arguments can be singleton values, or a list of values, e.g.
  makeExample(age=3.0, fruits=['apples', 'pears', 'oranges']).
  Empty lists are not allowed, since we won't be able to deduce the type.

  Args:
   **kwargs: Each key=value pair defines a field in the example to be
     constructed. The name of the field will be key, and the value will be
     value. The type will be deduced from the type of the value. Care must be
     taken for numeric types: 0 will be interpreted as an int, and 0.0 as a
       float.

  Returns:
    TensorFlow.Example with the corresponding fields set to the corresponding
    values.

  Raises:
    ValueError: One of the arguments was an empty list.
    TypeError: One of the elements (or one of the elements in a list) had an
      unsupported type.
  """

  result = example_pb2.Example()
  for key, value in kwargs.items():
    if isinstance(value, float):
      result.features.feature[key].float_list.value[:] = [value]
    elif isinstance(value, int):
      result.features.feature[key].int64_list.value[:] = [value]
    elif isinstance(value, six.binary_type):
      result.features.feature[key].bytes_list.value[:] = [value]
    elif isinstance(value, six.text_type):
      result.features.feature[key].bytes_list.value[:] = [value.encode('utf8')]
    elif isinstance(value, list):
      if len(value) == 0:  # pylint: disable=g-explicit-length-test
        raise ValueError('empty lists not allowed, but field %s was an empty '
                         'list' % key)
      if isinstance(value[0], float):
        result.features.feature[key].float_list.value[:] = value
      elif isinstance(value[0], int):
        result.features.feature[key].int64_list.value[:] = value
      elif isinstance(value[0], six.binary_type):
        result.features.feature[key].bytes_list.value[:] = value
      elif isinstance(value[0], six.text_type):
        result.features.feature[key].bytes_list.value[:] = [
            v.encode('utf8') for v in value
        ]
      else:
        raise TypeError(
            'field %s was a list, but the first element had '
            'unknown type %s' % key, type(value[0]))
    else:
      raise TypeError('unrecognised type for field %s: type %s' %
                      (key, type(value)))
  return result


def _copy_shape_zero_rows(shape: Tuple[int, ...]) -> Tuple[int, ...]:
  """Return a copy of given shape with the number of rows zeroed out."""
  temp = list(shape)
  temp[0] = 0
  return tuple(temp)


def _dense_concat_rows(arrays: List[np.ndarray]) -> np.ndarray:
  """Concat a list of np.arrays along rows.

  This is similar to (but not the same as) np.concatenate(arrays, axis=0),
  However, this assumes that each array represents a single row (so
  shape[0] == 1), but the other dimensions are allowed to
  vary across the list. The final shape will be
    len(arrays) x max(shape[:, 1]) x max(shape[:, 2]) x ...

  The dense tensors will be padded with the appropriate default value
  (e.g. 0 for ints, 0.0 for floats, '' for strings). We assume that this is
  the desired behaviour - some models may require different padding
  (b/111007595).

  We need this to support cases where predictions are of variable length.

  For example, if the values are:
    (1x2x1) [[[10], [11]]]
    (1x1x3) [[[20, 21, 22]]]
    (1x3x2) [[[30, 31], [32, 33], [34, 35]]]
    (1x1x2) [[[40, 41]]]

  Then the result will be:
    (4x3x3)

    [[[10,  0,  0],
      [11,  0,  0],
      [ 0,  0,  0]],

     [[20, 21, 22],
      [ 0,  0,  0],
      [ 0,  0,  0]],

     [[30, 31,  0],
      [32, 33,  0],
      [34, 35,  0],

     [[40, 41,  0],
      [ 0,  0,  0],
      [ 0,  0,  0]]])

  Args:
    arrays: List of np.arrays to concatenate together.

  Returns:
    A single np.array, representing the concatenated np.arrays

  Raises:
    ValueError: arrays was an empty list; or an np.array in the list was not
      for exactly one row.
  """
  if not arrays:
    raise ValueError('arrays must be a non-empty list.')

  shape_max = np.amax(np.array([a.shape for a in arrays]), axis=0)
  if arrays[0].dtype == np.object:
    # Assume if the dtype is object then the array contains strings.
    padding_value = ''
  else:
    padding_value = arrays[0].dtype.type()

  padded_arrays = []
  for array in arrays:
    if array.shape[0] != 1:
      raise ValueError(
          'each array should only have one row, but %s had shape %s' %
          (array, array.shape))
    # We use concatenation instead of padding because np.concatenate is much
    # faster than np.pad: see
    # https://stackoverflow.com/questions/12668027/
    # good-ways-to-expand-a-numpy-ndarray
    for i, actual in enumerate(array.shape[1:]):
      axis = i + 1
      num_to_pad = shape_max[axis] - actual
      if num_to_pad > 0:
        fill_shape = np.copy(array.shape)
        fill_shape[axis] = num_to_pad
        array = np.concatenate(
            [array, np.full(fill_shape, padding_value)], axis=axis)

    padded_arrays.append(array)

  return np.concatenate(padded_arrays, axis=0)


def _sparse_concat_rows(
    sparse_tensor_values: List[tf.compat.v1.SparseTensorValue]
) -> tf.compat.v1.SparseTensorValue:
  """Concat a list of SparseTensorValues along rows.

  This is similar to (but not the same as)
  tf.sparse_concat(axis=0, sp_inputs=sparse_tensor_values)
  except that this operates on NumPy arrays.

  More critically, this assumes that each sparse tensor value represents a
  single row (so dense_shape[0] == 1), but the other dimensions are allowed to
  vary across the list. The final shape will be
    batch_size x max(dense_shapes[:, 1]) x max(dense_shapes[:, 2]) x ...

  We need this to support cases where sparse features are of variable length,
  e.g. in sequence example cases, where the number of timesteps might be
  different across examples.

  For example, if the values are:
     indices=[[0, 0, 0], [0, 1, 1]], values=[10, 12], dense_shape=[0, 2, 2]
    indices=[[0, 2, 2]], values=[22], dense_shape=[0, 3, 3]
    indices=[[0, 0, 4]], values=[33], dense_shape=[0, 1, 5]

  Then the result will be:
    indices=[[0, 0, 0], [0, 1, 1],
             [1, 2, 2],
             [2, 0, 4]]
    values=[10, 12, 22, 33]
    dense_shape=[3, 3, 5]

  Args:
    sparse_tensor_values: List of SparseTensorValues to concatenate together.

  Returns:
    A single SparseTensorValue, representing the concatenated
    SparseTensorValues.

  Raises:
    ValueError: sparse_tensor_values was an empty list; or a sparse tensor value
     in the list was not for exactly one row.
  """
  if not sparse_tensor_values:
    raise ValueError('sparse_tensor_values must be a non-empty list.')

  # Create empty indices and values arrays with the same shape as the
  # sparse_tensor_value, except the number of rows (batch dimension) is 0.
  #
  # We need this because we need to preserve the shape of these arrays,
  # even if their batch dimension is 0.
  empty_indices_with_shape = np.zeros(
      _copy_shape_zero_rows(sparse_tensor_values[0].indices.shape),
      dtype=sparse_tensor_values[0].indices.dtype)
  empty_values_with_shape = np.zeros(
      _copy_shape_zero_rows(sparse_tensor_values[0].values.shape),
      dtype=sparse_tensor_values[0].values.dtype)

  indices = []

  # Make a copy here, so that in the case that we don't take any amaxes
  # in the loop below, we'll still be mutating a copy (rather than the original)
  # when we update the row size.
  dense_shape_max = np.array(sparse_tensor_values[0].dense_shape)
  values = []
  for row, sparse_tensor in enumerate(sparse_tensor_values):
    # Make a copy, so we can mutate it.
    cur_indices = np.array(sparse_tensor.indices)
    if cur_indices.size == 0:
      # Empty SparseTensorValue.
      continue
    cur_indices[:, 0, ...] += row
    indices.extend(cur_indices.tolist())
    values.extend(sparse_tensor.values)
    if sparse_tensor.dense_shape[0] != 1:
      raise ValueError(
          'each sparse_tensor_value should only have one row, but %s had '
          'shape %s' % (sparse_tensor, sparse_tensor.dense_shape))
    dense_shape_max = np.amax([dense_shape_max, sparse_tensor.dense_shape],
                              axis=0)

  # The final dense shape is the max of dense shapes of all the sparse tensor
  # values, except the number of rows should be the batch size.
  dense_shape_max[0] = len(sparse_tensor_values)

  # pylint: disable=g-long-ternary
  return tf.compat.v1.SparseTensorValue(
      indices=(np.array(indices, dtype=empty_indices_with_shape.dtype)
               if indices else empty_indices_with_shape),
      values=(np.array(values, dtype=empty_values_with_shape.dtype)
              if values else empty_values_with_shape),
      dense_shape=dense_shape_max)
  # pylint: enable=g-long-ternary


def _sparse_slice_rows(sparse_tensor_value: tf.compat.v1.SparseTensorValue
                      ) -> List[tf.compat.v1.SparseTensorValue]:
  """Returns a list of single rows of a SparseTensorValue.

  This is equivalent to:
  [tf.sparse_slice(sparse_tensor_value, [row, 0, 0, ...], [1, INF, INF, ...])
   for row in range(0, sparse_tensor_value.dense_shape[0])]
  except that this operates on NumPy arrays.

  Args:
    sparse_tensor_value: SparseTensorValue to slice.

  Returns:
    List of SparseTensorValue representing the sliced rows.
  """

  # Create empty indices and values arrays with the same shape as the
  # sparse_tensor_value, except the number of rows (batch dimension) is 0.
  #
  # We need this because we need to preserve the shape of these arrays,
  # even if their batch dimension is 0.
  empty_indices_with_shape = np.zeros(
      _copy_shape_zero_rows(sparse_tensor_value.indices.shape),
      dtype=sparse_tensor_value.indices.dtype)
  empty_values_with_shape = np.zeros(
      _copy_shape_zero_rows(sparse_tensor_value.values.shape),
      dtype=sparse_tensor_value.values.dtype)

  if sparse_tensor_value.indices.size > 0:
    indices = sparse_tensor_value.indices
    # Sort indices matrix by rows, treating each row as a coordinate
    argsort_indices = np.lexsort(np.transpose(indices)[::-1])
    sorted_indices = indices[argsort_indices]
    sorted_values = sparse_tensor_value.values[argsort_indices]
  else:
    sorted_indices = []
    sorted_values = []
  num_sorted_indices = len(sorted_indices)

  result = []
  original_dense_shape = list(sparse_tensor_value.dense_shape)

  # Offset into the indices/values arrays of the original SparseTensorValue
  offset = 0
  dense_shape = [1] + original_dense_shape[1:]

  for row in range(0, original_dense_shape[0]):
    # Process each output row one at a time.
    indices = []
    values = []

    # Collect all the elements for this output row.
    while offset < num_sorted_indices and sorted_indices[offset][0] == row:
      # Okay to not make a copy and mutate sorted_indices here.
      cur_index = sorted_indices[offset]
      cur_index[0] = 0  # Zero out the row number
      indices.append(cur_index)
      values.append(sorted_values[offset])
      offset += 1

    # We treat each split SparseTensorValue as having dense_shape equal to the
    # maximum index in each dimension (+1 for zero-index).
    if indices:
      dense_shape[1:] = [
          max([index[i]
               for index in indices]) + 1
          for i in range(1, len(indices[0]))
      ]
    # For empty examples, we should have 0 in all other dimensions for the
    # dense_shape.
    else:
      dense_shape[1:] = [0] * (len(original_dense_shape) - 1)

    # pylint: disable=g-long-ternary
    result.append(
        tf.compat.v1.SparseTensorValue(
            indices=(np.array(indices, dtype=empty_indices_with_shape.dtype)
                     if indices else empty_indices_with_shape),
            values=(np.array(values, dtype=empty_values_with_shape.dtype)
                    if values else empty_values_with_shape),
            dense_shape=np.array(dense_shape)))
    # pylint: enable=g-long-ternary

  return result


def split_tensor_value(tensor_value: types.TensorValue
                      ) -> List[types.TensorValue]:
  """Split a single batch of Tensor values into a list of Tensor values.

  Args:
    tensor_value: A single Tensor value that represents a batch of Tensor
      values. The zeroth dimension should be batch size.

  Returns:
    A list of Tensor values, one per element of the zeroth dimension.

  Raises:
    TypeError: tensor_value had unknown type.
  """
  if isinstance(tensor_value, tf.compat.v1.SparseTensorValue):
    return _sparse_slice_rows(tensor_value)
  elif isinstance(tensor_value, np.ndarray):
    if tensor_value.shape[0] != 0:
      return np.split(
          tensor_value, indices_or_sections=tensor_value.shape[0], axis=0)
    else:
      # The result value's shape must match the shape of `tensor_value`.
      return np.zeros_like(tensor_value)
  else:
    raise TypeError('tensor_value had unknown type: %s, value was: %s' %
                    (type(tensor_value), tensor_value))


def merge_tensor_values(tensor_values: List[types.TensorValue]
                       ) -> Optional[types.TensorValue]:
  """Merge a list of Tensor values into a single batch of Tensor values.

  Args:
    tensor_values: A list of Tensor values, all fetched from the same node in
      the same graph. Each Tensor value should be for a single example.

  Returns:
    A single Tensor value that represents a batch of all the Tensor values
    in the given list.

  Raises:
    ValueError: Got a SparseTensor with more than 1 row (i.e. that is likely
      to be for more than one example).
    TypeError: tensor_value had unknown type.
  """
  if not tensor_values:
    return None

  if isinstance(tensor_values[0], tf.compat.v1.SparseTensorValue):
    # Check batch sizes.
    for tensor_value in tensor_values:
      if tensor_value.dense_shape[0] > 1:
        raise ValueError('expecting SparseTensor to be for only 1 example. '
                         'but got dense_shape %s instead' %
                         tensor_value.dense_shape)
    return _sparse_concat_rows(tensor_values)
  elif isinstance(tensor_values[0], np.ndarray):
    return _dense_concat_rows(tensor_values)
  else:
    raise TypeError('tensor_values[0] had unknown type: %s, value was: %s' %
                    (type(tensor_values[0]), tensor_values[0]))


def add_build_data_collection():
  return
