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

from __future__ import print_function

import tensorflow as tf
from tensorflow_model_analysis import types


def wrap_tensor_or_dict_of_tensors_in_identity(
    tensor_or_dict_of_tensors
):
  """Wrap the given Tensor / dict of Tensors in tf.identity.

  Args:
    tensor_or_dict_of_tensors: Tensor or dict of Tensors to wrap around.

  Workaround for TensorFlow issue #17568.

  Returns:
    Tensor or dict of Tensors wrapped with tf.identity.

  Raises:
    ValueError: We could not wrap the given Tensor / dict of Tensors in
      tf.identity.
  """

  def _wrap_tensor_in_identity(tensor):
    if isinstance(tensor, tf.Tensor):
      return tf.identity(tensor)
    elif isinstance(tensor, tf.SparseTensor):
      return tf.SparseTensor(
          indices=tensor.indices,
          values=tensor.values,
          dense_shape=tensor.dense_shape)
    else:
      raise ValueError('could not wrap Tensor %s in identity' %
                       str(tensor_or_dict_of_tensors))

  if isinstance(tensor_or_dict_of_tensors, dict):
    result = {}
    for k, v in tensor_or_dict_of_tensors.items():
      # Dictionary elements should only be Tensors (and not dictionaries).
      result[k] = _wrap_tensor_in_identity(v)
    return result
  else:
    return _wrap_tensor_in_identity(tensor_or_dict_of_tensors)
