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
"""Slice accessor class.

For use within this directory only.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function


import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import types

from tensorflow_model_analysis.types_compat import List, Union


class SliceAccessor(object):
  """Wrapper around features dict for accessing keys and values for slicing."""

  def __init__(self, features_dict):
    self._features_dict = features_dict

  def has_key(self, key):
    return key in self._features_dict

  def get(self, key):
    """Get the values of the feature with the given key.

    Args:
      key: the key of the feature to get the values of

    Returns:
      The values of the feature.

    Raises:
      KeyError: If the feature was not present in the input example.
      ValueError: A dense feature was not a 1D array.
      ValueError: The feature had an unknown type.
    """
    feature = self._features_dict.get(key)
    if feature is None:
      raise KeyError('key %s not found' % key)

    value = feature['node']
    if isinstance(value, tf.SparseTensorValue):
      return value.values.tolist()
    if not isinstance(value, np.ndarray):
      raise ValueError(
          'feature had unsupported type: key: %s, value: %s, type: %s' %
          (key, value, type(value)))
    squeezed_value = np.squeeze(value)
    if squeezed_value.ndim > 1:
      raise ValueError(
          'all feature values must be convertible to 1D arrays, but %s was '
          'not. value was %s' % (key, value))
    if squeezed_value.ndim == 1:
      # For the multivalent columns, the squeezed values are in 1D arrays.
      # Convert the array to a list.
      return squeezed_value.tolist()
    else:  # squeezed_value.ndim == 0
      # For the univalent columns, we get a scalar after squeezing, which we
      # wrap in an list.
      return [np.asscalar(squeezed_value)]
