# Lint as: python3
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
# Standard __future__ imports
from __future__ import print_function

from typing import Iterable, List, Optional, Text, Union

import numpy as np
import pyarrow as pa
import tensorflow as tf
from tensorflow_model_analysis import types


class SliceAccessor(object):
  """Wrapper around features dict for accessing keys and values for slicing."""

  def __init__(self,
               features_dicts: Iterable[Union[types.DictOfTensorValue,
                                              types.DictOfFetchedTensorValues]],
               default_features_dict: Optional[
                   Union[types.DictOfTensorValue,
                         types.DictOfFetchedTensorValues]] = None):
    self._features_dicts = features_dicts
    self._default_features_dict = default_features_dict

  def has_key(self, key: Text):
    for d in self._features_dicts:
      if key in d and d[key] is not None:
        return True
    if (self._default_features_dict and key in self._default_features_dict and
        self._default_features_dict[key] is not None):
      return True
    return False

  def get(self, key: Text) -> List[Union[int, bytes, float]]:
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

    def normalize_value(value):
      if value is None:
        return None
      if isinstance(value, dict) and 'node' in value:
        # Backwards compatibility for features that were stored as FPL types
        # instead of native dicts.
        value = value['node']
      if isinstance(value, (tf.compat.v1.SparseTensorValue,
                            tf.compat.v1.ragged.RaggedTensorValue)):
        value = value.values
      if not isinstance(value, (np.ndarray, pa.Array, list)):
        raise ValueError(
            'feature had unsupported type: key: %s, value: %s, type: %s' %
            (key, value, type(value)))
      # Only np.array and multi-dimentional pa.array support flatten.
      if hasattr(value, 'flatten'):
        value = value.flatten()
      return value

    values = None
    for d in self._features_dicts:
      value = normalize_value(d.get(key))
      if value is None:
        continue
      if values is None:
        values = value
      else:
        values = np.concatenate((values, value))
    if values is None and self._default_features_dict:
      values = normalize_value(self._default_features_dict.get(key))
    if values is None:
      raise KeyError('key %s not found' % key)
    return np.unique(values).tolist()
