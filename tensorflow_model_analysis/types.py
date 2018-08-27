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
"""Types."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import copy

import numpy as np
import tensorflow as tf

from tensorflow_model_analysis.types_compat import Any, Dict, List, Text, Tuple, Union, NamedTuple

# pylint: disable=invalid-name
TensorType = Union[tf.Tensor, tf.SparseTensor]
DictOfTensorType = Dict[bytes, TensorType]
TensorTypeMaybeDict = Union[TensorType, DictOfTensorType]

# Type of keys we support for prediction, label and features dictionaries.
KeyType = Union[Union[bytes, Text], Tuple[Union[bytes, Text], Ellipsis]]

# Value of a Scalar fetched during session.run.
FetchedScalarValue = Union[np.float32, np.float64, np.int32, np.int64, bytes]

# Value of a Tensor fetched using session.run.
FetchedTensorValue = Union[tf.SparseTensorValue, np.ndarray]

# Value fechted using session.run.
FetchedValue = Union[FetchedScalarValue, FetchedTensorValue]

# Dictionary of Tensor values fetched.
# The dictionary maps original dictionary keys => ('node' => value).
DictOfFetchedTensorValues = Dict[KeyType, Dict[bytes, FetchedTensorValue]]

ListOfFetchedTensorValues = List[FetchedTensorValue]


def is_tensor(obj):
  return isinstance(obj, tf.Tensor) or isinstance(obj, tf.SparseTensor)


# Used in building the model diagnostics table, a MatrializedColumn is a value
# inside of ExampleAndExtract that will be emitted to file.
MaterializedColumn = NamedTuple(
    'MaterializedColumn',
    [('name', bytes),
     ('value', Union[List[bytes], List[int], List[float], bytes, int, float])])

# Used in building model diagnostics table, the ExampleAndExtracts holds an
# example and all its "extractions." Extractions that should be emitted to file.
# Each Extract has a name, stored as the key of the DictOfExtractedValues.
DictOfExtractedValues = Dict[Text, Any]


class ExampleAndExtracts(
    NamedTuple('ExampleAndExtracts', [('example', bytes),
                                      ('extracts', DictOfExtractedValues)])):
  """Example and extracts."""

  def create_copy_with_shallow_copy_of_extracts(self):
    """Returns a new copy of this with a shallow copy of extracts.

    This is NOT equivalent to making a shallow copy with copy.copy(this).
    That does NOT make a shallow copy of the dictionary. An illustration of
    the differences:
      a = ExampleAndExtracts(example='content', extracts=dict(apple=[1, 2]))

      # The dictionary is shared (and hence the elements are also shared)
      b = copy.copy(a)
      b.extracts['banana'] = 10
      assert a.extracts['banana'] == 10

      # The dictionary is not shared (but the elements are)
      c = a.create_copy_with_shallow_copy_of_extracts()
      c.extracts['cherry'] = 10
      assert 'cherry' not in a.extracts  # The dictionary is not shared
      c.extracts['apple'][0] = 100
      assert a.extracts['apple'][0] == 100  # But the elements are

    Returns:
      A shallow copy of this object.
    """
    return ExampleAndExtracts(
        example=self.example, extracts=copy.copy(self.extracts))
