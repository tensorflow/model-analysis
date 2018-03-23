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
"""Utilities for writing tests."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import math
import tempfile
import tensorflow as tf
from tensorflow_model_analysis.types_compat import Dict, Iterable, Union, Sequence, Tuple

from tensorflow.core.example import example_pb2


class TensorflowModelAnalysisTest(tf.test.TestCase):
  """Test class that extends tf.test.TestCase with extra functionality."""

  def setUp(self):
    self.longMessage = True  # pylint: disable=invalid-name

  def _getTempDir(self):
    return tempfile.mkdtemp()

  def _makeExample(self, **kwargs):
    """Make a TensorFlow Example with the given fields.

    The arguments can be singleton values, or a list of values, e.g.
    _makeExample(age=3.0, fruits=['apples', 'pears', 'oranges']).
    Empty lists are not allowed, since we won't be able to deduce the type.

    Args:
     **kwargs: Each key=value pair defines a field in the example to be
       constructed. The name of the field will be key, and the value will be
       value. The type will be deduced from the type of the value.

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
      if isinstance(value, float) or isinstance(value, int):
        result.features.feature[key].float_list.value[:] = [value]
      elif isinstance(value, str):
        result.features.feature[key].bytes_list.value[:] = [value]
      elif isinstance(value, list):
        if len(value) == 0:  # pylint: disable=g-explicit-length-test
          raise ValueError('empty lists not allowed, but field %s was an empty '
                           'list' % key)
        if isinstance(value[0], float) or isinstance(value[0], int):
          result.features.feature[key].float_list.value[:] = value
        elif isinstance(value[0], str):
          result.features.feature[key].bytes_list.value[:] = value
        else:
          raise TypeError('field %s was a list, but the first element had '
                          'unknown type %s' % key, type(value[0]))
      else:
        raise TypeError('unrecognised type for field %s: type %s' %
                        (key, type(value)))
    return result

  def assertHasKeyWithValueAlmostEqual(self,
                                       d,
                                       key,
                                       value,
                                       places = 5):
    self.assertIn(key, d)
    self.assertAlmostEqual(d[key], value, places=places, msg='key %s' % key)

  def assertDictElementsAlmostEqual(self,
                                    got_values_dict,
                                    expected_values_dict,
                                    places = 5):
    for key, expected_value in expected_values_dict.items():
      self.assertHasKeyWithValueAlmostEqual(got_values_dict, key,
                                            expected_value, places)

  def assertDictMatrixRowsAlmostEqual(
      self,
      got_values_dict,
      expected_values_dict,
      places = 5):
    """Fails if got_values_dict does not match values in expected_values_dict.

    For each entry, expected_values_dict provides the row index and the values
    of that row to be compared to the bucketing result in got_values_dict. For
    example:
      got_values_dict={'key', [[1,2,3],[4,5,6],[7,8,9]]}
    you can check the first and last row of got_values_dict[key] by setting
      expected_values_dict={'key', [(0,[1,2,3]), (2,[7,8,9])]}

    Args:
      got_values_dict: The dict got, where each value represents a full
        bucketing result.
      expected_values_dict: The expected dict. It may contain a subset of keys
        in got_values_dict. The value is of type "Iterable[Tuple[int,
        Iterable[scalar]]]", where each Tuple contains the index of a row to be
        checked and the expected values of that row.
      places: The number of decimal places to compare.
    """
    for key, expected_value in expected_values_dict.items():
      self.assertIn(key, got_values_dict)
      for (row, values) in expected_value:
        self.assertSequenceAlmostEqual(
            got_values_dict[key][row],
            values,
            places=places,
            msg_prefix='for key %s, row %d: ' % (key, row))

  def assertSequenceAlmostEqual(self,
                                got_seq,
                                expected_seq,
                                places = 5,
                                msg_prefix=''):
    got = list(got_seq)
    expected = list(expected_seq)
    self.assertEqual(
        len(got), len(expected), msg=msg_prefix + 'lengths do not match')
    for index, (a, b) in enumerate(zip(got, expected)):
      msg = msg_prefix + 'at index %d. sequences were: %s and %s' % (index, got,
                                                                     expected),
      if math.isnan(a) or math.isnan(b):
        self.assertEqual(math.isnan(a), math.isnan(b), msg=msg)
      else:
        self.assertAlmostEqual(a, b, msg=msg, places=places)
