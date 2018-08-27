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
from tensorflow_model_analysis.eval_saved_model import util
from tensorflow_model_analysis.types_compat import Dict, Iterable, Union, Sequence, Tuple

from tensorflow.core.example import example_pb2


class TensorflowModelAnalysisTest(tf.test.TestCase):
  """Test class that extends tf.test.TestCase with extra functionality."""

  def setUp(self):
    self.longMessage = True  # pylint: disable=invalid-name

  def _getTempDir(self):
    return tempfile.mkdtemp()

  def _makeExample(self, **kwargs):
    return util.make_example(**kwargs)

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

  def assertSparseTensorValueEqual(
      self, expected_sparse_tensor_value,
      got_sparse_tensor_value):
    self.assertAllEqual(expected_sparse_tensor_value.indices,
                        got_sparse_tensor_value.indices)
    self.assertAllEqual(expected_sparse_tensor_value.values,
                        got_sparse_tensor_value.values)
    self.assertAllEqual(expected_sparse_tensor_value.dense_shape,
                        got_sparse_tensor_value.dense_shape)
