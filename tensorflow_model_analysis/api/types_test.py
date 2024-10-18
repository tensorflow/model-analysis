# Copyright 2022 Google LLC
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
"""Tests for types."""

from absl.testing import absltest
import numpy as np
from tensorflow_model_analysis.api import types


class TypesTest(absltest.TestCase):

  def testVarLenTensorValueFromDenseRows(self):
    tensor = types.VarLenTensorValue.from_dense_rows(
        [np.array([]), np.array([1]), np.array([1, 2])]
    )
    np.testing.assert_array_equal(np.array([1, 1, 2]), tensor.values)
    np.testing.assert_array_equal(
        np.array([[1, 0], [2, 0], [2, 1]]), tensor.indices
    )
    np.testing.assert_array_equal(np.array([3, 2]), tensor.dense_shape)

  def testVarLenTensorValueToDenseRows(self):
    tensor = types.VarLenTensorValue(
        values=np.array([1, 2, 3, 4]),
        indices=np.array([[0, 0], [0, 1], [2, 0], [2, 1]]),
        dense_shape=np.array([3, 2]),
    )
    dense_rows = list(tensor.dense_rows())
    self.assertLen(dense_rows, 3)
    np.testing.assert_array_equal(np.array([1, 2]), dense_rows[0])
    np.testing.assert_array_equal(np.array([]), dense_rows[1])
    np.testing.assert_array_equal(np.array([3, 4]), dense_rows[2])

  def testVarLenTensorValueInvalidShape(self):
    with self.assertRaisesRegex(
        ValueError, r'A VarLenTensorValue .* \(\[2 2 2\]\)'
    ):
      types.VarLenTensorValue(
          values=np.array([1, 2, 3, 4, 5, 6, 7, 8]),
          indices=np.array([
              [0, 0, 0],
              [0, 0, 1],
              [0, 1, 0],
              [0, 1, 1],
              [1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1],
          ]),
          dense_shape=np.array([2, 2, 2]),
      )

  def testVarLenTensorValueInvalidRowIndices(self):
    with self.assertRaisesRegex(
        ValueError, r'The values and .* indices\[\[2\], :\]'
    ):
      types.VarLenTensorValue(
          values=np.array([1, 2, 3, 4]),
          # rows indices are reversed
          indices=np.array([[1, 0], [1, 1], [0, 0], [0, 1]]),
          dense_shape=np.array([2, 2]),
      )

  def testVarLenTensorValueInvalidColumnIndices(self):
    with self.assertRaisesRegex(
        ValueError, r'The values and .* indices\[\[1\], :\]'
    ):
      types.VarLenTensorValue(
          values=np.array([1, 2, 3, 4]),
          # columns indices in the first row are reversed
          indices=np.array([[0, 1], [0, 0], [1, 0], [1, 1]]),
          dense_shape=np.array([2, 2]),
      )

  def testVarLenTensorValueEmpty(self):
    types.VarLenTensorValue(
        values=np.array([]),
        indices=np.empty((0, 2)),
        dense_shape=np.array([2, 2]),
    )


