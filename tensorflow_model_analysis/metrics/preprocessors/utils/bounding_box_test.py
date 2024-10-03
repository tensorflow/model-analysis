# Copyright 2019 Google LLC
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
"""Tests for bounding_box."""
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from tensorflow_model_analysis.metrics.preprocessors.utils import bounding_box

# indices for the inputs, it should be arranged in the following format:
# [LEFT, RIGHT, TOP, BOTTOM, CLASS_ID, CONFIDENCE]
LEFT, RIGHT, TOP, BOTTOM, CLASS, CONFIDENCE = range(6)


class BoundingBoxTest(parameterized.TestCase):

  def test_input_check_bounding_box_area(self):
    # Input should not be empty
    expected_exception = ValueError
    expected_regex = 'Input boxes list should be a 2d array'
    self.assertRaisesRegex(expected_exception, expected_regex,
                           bounding_box.bounding_box_area,
                           np.array([20, 60, 290]))

  def test_input_value_check_bounding_box_area(self):
    boxes = np.array([[20, 300, 60, 290]])
    expected_exception = ValueError
    expected_regex = 'The BOTTOM boundary is less than the TOP boundary '
    self.assertRaisesRegex(expected_exception, expected_regex,
                           bounding_box.bounding_box_area, boxes)

  def test_compute_box_area(self):
    boxes = np.array([[30, 100, 70, 300], [50, 100, 80, 110]])
    np.testing.assert_allclose(
        np.array([8000, 300]), bounding_box.bounding_box_area(boxes))

  def test_input_check_filter_boxes_by_class(self):
    with self.assertRaisesRegex(ValueError,
                                'Input boxes list should be a 2d array'):
      _ = bounding_box.filter_boxes_by_class(np.array([20, 60, 290]), [3])

  def test_filter_boxes_by_one_class(self):
    boxes = np.array([[30, 100, 70, 300, 1], [50, 100, 80, 90, 2],
                      [40, 100, 100, 290, 1]])
    result = bounding_box.filter_boxes_by_class(boxes, [1])
    expected_result = np.array([[30, 100, 70, 300, 1], [40, 100, 100, 290, 1]])
    np.testing.assert_equal(result, expected_result)

  def test_filter_boxes_by_multi_classes(self):
    boxes = np.array([[30, 100, 70, 300, 1], [50, 100, 80, 90, 2],
                      [40, 100, 100, 290, 1], [55, 200, 88, 390, 0]])
    result = bounding_box.filter_boxes_by_class(boxes, [0, 1])
    expected_result = np.array([[30, 100, 70, 300, 1], [40, 100, 100, 290, 1],
                                [55, 200, 88, 390, 0]])
    np.testing.assert_equal(result, expected_result)

  @parameterized.named_parameters(
      ('_filtering_to_empty', np.array([[0, 100, 100, 300]
                                       ]), [50000, 80000], np.empty([0, 4])),
      ('_case1',
       np.array([[30, 100, 70, 300, 1], [50, 100, 80, 100, 2],
                 [40, 100, 100, 290, 1], [55, 200, 88, 390, 0]]), [7000, 12000],
       np.array([[30, 100, 70, 300, 1], [40, 100, 100, 290, 1]])))
  def test_filter_boxes_by_area_range(self, boxes, area_range, expected_result):
    result = bounding_box.filter_boxes_by_area_range(boxes, area_range)
    np.testing.assert_equal(result, expected_result)

  def test_input_check_sort_boxes_by_confidence(self):
    # Input should not be empty
    expected_exception = ValueError
    expected_regex = 'Input boxes list should be a 2d array'
    self.assertRaisesRegex(expected_exception, expected_regex,
                           bounding_box.sort_boxes_by_confidence,
                           np.array([20, 60, 290]))


