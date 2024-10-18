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
"""Tests for iou."""
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tensorflow_model_analysis.metrics.preprocessors.utils import box_match


class IouTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('_2dnot1d', np.array([30, 100, 70, 300]), np.array([[20, 130, 60, 290]
                                                          ])),
      ('_4cols', np.array([30, 100, 70, 300]), np.array([[20, 60, 290]])))
  def test_input_check_compute_iou(self, boxes1, boxes2):
    self.assertRaisesRegex(ValueError, 'Input boxes lists should be a 2d array',
                           box_match.compute_ious_for_image, boxes1, boxes2)

  def test_compute_single_iou(self):
    # Boxes are in the corners format [LEFT, RIGHT, TOP, BOTTOM]
    boxes1 = np.array([[30, 100, 70, 300]])
    boxes2 = np.array([[20, 130, 60, 290]])
    result = box_match.compute_ious_for_image(boxes1, boxes2)
    expected_result = np.array([[0.5]])
    np.testing.assert_allclose(result, expected_result)

  def test_compute_multiple_iou(self):
    boxes1 = np.array([[30, 100, 70, 300], [50, 100, 80, 200]])
    boxes2 = np.array([[20, 130, 60, 290], [30, 100, 70, 300],
                       [500, 100, 800, 300]])
    result = box_match.compute_ious_for_image(boxes1, boxes2)
    expected_result = np.array([[0.5, 1., 0.], [7 / 87, 2 / 9, 0.]])
    np.testing.assert_allclose(result, expected_result)


class BoundingBoxTest(parameterized.TestCase):

  def test_input_check_match_boxes(self):
    # Input should include class_id
    ious = np.array([20, 60, 290])
    thresholds = np.array(0.5)
    with self.assertRaisesRegex(ValueError, 'ious list should be a 2d array'):
      _ = box_match._match_boxes(ious, thresholds)

  @parameterized.named_parameters(('_one_gt_multi_pred', {
      'ious': np.array([[0.1], [0.8], [0.4]]),
      'thresholds': 0.5
  }, np.array([[1]]), np.array([[-1, 0, -1]])), ('_threshold_too_high', {
      'ious': np.array([[0.1], [0.8], [0.4]]),
      'thresholds': 0.9
  }, np.array([[-1]]), np.array([[-1, -1, -1]])), ('_multi_gt_multi_pred', {
      'ious': np.array([[0.1, 0.8, 0.4], [0.3, 0.1, 0.4], [0.6, 0.9, 0.4]]),
      'thresholds': [0., 0.5, 0.85]
  }, np.array([[2, 0, 1], [2, 0, -1], [-1, 2, -1]
              ]), np.array([[1, 2, 0], [1, -1, 0], [-1, -1, 1]])))
  def test_match_boxes(self, raw_input, expected_gt_match, expected_pred_match):
    gt_match, pred_match = box_match._match_boxes(**raw_input)
    np.testing.assert_equal(expected_gt_match, gt_match)
    np.testing.assert_equal(expected_pred_match, pred_match)

  @parameterized.named_parameters(
      ('_single_case_matched', {
          'boxes_gt': np.array([[0, 50, 30, 100, 0]]),
          'boxes_pred': np.array([[10, 60, 40, 80, 0, 0.5]]),
          'iou_threshold': 0.1
      }, {
          'labels': np.array([1.]),
          'predictions': np.array([0.5]),
          'example_weights': np.array([1.])
      }),
      ('_single_case_notmatched', {
          'boxes_gt': np.array([[0, 50, 30, 100, 0]]),
          'boxes_pred': np.array([[10, 60, 40, 80, 0, 0.5]]),
          'iou_threshold': 0.5
      }, {
          'labels': np.array([1., 0.]),
          'predictions': np.array([0., 0.5]),
          'example_weights': np.array([1., 1.])
      }),
      ('_empty_ground_truth', {
          'boxes_gt': np.empty([0, 5]),
          'boxes_pred': np.array([[10, 60, 40, 80, 0, 0.5]]),
          'iou_threshold': 0.5
      }, {
          'labels': np.array([0.]),
          'predictions': np.array([0.5]),
          'example_weights': np.array([1.])
      }),
      ('_empty_prediction', {
          'boxes_gt': np.array([[0, 50, 30, 100, 0]]),
          'boxes_pred': np.empty([0, 6]),
          'iou_threshold': 0.5
      }, {
          'labels': np.array([1]),
          'predictions': np.array([0]),
          'example_weights': np.array([1.])
      }),
      ('_empty_both_truth_and_prediction', {
          'boxes_gt': np.empty([0, 5]),
          'boxes_pred': np.empty([0, 6]),
          'iou_threshold': 0.5
      }, {
          'labels': np.array([]),
          'predictions': np.array([]),
          'example_weights': np.array([])
      }),
      # the following multi-example produces:(after_sorting)
      # ious: np.array([[0., 0., 0.], [0., 0.5, 7/87], [0., 1., 2/9]])
      # matches_gt: [-1, 1, -1]
      # matches_pred: [-1, 0, -1]
      ('_multi_cases', {
          'boxes_gt':
              np.array([[30, 1000, 70, 3000, 0], [30, 100, 70, 300, 0],
                        [50, 100, 80, 200, 0]]),
          'boxes_pred':
              np.array([[20, 130, 60, 290, 0, 0.5], [30, 100, 70, 300, 0, 0.3],
                        [500, 100, 800, 300, 0, 0.9]]),
          'iou_threshold':
              0.3
      }, {
          'labels': np.array([1., 1., 1., 0., 0.]),
          'predictions': np.array([0.5, 0., 0., 0.9, 0.3]),
          'example_weights': np.array([1., 1., 1., 1., 1.]),
      }))
  def test_boxes_to_label_prediction(self, raw_input, expected_result):
    result = box_match.boxes_to_label_prediction_example_weight(
        boxes_gt=raw_input['boxes_gt'],
        boxes_pred=raw_input['boxes_pred'],
        iou_threshold=raw_input['iou_threshold'],
        class_id=0)
    self.assertLen(result, len(expected_result))
    np.testing.assert_allclose(result[0], expected_result['labels'])
    np.testing.assert_allclose(result[1], expected_result['predictions'])
    np.testing.assert_allclose(result[2], expected_result['example_weights'])

  @parameterized.named_parameters((('_filter_by_class'), {
      'boxes_gt': np.array([[0, 50, 30, 100, 0]]),
      'boxes_pred': np.array([[10, 60, 40, 80, 1, 0.5]]),
      'iou_threshold': 0.1,
      'class_id': 1,
      'area_range': [0, 10000],
      'max_num_detections': 1
  }, {
      'labels': np.array([0.]),
      'predictions': np.array([0.5]),
      'example_weights': np.array([1.0]),
  }), (('_filter_by_area_range'), {
      'boxes_gt': np.array([[0, 50, 30, 100, 0]]),
      'boxes_pred': np.array([[10, 60, 40, 80, 1, 0.5]]),
      'iou_threshold': 0.1,
      'class_id': 0,
      'area_range': [100, 200],
      'max_num_detections': 1
  }, {
      'labels': np.array([]),
      'predictions': np.array([]),
      'example_weights': np.array([]),
  }), (('_filter_by_maximum_detections'), {
      'boxes_gt':
          np.array([[0, 50, 30, 100, 0]]),
      'boxes_pred':
          np.array([[10, 60, 40, 80, 1, 0.5], [10, 60, 40, 80, 0, 0.8]]),
      'iou_threshold':
          0.1,
      'class_id':
          1,
      'area_range': [0, 10000],
      'max_num_detections':
          1
  }, {
      'labels': np.array([0.]),
      'predictions': np.array([0.5]),
      'example_weights': np.array([1.0]),
  }))
  def test_boxes_to_label_prediction_filter(self, raw_input, expected_result):
    result = box_match.boxes_to_label_prediction_example_weight(**raw_input)
    self.assertLen(result, len(expected_result))
    np.testing.assert_allclose(result[0], expected_result['labels'])
    np.testing.assert_allclose(result[1], expected_result['predictions'])
    np.testing.assert_allclose(result[2], expected_result['example_weights'])


