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
"""Tests for object_detection_preprocessor."""

from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util as beam_testing_util
import numpy as np
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.metrics.preprocessors import object_detection_preprocessors
from tensorflow_model_analysis.utils import util

# Initialize test data

# This is a binary classification case, the iou matrix should be:
# [[0.5, 1., 0.], [7 / 87, 2 / 9, 0.]]
# The match at iou_threshold = 0.5 is
# gt_matches: [[0, -1]] dt_matches: [[0, -1, -1]]
_BOXMATCH_CASE1_BINARY = util.StandardExtracts({
    constants.LABELS_KEY:
        np.array([[30, 100, 70, 300, 0], [50, 100, 80, 200, 0]]),
    constants.PREDICTIONS_KEY:
        np.array([[20, 130, 60, 290, 0, 0.5], [30, 100, 70, 300, 0, 0.3],
                  [500, 100, 800, 300, 0, 0.1]])
})

_BOXMATCH_CASE1_SPLITFORMAT_BINARY = util.StandardExtracts({
    constants.FEATURES_KEY: {
        'xmin': np.array([30, 50]),
        'ymin': np.array([100, 100]),
        'xmax': np.array([70, 80]),
        'ymax': np.array([300, 200]),
        'class_id': np.array([0, 0])
    },
    constants.PREDICTIONS_KEY: {
        'xmin': np.array([20, 30, 500]),
        'ymin': np.array([130, 100, 100]),
        'xmax': np.array([60, 70, 800]),
        'ymax': np.array([290, 300, 300]),
        'class_id': np.array([0, 0, 0]),
        'score': np.array([0.5, 0.3, 0.1])
    },
})

_BOXMATCH_CASE1_BBOXFORMAT_BINARY = util.StandardExtracts({
    constants.LABELS_KEY: {
        'bbox': np.array([[30, 100, 70, 300], [50, 100, 80, 200]]),
        'class_id': np.array([0, 0])
    },
    constants.FEATURES_KEY: {
        'bbox':
            np.array([[20, 130, 60, 290], [30, 100, 70, 300],
                      [500, 100, 800, 300]]),
        'class_id':
            np.array([0, 0, 0]),
        'score':
            np.array([0.5, 0.3, 0.1])
    },
})

_BOXMATCH_CASE1_MULTI_MODEL_SPLITFORMAT_BINARY = util.StandardExtracts({
    constants.LABELS_KEY: {},
    # Searching labels in tranformed features
    constants.TRANSFORMED_FEATURES_KEY: {
        'baseline': {
            'xmin': np.array([30, 50]),
            'ymin': np.array([100, 100]),
            'xmax': np.array([70, 80]),
            'ymax': np.array([300, 200]),
            'class_id': np.array([0, 0])
        },
        'model1': {
            'xmin': np.array([30, 50]),
            'ymin': np.array([100, 100]),
            'xmax': np.array([70, 80]),
            'ymax': np.array([300, 200]),
            'class_id': np.array([0, 0])
        }
    },
    constants.PREDICTIONS_KEY: {
        'baseline': {
            'xmin': np.array([20, 30, 500]),
            'ymin': np.array([130, 100, 100]),
            'xmax': np.array([60, 70, 800]),
            'ymax': np.array([290, 300, 300]),
            'class_id': np.array([0, 0, 0]),
            'score': np.array([0.5, 0.3, 0.1])
        }
    },
})

_BOXMATCH_CASE1_MULTI_MODEL_BBOXFORMAT_BINARY = util.StandardExtracts({
    constants.LABELS_KEY: {
        'bbox': np.array([[30, 100, 70, 300], [50, 100, 80, 200]]),
        'class_id': np.array([0, 0])
    },
    constants.PREDICTIONS_KEY: {
        'baseline': {
            'bbox':
                np.array([[20, 130, 60, 290], [30, 100, 70, 300],
                          [500, 100, 800, 300]]),
            'class_id':
                np.array([0, 0, 0]),
            'score':
                np.array([0.5, 0.3, 0.1])
        },
        'model1': {
            'bbox':
                np.array([[120, 230, 60, 290], [30, 100, 70, 300],
                          [500, 100, 800, 300]]),
            'class_id':
                np.array([0, 0, 0]),
            'score':
                np.array([0.5, 0.3, 0.1])
        },
    },
})

_BOXMATCH_CASE1_RESULT = [{
    constants.LABELS_KEY: np.array([1.]),
    constants.PREDICTIONS_KEY: np.array([0.5]),
    constants.EXAMPLE_WEIGHTS_KEY: np.array([1.])
}, {
    constants.LABELS_KEY: np.array([1.]),
    constants.PREDICTIONS_KEY: np.array([0.]),
    constants.EXAMPLE_WEIGHTS_KEY: np.array([1.])
}, {
    constants.LABELS_KEY: np.array([0.]),
    constants.PREDICTIONS_KEY: np.array([0.3]),
    constants.EXAMPLE_WEIGHTS_KEY: np.array([1.])
}, {
    constants.LABELS_KEY: np.array([0.]),
    constants.PREDICTIONS_KEY: np.array([0.1]),
    constants.EXAMPLE_WEIGHTS_KEY: np.array([1.])
}]

_BOXMATCH_CASE2_PREDICT_NOT_FOUND = util.StandardExtracts({
    constants.LABELS_KEY: {
        'xmin': np.array([30, 50]),
        'ymin': np.array([100, 100]),
        'xmax': np.array([70, 80]),
        'ymax': np.array([300, 200]),
        'class_id': np.array([0, 0])
    },
    # Searching labels in tranformed features
    constants.TRANSFORMED_FEATURES_KEY: {},
    constants.PREDICTIONS_KEY: {
        'xmin': np.array([20, 30, 500]),
        'ymin': np.array([130, 100, 100]),
        'ymax': np.array([290, 300, 300]),
        'class_id': np.array([0, 0, 0]),
        'score': np.array([0.5, 0.3, 0.1])
    },
})

_BOXMATCH_CASE2_PREDICT_NOT_FOUND_RESULT = [{
    constants.LABELS_KEY: np.array([1.]),
    constants.PREDICTIONS_KEY: np.array([0]),
    constants.EXAMPLE_WEIGHTS_KEY: np.array([1.])
}, {
    constants.LABELS_KEY: np.array([1.]),
    constants.PREDICTIONS_KEY: np.array([0]),
    constants.EXAMPLE_WEIGHTS_KEY: np.array([1.])
}]


class ObjectDetectionPreprocessorTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('binary_classification', _BOXMATCH_CASE1_BINARY, 0, 0.5,
       _BOXMATCH_CASE1_RESULT))
  def testBoundingBoxMatchPreprocessor(self, extracts, class_id, iou_threshold,
                                       expected_inputs):
    with beam.Pipeline() as p:
      updated_pcoll = (
          p | 'Create' >> beam.Create([extracts])
          | 'Preprocess' >> beam.ParDo(
              object_detection_preprocessors.BoundingBoxMatchPreprocessor(
                  class_id=class_id, iou_threshold=iou_threshold)))

      def check_result(result):
        # Only single extract case is tested
        self.assertLen(result, 4)
        for updated_extracts, expected_input in zip(result, expected_inputs):
          self.assertIn(constants.PREDICTIONS_KEY, updated_extracts)
          np.testing.assert_allclose(
              updated_extracts[constants.PREDICTIONS_KEY],
              expected_input[constants.PREDICTIONS_KEY])
          self.assertIn(constants.LABELS_KEY, updated_extracts)
          np.testing.assert_allclose(updated_extracts[constants.LABELS_KEY],
                                     expected_input[constants.LABELS_KEY])
          self.assertIn(constants.EXAMPLE_WEIGHTS_KEY, updated_extracts)
          np.testing.assert_allclose(
              updated_extracts[constants.EXAMPLE_WEIGHTS_KEY],
              expected_input[constants.EXAMPLE_WEIGHTS_KEY])

      beam_testing_util.assert_that(updated_pcoll, check_result)

  @parameterized.named_parameters(
      ('split_format', _BOXMATCH_CASE1_SPLITFORMAT_BINARY, 0, 0.5, [
          'xmin', 'ymin', 'xmax', 'ymax', 'class_id'
      ], ['xmin', 'ymin', 'xmax', 'ymax', 'class_id', 'score'
         ], _BOXMATCH_CASE1_RESULT),
      ('bbox_format', _BOXMATCH_CASE1_BBOXFORMAT_BINARY, 0, 0.5, [
          'bbox', 'class_id'
      ], ['bbox', 'class_id', 'score'], _BOXMATCH_CASE1_RESULT))
  def testBoundingBoxMatchPreprocessorWithFormatChange(self, extracts, class_id,
                                                       iou_threshold,
                                                       labels_stack,
                                                       predictions_stack,
                                                       expected_inputs):
    with beam.Pipeline() as p:
      updated_pcoll = (
          p | 'Create' >> beam.Create([extracts])
          | 'Preprocess' >> beam.ParDo(
              object_detection_preprocessors.BoundingBoxMatchPreprocessor(
                  class_id=class_id,
                  iou_threshold=iou_threshold,
                  labels_to_stack=labels_stack,
                  predictions_to_stack=predictions_stack)))

      def check_result(result):
        # Only single extract case is tested
        self.assertLen(result, 4)
        for updated_extracts, expected_input in zip(result, expected_inputs):
          self.assertIn(constants.PREDICTIONS_KEY, updated_extracts)
          np.testing.assert_allclose(
              updated_extracts[constants.PREDICTIONS_KEY],
              expected_input[constants.PREDICTIONS_KEY])
          self.assertIn(constants.LABELS_KEY, updated_extracts)
          np.testing.assert_allclose(updated_extracts[constants.LABELS_KEY],
                                     expected_input[constants.LABELS_KEY])
          self.assertIn(constants.EXAMPLE_WEIGHTS_KEY, updated_extracts)
          np.testing.assert_allclose(
              updated_extracts[constants.EXAMPLE_WEIGHTS_KEY],
              expected_input[constants.EXAMPLE_WEIGHTS_KEY])

      beam_testing_util.assert_that(updated_pcoll, check_result)

  @parameterized.named_parameters(
      ('multi_output', _BOXMATCH_CASE1_MULTI_MODEL_SPLITFORMAT_BINARY, 0, 0.5, [
          'xmin', 'ymin', 'xmax', 'ymax', 'class_id'
      ], ['xmin', 'ymin', 'xmax', 'ymax', 'class_id', 'score'
         ], _BOXMATCH_CASE1_RESULT, 'baseline'),
      ('multi_model', _BOXMATCH_CASE1_MULTI_MODEL_BBOXFORMAT_BINARY, 0, 0.5, [
          'bbox', 'class_id'
      ], ['bbox', 'class_id', 'score'], _BOXMATCH_CASE1_RESULT, 'baseline'))
  def testBoundingBoxMatchPreprocessorWithMulitModel(
      self, extracts, class_id, iou_threshold, labels_stack, predictions_stack,
      expected_inputs, model_name):
    with beam.Pipeline() as p:
      updated_pcoll = (
          p | 'Create' >> beam.Create([extracts])
          | 'Preprocess' >> beam.ParDo(
              object_detection_preprocessors.BoundingBoxMatchPreprocessor(
                  class_id=class_id,
                  iou_threshold=iou_threshold,
                  labels_to_stack=labels_stack,
                  predictions_to_stack=predictions_stack,
                  model_name=model_name)))

      def check_result(result):
        # Only single extract case is tested
        self.assertLen(result, 4)
        for updated_extracts, expected_input in zip(result, expected_inputs):
          self.assertIn(constants.PREDICTIONS_KEY, updated_extracts)
          np.testing.assert_allclose(
              updated_extracts[constants.PREDICTIONS_KEY],
              expected_input[constants.PREDICTIONS_KEY])
          self.assertIn(constants.LABELS_KEY, updated_extracts)
          np.testing.assert_allclose(updated_extracts[constants.LABELS_KEY],
                                     expected_input[constants.LABELS_KEY])
          self.assertIn(constants.EXAMPLE_WEIGHTS_KEY, updated_extracts)
          np.testing.assert_allclose(
              updated_extracts[constants.EXAMPLE_WEIGHTS_KEY],
              expected_input[constants.EXAMPLE_WEIGHTS_KEY])

      beam_testing_util.assert_that(updated_pcoll, check_result)

  @parameterized.named_parameters(
      ('not_found', _BOXMATCH_CASE2_PREDICT_NOT_FOUND, 0, 0.5,
       ['xmin', 'ymin', 'xmax', 'ymax',
        'class_id'], ['xmin', 'ymin', 'xmax', 'ymax', 'class_id',
                      'score'], _BOXMATCH_CASE2_PREDICT_NOT_FOUND_RESULT))
  def testBoundingBoxMatchPreprocessorWithKeyNotFound(self, extracts, class_id,
                                                      iou_threshold,
                                                      labels_stack,
                                                      predictions_stack,
                                                      expected_inputs):
    with beam.Pipeline() as p:
      updated_pcoll = (
          p | 'Create' >> beam.Create([extracts])
          | 'Preprocess' >> beam.ParDo(
              object_detection_preprocessors.BoundingBoxMatchPreprocessor(
                  class_id=class_id,
                  iou_threshold=iou_threshold,
                  labels_to_stack=labels_stack,
                  predictions_to_stack=predictions_stack,
                  allow_missing_key=True)))

      def check_result(result):
        # Only single extract case is tested
        self.assertLen(result, 2)
        for updated_extracts, expected_input in zip(result, expected_inputs):
          self.assertIn(constants.PREDICTIONS_KEY, updated_extracts)
          np.testing.assert_allclose(
              updated_extracts[constants.PREDICTIONS_KEY],
              expected_input[constants.PREDICTIONS_KEY])
          self.assertIn(constants.LABELS_KEY, updated_extracts)
          np.testing.assert_allclose(updated_extracts[constants.LABELS_KEY],
                                     expected_input[constants.LABELS_KEY])
          self.assertIn(constants.EXAMPLE_WEIGHTS_KEY, updated_extracts)
          np.testing.assert_allclose(
              updated_extracts[constants.EXAMPLE_WEIGHTS_KEY],
              expected_input[constants.EXAMPLE_WEIGHTS_KEY])

      beam_testing_util.assert_that(updated_pcoll, check_result)


