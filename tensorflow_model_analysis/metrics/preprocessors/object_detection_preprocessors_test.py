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

# Initialize test data

# This is a binary classification case, the iou matrix should be:
# [[0.5, 1., 0.], [7 / 87, 2 / 9, 0.]]
# The match at iou_threshold = 0.5 is
# gt_matches: [[0, -1]] dt_matches: [[0, -1, -1]]
_BOXMATCH_CASE1_BINARY = {
    constants.LABELS_KEY:
        np.asarray([[30, 100, 70, 300, 0], [50, 100, 80, 200, 0]]),
    constants.PREDICTIONS_KEY:
        np.asarray([[20, 130, 60, 290, 0, 0.5], [30, 100, 70, 300, 0, 0.3],
                    [500, 100, 800, 300, 0, 0.1]])
}
_BOXMATCH_CASE1_RESULT = [{
    constants.LABELS_KEY: np.asarray([1.]),
    constants.PREDICTIONS_KEY: np.asarray([0.5]),
    constants.EXAMPLE_WEIGHTS_KEY: np.asarray([1.])
}, {
    constants.LABELS_KEY: np.asarray([1.]),
    constants.PREDICTIONS_KEY: np.asarray([0.]),
    constants.EXAMPLE_WEIGHTS_KEY: np.asarray([1.])
}, {
    constants.LABELS_KEY: np.asarray([0.]),
    constants.PREDICTIONS_KEY: np.asarray([0.3]),
    constants.EXAMPLE_WEIGHTS_KEY: np.asarray([1.])
}, {
    constants.LABELS_KEY: np.asarray([0.]),
    constants.PREDICTIONS_KEY: np.asarray([0.1]),
    constants.EXAMPLE_WEIGHTS_KEY: np.asarray([1.])
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


if __name__ == '__main__':
  absltest.main()
