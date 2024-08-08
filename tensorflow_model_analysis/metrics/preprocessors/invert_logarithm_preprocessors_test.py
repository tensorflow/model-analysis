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
"""Tests for invert logarithm preprocessors."""

from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util as beam_testing_util
import numpy as np
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.metrics.preprocessors import invert_logarithm_preprocessors
from tensorflow_model_analysis.utils import util


class InvertBinaryLogarithmPreprocessorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    values = np.array([[1, 2, 4], [1, 2, 4]], dtype=np.int32)
    processed_values = np.array([[1, 3, 15], [1, 3, 15]], dtype=np.float32)

    self._extract_inputs = [{
        constants.LABELS_KEY: values,
        constants.PREDICTIONS_KEY: values,
    }]
    self._expected_processed_inputs = [
        util.StandardExtracts({
            constants.LABELS_KEY: processed_values,
            constants.PREDICTIONS_KEY: processed_values,
        })
    ]

  def testInvertBinaryLogarithmPreprocessor(self):
    with beam.Pipeline() as pipeline:
      updated_pcoll = (
          pipeline
          | 'Create' >> beam.Create(self._extract_inputs)
          | 'Preprocess'
          >> beam.ParDo(
              invert_logarithm_preprocessors.InvertBinaryLogarithmPreprocessor()
          )
      )

      def check_result(result):
        # Only single extract case is tested
        self.assertLen(result, len(self._expected_processed_inputs))
        for updated_extracts, expected_input in zip(
            result, self._expected_processed_inputs
        ):
          self.assertIn(constants.PREDICTIONS_KEY, updated_extracts)
          np.testing.assert_allclose(
              updated_extracts[constants.PREDICTIONS_KEY],
              expected_input[constants.PREDICTIONS_KEY],
          )
          self.assertIn(constants.LABELS_KEY, updated_extracts)
          np.testing.assert_allclose(
              updated_extracts[constants.LABELS_KEY],
              expected_input[constants.LABELS_KEY],
          )
          if constants.EXAMPLE_WEIGHTS_KEY in expected_input:
            self.assertIn(constants.EXAMPLE_WEIGHTS_KEY, updated_extracts)
            np.testing.assert_allclose(
                updated_extracts[constants.EXAMPLE_WEIGHTS_KEY],
                expected_input[constants.EXAMPLE_WEIGHTS_KEY],
            )

      beam_testing_util.assert_that(updated_pcoll, check_result)

  def testName(self):
    preprocessor = (
        invert_logarithm_preprocessors.InvertBinaryLogarithmPreprocessor()
    )
    self.assertEqual(
        preprocessor.name, '_invert_binary_logarithm_preprocessor:'
    )

  def testLabelPreidictionSizeMismatch(self):
    extracts = {
        constants.LABELS_KEY: np.array([[1, 2]]),
        constants.PREDICTIONS_KEY: np.array([[1, 2, 3]]),
    }
    with self.assertRaisesRegex(ValueError, 'does not match'):
      _ = next(
          invert_logarithm_preprocessors.InvertBinaryLogarithmPreprocessor().process(
              extracts=extracts
          )
      )


if __name__ == '__main__':
  absltest.main()
