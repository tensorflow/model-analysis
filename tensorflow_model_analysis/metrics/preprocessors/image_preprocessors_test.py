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
"""Tests for image related preprocessors."""


import pytest
import io
from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util as beam_testing_util
import numpy as np
from PIL import Image
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.metrics.preprocessors import image_preprocessors
from tensorflow_model_analysis.utils import util


@pytest.mark.xfail(run=False, reason="PR 183 This class contains tests that fail and needs to be fixed. "
"If all tests pass, please remove this mark.")
class ImageDecodeTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    image_array = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.uint8)
    image = Image.fromarray(image_array)
    encoded_buffer = io.BytesIO()
    image.save(encoded_buffer, format='PNG')
    encoded_image = encoded_buffer.getvalue()

    self._extract_inputs = [
        util.StandardExtracts({
            constants.LABELS_KEY: {
                'image/encoded': encoded_image,
                'image/decoded': image_array,
                'number_of_classes': 2,
            },
            constants.PREDICTIONS_KEY: {
                'image/pred/encoded': encoded_image,
                'image/pred/decoded': image_array,
                'number_of_classes': 3,
            },
        })
    ]
    self._expected_processed_inputs = [
        util.StandardExtracts({
            constants.LABELS_KEY: image_array,
            constants.PREDICTIONS_KEY: image_array,
        })
    ]

  @parameterized.named_parameters(
      (
          'decode_encoded_image',
          image_preprocessors.DecodeImagePreprocessor(
              ground_truth_key='image/encoded',
              prediction_key='image/pred/encoded',
          ),
      ),
      (
          'no_decode_image',
          image_preprocessors.DecodeImagePreprocessor(
              ground_truth_key='image/decoded',
              prediction_key='image/pred/decoded',
              decode_ground_truth=False,
              decode_prediction=False,
          ),
      ),
  )
  def testImageDecodePreprocessor(self, preprocessor):
    with beam.Pipeline() as p:
      updated_pcoll = (
          p
          | 'Create' >> beam.Create(self._extract_inputs)
          | 'Preprocess' >> beam.ParDo(preprocessor)
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
    preprocessor = image_preprocessors.DecodeImagePreprocessor(
        ground_truth_key='image/encoded',
        prediction_key='image/pred/encoded',
    )
    self.assertEqual(
        preprocessor.name,
        (
            '_decode_image_preprocessor'
            ':ground_truth_key=image/encoded,'
            'prediction_key=image/pred/encoded'
        ),
    )

  def testLabelNotFoundImage(self):
    with self.assertRaisesRegex(ValueError, 'image/encodederror is not found'):
      _ = next(
          image_preprocessors.DecodeImagePreprocessor(
              ground_truth_key='image/encodederror',
              prediction_key='image/pred/encoded',
          ).process(extracts=self._extract_inputs[0])
      )

  def testPredictionNotFoundImage(self):
    with self.assertRaisesRegex(
        ValueError, 'image/pred/encodederror is not found'
    ):
      _ = next(
          image_preprocessors.DecodeImagePreprocessor(
              ground_truth_key='image/encoded',
              prediction_key='image/pred/encodederror',
          ).process(extracts=self._extract_inputs[0])
      )

  def testLabelPreidictionImageSizeMismatch(self):
    extracts = {
        constants.LABELS_KEY: {
            'image/encoded': np.array([[1, 2]]),
        },
        constants.PREDICTIONS_KEY: {
            'image/pred/encoded': np.array([[1, 2, 3]]),
        },
    }
    with self.assertRaisesRegex(ValueError, 'does not match'):
      _ = next(
          image_preprocessors.DecodeImagePreprocessor(
              ground_truth_key='image/encoded',
              prediction_key='image/pred/encoded',
              decode_ground_truth=False,
              decode_prediction=False,
          ).process(extracts=extracts)
      )


