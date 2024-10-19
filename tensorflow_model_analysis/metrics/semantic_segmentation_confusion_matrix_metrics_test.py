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
"""Tests for confusion matrix for semantic segmentation."""


import pytest
import io

from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
from PIL import Image
import tensorflow_model_analysis as tfma
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.contrib.aggregates import binary_confusion_matrices
from tensorflow_model_analysis.metrics import metric_types

from google.protobuf import text_format

Matrix = binary_confusion_matrices.Matrix


def _encode_image_from_nparray(image_array: np.ndarray) -> bytes:
  image = Image.fromarray(image_array)
  encoded_buffer = io.BytesIO()
  image.save(encoded_buffer, format='PNG')
  return encoded_buffer.getvalue()


@pytest.mark.xfail(run=False, reason="PR 183 This class contains tests that fail and needs to be fixed. "
"If all tests pass, please remove this mark.")
class SegmentationConfusionMatrixTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    label_image_array = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.uint8)
    prediction_image_array = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.uint8)
    label_encoded_image = _encode_image_from_nparray(label_image_array)
    prediction_encoded_image = _encode_image_from_nparray(
        prediction_image_array
    )

    label_image_array2 = np.array([[2, 2, 3], [1, 2, 3]], dtype=np.uint8)
    prediction_image_array2 = np.array([[2, 1, 1], [2, 2, 2]], dtype=np.uint8)
    label_encoded_image2 = _encode_image_from_nparray(label_image_array2)
    prediction_encoded_image2 = _encode_image_from_nparray(
        prediction_image_array2
    )

    self._extracts = [
        {
            'features': {
                constants.LABELS_KEY: {
                    'image/encoded': np.array([label_encoded_image]),
                },
                constants.PREDICTIONS_KEY: {
                    'image/pred/encoded': np.array([prediction_encoded_image]),
                },
            }
        },
        {
            'features': {
                constants.LABELS_KEY: {
                    'image/encoded': np.array([label_encoded_image2]),
                },
                constants.PREDICTIONS_KEY: {
                    'image/pred/encoded': np.array([prediction_encoded_image2]),
                },
            }
        },
    ]

  @parameterized.named_parameters(
      dict(
          testcase_name='_two_class',
          eval_config=text_format.Parse(
              """
               model_specs {
                  signature_name: "serving_default"
                  prediction_key: "predictions" # placeholder
                  label_key: "labels" # placeholder
               }
               slicing_specs {
               }
               metrics_specs {
                  metrics {
                     class_name: "SemanticSegmentationConfusionMatrix"
                     config:'"class_ids":[1, 2],'
                           '"ground_truth_key":"image/encoded", '
                           '"prediction_key":"image/pred/encoded", '
                           '"decode_prediction":true, '
                           '"name":"SegConfusionMatrix"'
                  }
               }
              """,
              tfma.EvalConfig(),
          ),
          name='SegConfusionMatrix',
          expected_result={
              1: Matrix(tp=1, tn=5, fp=4, fn=2),
              2: Matrix(tp=3, tn=3, fp=4, fn=2),
          },
      ),
      dict(
          testcase_name='_two_class_with_ignore',
          eval_config=text_format.Parse(
              """
               model_specs {
                  signature_name: "serving_default"
                  prediction_key: "predictions" # placeholder
                  label_key: "labels" # placeholder
               }
               slicing_specs {
               }
               metrics_specs {
                  metrics {
                     class_name: "SemanticSegmentationConfusionMatrix"
                     config:'"class_ids":[1, 2],'
                           '"ignore_ground_truth_id":3,'
                           '"ground_truth_key":"image/encoded", '
                           '"prediction_key":"image/pred/encoded", '
                           '"decode_prediction":true, '
                           '"name":"SegConfusionMatrix"'
                  }
               }
              """,
              tfma.EvalConfig(),
          ),
          name='SegConfusionMatrix',
          expected_result={
              1: Matrix(tp=1, tn=3, fp=2, fn=2),
              2: Matrix(tp=3, tn=1, fp=2, fn=2),
          },
      ),
      dict(
          testcase_name='_tp_two_class_with_ignore',
          eval_config=text_format.Parse(
              """
               model_specs {
                  signature_name: "serving_default"
                  prediction_key: "predictions" # placeholder
                  label_key: "labels" # placeholder
               }
               slicing_specs {
               }
               metrics_specs {
                  metrics {
                     class_name: "SemanticSegmentationTruePositive"
                     config:'"class_ids":[1, 2],'
                           '"ignore_ground_truth_id":3,'
                           '"ground_truth_key":"image/encoded", '
                           '"prediction_key":"image/pred/encoded", '
                           '"decode_prediction":true, '
                           '"name":"SegTruePositive"'
                  }
               }
              """,
              tfma.EvalConfig(),
          ),
          name='SegTruePositive',
          expected_result={
              1: np.array([1]),
              2: np.array([3]),
          },
      ),
      dict(
          testcase_name='_fp_two_class_with_ignore',
          eval_config=text_format.Parse(
              """
               model_specs {
                  signature_name: "serving_default"
                  prediction_key: "predictions" # placeholder
                  label_key: "labels" # placeholder
               }
               slicing_specs {
               }
               metrics_specs {
                  metrics {
                     class_name: "SemanticSegmentationFalsePositive"
                     config:'"class_ids":[1, 2],'
                           '"ignore_ground_truth_id":3,'
                           '"ground_truth_key":"image/encoded", '
                           '"prediction_key":"image/pred/encoded", '
                           '"decode_prediction":true, '
                           '"name":"SegFalsePositive"'
                  }
               }
              """,
              tfma.EvalConfig(),
          ),
          name='SegFalsePositive',
          expected_result={
              1: np.array([2]),
              2: np.array([2]),
          },
      ),
  )
  def testEncodedImage(self, eval_config, name, expected_result):
    extracts = self._extracts

    evaluators = tfma.default_evaluators(eval_config=eval_config)
    extractors = tfma.default_extractors(
        eval_shared_model=None, eval_config=eval_config
    )

    with beam.Pipeline() as p:
      result = (
          p
          | 'LoadData' >> beam.Create(extracts)
          | 'ExtractEval'
          >> tfma.ExtractAndEvaluate(
              extractors=extractors, evaluators=evaluators
          )
      )

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          self.assertLen(got_metrics, 2)

          for class_id, expected_matrix in expected_result.items():
            key = metric_types.MetricKey(
                name=name, sub_key=metric_types.SubKey(class_id=class_id)
            )
            self.assertIn(key, got_metrics)
            got_metric = got_metrics[key]
            np.testing.assert_allclose(
                expected_matrix,
                got_metric,
                rtol=1e-3,
                err_msg=f'This {name} metric fails.',
            )
        except AssertionError as err:
          raise util.BeamAssertException(err)

      self.assertIn('metrics', result)
      util.assert_that(result['metrics'], check_result, label='result')


