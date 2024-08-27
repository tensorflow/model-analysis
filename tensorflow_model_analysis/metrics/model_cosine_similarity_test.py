# Copyright 2024 Google LLC
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
"""Tests for model cosine similiarty metrics."""


import pytest
from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.metrics import model_cosine_similarity
from tensorflow_model_analysis.proto import config_pb2

from google.protobuf import text_format

_PREDICTION_A = np.array([1.0, 0.5, 0.5, 1.0])
_PREDICTION_B = np.array([0.5, 1.0, 1.0, 0.5])
_PREDICTION_C = np.array([0.25, 0.1, 0.9, 0.75])


@pytest.mark.xfail(run=False, reason="PR 183 This class contains tests that fail and needs to be fixed. "
"If all tests pass, please remove this mark.")
class ModelCosineSimilarityMetricsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='no_change',
          prediction_pairs=[
              (_PREDICTION_A, _PREDICTION_A),
              (_PREDICTION_B, _PREDICTION_B),
              (_PREDICTION_C, _PREDICTION_C),
          ],
          # cs(p1, p2):
          #     np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))
          # cs(_PREDICTION_A/B/C, _PREDICTION_A/B/C) = 1.0
          expected_average_cosine_similarity=1.0,
      ),
      dict(
          testcase_name='small_change',
          prediction_pairs=[
              (_PREDICTION_A, _PREDICTION_A),
              (_PREDICTION_B, _PREDICTION_A),
              (_PREDICTION_A, _PREDICTION_B),
          ],
          # cs(_PREDICTION_A, _PREDICTION_A) = 1.0
          # cs(_PREDICTION_B, _PREDICTION_A) = 0.8
          # cs(_PREDICTION_A, _PREDICTION_B) = 0.8
          expected_average_cosine_similarity=0.8666666666666666,
      ),
      dict(
          testcase_name='large_change',
          prediction_pairs=[
              (_PREDICTION_C, _PREDICTION_A),
              (_PREDICTION_A, _PREDICTION_B),
              (_PREDICTION_B, _PREDICTION_C),
          ],
          # cs(_PREDICTION_C, _PREDICTION_A) = 0.7892004626469845
          # cs(_PREDICTION_A, _PREDICTION_B) = 0.8
          # cs(_PREDICTION_B, _PREDICTION_C) = 0.7892004626469845
          expected_average_cosine_similarity=0.7928003084313229,
      ),
  )
  def test_cosine_similarity(
      self, prediction_pairs, expected_average_cosine_similarity
  ):
    baseline_model_name = 'baseline'
    candidate_model_name = 'candidate'

    eval_config = text_format.Parse(
        """
        model_specs {
          name: "baseline"
          is_baseline: true
        }
        model_specs {
          name: "candidate"
        }
        """,
        config_pb2.EvalConfig(),
    )

    computations = model_cosine_similarity.ModelCosineSimilarity().computations(
        eval_config=eval_config,
        model_names=['baseline', 'candidate'],
        output_names=[''],
    )
    self.assertLen(computations, 1)
    cosine_similarity = computations[0]

    examples = []
    for baseline_prediction, candidate_prediction in prediction_pairs:
      examples.append({
          constants.LABELS_KEY: {
              baseline_model_name: None,
              candidate_model_name: None,
          },
          constants.PREDICTIONS_KEY: {
              baseline_model_name: baseline_prediction,
              candidate_model_name: candidate_prediction,
          },
      })

    with beam.Pipeline() as pipeline:
      result = (
          pipeline
          | 'Create' >> beam.Create(examples)
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeMetric' >> beam.CombinePerKey(cosine_similarity.combiner)
      )

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())

          metric_key = metric_types.MetricKey(
              name=model_cosine_similarity._COSINE_SIMILARITY_METRIC_NAME,
              model_name=candidate_model_name,
              output_name='',
              is_diff=True,
          )

          self.assertIn(metric_key, got_metrics)
          self.assertIsInstance(got_metrics[metric_key], float)
          self.assertAlmostEqual(
              got_metrics[metric_key],
              expected_average_cosine_similarity,
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')


