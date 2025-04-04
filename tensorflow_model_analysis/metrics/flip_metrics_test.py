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
"""Tests for flip_metrics."""


import pytest
import copy

from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.metrics import flip_metrics
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.writers import metrics_plots_and_validations_writer

from google.protobuf import text_format


@pytest.mark.xfail(run=False, reason="PR 183 This class contains tests that fail and needs to be fixed. "
"If all tests pass, please remove this mark.")
class FlipRateMetricsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='symmetric_flip_rate',
          metric=flip_metrics.SymmetricFlipRate(threshold=0.5),
          metric_name=flip_metrics.SYMMETRIC_FLIP_RATE_NAME,
          expected_result=3 / 10,
      ),
      dict(
          testcase_name='neg_to_neg_flip_rate',
          metric=flip_metrics.NegToNegFlipRate(threshold=0.5),
          metric_name=flip_metrics.NEG_TO_NEG_FLIP_RATE_NAME,
          expected_result=3 / 10,
      ),
      dict(
          testcase_name='neg_to_pos_flip_rate',
          metric=flip_metrics.NegToPosFlipRate(threshold=0.5),
          metric_name=flip_metrics.NEG_TO_POS_FLIP_RATE_NAME,
          expected_result=1 / 10,
      ),
      dict(
          testcase_name='pos_to_neg_flip_rate',
          metric=flip_metrics.PosToNegFlipRate(threshold=0.5),
          metric_name=flip_metrics.POS_TO_NEG_FLIP_RATE_NAME,
          expected_result=2 / 10,
      ),
      dict(
          testcase_name='pos_to_pos_flip_rate',
          metric=flip_metrics.PosToPosFlipRate(threshold=0.5),
          metric_name=flip_metrics.POS_TO_POS_FLIP_RATE_NAME,
          expected_result=4 / 10,
      ),
  )
  def testIndividualFlipRates(self, metric, metric_name, expected_result):
    eval_config = text_format.Parse(
        """
        model_specs {
          name: "baseline"
          is_baseline: true
        }
        model_specs {
          name: "candidate"
        }
        """, config_pb2.EvalConfig())
    baseline_model_name = 'baseline'
    candidate_model_name = 'candidate'

    computations = metric.computations(
        eval_config=eval_config,
        model_names=['baseline', 'candidate'],
        output_names=[''],
        example_weighted=True,
    )
    self.assertLen(computations, 2)

    flip_counts = computations[0]
    flip_rate = computations[1]

    examples = [
        {
            constants.LABELS_KEY: [0],
            constants.PREDICTIONS_KEY: {
                baseline_model_name: [0.1],
                candidate_model_name: [0.9],
            },
            constants.EXAMPLE_WEIGHTS_KEY: [1],
        },
        {
            constants.LABELS_KEY: [0],
            constants.PREDICTIONS_KEY: {
                baseline_model_name: [0.9],
                candidate_model_name: [0.1],
            },
            constants.EXAMPLE_WEIGHTS_KEY: [2],
        },
        {
            constants.LABELS_KEY: [1],
            constants.PREDICTIONS_KEY: {
                baseline_model_name: [0.1],
                candidate_model_name: [0.2],
            },
            constants.EXAMPLE_WEIGHTS_KEY: [3],
        },
        {
            constants.LABELS_KEY: [1],
            constants.PREDICTIONS_KEY: {
                baseline_model_name: [0.9],
                candidate_model_name: [0.8],
            },
            constants.EXAMPLE_WEIGHTS_KEY: [4],
        },
    ]

    with beam.Pipeline() as pipeline:
      result = (
          pipeline
          | 'Create' >> beam.Create(examples)
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeFlipCounts' >> beam.CombinePerKey(flip_counts.combiner)
          | 'ComputeFlipRates'
          >> beam.Map(lambda x: (x[0], flip_rate.result(x[1])))
      )

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_proto = metrics_plots_and_validations_writer.convert_slice_metrics_to_proto(
              got[0], add_metrics_callbacks=None
          )
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          self.assertLen(got_proto.metric_keys_and_values, 1)

          model_name = candidate_model_name
          output_name = ''
          example_weighted = True
          metric_key = metric_types.MetricKey(
              name=metric_name,
              model_name=model_name,
              output_name=output_name,
              example_weighted=example_weighted,
              is_diff=True,
          )

          self.assertIn(metric_key, got_metrics)
          # Verify that metric is not a 0-D np.ndarray.
          self.assertIsInstance(got_metrics[metric_key], float)
          self.assertAlmostEqual(got_metrics[metric_key], expected_result)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testBooleanFlipRates(self):
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
    baseline_model_name = 'baseline'
    candidate_model_name = 'candidate'

    computations = flip_metrics.BooleanFlipRates(threshold=0.5).computations(
        eval_config=eval_config,
        model_names=['baseline', 'candidate'],
        output_names=[''],
        example_weighted=True,
    )
    self.assertLen(computations, 10)

    flip_counts = computations[0]

    symmetric_flip_rate = computations[1]
    neg_to_neg_flip_rate = computations[3]
    neg_to_pos_flip_rate = computations[5]
    pos_to_neg_flip_rate = computations[7]
    pos_to_pos_flip_rate = computations[9]

    all_flip_rates = (
        symmetric_flip_rate,
        neg_to_neg_flip_rate,
        neg_to_pos_flip_rate,
        pos_to_neg_flip_rate,
        pos_to_pos_flip_rate,
    )

    examples = [
        {
            constants.LABELS_KEY: [0],
            constants.PREDICTIONS_KEY: {
                baseline_model_name: [0.1],
                candidate_model_name: [0.9],
            },
            constants.EXAMPLE_WEIGHTS_KEY: [1],
        },
        {
            constants.LABELS_KEY: [0],
            constants.PREDICTIONS_KEY: {
                baseline_model_name: [0.9],
                candidate_model_name: [0.1],
            },
            constants.EXAMPLE_WEIGHTS_KEY: [2],
        },
        {
            constants.LABELS_KEY: [1],
            constants.PREDICTIONS_KEY: {
                baseline_model_name: [0.1],
                candidate_model_name: [0.2],
            },
            constants.EXAMPLE_WEIGHTS_KEY: [3],
        },
        {
            constants.LABELS_KEY: [1],
            constants.PREDICTIONS_KEY: {
                baseline_model_name: [0.9],
                candidate_model_name: [0.8],
            },
            constants.EXAMPLE_WEIGHTS_KEY: [4],
        },
    ]

    def _add_derived_metrics(sliced_metrics, derived_computations):
      slice_key, metrics = sliced_metrics
      result = copy.copy(metrics)
      for c in derived_computations:
        result.update(c.result(result))
      return slice_key, result

    with beam.Pipeline() as pipeline:
      result = (
          pipeline
          | 'Create' >> beam.Create(examples)
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeFlipCounts' >> beam.CombinePerKey(flip_counts.combiner)
          | 'AddDerivedMetrics'
          >> beam.Map(_add_derived_metrics, derived_computations=all_flip_rates)
      )

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_proto = (
              metrics_plots_and_validations_writer
              .convert_slice_metrics_to_proto(
                  got[0], add_metrics_callbacks=None))
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          self.assertLen(got_proto.metric_keys_and_values, 6)

          model_name = candidate_model_name
          output_name = ''
          example_weighted = True
          is_diff = True
          sym_fr_key = metric_types.MetricKey(
              name=flip_metrics.FLIP_RATE_NAME,
              model_name=model_name,
              output_name=output_name,
              example_weighted=example_weighted,
              is_diff=is_diff,
          )
          self.assertIn(sym_fr_key, got_metrics)
          # Verify that metric is not a 0-D np.ndarray.
          self.assertIsInstance(got_metrics[sym_fr_key], float)
          self.assertAlmostEqual(got_metrics[sym_fr_key], 3 / 10)

          n2n_fr_key = metric_types.MetricKey(
              name=flip_metrics.NEG_TO_NEG_FLIP_RATE_NAME,
              model_name=model_name,
              output_name=output_name,
              example_weighted=example_weighted,
              is_diff=is_diff,
          )
          self.assertIn(n2n_fr_key, got_metrics)
          # Verify that metric is not a 0-D np.ndarray.
          self.assertIsInstance(got_metrics[n2n_fr_key], float)
          self.assertAlmostEqual(got_metrics[n2n_fr_key], 3 / 10)

          n2p_fr_key = metric_types.MetricKey(
              name=flip_metrics.NEG_TO_POS_FLIP_RATE_NAME,
              model_name=model_name,
              output_name=output_name,
              example_weighted=example_weighted,
              is_diff=is_diff,
          )
          self.assertIn(n2p_fr_key, got_metrics)
          # Verify that metric is not a 0-D np.ndarray.
          self.assertIsInstance(got_metrics[n2p_fr_key], float)
          self.assertAlmostEqual(got_metrics[n2p_fr_key], 1 / 10)

          p2n_fr_key = metric_types.MetricKey(
              name=flip_metrics.POS_TO_NEG_FLIP_RATE_NAME,
              model_name=model_name,
              output_name=output_name,
              example_weighted=example_weighted,
              is_diff=is_diff,
          )
          self.assertIn(p2n_fr_key, got_metrics)
          # Verify that metric is not a 0-D np.ndarray.
          self.assertIsInstance(got_metrics[p2n_fr_key], float)
          self.assertAlmostEqual(got_metrics[p2n_fr_key], 2 / 10)

          p2p_fr_key = metric_types.MetricKey(
              name=flip_metrics.POS_TO_POS_FLIP_RATE_NAME,
              model_name=model_name,
              output_name=output_name,
              example_weighted=example_weighted,
              is_diff=is_diff,
          )
          self.assertIn(p2p_fr_key, got_metrics)
          # Verify that metric is not a 0-D np.ndarray.
          self.assertIsInstance(got_metrics[p2p_fr_key], float)
          self.assertAlmostEqual(got_metrics[p2p_fr_key], 4 / 10)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')


