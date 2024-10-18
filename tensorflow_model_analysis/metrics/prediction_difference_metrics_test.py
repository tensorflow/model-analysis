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
"""Tests for prediction difference metrics."""

from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing import util
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.metrics import prediction_difference_metrics
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.writers import metrics_plots_and_validations_writer

from google.protobuf import text_format


class SymmetricPredictionDifferenceTest(absltest.TestCase):

  def testSymmetricPredictionDifference(self):
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
    computations = prediction_difference_metrics.SymmetricPredictionDifference(
    ).computations(
        eval_config=eval_config,
        model_names=['baseline', 'candidate'],
        output_names=[''],
        example_weighted=True)
    self.assertLen(computations, 1)
    computation = computations[0]

    examples = [{
        'labels': [0],
        'example_weights': [1],
        'predictions': {
            baseline_model_name: [0.1],
            candidate_model_name: [0.2]
        }
    }, {
        'labels': [0],
        'example_weights': [2],
        'predictions': {
            baseline_model_name: [0.2],
            candidate_model_name: [0.3]
        }
    }, {
        'labels': [1],
        'example_weights': [3],
        'predictions': {
            baseline_model_name: [0.9],
            candidate_model_name: [0.8]
        }
    }]

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create(examples)
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeMetric' >> beam.CombinePerKey(computation.combiner))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_proto = (
              metrics_plots_and_validations_writer
              .convert_slice_metrics_to_proto(
                  got[0], add_metrics_callbacks=None))
          self.assertLen(got_proto.metric_keys_and_values, 1)
          got_kv_proto = got_proto.metric_keys_and_values[0]
          self.assertEqual(
              got_kv_proto.value.WhichOneof('type'), 'double_value')
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          pd_key = metric_types.MetricKey(
              name=prediction_difference_metrics
              .SYMMETRIC_PREDICITON_DIFFERENCE_NAME,
              model_name=candidate_model_name,
              output_name='',
              example_weighted=True,
              is_diff=True)
          self.assertIn(pd_key, got_metrics)
          self.assertAlmostEqual(
              got_metrics[pd_key],
              (2 * 0.1 / 0.3 * 1 + 2 * 0.1 / 0.5 * 2 + 2 * 0.1 / 1.7 * 3) / 6)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testSymmetricPredictionDifferenceEpsilon(self):
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
    computations = prediction_difference_metrics.SymmetricPredictionDifference(
    ).computations(
        eval_config=eval_config,
        model_names=['baseline', 'candidate'],
        output_names=[''],
        example_weighted=True)
    self.assertLen(computations, 1)
    computation = computations[0]

    examples = [{
        'labels': [0],
        'example_weights': [1],
        'predictions': {
            baseline_model_name: [0.1],
            candidate_model_name: [0.1]
        }
    }]

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create(examples)
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeMetric' >> beam.CombinePerKey(computation.combiner))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          pd_key = metric_types.MetricKey(
              name=prediction_difference_metrics
              .SYMMETRIC_PREDICITON_DIFFERENCE_NAME,
              model_name=candidate_model_name,
              output_name='',
              example_weighted=True,
              is_diff=True)
          self.assertIn(pd_key, got_metrics)
          self.assertAlmostEqual(got_metrics[pd_key], 0.0)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')


