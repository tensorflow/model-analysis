# Lint as: python3
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
"""Tests for calibration related metrics."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.metrics import calibration
from tensorflow_model_analysis.metrics import metric_util


class CalibrationMetricsTest(testutil.TensorflowModelAnalysisTest,
                             parameterized.TestCase):

  @parameterized.named_parameters(
      ('mean_label', calibration.MeanLabel(), 2.0 / 3.0),
      ('mean_prediction', calibration.MeanPrediction(), (0.3 + 0.9) / 3.0),
      ('calibration', calibration.Calibration(), (0.3 + 0.9) / 2.0))
  def testCalibrationMetricsWithoutWeights(self, metric, expected_value):
    computations = metric.computations()
    weighted_totals = computations[0]
    metric = computations[1]

    example1 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.0]),
        'example_weights': np.array([1.0]),
    }
    example2 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.3]),
        'example_weights': np.array([1.0]),
    }
    example3 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.9]),
        'example_weights': None,  # defaults to 1.0
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example1, example2, example3])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeWeightedTotals' >> beam.CombinePerKey(
              weighted_totals.combiner)
          | 'ComputeMetric' >> beam.Map(lambda x: (x[0], metric.result(x[1]))))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          key = metric.keys[0]
          self.assertDictElementsAlmostEqual(
              got_metrics, {key: expected_value}, places=5)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  @parameterized.named_parameters(
      ('mean_label', calibration.MeanLabel(), 1.0 * 0.7 / 2.1),
      ('mean_prediction', calibration.MeanPrediction(),
       (1.0 * 0.5 + 0.7 * 0.7 + 0.5 * 0.9) / 2.1),
      ('calibration', calibration.Calibration(),
       (1.0 * 0.5 + 0.7 * 0.7 + 0.5 * 0.9) / (1.0 * 0.7)))
  def testCalibrationMetricsWithWeights(self, metric, expected_value):
    computations = metric.computations()
    weighted_totals = computations[0]
    metric = computations[1]

    example1 = {
        'labels': np.array([0.0]),
        'predictions': np.array([1.0]),
        'example_weights': np.array([0.5]),
    }
    example2 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.7]),
        'example_weights': np.array([0.7]),
    }
    example3 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.5]),
        'example_weights': np.array([0.9]),
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example1, example2, example3])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeWeightedTotals' >> beam.CombinePerKey(
              weighted_totals.combiner)
          | 'ComputeMetric' >> beam.Map(lambda x: (x[0], metric.result(x[1]))))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          key = metric.keys[0]
          self.assertDictElementsAlmostEqual(
              got_metrics, {key: expected_value}, places=5)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')


if __name__ == '__main__':
  tf.test.main()
