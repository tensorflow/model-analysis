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
"""Tests for Tjur discrimination metrics."""

import math
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.metrics import tjur_discrimination
from tensorflow_model_analysis.utils import test_util


class TjurDisriminationTest(
    test_util.TensorflowModelAnalysisTest, parameterized.TestCase
):

  @parameterized.named_parameters(
      ('coefficient_of_discrimination',
       tjur_discrimination.CoefficientOfDiscrimination(),
       (1.2 / 2.0) - (0.8 / 1.0)),
      ('relative_coefficient_of_discrimination',
       tjur_discrimination.RelativeCoefficientOfDiscrimination(),
       (1.2 / 2.0) / (0.8 / 1.0)))
  def testTjuDicriminationMetricsWithoutWeights(self, metric, expected_value):
    computations = metric.computations()
    shared_metrics = computations[0]
    metric = computations[1]

    # Positive labels: 0.0 + 1.0 + 1.0  = 2.0
    # Negative labels: 1.0 + 0.0 + 0.0 = 1.0
    # Positive predictions: 0.0 * 0.8 + 1.0 * 0.3 + 1.0 * 0.9 = 1.2
    # Negative predictions: 1.0 * 0.8 + 0.0 * 0.3 + 0.0 * 0.9 = 0.8
    example1 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.8]),
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
        'example_weights': np.array([1.0]),
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example1, example2, example3])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          |
          'ComputeWeightedTotals' >> beam.CombinePerKey(shared_metrics.combiner)
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
      ('coefficient_of_discrimination',
       tjur_discrimination.CoefficientOfDiscrimination(),
       (3.3 / 5.0) - (1.6 / 5.0)),
      ('relative_coefficient_of_discrimination',
       tjur_discrimination.RelativeCoefficientOfDiscrimination(),
       (3.3 / 5.0) / (1.6 / 5.0)))
  def testTjuDicriminationMetricsWithWeights(self, metric, expected_value):
    computations = metric.computations(example_weighted=True)
    shared_metrics = computations[0]
    metric = computations[1]

    # Positive labels: 1.0 * 0.0 + 2.0 * 1.0 + 3.0 * 1.0 + 4.0 * 0.0 = 5.0
    # Negative labels: 1.0 * 1.0 + 2.0 * 0.0 + 3.0 * 0.0 + 4.0 * 1.0 = 5.0
    # Positive predictions: 1.0 * 0.0 * 0.8 + 2.0 * 1.0 * 0.3 + 3.0 * 1.0 * 0.9
    #                       + 4.0 * 0.0 * 0.2 = 3.3
    # Negative predictions: 1.0 * 1.0 * 0.8 + 2.0 * 0.0 * 0.7 + 3.0 * 0.0 * 0.1
    #                       + 4.0 * 1.0 * 0.2 = 1.6
    example1 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.8]),
        'example_weights': np.array([1.0]),
    }
    example2 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.3]),
        'example_weights': np.array([2.0]),
    }
    example3 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.9]),
        'example_weights': np.array([3.0]),
    }
    example4 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.2]),
        'example_weights': np.array([4.0]),
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example1, example2, example3, example4])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          |
          'ComputeWeightedTotals' >> beam.CombinePerKey(shared_metrics.combiner)
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
      ('coefficient_of_discrimination',
       tjur_discrimination.CoefficientOfDiscrimination()),
      ('relative_coefficient_of_discrimination',
       tjur_discrimination.RelativeCoefficientOfDiscrimination()))
  def testTjurDiscriminationMetricsWithNan(self, metric):
    computations = metric.computations()
    shared_metrics = computations[0]
    metric = computations[1]

    example = {
        'labels': np.array([0.0]),
        'predictions': np.array([1.0]),
        'example_weights': np.array([1.0]),
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          |
          'ComputeWeightedTotals' >> beam.CombinePerKey(shared_metrics.combiner)
          | 'ComputeMetric' >> beam.Map(lambda x: (x[0], metric.result(x[1]))))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          key = metric.keys[0]
          self.assertIn(key, got_metrics)
          self.assertTrue(math.isnan(got_metrics[key]))

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')


