# Copyright 2020 Google LLC
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
"""Tests for fairness indicators metrics."""


import pytest
import math

from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow_model_analysis.addons.fairness.metrics import fairness_indicators
from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.proto import config_pb2



@pytest.mark.xfail(run=False, reason="PR 183 This class contains tests that fail and needs to be fixed. "
"If all tests pass, please remove this mark.")
class FairnessIndicatorsTest(
    testutil.TensorflowModelAnalysisTest, parameterized.TestCase
):

  def testFairessIndicatorsMetricsGeneral(self):
    computations = fairness_indicators.FairnessIndicators(
        thresholds=[0.3, 0.7]
    ).computations()
    histogram = computations[0]
    matrices = computations[1]
    metrics = computations[2]
    examples = [
        {
            'labels': np.array([0.0]),
            'predictions': np.array([0.1]),
            'example_weights': np.array([1.0]),
        },
        {
            'labels': np.array([0.0]),
            'predictions': np.array([0.5]),
            'example_weights': np.array([1.0]),
        },
        {
            'labels': np.array([1.0]),
            'predictions': np.array([0.5]),
            'example_weights': np.array([1.0]),
        },
        {
            'labels': np.array([1.0]),
            'predictions': np.array([0.9]),
            'example_weights': np.array([1.0]),
        },
    ]

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create(examples)
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeHistogram' >> beam.CombinePerKey(histogram.combiner)
          | 'ComputeMatrices'
          >> beam.Map(
              lambda x: (x[0], matrices.result(x[1]))
          )  # pyformat: ignore
          | 'ComputeMetrics' >> beam.Map(lambda x: (x[0], metrics.result(x[1])))
      )  # pyformat: ignore

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          np.testing.assert_equal(
              got_metrics,
              {
                  metric_types.MetricKey(
                      name='fairness_indicators_metrics/false_positive_rate@0.3'
                  ): 0.5,
                  metric_types.MetricKey(
                      name='fairness_indicators_metrics/false_negative_rate@0.3'
                  ): 0.0,
                  metric_types.MetricKey(
                      name='fairness_indicators_metrics/true_positive_rate@0.3'
                  ): 1.0,
                  metric_types.MetricKey(
                      name='fairness_indicators_metrics/true_negative_rate@0.3'
                  ): 0.5,
                  metric_types.MetricKey(
                      name='fairness_indicators_metrics/positive_rate@0.3'
                  ): 0.75,
                  metric_types.MetricKey(
                      name='fairness_indicators_metrics/negative_rate@0.3'
                  ): 0.25,
                  metric_types.MetricKey(
                      name=(
                          'fairness_indicators_metrics/false_discovery_rate@0.3'
                      )
                  ): (1.0 / 3.0),
                  metric_types.MetricKey(
                      name='fairness_indicators_metrics/false_omission_rate@0.3'
                  ): 0.0,
                  metric_types.MetricKey(
                      name='fairness_indicators_metrics/precision@0.3'
                  ): (2.0 / 3.0),
                  metric_types.MetricKey(
                      name='fairness_indicators_metrics/recall@0.3'
                  ): 1.0,
                  metric_types.MetricKey(
                      name='fairness_indicators_metrics/false_positive_rate@0.7'
                  ): 0.0,
                  metric_types.MetricKey(
                      name='fairness_indicators_metrics/false_negative_rate@0.7'
                  ): 0.5,
                  metric_types.MetricKey(
                      name='fairness_indicators_metrics/true_positive_rate@0.7'
                  ): 0.5,
                  metric_types.MetricKey(
                      name='fairness_indicators_metrics/true_negative_rate@0.7'
                  ): 1.0,
                  metric_types.MetricKey(
                      name='fairness_indicators_metrics/positive_rate@0.7'
                  ): 0.25,
                  metric_types.MetricKey(
                      name='fairness_indicators_metrics/negative_rate@0.7'
                  ): 0.75,
                  metric_types.MetricKey(
                      name=(
                          'fairness_indicators_metrics/false_discovery_rate@0.7'
                      )
                  ): 0.0,
                  metric_types.MetricKey(
                      name='fairness_indicators_metrics/false_omission_rate@0.7'
                  ): (1.0 / 3.0),
                  metric_types.MetricKey(
                      name='fairness_indicators_metrics/precision@0.7'
                  ): 1.0,
                  metric_types.MetricKey(
                      name='fairness_indicators_metrics/recall@0.7'
                  ): 0.5,
              },
          )
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testFairessIndicatorsMetricsWithNanValue(self):
    computations = fairness_indicators.FairnessIndicators(
        thresholds=[0.5]
    ).computations()
    histogram = computations[0]
    matrices = computations[1]
    metrics = computations[2]
    examples = [
        {
            'labels': np.array([0.0]),
            'predictions': np.array([0.1]),
            'example_weights': np.array([1.0]),
        },
        {
            'labels': np.array([0.0]),
            'predictions': np.array([0.7]),
            'example_weights': np.array([1.0]),
        },
    ]

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create(examples)
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeHistogram' >> beam.CombinePerKey(histogram.combiner)
          | 'ComputeMatrices'
          >> beam.Map(
              lambda x: (x[0], matrices.result(x[1]))
          )  # pyformat: ignore
          | 'ComputeMetrics' >> beam.Map(lambda x: (x[0], metrics.result(x[1])))
      )  # pyformat: ignore
      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          self.assertLen(got_metrics, 10)  # 1 threshold * 10 metrics
          self.assertTrue(
              math.isnan(
                  got_metrics[
                      metric_types.MetricKey(
                          name='fairness_indicators_metrics/false_negative_rate@0.5'
                      )
                  ]
              )
          )
          self.assertTrue(
              math.isnan(
                  got_metrics[
                      metric_types.MetricKey(
                          name='fairness_indicators_metrics/true_positive_rate@0.5'
                      )
                  ]
              )
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  @parameterized.named_parameters(
      ('_default_threshold', {}, 90, ()),
      (
          '_thresholds_with_different_digits',
          {'thresholds': [0.1, 0.22, 0.333]},
          30,
          (
              metric_types.MetricKey(
                  name='fairness_indicators_metrics/false_positive_rate@0.100',
                  example_weighted=True,
              ),
              metric_types.MetricKey(
                  name='fairness_indicators_metrics/false_positive_rate@0.220',
                  example_weighted=True,
              ),
              metric_types.MetricKey(
                  name='fairness_indicators_metrics/false_positive_rate@0.333',
                  example_weighted=True,
              ),
          ),
      ),
  )
  def testFairessIndicatorsMetricsWithThresholds(
      self, kwargs, expected_metrics_nums, expected_metrics_keys
  ):
    # This is a parameterized test with following parameters.
    #   - metric parameters like thresholds.
    #   - expected number of metrics computed
    #   - expected list of metrics keys

    computations = fairness_indicators.FairnessIndicators(
        **kwargs
    ).computations(example_weighted=True)
    histogram = computations[0]
    matrices = computations[1]
    metrics = computations[2]
    examples = [
        {
            'labels': np.array([0.0]),
            'predictions': np.array([0.1]),
            'example_weights': np.array([1.0]),
        },
        {
            'labels': np.array([0.0]),
            'predictions': np.array([0.7]),
            'example_weights': np.array([3.0]),
        },
    ]

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create(examples)
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeHistogram' >> beam.CombinePerKey(histogram.combiner)
          | 'ComputeMatrices'
          >> beam.Map(
              lambda x: (x[0], matrices.result(x[1]))
          )  # pyformat: ignore
          | 'ComputeMetrics' >> beam.Map(lambda x: (x[0], metrics.result(x[1])))
      )  # pyformat: ignore

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          self.assertLen(got_metrics, expected_metrics_nums)
          for metrics_key in expected_metrics_keys:
            self.assertIn(metrics_key, got_metrics)
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  @parameterized.named_parameters(
      (
          '_has_weight',
          [
              {
                  'labels': np.array([0.0]),
                  'predictions': np.array([0.1]),
                  'example_weights': np.array([1.0]),
              },
              {
                  'labels': np.array([0.0]),
                  'predictions': np.array([0.7]),
                  'example_weights': np.array([3.0]),
              },
          ],
          {'example_weighted': True},
          {
              metric_types.MetricKey(
                  name='fairness_indicators_metrics/negative_rate@0.5',
                  example_weighted=True,
              ): 0.25,
              metric_types.MetricKey(
                  name='fairness_indicators_metrics/positive_rate@0.5',
                  example_weighted=True,
              ): 0.75,
              metric_types.MetricKey(
                  name='fairness_indicators_metrics/true_negative_rate@0.5',
                  example_weighted=True,
              ): 0.25,
              metric_types.MetricKey(
                  name='fairness_indicators_metrics/false_negative_rate@0.5',
                  example_weighted=True,
              ): float('nan'),
              metric_types.MetricKey(
                  name='fairness_indicators_metrics/true_positive_rate@0.5',
                  example_weighted=True,
              ): float('nan'),
              metric_types.MetricKey(
                  name='fairness_indicators_metrics/false_positive_rate@0.5',
                  example_weighted=True,
              ): 0.75,
              metric_types.MetricKey(
                  name='fairness_indicators_metrics/false_discovery_rate@0.5',
                  example_weighted=True,
              ): 1.0,
              metric_types.MetricKey(
                  name='fairness_indicators_metrics/false_omission_rate@0.5',
                  example_weighted=True,
              ): 0.0,
              metric_types.MetricKey(
                  name='fairness_indicators_metrics/precision@0.5',
                  example_weighted=True,
              ): 0.0,
              metric_types.MetricKey(
                  name='fairness_indicators_metrics/recall@0.5',
                  example_weighted=True,
              ): float('nan'),
          },
      ),
      (
          '_has_model_name',
          [
              {
                  'labels': np.array([0.0]),
                  'predictions': {
                      'model1': np.array([0.1]),
                  },
                  'example_weights': np.array([1.0]),
              },
              {
                  'labels': np.array([0.0]),
                  'predictions': {
                      'model1': np.array([0.7]),
                  },
                  'example_weights': np.array([3.0]),
              },
          ],
          {'model_names': ['model1'], 'example_weighted': True},
          {
              metric_types.MetricKey(
                  name='fairness_indicators_metrics/negative_rate@0.5',
                  model_name='model1',
                  example_weighted=True,
              ): 0.25,
              metric_types.MetricKey(
                  name='fairness_indicators_metrics/positive_rate@0.5',
                  model_name='model1',
                  example_weighted=True,
              ): 0.75,
              metric_types.MetricKey(
                  name='fairness_indicators_metrics/true_negative_rate@0.5',
                  model_name='model1',
                  example_weighted=True,
              ): 0.25,
              metric_types.MetricKey(
                  name='fairness_indicators_metrics/false_negative_rate@0.5',
                  model_name='model1',
                  example_weighted=True,
              ): float('nan'),
              metric_types.MetricKey(
                  name='fairness_indicators_metrics/true_positive_rate@0.5',
                  model_name='model1',
                  example_weighted=True,
              ): float('nan'),
              metric_types.MetricKey(
                  name='fairness_indicators_metrics/false_positive_rate@0.5',
                  model_name='model1',
                  example_weighted=True,
              ): 0.75,
              metric_types.MetricKey(
                  name='fairness_indicators_metrics/false_discovery_rate@0.5',
                  model_name='model1',
                  example_weighted=True,
              ): 1.0,
              metric_types.MetricKey(
                  name='fairness_indicators_metrics/false_omission_rate@0.5',
                  model_name='model1',
                  example_weighted=True,
              ): 0.0,
              metric_types.MetricKey(
                  name='fairness_indicators_metrics/precision@0.5',
                  model_name='model1',
                  example_weighted=True,
              ): 0.0,
              metric_types.MetricKey(
                  name='fairness_indicators_metrics/recall@0.5',
                  model_name='model1',
                  example_weighted=True,
              ): float('nan'),
          },
      ),
  )
  def testFairessIndicatorsMetricsWithInput(
      self, input_examples, computations_kwargs, expected_result
  ):
    # This is a parameterized test with following parameters.
    #   - input examples to be used in the test
    #   - parameters like model name etc.
    #   - expected result to assert on

    computations = fairness_indicators.FairnessIndicators(
        thresholds=[0.5]
    ).computations(**computations_kwargs)
    histogram = computations[0]
    matrices = computations[1]
    metrics = computations[2]

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create(input_examples)
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeHistogram' >> beam.CombinePerKey(histogram.combiner)
          | 'ComputeMatrices'
          >> beam.Map(
              lambda x: (x[0], matrices.result(x[1]))
          )  # pyformat: ignore
          | 'ComputeMetrics' >> beam.Map(lambda x: (x[0], metrics.result(x[1])))
      )  # pyformat: ignore

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          np.testing.assert_equal(got_metrics, expected_result)
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')



# TODO: b/147497357 - Add counter test once we have counter setup.

