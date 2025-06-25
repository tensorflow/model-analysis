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
"""Tests for squared pearson correlation metric."""


import pytest
import math

import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.metrics import squared_pearson_correlation
from tensorflow_model_analysis.utils import test_util


@pytest.mark.xfail(run=False, reason="PR 183 This class contains tests that fail and needs to be fixed. "
"If all tests pass, please remove this mark.")
class SquaredPearsonCorrelationTest(test_util.TensorflowModelAnalysisTest):

  def testSquaredPearsonCorrelationWithoutWeights(self):
    computations = (
        squared_pearson_correlation.SquaredPearsonCorrelation().computations())
    metric = computations[0]

    example1 = {
        'labels': np.array([2.0]),
        'predictions': np.array([1.0]),
        'example_weights': np.array([1.0]),
    }
    example2 = {
        'labels': np.array([1.0]),
        'predictions': np.array([2.0]),
        'example_weights': np.array([1.0]),
    }
    example3 = {
        'labels': np.array([2.0]),
        'predictions': np.array([3.0]),
        'example_weights': np.array([1.0]),
    }
    example4 = {
        'labels': np.array([3.0]),
        'predictions': np.array([4.0]),
        'example_weights': np.array([1.0]),
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example1, example2, example3, example4])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeMetric' >> beam.CombinePerKey(metric.combiner))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          key = metric.keys[0]
          # 1: prediction = 1, label = 2
          # 2: prediction = 2, label = 1
          # 3: prediction = 3, label = 2
          # 4: prediction = 4, label = 3
          #
          # pred_x_labels = 2 + 2 + 6 + 12 = 22
          # labels = 2 + 1 + 2 + 3 =  8
          # preds = 1 + 2 + 3 + 4 = 10
          # sq_labels = 4 + 1 + 4 + 9 = 18
          # sq_preds = 1 + 4 + 9 + 16 = 30
          # examples = 4
          #
          # r^2 = (22 - 8 * 10 / 4)^2 / (30 - 10^2 / 4) * (18 - 8^2 / 4)
          # r^2 = 4 / (5 * 2) = 0.4
          self.assertDictElementsAlmostEqual(got_metrics, {key: 0.4}, places=5)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testSquaredPearsonCorrelationWithWeights(self):
    computations = (
        squared_pearson_correlation.SquaredPearsonCorrelation().computations(
            example_weighted=True))
    metric = computations[0]

    example1 = {
        'labels': np.array([1.0]),
        'predictions': np.array([1.0]),
        'example_weights': np.array([1.0]),
    }
    example2 = {
        'labels': np.array([4.0]),
        'predictions': np.array([2.0]),
        'example_weights': np.array([2.0]),
    }
    example3 = {
        'labels': np.array([3.0]),
        'predictions': np.array([3.0]),
        'example_weights': np.array([3.0]),
    }
    example4 = {
        'labels': np.array([3.0]),
        'predictions': np.array([4.0]),
        'example_weights': np.array([4.0]),
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example1, example2, example3, example4])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeMetric' >> beam.CombinePerKey(metric.combiner))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          key = metric.keys[0]
          # 1: prediction = 1, label = 1
          # 2: prediction = 2, label = 4
          # 3: prediction = 3, label = 3
          # 4: prediction = 4, label = 3
          #
          # pred_x_labels = 1x1x1 + 2x2x4 + 3x3x3 + 4x4x3 = 92
          # labels = 1x1 + 2x4 + 3x3 + 4x3 = 30
          # preds = 1 + 2x2 + 3x3 + 4x4= 30
          # sq_labels = 1x1x1 + 2x4x4+ 3x3x3 + 4x3x3 = 96
          # sq_preds = 1x1x1 + 2x2x2 + 3x3x3 + 4x4x4 = 100
          # examples = 1 + 2 + 3 + 4 = 10
          #
          # r^2 = (92 - 30 * 30 / 10)^2 / (100 - 30^2 / 10) * (96 - 30^2 / 10)
          # r^2 = 4 / (10 * 6) = 0.06667
          self.assertDictElementsAlmostEqual(
              got_metrics, {key: 0.06667}, places=5)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testSquaredPearsonCorrelationMetricsWithNan(self):
    computations = (
        squared_pearson_correlation.SquaredPearsonCorrelation().computations())
    metric = computations[0]

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
          | 'ComputeMetric' >> beam.CombinePerKey(metric.combiner))

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


