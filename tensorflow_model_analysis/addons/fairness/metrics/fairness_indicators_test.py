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

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import math
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis.addons.fairness.metrics import fairness_indicators
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util


class FairnessIndicatorsTest(testutil.TensorflowModelAnalysisTest):

  def testFairessIndicatorsMetrics(self):
    computations = fairness_indicators.FairnessIndicators(
        thresholds=[0.3, 0.7]).computations()
    histogram = computations[0]
    matrices = computations[1]
    metrics = computations[2]
    examples = [{
        'labels': np.array([0.0]),
        'predictions': np.array([0.1]),
        'example_weights': np.array([1.0]),
    }, {
        'labels': np.array([0.0]),
        'predictions': np.array([0.5]),
        'example_weights': np.array([1.0]),
    }, {
        'labels': np.array([1.0]),
        'predictions': np.array([0.5]),
        'example_weights': np.array([1.0]),
    }, {
        'labels': np.array([1.0]),
        'predictions': np.array([0.9]),
        'example_weights': np.array([1.0]),
    }]

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create(examples)
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeHistogram' >> beam.CombinePerKey(histogram.combiner)
          | 'ComputeMatrices' >> beam.Map(
              lambda x: (x[0], matrices.result(x[1])))  # pyformat: ignore
          | 'ComputeMetrics' >> beam.Map(lambda x: (x[0], metrics.result(x[1])))
      )  # pyformat: ignore

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          self.assertLen(got_metrics, 12)  # 2 thresholds * 6 metrics
          self.assertDictElementsAlmostEqual(
              got_metrics, {
                  metric_types.MetricKey(
                      name='fairness_indicators_metrics/false_positive_rate@0.3',
                      model_name='',
                      output_name='',
                      sub_key=None):
                      0.5,
                  metric_types.MetricKey(
                      name='fairness_indicators_metrics/false_negative_rate@0.3',
                      model_name='',
                      output_name='',
                      sub_key=None):
                      0.0,
                  metric_types.MetricKey(
                      name='fairness_indicators_metrics/true_positive_rate@0.3',
                      model_name='',
                      output_name='',
                      sub_key=None):
                      1.0,
                  metric_types.MetricKey(
                      name='fairness_indicators_metrics/true_negative_rate@0.3',
                      model_name='',
                      output_name='',
                      sub_key=None):
                      0.5,
                  metric_types.MetricKey(
                      name='fairness_indicators_metrics/positive_rate@0.3',
                      model_name='',
                      output_name='',
                      sub_key=None):
                      0.75,
                  metric_types.MetricKey(
                      name='fairness_indicators_metrics/negative_rate@0.3',
                      model_name='',
                      output_name='',
                      sub_key=None):
                      0.25,
                  metric_types.MetricKey(
                      name='fairness_indicators_metrics/false_positive_rate@0.7',
                      model_name='',
                      output_name='',
                      sub_key=None):
                      0.0,
                  metric_types.MetricKey(
                      name='fairness_indicators_metrics/false_negative_rate@0.7',
                      model_name='',
                      output_name='',
                      sub_key=None):
                      0.5,
                  metric_types.MetricKey(
                      name='fairness_indicators_metrics/true_positive_rate@0.7',
                      model_name='',
                      output_name='',
                      sub_key=None):
                      0.5,
                  metric_types.MetricKey(
                      name='fairness_indicators_metrics/true_negative_rate@0.7',
                      model_name='',
                      output_name='',
                      sub_key=None):
                      1.0,
                  metric_types.MetricKey(
                      name='fairness_indicators_metrics/positive_rate@0.7',
                      model_name='',
                      output_name='',
                      sub_key=None):
                      0.25,
                  metric_types.MetricKey(
                      name='fairness_indicators_metrics/negative_rate@0.7',
                      model_name='',
                      output_name='',
                      sub_key=None):
                      0.75
              })
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testFairessIndicatorsMetricsWithNanValue(self):
    computations = fairness_indicators.FairnessIndicators(
        thresholds=[0.5]).computations()
    histogram = computations[0]
    matrices = computations[1]
    metrics = computations[2]
    examples = [{
        'labels': np.array([0.0]),
        'predictions': np.array([0.1]),
        'example_weights': np.array([1.0]),
    }, {
        'labels': np.array([0.0]),
        'predictions': np.array([0.7]),
        'example_weights': np.array([1.0]),
    }]

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create(examples)
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeHistogram' >> beam.CombinePerKey(histogram.combiner)
          | 'ComputeMatrices' >> beam.Map(
              lambda x: (x[0], matrices.result(x[1])))  # pyformat: ignore
          | 'ComputeMetrics' >> beam.Map(lambda x: (x[0], metrics.result(x[1])))
      )  # pyformat: ignore

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          self.assertLen(got_metrics, 6)  # 1 threshold * 6 metrics
          self.assertTrue(
              math.isnan(got_metrics[metric_types.MetricKey(
                  name='fairness_indicators_metrics/false_negative_rate@0.5',
                  model_name='',
                  output_name='',
                  sub_key=None)]))
          self.assertTrue(
              math.isnan(got_metrics[metric_types.MetricKey(
                  name='fairness_indicators_metrics/true_positive_rate@0.5',
                  model_name='',
                  output_name='',
                  sub_key=None)]))

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')


if __name__ == '__main__':
  tf.test.main()
