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
"""Tests for confusion matrix at thresholds."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import math

from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.metrics import confusion_matrix_metrics
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util


class ConfusionMatrixMetricsTest(testutil.TensorflowModelAnalysisTest,
                                 parameterized.TestCase):

  @parameterized.named_parameters(
      ('specificity', confusion_matrix_metrics.Specificity(), 2.0 /
       (2.0 + 3.0)),
      ('fall_out', confusion_matrix_metrics.FallOut(), 3.0 / (3.0 + 2.0)),
      ('miss_rate', confusion_matrix_metrics.MissRate(), 4.0 / (4.0 + 1.0)),
      ('negative_predictive_value',
       confusion_matrix_metrics.NegativePredictiveValue(), 2.0 / (2.0 + 4.0)),
      ('false_discovery_rate', confusion_matrix_metrics.FalseDiscoveryRate(),
       3.0 / (3.0 + 1.0)),
      ('false_omission_rate', confusion_matrix_metrics.FalseOmissionRate(),
       4.0 / (4.0 + 2.0)),
      ('prevalence', confusion_matrix_metrics.Prevalence(),
       (1.0 + 4.0) / (1.0 + 2.0 + 3.0 + 4.0)),
      ('prevalence_threshold', confusion_matrix_metrics.PrevalenceThreshold(),
       (math.sqrt((1.0 / (1.0 + 4.0)) * (1.0 - 1.0 * (2.0 / (2.0 + 3.0)))) +
        (2.0 / (2.0 + 3.0) - 1.0)) / ((1.0 / (1.0 + 4.0) +
                                       (2.0 / (2.0 + 3.0)) - 1.0))),
      ('threat_score', confusion_matrix_metrics.ThreatScore(), 1.0 /
       (1.0 + 4.0 + 3.0)),
      ('balanced_accuracy', confusion_matrix_metrics.BalancedAccuracy(),
       ((1.0 / (1.0 + 4.0)) + (2.0 / (2.0 + 3.0))) / 2),
      ('f1_score', confusion_matrix_metrics.F1Score(), 2 * 1.0 /
       (2 * 1.0 + 3.0 + 4.0)),
      ('matthews_correlation_coefficient',
       confusion_matrix_metrics.MatthewsCorrelationCoefficent(),
       (1.0 * 2.0 - 3.0 * 4.0) / math.sqrt(
           (1.0 + 3.0) * (1.0 + 4.0) * (2.0 + 3.0) * (2.0 + 4.0))),
      ('fowlkes_mallows_index', confusion_matrix_metrics.FowlkesMallowsIndex(),
       math.sqrt(1.0 / (1.0 + 3.0) * 1.0 / (1.0 + 4.0))),
      ('informedness', confusion_matrix_metrics.Informedness(),
       (1.0 / (1.0 + 4.0)) + (2.0 / (2.0 + 3.0)) - 1.0),
      ('markedness', confusion_matrix_metrics.Markedness(),
       (1.0 / (1.0 + 3.0)) + (2.0 / (2.0 + 4.0)) - 1.0),
      ('positive_likelihood_ratio',
       confusion_matrix_metrics.PositiveLikelihoodRatio(),
       (1.0 / (1.0 + 4.0)) / (3.0 / (3.0 + 2.0))),
      ('negative_likelihood_ratio',
       confusion_matrix_metrics.NegativeLikelihoodRatio(),
       (4.0 / (4.0 + 1.0)) / (2.0 / (2.0 + 3.0))),
      ('diagnostic_odds_ratio', confusion_matrix_metrics.DiagnosticOddsRatio(),
       ((1.0 / 3.0)) / (4.0 / 2.0)),
  )
  def testConfusionMatrixMetrics(self, metric, expected_value):
    computations = metric.computations()
    histogram = computations[0]
    matrices = computations[1]
    metrics = computations[2]

    # tp = 1
    # tn = 2
    # fp = 3
    # fn = 4
    example1 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.6]),
        'example_weights': np.array([1.0]),
    }
    example2 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.3]),
        'example_weights': np.array([1.0]),
    }
    example3 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.2]),
        'example_weights': np.array([1.0]),
    }
    example4 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.6]),
        'example_weights': np.array([1.0]),
    }
    example5 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.7]),
        'example_weights': np.array([1.0]),
    }
    example6 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.8]),
        'example_weights': np.array([1.0]),
    }
    example7 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.1]),
        'example_weights': np.array([1.0]),
    }
    example8 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.2]),
        'example_weights': np.array([1.0]),
    }
    example9 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.3]),
        'example_weights': np.array([1.0]),
    }
    example10 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.4]),
        'example_weights': np.array([1.0]),
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([
              example1, example2, example3, example4, example5, example6,
              example7, example8, example9, example10
          ])
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
          self.assertLen(got_metrics, 1)
          key = metrics.keys[0]
          self.assertDictElementsAlmostEqual(
              got_metrics, {key: expected_value}, places=5)
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testConfusionMatrixMetricsWithNan(self):
    computations = confusion_matrix_metrics.Specificity().computations()
    histogram = computations[0]
    matrices = computations[1]
    metrics = computations[2]

    example1 = {
        'labels': np.array([1.0]),
        'predictions': np.array([1.0]),
        'example_weights': np.array([1.0]),
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example1])
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
          self.assertLen(got_metrics, 1)
          key = metrics.keys[0]
          self.assertIn(key, got_metrics)
          self.assertTrue(math.isnan(got_metrics[key]))
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testConfusionMatrixAtThresholds(self):
    computations = confusion_matrix_metrics.ConfusionMatrixAtThresholds(
        thresholds=[0.3, 0.5, 0.8]).computations()
    histogram = computations[0]
    matrices = computations[1]
    metrics = computations[2]

    example1 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.0]),
        'example_weights': np.array([1.0]),
    }
    example2 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.5]),
        'example_weights': np.array([1.0]),
    }
    example3 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.3]),
        'example_weights': np.array([1.0]),
    }
    example4 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.9]),
        'example_weights': np.array([1.0]),
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example1, example2, example3, example4])
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
          self.assertLen(got_metrics, 1)
          key = metric_types.MetricKey(name='confusion_matrix_at_thresholds')
          self.assertIn(key, got_metrics)
          got_metric = got_metrics[key]
          self.assertProtoEquals(
              """
              matrices {
                threshold: 0.3
                false_negatives: 1.0
                true_negatives: 1.0
                false_positives: 1.0
                true_positives: 1.0
                precision: 0.5
                recall: 0.5
              }
              matrices {
                threshold: 0.5
                false_negatives: 1.0
                true_negatives: 2.0
                true_positives: 1.0
                precision: 1.0
                recall: 0.5
              }
              matrices {
                threshold: 0.8
                false_negatives: 1.0
                true_negatives: 2.0
                true_positives: 1.0
                precision: 1.0
                recall: 0.5
              }
          """, got_metric)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')


if __name__ == '__main__':
  tf.test.main()
