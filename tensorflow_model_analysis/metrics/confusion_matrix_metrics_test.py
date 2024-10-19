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


import pytest
import math

from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis.metrics import binary_confusion_matrices
from tensorflow_model_analysis.metrics import confusion_matrix_metrics
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.metrics import test_util as metric_test_util
from tensorflow_model_analysis.utils import test_util


_TF_MAJOR_VERSION = int(tf.version.VERSION.split('.')[0])
_TRUE_POISITIVE = (1, 1)
_TRUE_NEGATIVE = (0, 0)


@pytest.mark.xfail(run=False, reason="PR 183 This class contains tests that fail and needs to be fixed. "
"If all tests pass, please remove this mark.")
class ConfusionMatrixMetricsTest(
    test_util.TensorflowModelAnalysisTest,
    metric_test_util.TestCase,
    parameterized.TestCase,
):

  @parameterized.named_parameters(
      (
          'Precision',
          confusion_matrix_metrics.Precision(),
          _TRUE_NEGATIVE,
          float('nan'),
      ),
      (
          'Recall',
          confusion_matrix_metrics.Recall(),
          _TRUE_NEGATIVE,
          float('nan'),
      ),
      (
          'Specificity',
          confusion_matrix_metrics.Specificity(),
          _TRUE_POISITIVE,
          float('nan'),
      ),
      (
          'FallOut',
          confusion_matrix_metrics.FallOut(),
          _TRUE_POISITIVE,
          float('nan'),
      ),
      (
          'MissRate',
          confusion_matrix_metrics.MissRate(),
          _TRUE_NEGATIVE,
          float('nan'),
      ),
      (
          'NegativePredictiveValue',
          confusion_matrix_metrics.NegativePredictiveValue(),
          _TRUE_POISITIVE,
          float('nan'),
      ),
      (
          'FalseDiscoveryRate',
          confusion_matrix_metrics.FalseDiscoveryRate(),
          _TRUE_NEGATIVE,
          float('nan'),
      ),
      (
          'FalseOmissionRate',
          confusion_matrix_metrics.FalseOmissionRate(),
          _TRUE_POISITIVE,
          float('nan'),
      ),
      (
          'ThreatScore',
          confusion_matrix_metrics.ThreatScore(),
          _TRUE_NEGATIVE,
          float('nan'),
      ),
      (
          'F1Score',
          confusion_matrix_metrics.F1Score(),
          _TRUE_NEGATIVE,
          float('nan'),
      ),
      (
          'MatthewsCorrelationCoefficient',
          confusion_matrix_metrics.MatthewsCorrelationCoefficient(),
          _TRUE_NEGATIVE,
          float('nan'),
      ),
  )
  def testConfusionMatrixMetrics_DivideByZero_(
      self, metric, pred_label, expected_value
  ):
    if _TF_MAJOR_VERSION < 2 and metric.__class__.__name__ in (
        'SpecificityAtSensitivity',
        'SensitivityAtSpecificity',
        'PrecisionAtRecall',
        'RecallAtPrecision',
        'RecallAtFalsePositiveRate',
    ):
      self.skipTest('Not supported in TFv1.')

    computations = metric.computations(example_weighted=True)
    histogram = computations[0]
    matrices = computations[1]
    metrics = computations[2]

    # Using one example to create a situation where the denominator in the
    # corresponding calculation is 0.
    pred, label = pred_label
    example1 = {
        'labels': np.array([label]),
        'predictions': np.array([pred]),
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example1])
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
          self.assertLen(got_metrics, 1)
          key = metrics.keys[0]
          self.assertIn(key, got_metrics)
          # np.testing utils automatically cast floats to arrays which fails
          # to catch type mismatches.
          self.assertEqual(type(expected_value), type(got_metrics[key]))
          np.testing.assert_almost_equal(
              got_metrics[key], expected_value, decimal=5
          )
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  # LINT.IfChange(tfma_confusion_matrix_metrics_tests)
  @parameterized.named_parameters(
      ('auc', confusion_matrix_metrics.AUC(), np.float64(0.26)),
      (
          'auc_precision_recall',
          confusion_matrix_metrics.AUCPrecisionRecall(),
          np.float64(0.36205),
      ),
      (
          'specificity_at_sensitivity',
          confusion_matrix_metrics.SpecificityAtSensitivity(0.5),
          0.2,
      ),
      (
          'sensitivity_at_specificity',
          confusion_matrix_metrics.SensitivityAtSpecificity(0.5),
          0.0,
      ),
      (
          'precision_at_recall',
          confusion_matrix_metrics.PrecisionAtRecall(0.5),
          0.5,
      ),
      (
          'recall_at_precision',
          confusion_matrix_metrics.RecallAtPrecision(0.5),
          1.0,
      ),
      (
          'recall_at_false_positive_rate',
          confusion_matrix_metrics.RecallAtFalsePositiveRate(0.6),
          0.4,
      ),
      ('true_positives', confusion_matrix_metrics.TruePositives(), 1.0),
      ('tp', confusion_matrix_metrics.TP(), 1.0),
      ('false_positives', confusion_matrix_metrics.FalsePositives(), 3.0),
      ('fp', confusion_matrix_metrics.FP(), 3.0),
      ('true_negatives', confusion_matrix_metrics.TrueNegatives(), 2.0),
      ('tn', confusion_matrix_metrics.TN(), 2.0),
      ('false_negatives', confusion_matrix_metrics.FalseNegatives(), 4.0),
      ('fn', confusion_matrix_metrics.FN(), 4.0),
      (
          'binary_accuracy',
          confusion_matrix_metrics.BinaryAccuracy(),
          (1.0 + 2.0) / (1.0 + 2.0 + 3.0 + 4.0),
      ),
      ('precision', confusion_matrix_metrics.Precision(), 1.0 / (1.0 + 3.0)),
      ('ppv', confusion_matrix_metrics.PPV(), 1.0 / (1.0 + 3.0)),
      ('recall', confusion_matrix_metrics.Recall(), 1.0 / (1.0 + 4.0)),
      ('tpr', confusion_matrix_metrics.TPR(), 1.0 / (1.0 + 4.0)),
      (
          'specificity',
          confusion_matrix_metrics.Specificity(),
          2.0 / (2.0 + 3.0),
      ),
      ('tnr', confusion_matrix_metrics.TNR(), 2.0 / (2.0 + 3.0)),
      ('fall_out', confusion_matrix_metrics.FallOut(), 3.0 / (3.0 + 2.0)),
      ('fpr', confusion_matrix_metrics.FPR(), 3.0 / (3.0 + 2.0)),
      ('miss_rate', confusion_matrix_metrics.MissRate(), 4.0 / (4.0 + 1.0)),
      ('fnr', confusion_matrix_metrics.FNR(), 4.0 / (4.0 + 1.0)),
      (
          'negative_predictive_value',
          confusion_matrix_metrics.NegativePredictiveValue(),
          2.0 / (2.0 + 4.0),
      ),
      ('npv', confusion_matrix_metrics.NPV(), 2.0 / (2.0 + 4.0)),
      (
          'false_discovery_rate',
          confusion_matrix_metrics.FalseDiscoveryRate(),
          3.0 / (3.0 + 1.0),
      ),
      (
          'false_omission_rate',
          confusion_matrix_metrics.FalseOmissionRate(),
          4.0 / (4.0 + 2.0),
      ),
      (
          'prevalence',
          confusion_matrix_metrics.Prevalence(),
          (1.0 + 4.0) / (1.0 + 2.0 + 3.0 + 4.0),
      ),
      (
          'prevalence_threshold',
          confusion_matrix_metrics.PrevalenceThreshold(),
          (
              math.sqrt((1.0 / (1.0 + 4.0)) * (1.0 - (2.0 / (2.0 + 3.0))))
              + (2.0 / (2.0 + 3.0) - 1.0)
          )
          / ((1.0 / (1.0 + 4.0) + (2.0 / (2.0 + 3.0)) - 1.0)),
      ),
      (
          'threat_score',
          confusion_matrix_metrics.ThreatScore(),
          1.0 / (1.0 + 4.0 + 3.0),
      ),
      (
          'balanced_accuracy',
          confusion_matrix_metrics.BalancedAccuracy(),
          ((1.0 / (1.0 + 4.0)) + (2.0 / (2.0 + 3.0))) / 2,
      ),
      (
          'f1_score',
          confusion_matrix_metrics.F1Score(),
          2 * 1.0 / (2 * 1.0 + 3.0 + 4.0),
      ),
      (
          'matthews_correlation_coefficient',
          confusion_matrix_metrics.MatthewsCorrelationCoefficient(),
          (1.0 * 2.0 - 3.0 * 4.0)
          / math.sqrt((1.0 + 3.0) * (1.0 + 4.0) * (2.0 + 3.0) * (2.0 + 4.0)),
      ),
      (
          'fowlkes_mallows_index',
          confusion_matrix_metrics.FowlkesMallowsIndex(),
          math.sqrt(1.0 / (1.0 + 3.0) * 1.0 / (1.0 + 4.0)),
      ),
      (
          'informedness',
          confusion_matrix_metrics.Informedness(),
          (1.0 / (1.0 + 4.0)) + (2.0 / (2.0 + 3.0)) - 1.0,
      ),
      (
          'markedness',
          confusion_matrix_metrics.Markedness(),
          (1.0 / (1.0 + 3.0)) + (2.0 / (2.0 + 4.0)) - 1.0,
      ),
      (
          'positive_likelihood_ratio',
          confusion_matrix_metrics.PositiveLikelihoodRatio(),
          (1.0 / (1.0 + 4.0)) / (3.0 / (3.0 + 2.0)),
      ),
      (
          'negative_likelihood_ratio',
          confusion_matrix_metrics.NegativeLikelihoodRatio(),
          (4.0 / (4.0 + 1.0)) / (2.0 / (2.0 + 3.0)),
      ),
      (
          'diagnostic_odds_ratio',
          confusion_matrix_metrics.DiagnosticOddsRatio(),
          ((1.0 / 3.0)) / (4.0 / 2.0),
      ),
      (
          'predicted_positive_rate',
          confusion_matrix_metrics.PredictedPositiveRate(),
          (1.0 + 3.0) / (1.0 + 2.0 + 3.0 + 4.0),
      ),
      (
          'threshold_at_recall',
          confusion_matrix_metrics.ThresholdAtRecall(0.5),
          0.29993,
      ),
  )
  def testConfusionMatrixMetrics(self, metric, expected_value):
    if _TF_MAJOR_VERSION < 2 and metric.__class__.__name__ in (
        'SpecificityAtSensitivity',
        'SensitivityAtSpecificity',
        'PrecisionAtRecall',
        'RecallAtPrecision',
        'RecallAtFalsePositiveRate',
    ):
      self.skipTest('Not supported in TFv1.')

    computations = metric.computations(example_weighted=True)
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
          | 'Create'
          >> beam.Create([
              example1,
              example2,
              example3,
              example4,
              example5,
              example6,
              example7,
              example8,
              example9,
              example10,
          ])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeHistogram' >> beam.CombinePerKey(histogram.combiner)
          | 'ComputeMatrices'
          >> beam.Map(lambda x: (x[0], matrices.result(x[1])))
          | 'ComputeMetrics' >> beam.Map(lambda x: (x[0], metrics.result(x[1])))
      )

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          self.assertLen(got_metrics, 1)
          key = metrics.keys[0]
          self.assertIn(key, got_metrics)
          # np.testing utils automatically cast floats to arrays which fails
          # to catch type mismatches.
          self.assertEqual(type(expected_value), type(got_metrics[key]))
          np.testing.assert_almost_equal(
              got_metrics[key], expected_value, decimal=5)
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  @parameterized.named_parameters(
      ('auc', confusion_matrix_metrics.AUC(), np.float64(0.64286)),
      (
          'auc_precision_recall',
          confusion_matrix_metrics.AUCPrecisionRecall(),
          np.float64(0.37467),
      ),
      (
          'specificity_at_sensitivity',
          confusion_matrix_metrics.SpecificityAtSensitivity(0.5),
          0.642857,
      ),
      (
          'sensitivity_at_specificity',
          confusion_matrix_metrics.SensitivityAtSpecificity(0.5),
          1.0,
      ),
      (
          'precision_at_recall',
          confusion_matrix_metrics.PrecisionAtRecall(0.5),
          0.58333,
      ),
      (
          'recall_at_precision',
          confusion_matrix_metrics.RecallAtPrecision(0.5),
          1.0,
      ),
      (
          'recall_at_false_positive_rate',
          confusion_matrix_metrics.RecallAtFalsePositiveRate(0.5 / (0.5 + 0.9)),
          1.0,
      ),
      ('true_positives', confusion_matrix_metrics.TruePositives(), 0.7),
      ('false_positives', confusion_matrix_metrics.FalsePositives(), 0.5),
      ('true_negatives', confusion_matrix_metrics.TrueNegatives(), 0.9),
      ('false_negatives', confusion_matrix_metrics.FalseNegatives(), 0.0),
      (
          'binary_accuracy',
          confusion_matrix_metrics.BinaryAccuracy(),
          (0.7 + 0.9) / (0.7 + 0.9 + 0.5 + 0.0),
      ),
      ('precision', confusion_matrix_metrics.Precision(), 0.7 / (0.7 + 0.5)),
      ('recall', confusion_matrix_metrics.Recall(), 0.7 / (0.7 + 0.0)),
      (
          'specificity',
          confusion_matrix_metrics.Specificity(),
          0.9 / (0.9 + 0.5),
      ),
      ('fall_out', confusion_matrix_metrics.FallOut(), 0.5 / (0.5 + 0.9)),
      ('miss_rate', confusion_matrix_metrics.MissRate(), 0.0 / (0.0 + 0.7)),
      (
          'negative_predictive_value',
          confusion_matrix_metrics.NegativePredictiveValue(),
          0.9 / (0.9 + 0.0),
      ),
      (
          'false_discovery_rate',
          confusion_matrix_metrics.FalseDiscoveryRate(),
          0.5 / (0.5 + 0.7),
      ),
      (
          'false_omission_rate',
          confusion_matrix_metrics.FalseOmissionRate(),
          0.0 / (0.0 + 0.9),
      ),
      (
          'prevalence',
          confusion_matrix_metrics.Prevalence(),
          (0.7 + 0.0) / (0.7 + 0.9 + 0.5 + 0.0),
      ),
      (
          'prevalence_threshold',
          confusion_matrix_metrics.PrevalenceThreshold(),
          (
              math.sqrt((0.7 / (0.7 + 0.0)) * (1.0 - (0.9 / (0.9 + 0.5))))
              + (0.9 / (0.9 + 0.5) - 1.0)
          )
          / ((0.7 / (0.7 + 0.0) + (0.9 / (0.9 + 0.5)) - 1.0)),
      ),
      (
          'threat_score',
          confusion_matrix_metrics.ThreatScore(),
          0.7 / (0.7 + 0.0 + 0.5),
      ),
      (
          'balanced_accuracy',
          confusion_matrix_metrics.BalancedAccuracy(),
          ((0.7 / (0.7 + 0.0)) + (0.9 / (0.9 + 0.5))) / 2,
      ),
      (
          'f1_score',
          confusion_matrix_metrics.F1Score(),
          2 * 0.7 / (2 * 0.7 + 0.5 + 0.0),
      ),
      (
          'matthews_correlation_coefficient',
          confusion_matrix_metrics.MatthewsCorrelationCoefficient(),
          (0.7 * 0.9 - 0.5 * 0.0)
          / math.sqrt((0.7 + 0.5) * (0.7 + 0.0) * (0.9 + 0.5) * (0.9 + 0.0)),
      ),
      (
          'fowlkes_mallows_index',
          confusion_matrix_metrics.FowlkesMallowsIndex(),
          math.sqrt(0.7 / (0.7 + 0.5) * 0.7 / (0.7 + 0.0)),
      ),
      (
          'informedness',
          confusion_matrix_metrics.Informedness(),
          (0.7 / (0.7 + 0.0)) + (0.9 / (0.9 + 0.5)) - 1.0,
      ),
      (
          'markedness',
          confusion_matrix_metrics.Markedness(),
          (0.7 / (0.7 + 0.5)) + (0.9 / (0.9 + 0.0)) - 1.0,
      ),
      (
          'positive_likelihood_ratio',
          confusion_matrix_metrics.PositiveLikelihoodRatio(),
          (0.7 / (0.7 + 0.0)) / (0.5 / (0.5 + 0.9)),
      ),
      (
          'negative_likelihood_ratio',
          confusion_matrix_metrics.NegativeLikelihoodRatio(),
          (0.0 / (0.0 + 0.7)) / (0.9 / (0.9 + 0.5)),
      ),
      (
          'predicted_positive_rate',
          confusion_matrix_metrics.PredictedPositiveRate(),
          (0.7 + 0.5) / (0.7 + 0.9 + 0.5 + 0.0),
      ),
  )
  def testConfusionMatrixMetricsWithWeights(self, metric, expected_value):
    if _TF_MAJOR_VERSION < 2 and metric.__class__.__name__ in (
        'SpecificityAtSensitivity',
        'SensitivityAtSpecificity',
        'PrecisionAtRecall',
        'RecallAtPrecision',
        'RecallAtFalsePositiveRate',
    ):
      self.skipTest('Not supported in TFv1.')

    computations = metric.computations(example_weighted=True)
    histogram = computations[0]
    matrix = computations[1]
    derived_metric = computations[2]

    # tp = 0.7
    # tn = 0.9
    # fp = 0.5
    # fn = 0.0
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
          | 'ComputeHistogram' >> beam.CombinePerKey(histogram.combiner)
          | 'ComputeConfusionMatrix'
          >> beam.Map(lambda x: (x[0], matrix.result(x[1])))
          | 'ComputeMetric'
          >> beam.Map(lambda x: (x[0], derived_metric.result(x[1])))
      )

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          key = metric_types.MetricKey(name=metric.name, example_weighted=True)
          self.assertIn(key, got_metrics)
          # np.testing utils automatically cast floats to arrays which fails
          # to catch type mismatches.
          self.assertEqual(type(expected_value), type(got_metrics[key]))
          np.testing.assert_almost_equal(
              np.array(got_metrics[key]), np.array(expected_value), decimal=5)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  # LINT.ThenChange(../google/sql:uda_auc_tests)

  @parameterized.named_parameters(
      ('auc', confusion_matrix_metrics.AUC(), 0.8571428),
      ('auc_precision_recall', confusion_matrix_metrics.AUCPrecisionRecall(),
       0.77369833),
      ('true_positives', confusion_matrix_metrics.TruePositives(), 1.4),
      ('false_positives', confusion_matrix_metrics.FalsePositives(), 0.6),
      ('true_negatives', confusion_matrix_metrics.TrueNegatives(), 1.0),
      ('false_negatives', confusion_matrix_metrics.FalseNegatives(), 0.0),
  )
  def testConfusionMatrixMetricsWithFractionalLabels(self, metric,
                                                     expected_value):
    computations = metric.computations(example_weighted=True)
    histogram = computations[0]
    matrix = computations[1]
    derived_metric = computations[2]

    # The following examples will be expanded to:
    #
    # prediction | label | weight
    #     0.0    |   -   |  1.0
    #     0.7    |   -   |  0.4
    #     0.7    |   +   |  0.6
    #     1.0    |   -   |  0.2
    #     1.0    |   +   |  0.8
    example1 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.0]),
        'example_weights': np.array([1.0]),
    }
    example2 = {
        'labels': np.array([0.6]),
        'predictions': np.array([0.7]),
        'example_weights': np.array([1.0]),
    }
    example3 = {
        'labels': np.array([0.8]),
        'predictions': np.array([1.0]),
        'example_weights': np.array([1.0]),
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example1, example2, example3])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeHistogram' >> beam.CombinePerKey(histogram.combiner)
          | 'ComputeConfusionMatrix'
          >> beam.Map(lambda x: (x[0], matrix.result(x[1])))
          | 'ComputeMetric'
          >> beam.Map(lambda x: (x[0], derived_metric.result(x[1])))
      )

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          key = metric_types.MetricKey(name=metric.name, example_weighted=True)
          self.assertDictElementsAlmostEqual(
              got_metrics, {key: expected_value}, places=5)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  @parameterized.named_parameters(
      ('precision@2 (using sub_key)', confusion_matrix_metrics.Precision(), 2,
       1.6 / (1.6 + 3.2)),
      ('precision@2 (using param)', confusion_matrix_metrics.Precision(top_k=2),
       None, 1.6 / (1.6 + 3.2)),
      ('recall@2 (using sub_key)', confusion_matrix_metrics.Recall(), 2, 1.6 /
       (1.6 + 0.8)),
      ('recall@2 (using param)', confusion_matrix_metrics.Recall(top_k=2), None,
       1.6 / (1.6 + 0.8)),
      ('precision@3 (using sub_key)', confusion_matrix_metrics.Precision(), 3,
       1.9 / (1.9 + 5.3)),
      ('recall@3 (using sub_key)', confusion_matrix_metrics.Recall(), 3, 1.9 /
       (1.9 + 0.5)),
  )
  def testConfusionMatrixMetricsWithTopK(self, metric, top_k, expected_value):
    computations = metric.computations(
        sub_keys=[metric_types.SubKey(top_k=top_k)], example_weighted=True)
    histogram = computations[0]
    matrix = computations[1]
    derived_metric = computations[2]

    # top_k = 2
    #   TP = 0.5*0 + 0.7*1 + 0.9*1 + 0.3*0 = 1.6
    #   FP = 0.5*2 + 0.7*1 + 0.9*1 + 0.3*2 = 3.2
    #   FN = 0.5*1 + 0.7*0 + 0.9*0 + 0.3*1 = 0.8
    #
    # top_k = 3
    #   TP = 0.5*0 + 0.7*1 + 0.9*1 + 0.3*1 = 1.9
    #   FP = 0.5*3 + 0.7*2 + 0.9*2 + 0.3*2 = 5.3
    #   FN = 0.5*1 + 0.7*0 + 0.9*0 + 0.3*0 = 0.5
    example1 = {
        'labels': np.array([2]),
        'predictions': np.array([0.1, 0.2, 0.1, 0.25, 0.35]),
        'example_weights': np.array([0.5]),
    }
    example2 = {
        'labels': np.array([1]),
        'predictions': np.array([0.2, 0.3, 0.05, 0.15, 0.3]),
        'example_weights': np.array([0.7]),
    }
    example3 = {
        'labels': np.array([3]),
        'predictions': np.array([0.01, 0.2, 0.09, 0.5, 0.2]),
        'example_weights': np.array([0.9]),
    }
    example4 = {
        'labels': np.array([1]),
        'predictions': np.array([0.3, 0.2, 0.05, 0.4, 0.05]),
        'example_weights': np.array([0.3]),
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example1, example2, example3, example4])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeHistogram' >> beam.CombinePerKey(histogram.combiner)
          | 'ComputeConfusionMatrix'
          >> beam.Map(lambda x: (x[0], matrix.result(x[1])))
          | 'ComputeMetric'
          >> beam.Map(lambda x: (x[0], derived_metric.result(x[1])))
      )

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          if top_k:
            sub_key = metric_types.SubKey(top_k=top_k)
          else:
            sub_key = metric_types.SubKey(top_k=metric.get_config()['top_k'])
          key = metric_types.MetricKey(
              name=metric.name, sub_key=sub_key, example_weighted=True)
          self.assertDictElementsAlmostEqual(
              got_metrics, {key: expected_value}, places=5)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  @parameterized.named_parameters(
      ('precision (class_id=1 using sub_key)',
       confusion_matrix_metrics.Precision(thresholds=[0.1]), 1, 0.5 /
       (0.5 + 1.6)),
      ('precision (class_id=1 using param)',
       confusion_matrix_metrics.Precision(
           class_id=1, thresholds=[0.1]), None, 0.5 / (0.5 + 1.6)),
      ('recall (class_id=3 using sub_key)',
       confusion_matrix_metrics.Recall(thresholds=[0.1]), 3, 0.7 / (0.7 + 0.9)),
      ('recall (class_id=3 using param)',
       confusion_matrix_metrics.Recall(
           class_id=3, thresholds=[0.1]), None, 0.7 / (0.7 + 0.9)),
  )
  def testConfusionMatrixMetricsWithClassId(self, metric, class_id,
                                            expected_value):
    computations = metric.computations(
        sub_keys=[metric_types.SubKey(class_id=class_id)],
        example_weighted=True)
    histogram = computations[0]
    matrix = computations[1]
    derived_metric = computations[2]

    # class_id = 1, threshold = 0.1
    #   TP = 0.5*1 + 0.7*0 + 0.9*0 + 0.3*0 = 0.5
    #   FP = 0.5*0 + 0.7*1 + 0.9*1 + 0.3*0 = 1.6
    #   FN = 0.5*0 + 0.7*0 + 0.9*0 + 0.3*1 = 0.3
    #
    # class_id = 3, threshold = 0.1
    #   TP = 0.5*0 + 0.7*1 + 0.9*0 + 0.3*0 = 0.7
    #   FP = 0.5*1 + 0.7*0 + 0.9*0 + 0.3*1 = 0.8
    #   FN = 0.5*0 + 0.7*0 + 0.9*1 + 0.3*0 = 0.9
    example1 = {
        'labels': np.array([1]),
        'predictions': np.array([0.1, 0.2, 0.1, 0.25, 0.35]),
        'example_weights': np.array([0.5]),
    }
    example2 = {
        'labels': np.array([3]),
        'predictions': np.array([0.2, 0.3, 0.05, 0.15, 0.3]),
        'example_weights': np.array([0.7]),
    }
    example3 = {
        'labels': np.array([3]),
        'predictions': np.array([0.01, 0.2, 0.2, 0.09, 0.5]),
        'example_weights': np.array([0.9]),
    }
    example4 = {
        'labels': np.array([1]),
        'predictions': np.array([0.1, 0.05, 0.3, 0.4, 0.05]),
        'example_weights': np.array([0.3]),
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example1, example2, example3, example4])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeHistogram' >> beam.CombinePerKey(histogram.combiner)
          | 'ComputeConfusionMatrix'
          >> beam.Map(lambda x: (x[0], matrix.result(x[1])))
          | 'ComputeMetric'
          >> beam.Map(lambda x: (x[0], derived_metric.result(x[1])))
      )

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          if class_id:
            sub_key = metric_types.SubKey(class_id=class_id)
          else:
            sub_key = metric_types.SubKey(
                class_id=metric.get_config()['class_id'])
          key = metric_types.MetricKey(
              name=metric.name, sub_key=sub_key, example_weighted=True)
          self.assertDictElementsAlmostEqual(
              got_metrics, {key: expected_value}, places=5)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testConfusionMatrixMetricsWithNan(self):
    computations = confusion_matrix_metrics.Specificity().computations(
        example_weighted=True)
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
          | 'ComputeMatrices'
          >> beam.Map(lambda x: (x[0], matrices.result(x[1])))
          | 'ComputeMetrics' >> beam.Map(lambda x: (x[0], metrics.result(x[1])))
      )

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

  @parameterized.named_parameters(
      ('class_id as param and class_id as sub_key',
       confusion_matrix_metrics.Precision(class_id=2), 2, None),
      ('top_k as param and top_k as sub_key',
       confusion_matrix_metrics.Precision(top_k=2), None, 2),
  )
  def testRaisesErrorIfOverlappingSettings(self, metric, class_id, top_k):
    with self.assertRaisesRegex(ValueError,
                                '.*is configured with overlapping settings.*'):
      metric.computations(
          sub_keys=[metric_types.SubKey(class_id=class_id, top_k=top_k)])

  def testConfusionMatrixAtThresholds(self):
    computations = confusion_matrix_metrics.ConfusionMatrixAtThresholds(
        thresholds=[0.3, 0.5, 0.8]).computations(example_weighted=True)
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
          | 'ComputeMatrices'
          >> beam.Map(lambda x: (x[0], matrices.result(x[1])))
          | 'ComputeMetrics' >> beam.Map(lambda x: (x[0], metrics.result(x[1])))
      )

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          self.assertLen(got_metrics, 1)
          key = metric_types.MetricKey(
              name='confusion_matrix_at_thresholds', example_weighted=True)
          self.assertIn(key, got_metrics)
          got_metric = got_metrics[key]
          self.assertEqual(
              binary_confusion_matrices.Matrices(
                  thresholds=[0.3, 0.5, 0.8],
                  tp=[1.0, 1.0, 1.0],
                  tn=[1.0, 2.0, 2.0],
                  fp=[1.0, 0.0, 0.0],
                  fn=[1.0, 1.0, 1.0]), got_metric)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  @parameterized.named_parameters(
      (
          'precision (class_id=1 top_k=2 using sub_key)',
          confusion_matrix_metrics.Precision(thresholds=[0.1]),
          1,
          2,
          0.5 / (0.5 + 1.6),
      ),
      (
          'precision (class_id=1 using param and top_k=2 using sub_key)',
          confusion_matrix_metrics.Precision(class_id=1, thresholds=[0.1]),
          None,
          2,
          0.5 / (0.5 + 1.6),
      ),
      (
          'recall (class_id=3 using sub_key and top_k=2 using param)',
          confusion_matrix_metrics.Recall(thresholds=[0.1], top_k=2),
          3,
          None,
          0.7 / (0.7 + 0.9),
      ),
      (
          'recall (class_id=3 top_k=2 using param)',
          confusion_matrix_metrics.Recall(
              class_id=3, top_k=2, thresholds=[0.1]
          ),
          None,
          None,
          0.7 / (0.7 + 0.9),
      ),
  )
  def testConfusionMatrixMetricsWithClassIdAndTopK(
      self, metric, class_id, top_k, expected_value
  ):
    computations = metric.computations(
        sub_keys=[metric_types.SubKey(class_id=class_id, top_k=top_k)],
        example_weighted=True,
    )
    histogram = computations[0]
    matrix = computations[1]
    derived_metric = computations[2]

    # class_id = 1, top_k=2
    #   TP = 0.5*1 + 0.7*0 + 0.9*0 + 0.3*0 = 0.5
    #   FP = 0.5*0 + 0.7*1 + 0.9*1 + 0.3*0 = 1.6
    #   FN = 0.5*0 + 0.7*0 + 0.9*0 + 0.3*1 = 0.3
    #
    # class_id = 3, top_k=2
    #   TP = 0.5*0 + 0.7*1 + 0.9*0 + 0.3*0 = 0.7
    #   FP = 0.5*1 + 0.7*0 + 0.9*0 + 0.3*1 = 0.8
    #   FN = 0.5*0 + 0.7*0 + 0.9*1 + 0.3*0 = 0.9
    example1 = {
        'labels': np.array([1]),
        'predictions': np.array([0.1, 0.5, 0.1, 0.45, 0.35]),
        'example_weights': np.array([0.5]),
    }
    example2 = {
        'labels': np.array([3]),
        'predictions': np.array([0.2, 0.3, 0.05, 0.31, 0.3]),
        'example_weights': np.array([0.7]),
    }
    example3 = {
        'labels': np.array([3]),
        'predictions': np.array([0.01, 0.2, 0.2, 0.09, 0.5]),
        'example_weights': np.array([0.9]),
    }
    example4 = {
        'labels': np.array([1]),
        'predictions': np.array([0.1, 0.05, 0.3, 0.4, 0.05]),
        'example_weights': np.array([0.3]),
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example1, example2, example3, example4])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeHistogram' >> beam.CombinePerKey(histogram.combiner)
          | 'ComputeConfusionMatrix'
          >> beam.Map(lambda x: (x[0], matrix.result(x[1])))
          | 'ComputeMetric'
          >> beam.Map(lambda x: (x[0], derived_metric.result(x[1])))
      )

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          subkey_class_id = class_id or metric.get_config()['class_id']
          subkey_top_k = top_k or metric.get_config()['top_k']
          sub_key = metric_types.SubKey(
              class_id=subkey_class_id, top_k=subkey_top_k
          )
          key = metric_types.MetricKey(
              name=metric.name, sub_key=sub_key, example_weighted=True
          )
          self.assertDictElementsAlmostEqual(
              got_metrics, {key: expected_value}, places=5
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  @parameterized.named_parameters(
      (
          'false_positives',
          confusion_matrix_metrics.FalsePositiveFeatureSampler(
              threshold=0.5, feature_key='example_id', sample_size=2
          ),
          'false_positive_feature_sampler',
          np.array(['example1', 'example2'], dtype=str),
      ),
      (
          'false_negatives',
          confusion_matrix_metrics.FalseNegativeFeatureSampler(
              threshold=0.5, feature_key='example_id', sample_size=2
          ),
          'false_negative_feature_sampler',
          np.array(['example3', 'example4'], dtype=str),
      ),
  )
  def testConfusionMatrixFeatureSamplers(
      self, metric, expected_metric_name, expected_value
  ):
    # false positive
    example1 = {
        'labels': np.array([0.0]),
        'predictions': np.array([1.0]),
        'example_weights': np.array([1.0]),
        'features': {'example_id': np.array(['example1'])},
    }
    # false positive
    example2 = {
        'labels': np.array([0.0]),
        'predictions': np.array([1.0]),
        'example_weights': np.array([1.0]),
        'features': {'example_id': np.array(['example2'])},
    }
    # false negative
    example3 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.0]),
        'example_weights': np.array([1.0]),
        'features': {'example_id': np.array(['example3'])},
    }
    # false negative
    example4 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.0]),
        'example_weights': np.array([1.0]),
        'features': {'example_id': np.array(['example4'])},
    }

    expected_metrics = {
        metric_types.MetricKey(
            name=expected_metric_name, example_weighted=True
        ): expected_value,
    }
    self.assertDerivedMetricsEqual(
        expected_metrics=expected_metrics,
        extracts=[example1, example2, example3, example4],
        metric=metric,
        enable_debug_print=True,
    )


