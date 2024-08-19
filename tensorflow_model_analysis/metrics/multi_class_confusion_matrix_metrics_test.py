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
"""Tests for multi-class confusion matrix metrics at thresholds."""

from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.metrics import multi_class_confusion_matrix_metrics
from tensorflow_model_analysis.utils import test_util


class MultiClassConfusionMatrixMetricsTest(
    test_util.TensorflowModelAnalysisTest, parameterized.TestCase
):

  @parameterized.named_parameters(
      {
          'testcase_name': '_empty_thresholds',
          'left': multi_class_confusion_matrix_metrics.Matrices({}),
          'right': multi_class_confusion_matrix_metrics.Matrices({}),
          'expected': multi_class_confusion_matrix_metrics.Matrices({})
      }, {
          'testcase_name': '_empty_entries',
          'left': multi_class_confusion_matrix_metrics.Matrices({0.5: {}}),
          'right': multi_class_confusion_matrix_metrics.Matrices({0.5: {}}),
          'expected': multi_class_confusion_matrix_metrics.Matrices({0.5: {}})
      }, {
          'testcase_name':
              '_different_thresholds',
          'left':
              multi_class_confusion_matrix_metrics.Matrices({
                  0.5: {
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=0, predicted_class_id=0):
                          1.0
                  }
              }),
          'right':
              multi_class_confusion_matrix_metrics.Matrices({
                  0.75: {
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=0, predicted_class_id=0):
                          2.0
                  }
              }),
          'expected':
              multi_class_confusion_matrix_metrics.Matrices({
                  0.5: {
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=0, predicted_class_id=0):
                          1.0
                  },
                  0.75: {
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=0, predicted_class_id=0):
                          2.0
                  }
              }),
      }, {
          'testcase_name':
              '_different_entries',
          'left':
              multi_class_confusion_matrix_metrics.Matrices({
                  0.5: {
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=0, predicted_class_id=0):
                          1.0
                  }
              }),
          'right':
              multi_class_confusion_matrix_metrics.Matrices({
                  0.5: {
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=0, predicted_class_id=1):
                          2.0
                  }
              }),
          'expected':
              multi_class_confusion_matrix_metrics.Matrices({
                  0.5: {
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=0, predicted_class_id=0):
                          1.0,
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=0, predicted_class_id=1):
                          2.0
                  }
              }),
      }, {
          'testcase_name':
              '_same_thresholds_and_entries',
          'left':
              multi_class_confusion_matrix_metrics.Matrices({
                  0.5: {
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=0, predicted_class_id=0):
                          1.0,
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=0, predicted_class_id=1):
                          2.0,
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=1, predicted_class_id=0):
                          3.0,
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=1, predicted_class_id=1):
                          4.0,
                  },
                  0.75: {
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=0, predicted_class_id=0):
                          2.0,
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=0, predicted_class_id=1):
                          4.0,
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=1, predicted_class_id=0):
                          6.0,
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=1, predicted_class_id=1):
                          8.0,
                  }
              }),
          'right':
              multi_class_confusion_matrix_metrics.Matrices({
                  0.5: {
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=0, predicted_class_id=0):
                          1.0,
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=0, predicted_class_id=1):
                          3.0,
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=1, predicted_class_id=0):
                          5.0,
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=1, predicted_class_id=1):
                          7.0,
                  },
                  0.75: {
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=0, predicted_class_id=0):
                          2.0,
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=0, predicted_class_id=1):
                          6.0,
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=1, predicted_class_id=0):
                          10.0,
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=1, predicted_class_id=1):
                          14.0,
                  }
              }),
          'expected':
              multi_class_confusion_matrix_metrics.Matrices({
                  0.5: {
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=0, predicted_class_id=0):
                          2.0,
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=0, predicted_class_id=1):
                          5.0,
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=1, predicted_class_id=0):
                          8.0,
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=1, predicted_class_id=1):
                          11.0,
                  },
                  0.75: {
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=0, predicted_class_id=0):
                          4.0,
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=0, predicted_class_id=1):
                          10.0,
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=1, predicted_class_id=0):
                          16.0,
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=1, predicted_class_id=1):
                          22.0,
                  }
              }),
      }, {
          'testcase_name': '_empty_thresholds_broadcast',
          'left': multi_class_confusion_matrix_metrics.Matrices({}),
          'right': 1.0,
          'expected': multi_class_confusion_matrix_metrics.Matrices({})
      }, {
          'testcase_name': '_empty_entries_broadcast',
          'left': multi_class_confusion_matrix_metrics.Matrices({0.5: {}}),
          'right': 1.0,
          'expected': multi_class_confusion_matrix_metrics.Matrices({0.5: {}})
      }, {
          'testcase_name':
              '_nonempty_thresholds_and_entries_broadcast',
          'left':
              multi_class_confusion_matrix_metrics.Matrices({
                  0.5: {
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=0, predicted_class_id=0):
                          1.0,
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=0, predicted_class_id=1):
                          2.0,
                  },
              }),
          'right':
              3.0,
          'expected':
              multi_class_confusion_matrix_metrics.Matrices({
                  0.5: {
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=0, predicted_class_id=0):
                          4.0,
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=0, predicted_class_id=1):
                          5.0,
                  },
              }),
      })
  def testAddMatrices(self, left, right, expected):
    self.assertEqual(expected, left + right)

  @parameterized.named_parameters(('using_default_thresholds', {}),
                                  ('setting_thresholds', {
                                      'thresholds': [0.5]
                                  }))
  def testMultiClassConfusionMatrixAtThresholds(self, kwargs):
    computations = (
        multi_class_confusion_matrix_metrics
        .MultiClassConfusionMatrixAtThresholds(**kwargs).computations(
            example_weighted=True))
    matrices = computations[0]
    metrics = computations[1]

    example1 = {
        'labels': np.array([2.0]),
        'predictions': np.array([0.2, 0.3, 0.5]),
        'example_weights': np.array([0.5])
    }
    example2 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.1, 0.3, 0.6]),
        'example_weights': np.array([1.0])
    }
    example3 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.3, 0.1, 0.6]),
        'example_weights': np.array([0.25])
    }
    example4 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.1, 0.9, 0.0]),
        'example_weights': np.array([1.0])
    }
    example5 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.1, 0.8, 0.1]),
        'example_weights': np.array([1.0])
    }
    example6 = {
        'labels': np.array([2.0]),
        'predictions': np.array([0.3, 0.1, 0.6]),
        'example_weights': np.array([1.0])
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create(
              [example1, example2, example3, example4, example5, example6])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeMatrices' >> beam.CombinePerKey(matrices.combiner)
          |
          'ComputeMetrics' >> beam.Map(lambda x: (x[0], metrics.result(x[1]))))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          self.assertLen(got_metrics, 1)
          key = metric_types.MetricKey(
              name='multi_class_confusion_matrix_at_thresholds',
              example_weighted=True)
          got_matrix = got_metrics[key]
          self.assertEqual(
              multi_class_confusion_matrix_metrics.Matrices({
                  0.5: {
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=0, predicted_class_id=2):
                          1.0,
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=1, predicted_class_id=1):
                          2.0,
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=1, predicted_class_id=2):
                          0.25,
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=2, predicted_class_id=-1):
                          0.5,
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=2, predicted_class_id=2):
                          1.0
                  }
              }), got_matrix)
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testMultiClassConfusionMatrixAtThresholdsWithStringLabels(self):
    computations = (
        multi_class_confusion_matrix_metrics
        .MultiClassConfusionMatrixAtThresholds().computations(
            example_weighted=True))
    matrices = computations[0]
    metrics = computations[1]

    example1 = {
        'labels': np.array([['unacc']]),
        'predictions': {
            'probabilities':
                np.array([[
                    1.0000000e+00, 6.9407083e-24, 2.7419115e-38, 0.0000000e+00
                ]]),
            'all_classes':
                np.array([['unacc', 'acc', 'vgood', 'good']]),
        },
        'example_weights': np.array([0.5])
    }
    example2 = {
        'labels': np.array([['vgood']]),
        'predictions': {
            'probabilities': np.array([[0.2, 0.3, 0.4, 0.1]]),
            'all_classes': np.array([['unacc', 'acc', 'vgood', 'good']]),
        },
        'example_weights': np.array([1.0])
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example1, example2])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeMatrices' >> beam.CombinePerKey(matrices.combiner)
          |
          'ComputeMetrics' >> beam.Map(lambda x: (x[0], metrics.result(x[1]))))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          self.assertLen(got_metrics, 1)
          key = metric_types.MetricKey(
              name='multi_class_confusion_matrix_at_thresholds',
              example_weighted=True)
          got_matrix = got_metrics[key]
          self.assertEqual(
              multi_class_confusion_matrix_metrics.Matrices({
                  0.5: {
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=0, predicted_class_id=0):
                          0.5,
                      multi_class_confusion_matrix_metrics.MatrixEntryKey(
                          actual_class_id=2, predicted_class_id=-1):
                          1.0
                  }
              }), got_matrix)
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')


