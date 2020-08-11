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
"""Tests for multi-class confusion matrix plot at thresholds."""

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
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.metrics import multi_class_confusion_matrix_plot


class MultiClassConfusionMatrixPlotTest(testutil.TensorflowModelAnalysisTest,
                                        parameterized.TestCase):

  def testMultiClassConfusionMatrixPlot(self):
    computations = (
        multi_class_confusion_matrix_plot.MultiClassConfusionMatrixPlot()
        .computations())
    matrices = computations[0]
    plot = computations[1]

    example1 = {
        'labels': np.array([2.0]),
        'predictions': np.array([0.2, 0.3, 0.5]),
        'example_weights': np.array([0.5])
    }
    example2 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.1, 0.4, 0.5]),
        'example_weights': np.array([1.0])
    }
    example3 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.3, 0.2, 0.5]),
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
        'predictions': np.array([0.3, 0.2, 0.5]),
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
          | 'ComputePlot' >> beam.Map(lambda x: (x[0], plot.result(x[1]))))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_plots = got[0]
          self.assertEqual(got_slice_key, ())
          self.assertLen(got_plots, 1)
          key = metric_types.PlotKey(name='multi_class_confusion_matrix_plot')
          got_matrix = got_plots[key]
          self.assertProtoEquals(
              """
              matrices {
                threshold: 0.0
                entries {
                  actual_class_id: 0
                  predicted_class_id: 2
                  num_weighted_examples: 1.0
                }
                entries {
                  actual_class_id: 1
                  predicted_class_id: 1
                  num_weighted_examples: 2.0
                }
                entries {
                  actual_class_id: 1
                  predicted_class_id: 2
                  num_weighted_examples: 0.25
                }
                entries {
                  actual_class_id: 2
                  predicted_class_id: 2
                  num_weighted_examples: 1.5
                }
              }
          """, got_matrix)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  @parameterized.named_parameters(('using_num_thresholds', {
      'num_thresholds': 3
  }), ('using_thresholds', {
      'thresholds': [0.0, 0.5, 1.0]
  }))
  def testMultiClassConfusionMatrixPlotWithThresholds(self, kwargs):
    computations = (
        multi_class_confusion_matrix_plot.MultiClassConfusionMatrixPlot(
            **kwargs).computations())
    matrices = computations[0]
    plot = computations[1]

    example1 = {
        'labels': np.array([2.0]),
        'predictions': np.array([0.2, 0.35, 0.45]),
        'example_weights': np.array([1.0])
    }
    example2 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.1, 0.35, 0.55]),
        'example_weights': np.array([1.0])
    }
    example3 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.3, 0.25, 0.45]),
        'example_weights': np.array([1.0])
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
        'predictions': np.array([0.3, 0.25, 0.45]),
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
          | 'ComputePlot' >> beam.Map(lambda x: (x[0], plot.result(x[1]))))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_plots = got[0]
          self.assertEqual(got_slice_key, ())
          self.assertLen(got_plots, 1)
          key = metric_types.PlotKey(name='multi_class_confusion_matrix_plot')
          got_matrix = got_plots[key]
          self.assertProtoEquals(
              """
              matrices {
                threshold: 0.0
                entries {
                  actual_class_id: 0
                  predicted_class_id: 2
                  num_weighted_examples: 1.0
                }
                entries {
                  actual_class_id: 1
                  predicted_class_id: 1
                  num_weighted_examples: 2.0
                }
                entries {
                  actual_class_id: 1
                  predicted_class_id: 2
                  num_weighted_examples: 1.0
                }
                entries {
                  actual_class_id: 2
                  predicted_class_id: 2
                  num_weighted_examples: 2.0
                }
              }
              matrices {
                threshold: 0.5
                entries {
                  actual_class_id: 0
                  predicted_class_id: 2
                  num_weighted_examples: 1.0
                }
                entries {
                  actual_class_id: 1
                  predicted_class_id: -1
                  num_weighted_examples: 1.0
                }
                entries {
                  actual_class_id: 1
                  predicted_class_id: 1
                  num_weighted_examples: 2.0
                }
                entries {
                  actual_class_id: 2
                  predicted_class_id: -1
                  num_weighted_examples: 2.0
                }
              }
              matrices {
                threshold: 1.0
                entries {
                  predicted_class_id: -1
                  num_weighted_examples: 1.0
                }
                entries {
                  actual_class_id: 1
                  predicted_class_id: -1
                  num_weighted_examples: 3.0
                }
                entries {
                  actual_class_id: 2
                  predicted_class_id: -1
                  num_weighted_examples: 2.0
               }
             }
          """, got_matrix)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testMultiClassConfusionMatrixPlotWithStringLabels(self):
    computations = (
        multi_class_confusion_matrix_plot.MultiClassConfusionMatrixPlot()
        .computations())
    matrices = computations[0]
    plot = computations[1]

    # Examples from b/149558504.
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
          | 'ComputePlot' >> beam.Map(lambda x: (x[0], plot.result(x[1]))))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_plots = got[0]
          self.assertEqual(got_slice_key, ())
          self.assertLen(got_plots, 1)
          key = metric_types.PlotKey(name='multi_class_confusion_matrix_plot')
          got_matrix = got_plots[key]
          self.assertProtoEquals(
              """
              matrices {
                threshold: 0.0
                entries {
                  actual_class_id: 0
                  predicted_class_id: 0
                  num_weighted_examples: 0.5
                }
                entries {
                  actual_class_id: 2
                  predicted_class_id: 2
                  num_weighted_examples: 1.0
                }
              }
          """, got_matrix)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')


if __name__ == '__main__':
  tf.test.main()
