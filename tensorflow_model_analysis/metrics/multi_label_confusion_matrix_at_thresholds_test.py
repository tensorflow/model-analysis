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
"""Tests for multi-label confusion matrix at thresholds."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.metrics import multi_label_confusion_matrix_at_thresholds


class MultiLabelConfusionMatrixAtThresholdsTest(
    testutil.TensorflowModelAnalysisTest):

  def testMultiLabelConfusionMatrixAtThresholds(self):
    computation = (
        multi_label_confusion_matrix_at_thresholds
        .MultiLabelConfusionMatrixAtThresholds().computations()[0])

    example1 = {
        'labels': np.array([1.0, 1.0, 0.0]),
        'predictions': np.array([0.7, 0.5, 0.2]),
        'example_weights': np.array([1.0])
    }
    example2 = {
        'labels': np.array([0.0, 1.0, 0.0]),
        'predictions': np.array([0.3, 0.6, 0.1]),
        'example_weights': np.array([1.0])
    }
    example3 = {
        'labels': np.array([0.0, 0.0, 0.0]),
        'predictions': np.array([0.2, 0.4, 0.5]),
        'example_weights': np.array([1.0])
    }
    example4 = {
        'labels': np.array([1.0, 0.0, 0.0]),
        'predictions': np.array([1.0, 0.4, 0.1]),
        'example_weights': np.array([1.0])
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example1, example2, example3, example4])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputePlot' >> beam.CombinePerKey(computation.combiner))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_plots = got[0]
          self.assertEqual(got_slice_key, ())
          self.assertLen(got_plots, 1)
          key = metric_types.PlotKey(
              name='multi_label_confusion_matrix_at_thresholds')
          got_matrix = got_plots[key]
          self.assertProtoEquals(
              """
              matrices {
                threshold: 0.5
                entries {
                  actual_class_id: 0
                  predicted_class_id: 0
                  false_negatives: 0.0
                  true_negatives: 0.0
                  false_positives: 0.0
                  true_positives: 2.0
                }
                entries {
                  actual_class_id: 0
                  predicted_class_id: 1
                  false_negatives: 1.0
                  true_negatives: 1.0
                  false_positives: 0.0
                  true_positives: 0.0
                }
                entries {
                  actual_class_id: 0
                  predicted_class_id: 2
                  false_negatives: 0.0
                  true_negatives: 2.0
                  false_positives: 0.0
                  true_positives: 0.0
                }
                entries {
                  actual_class_id: 1
                  predicted_class_id: 0
                  false_negatives: 0.0
                  true_negatives: 1.0
                  false_positives: 0.0
                  true_positives: 1.0
                }
                entries {
                  actual_class_id: 1
                  predicted_class_id: 1
                  false_negatives: 1.0
                  true_negatives: 0.0
                  false_positives: 0.0
                  true_positives: 1.0
                }
                entries {
                  actual_class_id: 1
                  predicted_class_id: 2
                  false_negatives: 0.0
                  false_positives: 0.0
                  true_negatives: 2.0
                  true_positives: 0.0
                }
              }
          """, got_matrix)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testMultiLabelConfusionMatrixAtThresholdsWithThresholds(self):
    computation = (
        multi_label_confusion_matrix_at_thresholds
        .MultiLabelConfusionMatrixAtThresholds(
            thresholds=[0.2, 0.5]).computations()[0])

    example1 = {
        'labels': np.array([1.0, 1.0, 0.0]),
        'predictions': np.array([0.7, 0.5, 0.2]),
        'example_weights': np.array([0.25])
    }
    example2 = {
        'labels': np.array([0.0, 1.0, 0.0]),
        'predictions': np.array([0.3, 0.6, 0.1]),
        'example_weights': np.array([0.5])
    }
    example3 = {
        'labels': np.array([0.0, 0.0, 0.0]),
        'predictions': np.array([0.2, 0.4, 0.5]),
        'example_weights': np.array([0.75])
    }
    example4 = {
        'labels': np.array([1.0, 0.0, 0.0]),
        'predictions': np.array([1.0, 0.4, 0.1]),
        'example_weights': np.array([1.0])
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example1, example2, example3, example4])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputePlot' >> beam.CombinePerKey(computation.combiner))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_plots = got[0]
          self.assertEqual(got_slice_key, ())
          self.assertLen(got_plots, 1)
          key = metric_types.PlotKey(
              name='multi_label_confusion_matrix_at_thresholds')
          got_matrix = got_plots[key]
          self.assertProtoEquals(
              """
              matrices {
                threshold: 0.2
                entries {
                  actual_class_id: 0
                  predicted_class_id: 0
                  false_negatives: 0.0
                  true_negatives: 0.0
                  false_positives: 0.0
                  true_positives: 1.25
                }
                entries {
                  actual_class_id: 0
                  predicted_class_id: 1
                  false_negatives: 0.0
                  true_negatives: 0.0
                  false_positives: 1.0
                  true_positives: 0.25
                }
                entries {
                  actual_class_id: 0
                  predicted_class_id: 2
                  false_negatives: 0.0
                  true_negatives: 1.25
                  false_positives: 0.0
                  true_positives: 0.0
                }
                entries {
                  actual_class_id: 1
                  predicted_class_id: 0
                  false_negatives: 0.0
                  true_negatives: 0.0
                  false_positives: 0.5
                  true_positives: 0.25
                }
                entries {
                  actual_class_id: 1
                  predicted_class_id: 1
                  false_negatives: 0.0
                  true_negatives: 0.0
                  false_positives: 0.0
                  true_positives: 0.75
                }
                entries {
                  actual_class_id: 1
                  predicted_class_id: 2
                  false_negatives: 0.0
                  false_positives: 0.0
                  true_negatives: 0.75
                  true_positives: 0.0
                }
              }
              matrices {
                threshold: 0.5
                entries {
                  actual_class_id: 0
                  predicted_class_id: 0
                  false_negatives: 0.0
                  true_negatives: 0.0
                  false_positives: 0.0
                  true_positives: 1.25
                }
                entries {
                  actual_class_id: 0
                  predicted_class_id: 1
                  false_negatives: 0.25
                  true_negatives: 1.0
                  false_positives: 0.0
                  true_positives: 0.0
                }
                entries {
                  actual_class_id: 0
                  predicted_class_id: 2
                  false_negatives: 0.0
                  true_negatives: 1.25
                  false_positives: 0.0
                  true_positives: 0.0
                }
                entries {
                  actual_class_id: 1
                  predicted_class_id: 0
                  false_negatives: 0.0
                  true_negatives: 0.5
                  false_positives: 0.0
                  true_positives: 0.25
                }
                entries {
                  actual_class_id: 1
                  predicted_class_id: 1
                  false_negatives: 0.25
                  true_negatives: 0.0
                  false_positives: 0.0
                  true_positives: 0.5
                }
                entries {
                  actual_class_id: 1
                  predicted_class_id: 2
                  false_negatives: 0.0
                  false_positives: 0.0
                  true_negatives: 0.75
                  true_positives: 0.0
                }
              }
          """, got_matrix)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')


if __name__ == '__main__':
  tf.test.main()
