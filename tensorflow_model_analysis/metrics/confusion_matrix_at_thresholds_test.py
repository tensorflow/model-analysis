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

import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.metrics import confusion_matrix_at_thresholds
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util


class ConfusionMatrixAtThresholdsTest(testutil.TensorflowModelAnalysisTest):

  def testConfusionMatrixAtThresholds(self):
    computations = confusion_matrix_at_thresholds.ConfusionMatrixAtThresholds(
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
                false_positives: 2.0
                true_positives: 2.0
                precision: 0.5
                recall: 1.0
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
                false_negatives: 2.0
                true_negatives: 2.0
                true_positives: 1.0
                precision: 1.0
                recall: 0.3333333
              }
          """, got_metric)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')


if __name__ == '__main__':
  tf.test.main()
