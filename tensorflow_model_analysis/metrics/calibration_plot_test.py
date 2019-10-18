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
"""Tests for calibration plot."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.metrics import calibration_plot
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util


class CalibrationPlotTest(testutil.TensorflowModelAnalysisTest):

  def testCalibrationPlot(self):
    computations = calibration_plot.CalibrationPlot(
        num_buckets=10).computations()
    histogram = computations[0]
    plot = computations[1]

    example1 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.2]),
        'example_weights': np.array([1.0])
    }
    example2 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.8]),
        'example_weights': np.array([2.0])
    }
    example3 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.5]),
        'example_weights': np.array([3.0])
    }
    example4 = {
        'labels': np.array([1.0]),
        'predictions': np.array([-0.1]),
        'example_weights': np.array([4.0])
    }
    example5 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.5]),
        'example_weights': np.array([5.0])
    }
    example6 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.8]),
        'example_weights': np.array([6.0])
    }
    example7 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.2]),
        'example_weights': np.array([7.0])
    }
    example8 = {
        'labels': np.array([1.0]),
        'predictions': np.array([1.1]),
        'example_weights': np.array([8.0])
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([
              example1, example2, example3, example4, example5, example6,
              example7, example8
          ])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeHistogram' >> beam.CombinePerKey(histogram.combiner)
          | 'ComputePlot' >> beam.Map(lambda x: (x[0], plot.result(x[1]))))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_plots = got[0]
          self.assertEqual(got_slice_key, ())
          self.assertLen(got_plots, 1)
          key = metric_types.PlotKey(name='calibration_plot')
          self.assertIn(key, got_plots)
          got_plot = got_plots[key]
          self.assertProtoEquals(
              """
              buckets {
                lower_threshold_inclusive: -inf
                upper_threshold_exclusive: 0.0
                total_weighted_label {
                  value: 4.0
                }
                total_weighted_refined_prediction {
                  value: -0.4
                }
                num_weighted_examples {
                  value: 4.0
                }
              }
              buckets {
                lower_threshold_inclusive: 0.0
                upper_threshold_exclusive: 0.1
                total_weighted_label {
                }
                total_weighted_refined_prediction {
                }
                num_weighted_examples {
                }
              }
              buckets {
                lower_threshold_inclusive: 0.1
                upper_threshold_exclusive: 0.2
                total_weighted_label {
                }
                total_weighted_refined_prediction {
                }
                num_weighted_examples {
                }
              }
              buckets {
                lower_threshold_inclusive: 0.2
                upper_threshold_exclusive: 0.3
                total_weighted_label {
                }
                total_weighted_refined_prediction {
                  value: 1.6
                }
                num_weighted_examples {
                  value: 8.0
                }
              }
              buckets {
                lower_threshold_inclusive: 0.3
                upper_threshold_exclusive: 0.4
                total_weighted_label {
                }
                total_weighted_refined_prediction {
                }
                num_weighted_examples {
                }
              }
              buckets {
                lower_threshold_inclusive: 0.4
                upper_threshold_exclusive: 0.5
                total_weighted_label {
                }
                total_weighted_refined_prediction {
                }
                num_weighted_examples {
                }
              }
              buckets {
                lower_threshold_inclusive: 0.5
                upper_threshold_exclusive: 0.6
                total_weighted_label {
                  value: 5.0
                }
                total_weighted_refined_prediction {
                  value: 4.0
                }
                num_weighted_examples {
                  value: 8.0
                }
              }
              buckets {
                lower_threshold_inclusive: 0.6
                upper_threshold_exclusive: 0.7
                total_weighted_label {
                }
                total_weighted_refined_prediction {
                }
                num_weighted_examples {
                }
              }
              buckets {
                lower_threshold_inclusive: 0.7
                upper_threshold_exclusive: 0.8
                total_weighted_label {
                }
                total_weighted_refined_prediction {
                }
                num_weighted_examples {
                }
              }
              buckets {
                lower_threshold_inclusive: 0.8
                upper_threshold_exclusive: 0.9
                total_weighted_label {
                  value: 8.0
                }
                total_weighted_refined_prediction {
                  value: 6.4
                }
                num_weighted_examples {
                  value: 8.0
                }
              }
              buckets {
                lower_threshold_inclusive: 0.9
                upper_threshold_exclusive: 1.0
                total_weighted_label {
                }
                total_weighted_refined_prediction {
                }
                num_weighted_examples {
                }
              }
              buckets {
                lower_threshold_inclusive: 1.0
                upper_threshold_exclusive: inf
                total_weighted_label {
                  value: 8.0
                }
                total_weighted_refined_prediction {
                  value: 8.8
                }
                num_weighted_examples {
                  value: 8.0
                }
              }
          """, got_plot)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')


if __name__ == '__main__':
  tf.test.main()
