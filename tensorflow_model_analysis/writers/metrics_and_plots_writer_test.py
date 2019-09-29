# Copyright 2018 Google LLC
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
"""Test for using the MetricsAndPlotsWriter API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

# Standard Imports

import apache_beam as beam
import tensorflow as tf
from tensorflow_model_analysis import config
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.eval_saved_model.example_trainers import fixed_prediction_estimator
from tensorflow_model_analysis.evaluators import metrics_and_plots_evaluator
from tensorflow_model_analysis.extractors import predict_extractor
from tensorflow_model_analysis.extractors import slice_key_extractor
from tensorflow_model_analysis.post_export_metrics import post_export_metrics
from tensorflow_model_analysis.proto import metrics_for_slice_pb2
from tensorflow_model_analysis.writers import metrics_and_plots_writer

from google.protobuf import text_format


class MetricsAndPlotsWriterTest(testutil.TensorflowModelAnalysisTest):

  def setUp(self):
    super(MetricsAndPlotsWriterTest, self).setUp()
    self.longMessage = True  # pylint: disable=invalid-name

  def _getTempDir(self):
    return tempfile.mkdtemp()

  def testWriteMetricsAndPlots(self):
    metrics_file = os.path.join(self._getTempDir(), 'metrics')
    plots_file = os.path.join(self._getTempDir(), 'plots')
    temp_eval_export_dir = os.path.join(self._getTempDir(), 'eval_export_dir')

    _, eval_export_dir = (
        fixed_prediction_estimator.simple_fixed_prediction_estimator(
            None, temp_eval_export_dir))
    eval_config = config.EvalConfig(
        input_data_specs=[config.InputDataSpec()],
        model_specs=[config.ModelSpec()],
        output_data_specs=[
            config.OutputDataSpec(disabled_outputs=['eval_config.json'])
        ])
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=eval_export_dir,
        add_metrics_callbacks=[
            post_export_metrics.example_count(),
            post_export_metrics.calibration_plot_and_prediction_histogram(
                num_buckets=2)
        ])
    extractors = [
        predict_extractor.PredictExtractor(eval_shared_model),
        slice_key_extractor.SliceKeyExtractor()
    ]
    evaluators = [
        metrics_and_plots_evaluator.MetricsAndPlotsEvaluator(eval_shared_model)
    ]
    output_paths = {
        constants.METRICS_KEY: metrics_file,
        constants.PLOTS_KEY: plots_file
    }
    writers = [
        metrics_and_plots_writer.MetricsAndPlotsWriter(eval_shared_model,
                                                       output_paths)
    ]

    with beam.Pipeline() as pipeline:
      example1 = self._makeExample(prediction=0.0, label=1.0)
      example2 = self._makeExample(prediction=1.0, label=1.0)

      # pylint: disable=no-value-for-parameter
      _ = (
          pipeline
          | 'Create' >> beam.Create([
              example1.SerializeToString(),
              example2.SerializeToString(),
          ])
          | 'ExtractEvaluateAndWriteResults' >>
          model_eval_lib.ExtractEvaluateAndWriteResults(
              eval_config=eval_config,
              eval_shared_models=[eval_shared_model],
              extractors=extractors,
              evaluators=evaluators,
              writers=writers))
      # pylint: enable=no-value-for-parameter

    expected_metrics_for_slice = text_format.Parse(
        """
        slice_key {}
        metrics {
          key: "average_loss"
          value {
            double_value {
              value: 0.5
            }
          }
        }
        metrics {
          key: "post_export_metrics/example_count"
          value {
            double_value {
              value: 2.0
            }
          }
        }
        """, metrics_for_slice_pb2.MetricsForSlice())

    metric_records = []
    for record in tf.compat.v1.python_io.tf_record_iterator(metrics_file):
      metric_records.append(
          metrics_for_slice_pb2.MetricsForSlice.FromString(record))
    self.assertEqual(1, len(metric_records), 'metrics: %s' % metric_records)
    self.assertProtoEquals(expected_metrics_for_slice, metric_records[0])

    expected_plots_for_slice = text_format.Parse(
        """
      slice_key {}
      plots {
        key: "post_export_metrics"
        value {
          calibration_histogram_buckets {
            buckets {
              lower_threshold_inclusive: -inf
              num_weighted_examples {}
              total_weighted_label {}
              total_weighted_refined_prediction {}
            }
            buckets {
              upper_threshold_exclusive: 0.5
              num_weighted_examples {
                value: 1.0
              }
              total_weighted_label {
                value: 1.0
              }
              total_weighted_refined_prediction {}
            }
            buckets {
              lower_threshold_inclusive: 0.5
              upper_threshold_exclusive: 1.0
              num_weighted_examples {
              }
              total_weighted_label {}
              total_weighted_refined_prediction {}
            }
            buckets {
              lower_threshold_inclusive: 1.0
              upper_threshold_exclusive: inf
              num_weighted_examples {
                value: 1.0
              }
              total_weighted_label {
                value: 1.0
              }
              total_weighted_refined_prediction {
                value: 1.0
              }
            }
         }
        }
      }
    """, metrics_for_slice_pb2.PlotsForSlice())

    plot_records = []
    for record in tf.compat.v1.python_io.tf_record_iterator(plots_file):
      plot_records.append(
          metrics_for_slice_pb2.PlotsForSlice.FromString(record))
    self.assertEqual(1, len(plot_records), 'plots: %s' % plot_records)
    self.assertProtoEquals(expected_plots_for_slice, plot_records[0])


if __name__ == '__main__':
  tf.test.main()
