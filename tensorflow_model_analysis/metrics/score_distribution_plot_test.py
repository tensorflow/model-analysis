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
"""Tests for confusion matrix plot."""

import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
import tensorflow_model_analysis as tfma  # pylint: disable=unused-import
from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.utils import test_util

from google.protobuf import text_format


class ScoreDistributionPlotTest(test_util.TensorflowModelAnalysisTest):

  def testScoreDistributionPlot(self):

    extracts = [{
        'features': {
            'my_predictions': np.array([0.0]),
            'my_weights': np.array([1.0]),
        }
    }, {
        'features': {
            'my_predictions': np.array([0.5]),
            'my_weights': np.array([1.0]),
        }
    }, {
        'features': {
            'my_predictions': np.array([0.3]),
            'my_weights': np.array([1.0]),
        }
    }, {
        'features': {
            'my_predictions': np.array([0.9]),
            'my_weights': np.array([1.0]),
        }
    }]

    eval_config = text_format.Parse(
        """
        model_specs {
          name: "baseline"
          prediction_key: "my_predictions"
          is_baseline: true
        }
        metrics_specs {
          metrics {
            class_name: "ScoreDistributionPlot"
            config: '"num_thresholds": 4'
          }
        }
        options {
          compute_confidence_intervals {
          }
        }""", config_pb2.EvalConfig())

    evaluators = model_eval_lib.default_evaluators(eval_config=eval_config)
    extractors = model_eval_lib.default_extractors(
        eval_shared_model=None, eval_config=eval_config)

    with beam.Pipeline() as pipeline:
      result = (
          pipeline
          | 'LoadData' >> beam.Create(extracts)
          | 'ExtractEval' >> model_eval_lib.ExtractAndEvaluate(
              extractors=extractors, evaluators=evaluators))

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_plots = got[0]
          self.assertEqual(got_slice_key, ())
          self.assertLen(got_plots, 1)
          key = metric_types.PlotKey(name='score_distribution_plot')
          self.assertIn(key, got_plots)
          got_plot = got_plots[key]
          self.assertProtoEquals(
              """
              matrices {
                threshold: -1e-06
                true_positives: 4.0
                false_positive_rate: 1.0
                f1: 0.5964912
                accuracy: 0.425
                false_omission_rate: nan
              }
              matrices {
                true_negatives: 1.0
                true_positives: 3.0
                false_positive_rate: 0.5652174
                f1: 0.7234043
                accuracy: 0.675
              }
              matrices {
                threshold: 0.25
                true_negatives: 1.0
                true_positives: 3.0
                false_positive_rate: 0.5652174
                f1: 0.7234043
                accuracy: 0.675
              }
              matrices {
                threshold: 0.5
                true_negatives: 3.0
                true_positives: 1.0
                false_positive_rate: 0.0434783
                f1: 0.6666667
                accuracy: 0.775
                false_omission_rate: 0.2666667
              }
              matrices {
                threshold: 0.75
                true_negatives: 3.0
                true_positives: 1.0
                false_positive_rate: 0.0434783
                f1: 0.6666667
                accuracy: 0.775
                false_omission_rate: 0.2666667
              }
              matrices {
                threshold: 1.0
                true_negatives: 4.0
                accuracy: 0.575
                false_omission_rate: 0.425
              }
          """,
              got_plot,
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result['plots'], check_result, label='result')


