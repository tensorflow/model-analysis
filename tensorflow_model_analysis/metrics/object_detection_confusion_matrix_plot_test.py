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
"""Tests for object detection confusion matrix plot."""


import pytest
from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow_model_analysis as tfma
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.utils import test_util

from google.protobuf import text_format


@pytest.mark.xfail(run=False, reason="PR 183 This class contains tests that fail and needs to be fixed. "
"If all tests pass, please remove this mark.")
class ObjectDetectionConfusionMatrixPlotTest(
    test_util.TensorflowModelAnalysisTest, absltest.TestCase
):

  def testConfusionMatrixPlot(self):
    eval_config = text_format.Parse(
        """
        model_specs {
          signature_name: "serving_default"
          prediction_key: "predictions"
          label_key: "labels"
        }
        slicing_specs {
        }
        metrics_specs {
          metrics {
            class_name: "ObjectDetectionConfusionMatrixPlot"
            config:'"num_thresholds": 5, "iou_threshold":0.5, "class_id":1,'
                   '"max_num_detections":100, "name":"iou0.5"'
          }
        }
        """, tfma.EvalConfig())
    extracts = [
        # The match at iou_threshold = 0.5 is
        # gt_matches: [[0]] dt_matches: [[0, -1]]
        # Results after preprocess:
        #   'labels': np.asarray([1., 0.]),
        #   'predictions': np.asarray([0.7, 0.3])
        {
            'features': {
                'labels':
                    np.asarray([[[30, 100, 70, 300, 0], [50, 100, 80, 200,
                                                         1]]]),
                'predictions':
                    np.asarray([[[20, 130, 60, 290, 0, 0.7],
                                 [30, 100, 70, 300, 0, 0.3],
                                 [500, 100, 800, 300, 1, 0.1]]])
            }
        },
        # This is a binary classification case, the iou matrix should be:
        # [[0., 2/3], [0., 4/11]]
        # The match at iou_threshold = 0.5 is
        # gt_matches: [[-1, 0]] dt_matches: [[1, -1]]
        # Results after preprocess:
        #   'labels': np.asarray([1., 1., 0.]),
        #   'predictions': np.asarray([0., 0.4, 0.3])
        #    thresholds=[-1e-7, 0.5, 1.0 + 1e-7],
        #    tp=[3.0, 1.0, 0.0],
        #    fp=[2.0, 0.0, 0.0],
        #    tn=[0.0, 2.0, 2.0],
        #    fn=[0.0, 2.0, 3.0])
        # Precision: [3/5, 1.0, 'nan']
        # Recall: [1.0, 1/3, 0.0]
        {
            'features': {
                'labels':
                    np.asarray([[[30, 100, 70, 400, 0], [10, 200, 80, 300,
                                                         0]]]),
                'predictions':
                    np.asarray([[[100, 130, 160, 290, 0, 0.4],
                                 [30, 100, 70, 300, 0, 0.3]]])
            }
        }
    ]
    evaluators = tfma.default_evaluators(eval_config=eval_config)
    extractors = tfma.default_extractors(
        eval_shared_model=None, eval_config=eval_config)

    with beam.Pipeline() as p:
      result = (
          p | 'LoadData' >> beam.Create(extracts)
          | 'ExtractEval' >> tfma.ExtractAndEvaluate(
              extractors=extractors, evaluators=evaluators))

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_plots = got[0]
          self.assertEqual(got_slice_key, ())
          key = metric_types.PlotKey(
              name='iou0.5', sub_key=metric_types.SubKey(class_id=1)
          )
          self.assertIn(key, got_plots)
          got_plot = got_plots[key]
          self.assertProtoEquals(
              """
              matrices {
                threshold: -1e-06
                false_positives: 1.0
                true_positives: 1.0
                precision: 0.5
                recall: 1.0
                false_positive_rate: 1.0
                f1: 0.6666667
                accuracy: 0.5
                false_omission_rate: nan
              }
              matrices {
                false_positives: 1.0
                true_positives: 1.0
                precision: 0.5
                recall: 1.0
                false_positive_rate: 1.0
                f1: 0.6666667
                accuracy: 0.5
                false_omission_rate: nan
              }
              matrices {
                threshold: 0.2
                false_negatives: 1.0
                true_negatives: 1.0
                precision: 1.0
                accuracy: 0.5
                false_omission_rate: 0.5
              }
              matrices {
                threshold: 0.4
                false_negatives: 1.0
                true_negatives: 1.0
                precision: 1.0
                accuracy: 0.5
                false_omission_rate: 0.5
              }
              matrices {
                threshold: 0.6
                false_negatives: 1.0
                true_negatives: 1.0
                precision: 1.0
                accuracy: 0.5
                false_omission_rate: 0.5
              }
              matrices {
                threshold: 0.8
                false_negatives: 1.0
                true_negatives: 1.0
                precision: 1.0
                accuracy: 0.5
                false_omission_rate: 0.5
              }
              matrices {
                threshold: 1.0
                false_negatives: 1.0
                true_negatives: 1.0
                precision: 1.0
                accuracy: 0.5
                false_omission_rate: 0.5
              }
          """,
              got_plot,
          )
        except AssertionError as err:
          raise util.BeamAssertException(err)

      self.assertIn('plots', result)
      util.assert_that(result['plots'], check_result, label='result')


