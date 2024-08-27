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
"""Tests for object detection related confusion matrix metrics."""

import pytest
from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow_model_analysis as tfma
from tensorflow_model_analysis.metrics import metric_types
from google.protobuf import text_format


@pytest.mark.xfail(run=False, reason="PR 183 This class contains tests that fail and needs to be fixed. "
"If all tests pass, please remove this mark.")
class ObjectDetectionConfusionMatrixMetricsTest(parameterized.TestCase):

  @parameterized.named_parameters(('_max_recall',
                                   text_format.Parse(
                                       """
        model_specs {
          signature_name: "serving_default"
          prediction_key: "predictions" # placeholder
          label_key: "labels" # placeholder
        }
        slicing_specs {
        }
        metrics_specs {
          metrics {
            class_name: "ObjectDetectionMaxRecall"
            config:'"class_id":0, '
                   '"max_num_detections":100, "name":"maxrecall"'
          }
        }
        """, tfma.EvalConfig()), ['maxrecall'], [2 / 3]),
                                  ('_precision_at_recall',
                                   text_format.Parse(
                                       """
        model_specs {
          signature_name: "serving_default"
          prediction_key: "predictions" # placeholder
          label_key: "labels" # placeholder
        }
        slicing_specs {
        }
        metrics_specs {
          metrics {
            class_name: "ObjectDetectionPrecisionAtRecall"
            config:'"class_id":0, "recall":[0.4], "num_thresholds":3, '
                   '"max_num_detections":100, "name":"precisionatrecall"'
          }
        }
        """, tfma.EvalConfig()), ['precisionatrecall'], [3 / 5]),
                                  ('_recall',
                                   text_format.Parse(
                                       """
        model_specs {
          signature_name: "serving_default"
          prediction_key: "predictions" # placeholder
          label_key: "labels" # placeholder
        }
        slicing_specs {
        }
        metrics_specs {
          metrics {
            class_name: "ObjectDetectionRecall"
            config:'"class_id":0, "thresholds":0.1, '
                   '"max_num_detections":100, "name":"recall"'
          }
        }
        """, tfma.EvalConfig()), ['recall'], [2 / 3]), ('_precision',
                                                        text_format.Parse(
                                                            """
        model_specs {
          signature_name: "serving_default"
          prediction_key: "predictions" # placeholder
          label_key: "labels" # placeholder
        }
        slicing_specs {
        }
        metrics_specs {
          metrics {
            class_name: "ObjectDetectionPrecision"
            config:'"class_id":0, "thresholds":0.1, '
                   '"max_num_detections":100, "name":"precision"'
          }
        }
        """, tfma.EvalConfig()), ['precision'], [0.5]), ('_threshold_at_recall',
                                                         text_format.Parse(
                                                             """
        model_specs {
          signature_name: "serving_default"
          prediction_key: "predictions" # placeholder
          label_key: "labels" # placeholder
        }
        slicing_specs {
        }
        metrics_specs {
          metrics {
            class_name: "ObjectDetectionThresholdAtRecall"
            config:'"recall":[0.4], "class_id":0, '
                   '"max_num_detections":100, "name":"thresholdatrecall"'
          }
        }
        """, tfma.EvalConfig()), ['thresholdatrecall'], [0.3]))
  def testObjectDetectionMetrics(self, eval_config, name_list,
                                 expected_results):

    extracts = [
        {
            # The match at iou_threshold = 0.5 is
            # gt_matches: [[0]] dt_matches: [[0, -1]]
            # Results after preprocess:
            #   'labels': np.asarray([1., 0.]),
            #   'predictions': np.asarray([0.7, 0.3])
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
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          self.assertLen(got_metrics, len(name_list))
          for name, expected_result in zip(name_list, expected_results):
            key = metric_types.MetricKey(
                name=name, sub_key=metric_types.SubKey(class_id=0)
            )
            self.assertIn(key, got_metrics)
            got_metric = got_metrics[key]
            np.testing.assert_allclose(
                expected_result,
                got_metric,
                rtol=1e-3,
                err_msg=f'This {name} metric fails.')
        except AssertionError as err:
          raise util.BeamAssertException(err)

      self.assertIn('metrics', result)
      util.assert_that(result['metrics'], check_result, label='result')


