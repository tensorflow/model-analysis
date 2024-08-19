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
"""Tests for object detection related metrics."""
from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow_model_analysis as tfma
from tensorflow_model_analysis.metrics import metric_types
from google.protobuf import text_format


class ObjectDetectionMetricsTest(parameterized.TestCase):
  """This tests the object detection metrics.

   Results provided from COCOAPI: AP with all IoUs causes overflow of memory,
   thus we do not check it here, but check the single value instead and the
   average of two IoUs.
   Average Precision @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.533
   Average Precision @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.916
   Average Precision @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.416
   Average Precision @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.500
   Average Precision @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.303
   Average Precision @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.701
   Average Recall @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.375
   Average Recall @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.533
   Average Recall @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.533
   Average Recall @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.500
   Average Recall @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.300
   Average Recall @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.700
  """

  @parameterized.named_parameters(('_average_precision_iou0.5',
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
            class_name: "COCOMeanAveragePrecision"
            config:'"iou_thresholds":[0.5], "class_ids":[1,2],'
                   '"max_num_detections":100, "name":"iou0.5"'
          }
        }
        """, tfma.EvalConfig()), ['iou0.5'], [0.916]),
                                  ('_average_precision_iou0.75',
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
           class_name: "COCOMeanAveragePrecision"
           config:'"iou_thresholds":[0.75], "class_ids":[1, 2], '
                  '"max_num_detections":100, "name":"iou0.75"'
         }
       }
       """, tfma.EvalConfig()), ['iou0.75'], [0.416]),
                                  ('_average_precision_ave',
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
           class_name: "COCOMeanAveragePrecision"
           config:'"iou_thresholds":[0.5, 0.75], "class_ids":[1, 2], '
                  '"max_num_detections":100, "name":"iouave"'
         }
       }
       """, tfma.EvalConfig()), ['iouave'], [0.666]), ('_average_recall_mdet1',
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
           class_name: "COCOMeanAverageRecall"
           config:'"class_ids":[1, 2], "max_num_detections":1, '
                  '"name":"mdet1"'
         }
       }
       """, tfma.EvalConfig()), ['mdet1'], [0.375]), ('_average_recall_mdet10',
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
           class_name: "COCOMeanAverageRecall"
           config:'"class_ids":[1, 2], "max_num_detections":10, '
                  '"name":"mdet10"'
         }
       }
       """, tfma.EvalConfig()), ['mdet10'], [0.533]),
                                  ('_average_recall_mdet100',
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
           class_name: "COCOMeanAverageRecall"
           config:'"class_ids":[1, 2], "max_num_detections":100, '
                  '"name":"mdet100"'
         }
       }
       """, tfma.EvalConfig()), ['mdet100'], [0.533]),
                                  ('_average_recall_arsmall',
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
           class_name: "COCOMeanAverageRecall"
           config:'"class_ids":[1, 2], "area_range":[0, 1024], '
                  '"max_num_detections":100, "name":"arsmall"'
         }
       }
       """, tfma.EvalConfig()), ['arsmall'], [0.500]),
                                  ('_average_recall_armedium',
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
           class_name: "COCOMeanAverageRecall"
           config:'"class_ids":[1, 2], "area_range":[1024, 9216], '
                  '"max_num_detections":100, "name":"armedium"'
         }
       }
       """, tfma.EvalConfig()), ['armedium'], [0.300]),
                                  ('_average_recall_arlarge',
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
           class_name: "COCOMeanAverageRecall"
           config:'"class_ids":[1, 2], "area_range":[9216, 99999], '
                  '"max_num_detections":100, "name":"arlarge"'
         }
       }
       """, tfma.EvalConfig()), ['arlarge'], [0.700]))
  def testMetricValuesWithLargerData(self, eval_config, name_list,
                                     expected_results):

    extracts = [{
        'features': {
            'labels':
                np.array([[[272.1, 200.23, 424.07, 480., 2.],
                           [181.23, 86.28, 208.67, 159.81, 2.],
                           [174.74, 0., 435.78, 220.79, 2.]]]),
            'predictions':
                np.array([[[271.2, 178.86, 429.52, 459.57, 2., 0.64],
                           [178.53, 92.57, 206.39, 159.71, 2., 0.38],
                           [167.96, 9.97, 442.79, 235.07, 2., 0.95]]])
        }
    }, {
        'features': {
            'labels':
                np.array([[[473.07, 395.93, 503.07, 424.6, 1.],
                           [204.01, 235.08, 264.85, 412.44, 2.],
                           [0.43, 499.79, 340.22, 606.24, 2.],
                           [204.42, 304.1, 256.93, 456.86, 2.]]]),
            'predictions':
                np.array([[[471.15, 398.57, 502.29, 428.26, 1., 0.54],
                           [198.53, 242.14, 263.93, 427.51, 2., 0.95],
                           [-32.86, 505.75, 338.82, 619.66, 2., 0.17],
                           [201.59, 299.39, 258.4, 452.88, 1., 0.05]]])
        }
    }]

    evaluators = tfma.default_evaluators(eval_config=eval_config)
    extractors = tfma.default_extractors(
        eval_shared_model=None, eval_config=eval_config)

    with beam.Pipeline() as p:
      result = (
          p | 'LoadData' >> beam.Create(extracts)
          | 'ExtractEval' >> tfma.ExtractAndEvaluate(
              extractors=extractors, evaluators=evaluators))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          self.assertLen(got_metrics, len(name_list))
          for name, expected_result in zip(name_list, expected_results):
            key = metric_types.MetricKey(name=name)
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

  @parameterized.named_parameters(('_average_precision_iou0.5',
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
           class_name: "COCOMeanAveragePrecision"
           config:'"iou_thresholds":[0.5], "class_ids":[1,2], "num_thresholds":3,'
                  '"max_num_detections":100, "name":"iou0.5", '
                  '"labels_to_stack":["xmin", "ymin", "xmax", "ymax", "class_id"], '
                  '"predictions_to_stack":["bbox", "class_id", "scores"]'
         }
       }
       """, tfma.EvalConfig()), ['iou0.5'], [0.916]))
  def testMetricValuesWithSplittedData(self, eval_config, name_list,
                                       expected_results):

    extracts = [{
        'features': {
            'labels': {
                'xmin': np.array([[272.1, 181.23, 174.74]]),
                'ymin': np.array([[200.23, 86.28, 0.]]),
                'xmax': np.array([[424.07, 208.67, 435.78]]),
                'ymax': np.array([[480., 159.81, 220.79]]),
                'class_id': np.array([[2., 2., 2.]]),
            },
            'predictions': {
                'bbox':
                    np.array([[[271.2, 178.86, 429.52, 459.57],
                               [178.53, 92.57, 206.39, 159.71],
                               [167.96, 9.97, 442.79, 235.07]]]),
                'class_id':
                    np.array([[2., 2., 2.]]),
                'scores':
                    np.array([[0.64, 0.38, 0.95]]),
            }
        }
    }, {
        'features': {
            'labels': {
                'xmin': np.array([[473.07, 204.01, 0.43, 204.42]]),
                'ymin': np.array([[395.93, 235.08, 499.79, 304.1]]),
                'xmax': np.array([[503.07, 264.85, 340.22, 256.93]]),
                'ymax': np.array([[424.6, 412.44, 606.24, 456.86]]),
                'class_id': np.array([[1., 2., 2., 2.]]),
            },
            'predictions': {
                'bbox':
                    np.array([[[471.15, 398.57, 502.29, 428.26],
                               [198.53, 242.14, 263.93, 427.51],
                               [-32.86, 505.75, 338.82, 619.66],
                               [201.59, 299.39, 258.4, 452.88]]]),
                'class_id':
                    np.array([[1., 2., 2., 1.]]),
                'scores':
                    np.array([[0.54, 0.95, 0.17, 0.05]]),
            }
        }
    }]

    evaluators = tfma.default_evaluators(eval_config=eval_config)
    extractors = tfma.default_extractors(
        eval_shared_model=None, eval_config=eval_config)

    with beam.Pipeline() as p:
      result = (
          p | 'LoadData' >> beam.Create(extracts)
          | 'ExtractEval' >> tfma.ExtractAndEvaluate(
              extractors=extractors, evaluators=evaluators))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          self.assertLen(got_metrics, len(name_list))
          for name, expected_result in zip(name_list, expected_results):
            key = metric_types.MetricKey(name=name)
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


