# Copyright 2023 Google LLC
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
"""Tests for set match related confusion matrix metrics."""
from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow_model_analysis as tfma
from tensorflow_model_analysis.metrics import metric_types
from google.protobuf import text_format


class SetMatchConfusionMatrixMetricsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          '_precision',
          text_format.Parse(
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
            class_name: "SetMatchPrecision"
            config:'"thresholds":[0.01]'
          }
        }
        """,
              tfma.EvalConfig(),
          ),
          ['set_match_precision'],
          [0.4],
      ),
      (
          '_recall',
          text_format.Parse(
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
            class_name: "SetMatchRecall"
            config:'"name":"recall", "thresholds":[0.01]'
          }
        }
        """,
              tfma.EvalConfig(),
          ),
          ['recall'],
          [0.5],
      ),
      (
          '_precision_top_k',
          text_format.Parse(
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
            class_name: "SetMatchPrecision"
            config:'"name":"precision", "top_k":2, "thresholds":[0.01]'
          }
        }
        """,
              tfma.EvalConfig(),
          ),
          ['precision'],
          [0.25],
      ),
      (
          '_recall_top_k',
          text_format.Parse(
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
            class_name: "SetMatchRecall"
            config:'"name":"recall", "top_k":2'
          }
        }
        """,
              tfma.EvalConfig(),
          ),
          ['recall'],
          [0.25],
      ),
      (
          '_recall_top_k_with_threshold_set',
          text_format.Parse(
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
            class_name: "SetMatchRecall"
            config:'"name":"recall", "top_k":2, "thresholds":[0.01]'
          }
        }
        """,
              tfma.EvalConfig(),
          ),
          ['recall'],
          [0.25],
      ),
  )
  def testSetMatchMetrics(self, eval_config, name_list, expected_results):
    extracts = [
        {
            'features': {
                'labels': np.array([['dogs', 'cats']]),
                'predictions': {
                    'classes': np.array([['dogs', 'pigs']]),
                    'scores': np.array([[0.1, 0.3]]),
                },
            }
        },
        {
            'features': {
                'labels': np.array([['birds', 'cats']]),
                'predictions': {
                    'classes': np.array([['dogs', 'pigs', 'birds']]),
                    'scores': np.array([[0.1, 0.3, 0.4]]),
                },
            }
        },
    ]

    evaluators = tfma.default_evaluators(eval_config=eval_config)
    extractors = tfma.default_extractors(
        eval_shared_model=None, eval_config=eval_config
    )

    with beam.Pipeline() as p:
      result = (
          p
          | 'LoadData' >> beam.Create(extracts)
          | 'ExtractEval'
          >> tfma.ExtractAndEvaluate(
              extractors=extractors, evaluators=evaluators
          )
      )

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
                err_msg=f'This {name} metric fails.',
            )
        except AssertionError as err:
          raise util.BeamAssertException(err)

      self.assertIn('metrics', result)
      util.assert_that(result['metrics'], check_result, label='result')

  @parameterized.named_parameters(
      (
          '_precision_with_class_weight',
          text_format.Parse(
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
            class_name: "SetMatchPrecision"
            config:'"thresholds":[0.01], "class_key":"classes", '
              '"weight_key":"weights"'
          }
        }
        """,
              tfma.EvalConfig(),
          ),
          ['set_match_precision'],
          [0.25],
      ),
      (
          '_recall_with_class_weight',
          text_format.Parse(
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
            class_name: "SetMatchRecall"
            config:'"name":"recall", "thresholds":[0.01],'
              '"weight_key":"weights"'
          }
        }
        """,
              tfma.EvalConfig(),
          ),
          ['recall'],
          [0.294118],
      ),
  )
  def testSetMatchMetricsWithClassWeights(
      self, eval_config, name_list, expected_results
  ):
    extracts = [
        {
            'features': {
                'labels': np.array([['dogs', 'cats']]),
                'predictions': {
                    'classes': np.array([['dogs', 'pigs']]),
                    'scores': np.array([[0.1, 0.3]]),
                    'weights': np.array([[0.1, 0.9]]),
                },
                'classes': np.array([['dogs', 'cats']]),
                'weights': np.array([[0.5, 1.2]]),
            }
        },
        {
            'features': {
                'labels': np.array([['birds', 'cats']]),
                'predictions': {
                    'classes': np.array([['dogs', 'pigs', 'birds']]),
                    'scores': np.array([[0.1, 0.3, 0.4]]),
                },
                'classes': np.array([['birds', 'cats']]),
                'weights': np.array([[0.5, 1.2]]),
            }
        },
    ]

    evaluators = tfma.default_evaluators(eval_config=eval_config)
    extractors = tfma.default_extractors(
        eval_shared_model=None, eval_config=eval_config
    )

    with beam.Pipeline() as p:
      result = (
          p
          | 'LoadData' >> beam.Create(extracts)
          | 'ExtractEval'
          >> tfma.ExtractAndEvaluate(
              extractors=extractors, evaluators=evaluators
          )
      )

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          self.assertLen(got_metrics, len(name_list))
          for name, expected_result in zip(name_list, expected_results):
            key = metric_types.MetricKey(name=name, example_weighted=True)
            self.assertIn(key, got_metrics)
            got_metric = got_metrics[key]
            np.testing.assert_allclose(
                expected_result,
                got_metric,
                rtol=1e-3,
                err_msg=f'This {name} metric fails.',
            )
        except AssertionError as err:
          raise util.BeamAssertException(err)

      self.assertIn('metrics', result)
      util.assert_that(result['metrics'], check_result, label='result')


