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
"""Test for MetricsAndPlotsEvaluator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_model_analysis import config
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.evaluators import metrics_validator
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.proto import validation_result_pb2
from google.protobuf import text_format


class MetricsValidatorTest(testutil.TensorflowModelAnalysisTest):

  def testValidateMetricsMetricValueAndThreshold(self):
    eval_config = config.EvalConfig(
        model_specs=[
            config.ModelSpec(),
        ],
        slicing_specs=[config.SlicingSpec()],
        metrics_specs=[
            config.MetricsSpec(
                metrics=[
                    config.MetricConfig(
                        class_name='WeightedExampleCount',
                        # 1.5 < 1, NOT OK.
                        threshold=config.MetricThreshold(
                            value_threshold=config.GenericValueThreshold(
                                upper_bound={'value': 1}))),
                ],
                model_names=['']),
        ],
    )
    sliced_metrics = ((()), {
        metric_types.MetricKey(name='weighted_example_count'): 1.5,
    })
    result = metrics_validator.validate_metrics(sliced_metrics, eval_config)
    self.assertFalse(result.validation_ok)
    expected = text_format.Parse(
        """
        metric_validations_per_slice {
          slice_key {
          }
          failures {
            metric_key {
              name: "weighted_example_count"
            }
            metric_threshold {
              value_threshold {
                upper_bound {
                  value: 1.0
                }
              }
            }
            metric_value {
              double_value {
                value: 1.5
              }
            }
          }
        }""", validation_result_pb2.ValidationResult())
    self.assertEqual(result, expected)

  def testValidateMetricsValueThresholdUpperBoundFail(self):
    eval_config = config.EvalConfig(
        model_specs=[
            config.ModelSpec(),
        ],
        slicing_specs=[config.SlicingSpec()],
        metrics_specs=[
            config.MetricsSpec(
                metrics=[
                    config.MetricConfig(
                        class_name='WeightedExampleCount',
                        # 1.5 < 1, NOT OK.
                        threshold=config.MetricThreshold(
                            value_threshold=config.GenericValueThreshold(
                                upper_bound={'value': 1}))),
                ],
                model_names=['']),
        ],
    )
    sliced_metrics = ((()), {
        metric_types.MetricKey(name='weighted_example_count'): 1.5,
    })
    result = metrics_validator.validate_metrics(sliced_metrics, eval_config)
    self.assertFalse(result.validation_ok)

  def testValidateMetricsValueThresholdLowerBoundFail(self):
    eval_config = config.EvalConfig(
        model_specs=[
            config.ModelSpec(),
        ],
        slicing_specs=[config.SlicingSpec()],
        metrics_specs=[
            config.MetricsSpec(
                metrics=[
                    config.MetricConfig(
                        class_name='WeightedExampleCount',
                        # 0 > 1, NOT OK.
                        threshold=config.MetricThreshold(
                            value_threshold=config.GenericValueThreshold(
                                lower_bound={'value': 1}))),
                ],
                model_names=['']),
        ],
    )
    sliced_metrics = ((()), {
        metric_types.MetricKey(name='weighted_example_count'): 0,
    })
    result = metrics_validator.validate_metrics(sliced_metrics, eval_config)
    self.assertFalse(result.validation_ok)

  def testValidateMetricsValueThresholdUpperBoundPass(self):
    eval_config = config.EvalConfig(
        model_specs=[
            config.ModelSpec(),
        ],
        slicing_specs=[config.SlicingSpec()],
        metrics_specs=[
            config.MetricsSpec(
                metrics=[
                    config.MetricConfig(
                        class_name='WeightedExampleCount',
                        # 0 < 1, OK.
                        threshold=config.MetricThreshold(
                            value_threshold=config.GenericValueThreshold(
                                upper_bound={'value': 1}))),
                ],
                model_names=['']),
        ],
    )
    sliced_metrics = ((()), {
        metric_types.MetricKey(name='weighted_example_count'): 0,
    })
    result = metrics_validator.validate_metrics(sliced_metrics, eval_config)
    self.assertTrue(result.validation_ok)

  def testValidateMetricsValueThresholdLowerBoundPass(self):
    eval_config = config.EvalConfig(
        model_specs=[
            config.ModelSpec(),
        ],
        slicing_specs=[config.SlicingSpec()],
        metrics_specs=[
            config.MetricsSpec(
                metrics=[
                    config.MetricConfig(
                        class_name='WeightedExampleCount',
                        # 2 > 1, OK.
                        threshold=config.MetricThreshold(
                            value_threshold=config.GenericValueThreshold(
                                lower_bound={'value': 1}))),
                ],
                model_names=['']),
        ],
    )
    sliced_metrics = ((()), {
        metric_types.MetricKey(name='weighted_example_count'): 2,
    })
    result = metrics_validator.validate_metrics(sliced_metrics, eval_config)
    self.assertTrue(result.validation_ok)

  def testValidateMetricsChangeThresholdAbsoluteFail(self):
    eval_config = config.EvalConfig(
        model_specs=[
            config.ModelSpec(),
            config.ModelSpec(name='baseline', is_baseline=True)
        ],
        slicing_specs=[config.SlicingSpec()],
        metrics_specs=[
            config.MetricsSpec(
                metrics=[
                    config.MetricConfig(
                        class_name='MeanPrediction',
                        # Diff = 0 - .333 = -.333 < -1, NOT OK.
                        threshold=config.MetricThreshold(
                            change_threshold=config.GenericChangeThreshold(
                                direction=config.MetricDirection
                                .LOWER_IS_BETTER,
                                absolute={'value': -1})))
                ],
                model_names=['']),
        ],
    )
    sliced_metrics = ((()), {
        metric_types.MetricKey(name='mean_prediction', model_name='baseline'):
            0.333,
        metric_types.MetricKey(name='mean_prediction', is_diff=True):
            -0.333,
    })
    result = metrics_validator.validate_metrics(sliced_metrics, eval_config)
    self.assertFalse(result.validation_ok)

  def testValidateMetricsChangeThresholdRelativeFail(self):
    eval_config = config.EvalConfig(
        model_specs=[
            config.ModelSpec(),
            config.ModelSpec(name='baseline', is_baseline=True)
        ],
        slicing_specs=[config.SlicingSpec()],
        metrics_specs=[
            config.MetricsSpec(
                metrics=[
                    config.MetricConfig(
                        class_name='MeanPrediction',
                        # Diff = -.333
                        # Diff% = -.333/.333 = -100% < -200%, NOT OK.
                        threshold=config.MetricThreshold(
                            change_threshold=config.GenericChangeThreshold(
                                direction=config.MetricDirection
                                .LOWER_IS_BETTER,
                                relative={'value': -2}))),
                ],
                model_names=['']),
        ],
    )
    sliced_metrics = ((()), {
        metric_types.MetricKey(name='mean_prediction', model_name='baseline'):
            0.333,
        metric_types.MetricKey(name='mean_prediction', is_diff=True):
            -0.333,
    })
    result = metrics_validator.validate_metrics(sliced_metrics, eval_config)
    self.assertFalse(result.validation_ok)

  def testValidateMetricsChangeThresholdAbsolutePass(self):
    eval_config = config.EvalConfig(
        model_specs=[
            config.ModelSpec(),
            config.ModelSpec(name='baseline', is_baseline=True)
        ],
        slicing_specs=[config.SlicingSpec()],
        metrics_specs=[
            config.MetricsSpec(
                metrics=[
                    config.MetricConfig(
                        class_name='MeanPrediction',
                        # Diff = 0 - .333 = -.333 < 0, OK.
                        threshold=config.MetricThreshold(
                            change_threshold=config.GenericChangeThreshold(
                                direction=config.MetricDirection
                                .LOWER_IS_BETTER,
                                absolute={'value': 0})))
                ],
                model_names=['']),
        ],
    )
    sliced_metrics = ((()), {
        metric_types.MetricKey(name='mean_prediction', model_name='baseline'):
            0.333,
        metric_types.MetricKey(name='mean_prediction', is_diff=True):
            -0.333,
    })
    result = metrics_validator.validate_metrics(sliced_metrics, eval_config)
    self.assertTrue(result.validation_ok)

  def testValidateMetricsChangeThresholdRelativePass(self):
    eval_config = config.EvalConfig(
        model_specs=[
            config.ModelSpec(),
            config.ModelSpec(name='baseline', is_baseline=True)
        ],
        slicing_specs=[config.SlicingSpec()],
        metrics_specs=[
            config.MetricsSpec(
                metrics=[
                    config.MetricConfig(
                        class_name='MeanPrediction',
                        # Diff = -.333
                        # Diff% = -.333/.333 = -100% < 0%, OK.
                        threshold=config.MetricThreshold(
                            change_threshold=config.GenericChangeThreshold(
                                direction=config.MetricDirection
                                .LOWER_IS_BETTER,
                                relative={'value': 0}))),
                ],
                model_names=['']),
        ],
    )
    sliced_metrics = ((()), {
        metric_types.MetricKey(name='mean_prediction', model_name='baseline'):
            0.333,
        metric_types.MetricKey(name='mean_prediction', is_diff=True):
            -0.333,
    })
    result = metrics_validator.validate_metrics(sliced_metrics, eval_config)
    self.assertTrue(result.validation_ok)

  def testValidateMetricsChangeThresholdHigherIsBetterPass(self):
    eval_config = config.EvalConfig(
        model_specs=[
            config.ModelSpec(),
            config.ModelSpec(name='baseline', is_baseline=True)
        ],
        slicing_specs=[config.SlicingSpec()],
        metrics_specs=[
            config.MetricsSpec(
                metrics=[
                    config.MetricConfig(
                        class_name='MeanPrediction',
                        # Diff = -.333 > -1, OK.
                        threshold=config.MetricThreshold(
                            change_threshold=config.GenericChangeThreshold(
                                direction=config.MetricDirection
                                .HIGHER_IS_BETTER,
                                absolute={'value': -1}))),
                ],
                model_names=['']),
        ],
    )
    sliced_metrics = ((()), {
        metric_types.MetricKey(name='mean_prediction', model_name='baseline'):
            0.333,
        metric_types.MetricKey(name='mean_prediction', is_diff=True):
            -0.333,
    })
    result = metrics_validator.validate_metrics(sliced_metrics, eval_config)
    self.assertTrue(result.validation_ok)

  def testValidateMetricsChangeThresholdHigherIsBetterFail(self):
    eval_config = config.EvalConfig(
        model_specs=[
            config.ModelSpec(),
            config.ModelSpec(name='baseline', is_baseline=True)
        ],
        slicing_specs=[config.SlicingSpec()],
        metrics_specs=[
            config.MetricsSpec(
                metrics=[
                    config.MetricConfig(
                        class_name='MeanPrediction',
                        # Diff = -.333 > 0, NOT OK.
                        threshold=config.MetricThreshold(
                            change_threshold=config.GenericChangeThreshold(
                                direction=config.MetricDirection
                                .HIGHER_IS_BETTER,
                                absolute={'value': 0}))),
                ],
                model_names=['']),
        ],
    )
    sliced_metrics = ((()), {
        metric_types.MetricKey(name='mean_prediction', model_name='baseline'):
            0.333,
        metric_types.MetricKey(name='mean_prediction', is_diff=True):
            -0.333,
    })
    result = metrics_validator.validate_metrics(sliced_metrics, eval_config)
    self.assertFalse(result.validation_ok)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
