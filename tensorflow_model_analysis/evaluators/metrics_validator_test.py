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

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_model_analysis import config
from tensorflow_model_analysis import types
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.evaluators import metrics_validator
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.proto import validation_result_pb2
from tensorflow_model_analysis.slicer import slicer_lib as slicer
from google.protobuf import text_format

# Tests involiving slices: (<test_name>, <slice_config> , <slice_key>)
_NO_SLICE_TEST = ('no_slice', None, (()))
_GLOBAL_SLICE_TEST = ('global_slice', [config.SlicingSpec()], (()))
_FEATURE_SLICE_TEST = ('feature_slice',
                       [config.SlicingSpec(feature_keys=['feature1'])
                       ], (('feature1', 'value1'),))
_FEATURE_VALUE_SLICE_TEST = ('feature_value_slice', [
    config.SlicingSpec(feature_values={'feature1': 'value1'})
], (('feature1', 'value1'),))
_MULTIPLE_SLICES_TEST = ('multiple_slices', [
    config.SlicingSpec(feature_values={'feature1': 'value1'}),
    config.SlicingSpec(feature_values={'feature2': 'value2'})
], (('feature1', 'value1'),))
_UNMATCHED_SINGLE_SLICE_TEST = ('single_slice',
                                [config.SlicingSpec(feature_keys='feature1')
                                ], (('unmatched_feature', 'unmatched_value'),))
_UNMATCHED_MULTIPLE_SLICES_TEST = ('multiple_slices', [
    config.SlicingSpec(feature_values={'feature1': 'value1'}),
    config.SlicingSpec(feature_values={'feature2': 'value2'})
], (('unmatched_feature', 'unmatched_value'),))

# Cross slice tests: (<test_name>, <cross_slice_config>, <cross_slice_key>)
_CROSS_SLICE_GLOBAL_TEST = ('global_slice', [
    config.CrossSlicingSpec(
        baseline_spec=config.SlicingSpec(),
        slicing_specs=[
            config.SlicingSpec(feature_values={'feature2': 'value2'})
        ])
], ((()), (('feature2', 'value2'),)))
_SINGLE_CROSS_SLICE_TEST = ('single_slice', [
    config.CrossSlicingSpec(
        baseline_spec=config.SlicingSpec(feature_keys=['feature1']),
        slicing_specs=[
            config.SlicingSpec(feature_values={'feature2': 'value2'})
        ])
], ((('feature1', 'value1'),), (('feature2', 'value2'),)))
_MULTIPLE_CROSS_SLICE_TEST = ('multiple_slice', [
    config.CrossSlicingSpec(
        baseline_spec=config.SlicingSpec(feature_keys=['feature1']),
        slicing_specs=[
            config.SlicingSpec(feature_values={'feature2': 'value2'})
        ]),
    config.CrossSlicingSpec(
        baseline_spec=config.SlicingSpec(feature_keys=['feature2']),
        slicing_specs=[
            config.SlicingSpec(feature_values={'feature3': 'value3'})
        ])
], ((('feature2', 'value2'),), (('feature3', 'value3'),)))
_CROSS_SLICE_MULTIPLE_SLICING_SPEC_TEST = ('multiple_slicing_spec', [
    config.CrossSlicingSpec(
        baseline_spec=config.SlicingSpec(feature_keys=['feature1']),
        slicing_specs=[
            config.SlicingSpec(feature_values={'feature2': 'value2'}),
            config.SlicingSpec(feature_keys=['feature3'])
        ])
], ((('feature1', 'value1'),), (('feature3', 'value3'),)))
_UNMATCHED_CROSS_SLICE_TEST = ('unmatched_cross_slice', [
    config.CrossSlicingSpec(
        baseline_spec=config.SlicingSpec(feature_keys=['feature1']),
        slicing_specs=[
            config.SlicingSpec(feature_values={'feature2': 'value2'})
        ]),
    config.CrossSlicingSpec(
        baseline_spec=config.SlicingSpec(feature_keys=['feature2']),
        slicing_specs=[
            config.SlicingSpec(feature_values={'feature3': 'value3'})
        ])
], ((('feature1', 'value1'),), (('feature3', 'value3'),)))


class MetricsValidatorTest(testutil.TensorflowModelAnalysisTest,
                           parameterized.TestCase):

  def testValidateMetricsInvalidThreshold(self):
    eval_config = config.EvalConfig(
        model_specs=[
            config.ModelSpec(),
        ],
        slicing_specs=[config.SlicingSpec()],
        metrics_specs=[
            config.MetricsSpec(
                thresholds={
                    'invalid_threshold':
                        config.MetricThreshold(
                            value_threshold=config.GenericValueThreshold(
                                lower_bound={'value': 0.2}))
                })
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
              name: "invalid_threshold"
            }
            metric_threshold {
              value_threshold {
                lower_bound {
                  value: 0.2
                }
              }
            }
            message: 'Metric not found.'
          }
        }""", validation_result_pb2.ValidationResult())
    self.assertProtoEquals(expected, result)

  @parameterized.named_parameters(_NO_SLICE_TEST, _GLOBAL_SLICE_TEST,
                                  _FEATURE_SLICE_TEST,
                                  _FEATURE_VALUE_SLICE_TEST,
                                  _MULTIPLE_SLICES_TEST)
  def testValidateMetricsMetricTDistributionValueAndThreshold(
      self, slicing_specs, slice_key):
    threshold = config.MetricThreshold(
        value_threshold=config.GenericValueThreshold(
            lower_bound={'value': 0.9}))
    eval_config = config.EvalConfig(
        model_specs=[
            config.ModelSpec(),
        ],
        slicing_specs=slicing_specs,
        metrics_specs=[
            config.MetricsSpec(
                metrics=[
                    config.MetricConfig(
                        class_name='AUC',
                        threshold=threshold if slicing_specs is None else None,
                        per_slice_thresholds=[
                            config.PerSliceMetricThreshold(
                                slicing_specs=slicing_specs,
                                threshold=threshold)
                        ]),
                ],
                model_names=['']),
        ],
    )
    sliced_metrics = (slice_key, {
        metric_types.MetricKey(name='auc'):
            types.ValueWithTDistribution(sample_mean=0.91, unsampled_value=0.8)
    })
    result = metrics_validator.validate_metrics(sliced_metrics, eval_config)
    self.assertFalse(result.validation_ok)
    expected = text_format.Parse(
        """
        metric_validations_per_slice {
          failures {
            metric_key {
              name: "auc"
            }
            metric_value {
              double_value {
                value: 0.8
              }
            }
          }
        }""", validation_result_pb2.ValidationResult())
    expected.metric_validations_per_slice[0].failures[
        0].metric_threshold.CopyFrom(threshold)
    expected.metric_validations_per_slice[0].slice_key.CopyFrom(
        slicer.serialize_slice_key(slice_key))
    for spec in slicing_specs or [None]:
      if (spec is None or
          slicer.SingleSliceSpec(spec=spec).is_slice_applicable(slice_key)):
        slicing_details = expected.validation_details.slicing_details.add()
        if spec is not None:
          slicing_details.slicing_spec.CopyFrom(spec)
        else:
          slicing_details.slicing_spec.CopyFrom(config.SlicingSpec())
        slicing_details.num_matching_slices = 1
    self.assertEqual(result, expected)

  @parameterized.named_parameters(_NO_SLICE_TEST, _GLOBAL_SLICE_TEST,
                                  _FEATURE_SLICE_TEST,
                                  _FEATURE_VALUE_SLICE_TEST,
                                  _MULTIPLE_SLICES_TEST)
  def testValidateMetricsMetricTDistributionChangeAndThreshold(
      self, slicing_specs, slice_key):
    threshold = config.MetricThreshold(
        change_threshold=config.GenericChangeThreshold(
            direction=config.MetricDirection.LOWER_IS_BETTER,
            absolute={'value': -1}))
    eval_config = config.EvalConfig(
        model_specs=[
            config.ModelSpec(),
            config.ModelSpec(name='baseline', is_baseline=True)
        ],
        slicing_specs=slicing_specs,
        metrics_specs=[
            config.MetricsSpec(
                metrics=[
                    config.MetricConfig(
                        class_name='AUC',
                        threshold=threshold if slicing_specs is None else None,
                        per_slice_thresholds=[
                            config.PerSliceMetricThreshold(
                                slicing_specs=slicing_specs,
                                threshold=threshold)
                        ]),
                ],
                model_names=['']),
        ],
    )
    sliced_metrics = (
        slice_key,
        {
            # This is the mean of the diff.
            metric_types.MetricKey(name='auc', model_name='baseline'):
                types.ValueWithTDistribution(
                    sample_mean=0.91, unsampled_value=0.6),
            metric_types.MetricKey(name='auc', is_diff=True):
                types.ValueWithTDistribution(
                    sample_mean=0.1, unsampled_value=0.1),
        })
    result = metrics_validator.validate_metrics(sliced_metrics, eval_config)
    self.assertFalse(result.validation_ok)
    expected = text_format.Parse(
        """
        metric_validations_per_slice {
          failures {
            metric_key {
              name: "auc"
              is_diff: true
            }
            metric_value {
              double_value {
                value: 0.1
              }
            }
          }
        }""", validation_result_pb2.ValidationResult())
    expected.metric_validations_per_slice[0].failures[
        0].metric_threshold.CopyFrom(threshold)
    expected.metric_validations_per_slice[0].slice_key.CopyFrom(
        slicer.serialize_slice_key(slice_key))
    for spec in slicing_specs or [None]:
      if (spec is None or
          slicer.SingleSliceSpec(spec=spec).is_slice_applicable(slice_key)):
        slicing_details = expected.validation_details.slicing_details.add()
        if spec is not None:
          slicing_details.slicing_spec.CopyFrom(spec)
        else:
          slicing_details.slicing_spec.CopyFrom(config.SlicingSpec())
        slicing_details.num_matching_slices = 1
    self.assertAlmostEqual(result, expected)

  @parameterized.named_parameters(_NO_SLICE_TEST, _GLOBAL_SLICE_TEST,
                                  _FEATURE_SLICE_TEST,
                                  _FEATURE_VALUE_SLICE_TEST,
                                  _MULTIPLE_SLICES_TEST)
  def testValidateMetricsMetricValueAndThreshold(self, slicing_specs,
                                                 slice_key):
    threshold = config.MetricThreshold(
        value_threshold=config.GenericValueThreshold(upper_bound={'value': 1}))
    eval_config = config.EvalConfig(
        model_specs=[
            config.ModelSpec(),
        ],
        slicing_specs=slicing_specs,
        metrics_specs=[
            config.MetricsSpec(
                metrics=[
                    config.MetricConfig(
                        class_name='WeightedExampleCount',
                        # 1.5 < 1, NOT OK.
                        threshold=threshold if slicing_specs is None else None,
                        per_slice_thresholds=[
                            config.PerSliceMetricThreshold(
                                slicing_specs=slicing_specs,
                                threshold=threshold)
                        ]),
                ],
                model_names=['']),
        ],
    )
    sliced_metrics = (slice_key, {
        metric_types.MetricKey(name='weighted_example_count'): 1.5,
    })
    result = metrics_validator.validate_metrics(sliced_metrics, eval_config)
    self.assertFalse(result.validation_ok)
    expected = text_format.Parse(
        """
        metric_validations_per_slice {
          failures {
            metric_key {
              name: "weighted_example_count"
            }
            metric_value {
              double_value {
                value: 1.5
              }
            }
          }
        }""", validation_result_pb2.ValidationResult())
    expected.metric_validations_per_slice[0].failures[
        0].metric_threshold.CopyFrom(threshold)
    expected.metric_validations_per_slice[0].slice_key.CopyFrom(
        slicer.serialize_slice_key(slice_key))
    for spec in slicing_specs or [None]:
      if (spec is None or
          slicer.SingleSliceSpec(spec=spec).is_slice_applicable(slice_key)):
        slicing_details = expected.validation_details.slicing_details.add()
        if spec is not None:
          slicing_details.slicing_spec.CopyFrom(spec)
        else:
          slicing_details.slicing_spec.CopyFrom(config.SlicingSpec())
        slicing_details.num_matching_slices = 1
    self.assertEqual(result, expected)

  @parameterized.named_parameters(_UNMATCHED_SINGLE_SLICE_TEST,
                                  _UNMATCHED_MULTIPLE_SLICES_TEST)
  def testValidateMetricsMetricValueAndThresholdIgnoreUnmatchedSlice(
      self, slicing_specs, slice_key):
    threshold = config.MetricThreshold(
        value_threshold=config.GenericValueThreshold(upper_bound={'value': 1}))
    eval_config = config.EvalConfig(
        model_specs=[
            config.ModelSpec(),
        ],
        slicing_specs=slicing_specs,
        metrics_specs=[
            config.MetricsSpec(
                metrics=[
                    config.MetricConfig(
                        class_name='WeightedExampleCount',
                        # 1.5 < 1, NOT OK.
                        per_slice_thresholds=[
                            config.PerSliceMetricThreshold(
                                slicing_specs=slicing_specs,
                                threshold=threshold)
                        ]),
                ],
                model_names=['']),
        ],
    )
    sliced_metrics = (slice_key, {
        metric_types.MetricKey(name='weighted_example_count'): 1.5,
    })
    result = metrics_validator.validate_metrics(sliced_metrics, eval_config)
    self.assertTrue(result.validation_ok)

  @parameterized.named_parameters(_NO_SLICE_TEST, _GLOBAL_SLICE_TEST,
                                  _FEATURE_SLICE_TEST,
                                  _FEATURE_VALUE_SLICE_TEST,
                                  _MULTIPLE_SLICES_TEST)
  def testValidateMetricsValueThresholdUpperBoundFail(self, slicing_specs,
                                                      slice_key):
    threshold = config.MetricThreshold(
        value_threshold=config.GenericValueThreshold(upper_bound={'value': 1}))
    eval_config = config.EvalConfig(
        model_specs=[
            config.ModelSpec(),
        ],
        slicing_specs=slicing_specs,
        metrics_specs=[
            config.MetricsSpec(
                metrics=[
                    config.MetricConfig(
                        class_name='WeightedExampleCount',
                        # 1.5 < 1, NOT OK.
                        threshold=threshold if slicing_specs is None else None,
                        per_slice_thresholds=[
                            config.PerSliceMetricThreshold(
                                slicing_specs=slicing_specs,
                                threshold=threshold)
                        ]),
                ],
                model_names=['']),
        ],
    )
    sliced_metrics = (slice_key, {
        metric_types.MetricKey(name='weighted_example_count'): 1.5,
    })
    result = metrics_validator.validate_metrics(sliced_metrics, eval_config)
    self.assertFalse(result.validation_ok)

  @parameterized.named_parameters(_NO_SLICE_TEST, _GLOBAL_SLICE_TEST,
                                  _FEATURE_SLICE_TEST,
                                  _FEATURE_VALUE_SLICE_TEST,
                                  _MULTIPLE_SLICES_TEST)
  def testValidateMetricsValueThresholdLowerBoundFail(self, slicing_specs,
                                                      slice_key):
    threshold = config.MetricThreshold(
        value_threshold=config.GenericValueThreshold(lower_bound={'value': 1}))
    eval_config = config.EvalConfig(
        model_specs=[
            config.ModelSpec(),
        ],
        slicing_specs=slicing_specs,
        metrics_specs=[
            config.MetricsSpec(
                metrics=[
                    config.MetricConfig(
                        class_name='WeightedExampleCount',
                        # 0 > 1, NOT OK.
                        threshold=threshold if slicing_specs is None else None,
                        per_slice_thresholds=[
                            config.PerSliceMetricThreshold(
                                slicing_specs=slicing_specs,
                                threshold=threshold)
                        ]),
                ],
                model_names=['']),
        ],
    )
    sliced_metrics = (slice_key, {
        metric_types.MetricKey(name='weighted_example_count'): 0,
    })
    result = metrics_validator.validate_metrics(sliced_metrics, eval_config)
    self.assertFalse(result.validation_ok)

  @parameterized.named_parameters(_NO_SLICE_TEST, _GLOBAL_SLICE_TEST,
                                  _FEATURE_SLICE_TEST,
                                  _FEATURE_VALUE_SLICE_TEST,
                                  _MULTIPLE_SLICES_TEST)
  def testValidateMetricsValueThresholdUpperBoundPass(self, slicing_specs,
                                                      slice_key):
    threshold = config.MetricThreshold(
        value_threshold=config.GenericValueThreshold(upper_bound={'value': 1}))
    eval_config = config.EvalConfig(
        model_specs=[
            config.ModelSpec(),
        ],
        slicing_specs=slicing_specs,
        metrics_specs=[
            config.MetricsSpec(
                metrics=[
                    config.MetricConfig(
                        class_name='WeightedExampleCount',
                        # 0 < 1, OK.
                        threshold=threshold if slicing_specs is None else None,
                        per_slice_thresholds=[
                            config.PerSliceMetricThreshold(
                                slicing_specs=slicing_specs,
                                threshold=threshold)
                        ]),
                ],
                model_names=['']),
        ],
    )
    sliced_metrics = (slice_key, {
        metric_types.MetricKey(name='weighted_example_count'): 0,
    })
    result = metrics_validator.validate_metrics(sliced_metrics, eval_config)
    self.assertTrue(result.validation_ok)

  @parameterized.named_parameters(_NO_SLICE_TEST, _GLOBAL_SLICE_TEST,
                                  _FEATURE_SLICE_TEST,
                                  _FEATURE_VALUE_SLICE_TEST,
                                  _MULTIPLE_SLICES_TEST)
  def testValidateMetricsValueThresholdLowerBoundPass(self, slicing_specs,
                                                      slice_key):
    threshold = config.MetricThreshold(
        value_threshold=config.GenericValueThreshold(lower_bound={'value': 1}))
    eval_config = config.EvalConfig(
        model_specs=[
            config.ModelSpec(),
        ],
        slicing_specs=slicing_specs,
        metrics_specs=[
            config.MetricsSpec(
                metrics=[
                    config.MetricConfig(
                        class_name='WeightedExampleCount',
                        # 2 > 1, OK.
                        threshold=threshold if slicing_specs is None else None,
                        per_slice_thresholds=[
                            config.PerSliceMetricThreshold(
                                slicing_specs=slicing_specs,
                                threshold=threshold)
                        ]),
                ],
                model_names=['']),
        ],
    )
    sliced_metrics = (slice_key, {
        metric_types.MetricKey(name='weighted_example_count'): 2,
    })
    result = metrics_validator.validate_metrics(sliced_metrics, eval_config)
    self.assertTrue(result.validation_ok)

  @parameterized.named_parameters(_NO_SLICE_TEST, _GLOBAL_SLICE_TEST,
                                  _FEATURE_SLICE_TEST,
                                  _FEATURE_VALUE_SLICE_TEST,
                                  _MULTIPLE_SLICES_TEST)
  def testValidateMetricsChangeThresholdAbsoluteFail(self, slicing_specs,
                                                     slice_key):
    threshold = config.MetricThreshold(
        change_threshold=config.GenericChangeThreshold(
            direction=config.MetricDirection.LOWER_IS_BETTER,
            absolute={'value': -1}))
    eval_config = config.EvalConfig(
        model_specs=[
            config.ModelSpec(),
            config.ModelSpec(name='baseline', is_baseline=True)
        ],
        slicing_specs=slicing_specs,
        metrics_specs=[
            config.MetricsSpec(
                metrics=[
                    config.MetricConfig(
                        class_name='MeanPrediction',
                        # Diff = 0 - .333 = -.333 < -1, NOT OK.
                        threshold=threshold if slicing_specs is None else None,
                        per_slice_thresholds=[
                            config.PerSliceMetricThreshold(
                                slicing_specs=slicing_specs,
                                threshold=threshold)
                        ])
                ],
                model_names=['']),
        ],
    )
    sliced_metrics = (slice_key, {
        metric_types.MetricKey(name='mean_prediction', model_name='baseline'):
            0.333,
        metric_types.MetricKey(name='mean_prediction', is_diff=True):
            -0.333,
    })
    result = metrics_validator.validate_metrics(sliced_metrics, eval_config)
    self.assertFalse(result.validation_ok)

  @parameterized.named_parameters(_NO_SLICE_TEST, _GLOBAL_SLICE_TEST,
                                  _FEATURE_SLICE_TEST,
                                  _FEATURE_VALUE_SLICE_TEST,
                                  _MULTIPLE_SLICES_TEST)
  def testValidateMetricsChangeThresholdRelativeFail(self, slicing_specs,
                                                     slice_key):
    threshold = config.MetricThreshold(
        change_threshold=config.GenericChangeThreshold(
            direction=config.MetricDirection.LOWER_IS_BETTER,
            relative={'value': -2}))
    eval_config = config.EvalConfig(
        model_specs=[
            config.ModelSpec(),
            config.ModelSpec(name='baseline', is_baseline=True)
        ],
        slicing_specs=slicing_specs,
        metrics_specs=[
            config.MetricsSpec(
                metrics=[
                    config.MetricConfig(
                        class_name='MeanPrediction',
                        # Diff = -.333
                        # Diff% = -.333/.333 = -100% < -200%, NOT OK.
                        threshold=threshold if slicing_specs is None else None,
                        per_slice_thresholds=[
                            config.PerSliceMetricThreshold(
                                slicing_specs=slicing_specs,
                                threshold=threshold)
                        ])
                ],
                model_names=['']),
        ],
    )
    sliced_metrics = (slice_key, {
        metric_types.MetricKey(name='mean_prediction', model_name='baseline'):
            0.333,
        metric_types.MetricKey(name='mean_prediction', is_diff=True):
            -0.333,
    })
    result = metrics_validator.validate_metrics(sliced_metrics, eval_config)
    self.assertFalse(result.validation_ok)

  @parameterized.named_parameters(_NO_SLICE_TEST, _GLOBAL_SLICE_TEST,
                                  _FEATURE_SLICE_TEST,
                                  _FEATURE_VALUE_SLICE_TEST,
                                  _MULTIPLE_SLICES_TEST)
  def testValidateMetricsChangeThresholdAbsolutePass(self, slicing_specs,
                                                     slice_key):
    threshold = config.MetricThreshold(
        change_threshold=config.GenericChangeThreshold(
            direction=config.MetricDirection.LOWER_IS_BETTER,
            absolute={'value': 0}))
    eval_config = config.EvalConfig(
        model_specs=[
            config.ModelSpec(),
            config.ModelSpec(name='baseline', is_baseline=True)
        ],
        slicing_specs=slicing_specs,
        metrics_specs=[
            config.MetricsSpec(
                metrics=[
                    config.MetricConfig(
                        class_name='MeanPrediction',
                        # Diff = 0 - .333 = -.333 < 0, OK.
                        threshold=threshold if slicing_specs is None else None,
                        per_slice_thresholds=[
                            config.PerSliceMetricThreshold(
                                slicing_specs=slicing_specs,
                                threshold=threshold)
                        ])
                ],
                model_names=['']),
        ],
    )
    sliced_metrics = (slice_key, {
        metric_types.MetricKey(name='mean_prediction', model_name='baseline'):
            0.333,
        metric_types.MetricKey(name='mean_prediction', is_diff=True):
            -0.333,
    })
    result = metrics_validator.validate_metrics(sliced_metrics, eval_config)
    self.assertTrue(result.validation_ok)

  @parameterized.named_parameters(_NO_SLICE_TEST, _GLOBAL_SLICE_TEST,
                                  _FEATURE_SLICE_TEST,
                                  _FEATURE_VALUE_SLICE_TEST,
                                  _MULTIPLE_SLICES_TEST)
  def testValidateMetricsChangeThresholdRelativePass(self, slicing_specs,
                                                     slice_key):
    threshold = config.MetricThreshold(
        change_threshold=config.GenericChangeThreshold(
            direction=config.MetricDirection.LOWER_IS_BETTER,
            relative={'value': 0}))
    eval_config = config.EvalConfig(
        model_specs=[
            config.ModelSpec(),
            config.ModelSpec(name='baseline', is_baseline=True)
        ],
        slicing_specs=slicing_specs,
        metrics_specs=[
            config.MetricsSpec(
                metrics=[
                    config.MetricConfig(
                        class_name='MeanPrediction',
                        # Diff = -.333
                        # Diff% = -.333/.333 = -100% < 0%, OK.
                        threshold=threshold if slicing_specs is None else None,
                        per_slice_thresholds=[
                            config.PerSliceMetricThreshold(
                                slicing_specs=slicing_specs,
                                threshold=threshold)
                        ])
                ],
                model_names=['']),
        ],
    )
    sliced_metrics = (slice_key, {
        metric_types.MetricKey(name='mean_prediction', model_name='baseline'):
            0.333,
        metric_types.MetricKey(name='mean_prediction', is_diff=True):
            -0.333,
    })
    result = metrics_validator.validate_metrics(sliced_metrics, eval_config)
    self.assertTrue(result.validation_ok)

  @parameterized.named_parameters(_NO_SLICE_TEST, _GLOBAL_SLICE_TEST,
                                  _FEATURE_SLICE_TEST,
                                  _FEATURE_VALUE_SLICE_TEST,
                                  _MULTIPLE_SLICES_TEST)
  def testValidateMetricsChangeThresholdHigherIsBetterPass(
      self, slicing_specs, slice_key):
    threshold = config.MetricThreshold(
        change_threshold=config.GenericChangeThreshold(
            direction=config.MetricDirection.HIGHER_IS_BETTER,
            absolute={'value': -1}))
    eval_config = config.EvalConfig(
        model_specs=[
            config.ModelSpec(),
            config.ModelSpec(name='baseline', is_baseline=True)
        ],
        slicing_specs=slicing_specs,
        metrics_specs=[
            config.MetricsSpec(
                metrics=[
                    config.MetricConfig(
                        class_name='MeanPrediction',
                        # Diff = -.333 > -1, OK.
                        threshold=threshold if slicing_specs is None else None,
                        per_slice_thresholds=[
                            config.PerSliceMetricThreshold(
                                slicing_specs=slicing_specs,
                                threshold=threshold)
                        ])
                ],
                model_names=['']),
        ],
    )
    sliced_metrics = (slice_key, {
        metric_types.MetricKey(name='mean_prediction', model_name='baseline'):
            0.333,
        metric_types.MetricKey(name='mean_prediction', is_diff=True):
            -0.333,
    })
    result = metrics_validator.validate_metrics(sliced_metrics, eval_config)
    self.assertTrue(result.validation_ok)

  @parameterized.named_parameters(_NO_SLICE_TEST, _GLOBAL_SLICE_TEST,
                                  _FEATURE_SLICE_TEST,
                                  _FEATURE_VALUE_SLICE_TEST,
                                  _MULTIPLE_SLICES_TEST)
  def testValidateMetricsChangeThresholdHigherIsBetterFail(
      self, slicing_specs, slice_key):
    threshold = config.MetricThreshold(
        change_threshold=config.GenericChangeThreshold(
            direction=config.MetricDirection.HIGHER_IS_BETTER,
            absolute={'value': 0}))
    eval_config = config.EvalConfig(
        model_specs=[
            config.ModelSpec(),
            config.ModelSpec(name='baseline', is_baseline=True)
        ],
        slicing_specs=slicing_specs,
        metrics_specs=[
            config.MetricsSpec(
                metrics=[
                    config.MetricConfig(
                        class_name='MeanPrediction',
                        # Diff = -.333 > 0, NOT OK.
                        threshold=threshold if slicing_specs is None else None,
                        per_slice_thresholds=[
                            config.PerSliceMetricThreshold(
                                slicing_specs=slicing_specs,
                                threshold=threshold)
                        ])
                ],
                model_names=['']),
        ],
    )
    sliced_metrics = (slice_key, {
        metric_types.MetricKey(name='mean_prediction', model_name='baseline'):
            0.333,
        metric_types.MetricKey(name='mean_prediction', is_diff=True):
            -0.333,
    })
    result = metrics_validator.validate_metrics(sliced_metrics, eval_config)
    self.assertFalse(result.validation_ok)

  def testMergeDetails(self):
    a = text_format.Parse(
        """
        validation_details {
          slicing_details {
            slicing_spec {}
            num_matching_slices: 1
          }
          slicing_details {
            slicing_spec {
              feature_keys: ["x", "y"]
            }
            num_matching_slices: 1
          }
        }""", validation_result_pb2.ValidationResult())

    b = text_format.Parse(
        """
        validation_details {
          slicing_details {
            slicing_spec {
              feature_keys: ["x"]
            }
            num_matching_slices: 1
          }
          slicing_details {
            slicing_spec {
              feature_keys: ["x", "y"]
            }
            num_matching_slices: 2
          }
        }""", validation_result_pb2.ValidationResult())

    expected = text_format.Parse(
        """
        validation_details {
          slicing_details {
            slicing_spec {}
            num_matching_slices: 1
          }
          slicing_details {
            slicing_spec {
              feature_keys: ["x", "y"]
            }
            num_matching_slices: 3
          }
          slicing_details {
            slicing_spec {
              feature_keys: ["x"]
            }
            num_matching_slices: 1
          }
        }""", validation_result_pb2.ValidationResult())

    metrics_validator.merge_details(a, b)
    self.assertProtoEquals(expected, a)

  def testGetMissingSlices(self):
    slicing_specs = [
        config.SlicingSpec(),
        config.SlicingSpec(feature_values={'feature1': 'value1'}),
        config.SlicingSpec(feature_values={'feature2': 'value2'})
    ]
    threshold = config.MetricThreshold(
        value_threshold=config.GenericValueThreshold(upper_bound={'value': 1}))
    eval_config = config.EvalConfig(
        model_specs=[
            config.ModelSpec(),
        ],
        slicing_specs=slicing_specs,
        metrics_specs=[
            config.MetricsSpec(
                metrics=[
                    config.MetricConfig(
                        class_name='WeightedExampleCount',
                        # 1.5 < 1, NOT OK.
                        per_slice_thresholds=[
                            config.PerSliceMetricThreshold(
                                slicing_specs=slicing_specs,
                                threshold=threshold)
                        ]),
                ],
                model_names=['']),
        ],
    )
    sliced_metrics = ((('feature1', 'value1'),), {
        metric_types.MetricKey(name='weighted_example_count'): 0,
    })
    result = metrics_validator.validate_metrics(sliced_metrics, eval_config)

    expected_checks = text_format.Parse(
        """
        validation_ok: true
        validation_details {
          slicing_details {
            slicing_spec {
              feature_values {
                key: "feature1"
                value: "value1"
              }
            }
            num_matching_slices: 1
          }
        }""", validation_result_pb2.ValidationResult())

    self.assertProtoEquals(expected_checks, result)

    missing = metrics_validator.get_missing_slices(
        result.validation_details.slicing_details, eval_config)
    self.assertLen(missing, 2)
    self.assertProtoEquals(missing[0], slicing_specs[0])
    self.assertProtoEquals(missing[1], slicing_specs[2])

  @parameterized.named_parameters(_NO_SLICE_TEST, _SINGLE_CROSS_SLICE_TEST,
                                  _CROSS_SLICE_GLOBAL_TEST,
                                  _MULTIPLE_CROSS_SLICE_TEST,
                                  _CROSS_SLICE_MULTIPLE_SLICING_SPEC_TEST)
  def testValidateMetricsCrossSliceThresholdPass(self, cross_slicing_specs,
                                                 slice_key):
    threshold = config.MetricThreshold(
        value_threshold=config.GenericValueThreshold(upper_bound={'value': 1}))
    eval_config = config.EvalConfig(
        model_specs=[
            config.ModelSpec(),
        ],
        cross_slicing_specs=cross_slicing_specs,
        metrics_specs=[
            config.MetricsSpec(
                metrics=[
                    config.MetricConfig(
                        class_name='WeightedExampleCount',
                        # 1.5 < 1, NOT OK.
                        threshold=(threshold
                                   if cross_slicing_specs is None else None),
                        cross_slice_thresholds=[
                            config.CrossSliceMetricThreshold(
                                cross_slicing_specs=cross_slicing_specs,
                                threshold=threshold)
                        ]),
                ],
                model_names=['']),
        ],
    )
    sliced_metrics = (slice_key, {
        metric_types.MetricKey(name='weighted_example_count'): 0,
    })
    result = metrics_validator.validate_metrics(sliced_metrics, eval_config)
    self.assertTrue(result.validation_ok)

  @parameterized.named_parameters(_NO_SLICE_TEST, _SINGLE_CROSS_SLICE_TEST,
                                  _CROSS_SLICE_GLOBAL_TEST,
                                  _MULTIPLE_CROSS_SLICE_TEST,
                                  _CROSS_SLICE_MULTIPLE_SLICING_SPEC_TEST)
  def testValidateMetricsCrossSliceThresholdFail(self, cross_slicing_specs,
                                                 slice_key):
    threshold = config.MetricThreshold(
        value_threshold=config.GenericValueThreshold(upper_bound={'value': 1}))
    eval_config = config.EvalConfig(
        model_specs=[
            config.ModelSpec(),
        ],
        cross_slicing_specs=cross_slicing_specs,
        metrics_specs=[
            config.MetricsSpec(
                metrics=[
                    config.MetricConfig(
                        class_name='WeightedExampleCount',
                        # 1.5 < 1, NOT OK.
                        threshold=(threshold
                                   if cross_slicing_specs is None else None),
                        cross_slice_thresholds=[
                            config.CrossSliceMetricThreshold(
                                cross_slicing_specs=cross_slicing_specs,
                                threshold=threshold)
                        ]),
                ],
                model_names=['']),
        ],
    )
    sliced_metrics = (slice_key, {
        metric_types.MetricKey(name='weighted_example_count'): 1.5,
    })
    result = metrics_validator.validate_metrics(sliced_metrics, eval_config)
    self.assertFalse(result.validation_ok)

  @parameterized.named_parameters(_UNMATCHED_CROSS_SLICE_TEST)
  def testValidateMetricsCrossSliceThresholdUnmacthed(self, cross_slicing_specs,
                                                      slice_key):
    threshold = config.MetricThreshold(
        value_threshold=config.GenericValueThreshold(upper_bound={'value': 1}))
    eval_config = config.EvalConfig(
        model_specs=[
            config.ModelSpec(),
        ],
        cross_slicing_specs=cross_slicing_specs,
        metrics_specs=[
            config.MetricsSpec(
                metrics=[
                    config.MetricConfig(
                        class_name='WeightedExampleCount',
                        # 1.5 < 1, NOT OK.
                        threshold=(threshold
                                   if cross_slicing_specs is None else None),
                        cross_slice_thresholds=[
                            config.CrossSliceMetricThreshold(
                                cross_slicing_specs=cross_slicing_specs,
                                threshold=threshold)
                        ]),
                ],
                model_names=['']),
        ],
    )
    sliced_metrics = (slice_key, {
        metric_types.MetricKey(name='weighted_example_count'): 0,
    })
    result = metrics_validator.validate_metrics(sliced_metrics, eval_config)
    self.assertTrue(result.validation_ok)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
