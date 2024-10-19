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
"""Test for using the MetricsPlotsAndValidationsWriter API."""


import pytest
import os
import tempfile

from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.api import types
from tensorflow_model_analysis.evaluators import metrics_plots_and_validations_evaluator
from tensorflow_model_analysis.extractors import example_weights_extractor
from tensorflow_model_analysis.extractors import features_extractor
from tensorflow_model_analysis.extractors import labels_extractor
from tensorflow_model_analysis.extractors import predictions_extractor
from tensorflow_model_analysis.extractors import slice_key_extractor
from tensorflow_model_analysis.extractors import unbatch_extractor
from tensorflow_model_analysis.metrics import attributions
from tensorflow_model_analysis.metrics import binary_confusion_matrices
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.proto import metrics_for_slice_pb2
from tensorflow_model_analysis.proto import validation_result_pb2
from tensorflow_model_analysis.slicer import slicer_lib as slicer
from tensorflow_model_analysis.utils import test_util as testutil
from tensorflow_model_analysis.utils.keras_lib import tf_keras
from tensorflow_model_analysis.writers import metrics_plots_and_validations_writer
from tfx_bsl.tfxio import tensor_adapter
from tfx_bsl.tfxio import test_util

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2


def _make_slice_key(*args):
  if len(args) % 2 != 0:
    raise ValueError('number of arguments should be even')

  result = []
  for i in range(0, len(args), 2):
    result.append((args[i], args[i + 1]))
  result = tuple(result)
  return result


@pytest.mark.xfail(run=False, reason="PR 183 This class contains tests that fail and needs to be fixed. "
"If all tests pass, please remove this mark.")
class MetricsPlotsAndValidationsWriterTest(testutil.TensorflowModelAnalysisTest,
                                           parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.longMessage = True  # pylint: disable=invalid-name

  def _getTempDir(self):
    return tempfile.mkdtemp()

  def _getExportDir(self):
    return os.path.join(self._getTempDir(), 'export_dir')

  def _getBaselineDir(self):
    return os.path.join(self._getTempDir(), 'baseline_export_dir')

  def _build_keras_model(self, model_dir, mul):
    input_layer = tf_keras.layers.Input(shape=(1,), name='input_1')
    output_layer = tf_keras.layers.Lambda(
        lambda x, mul: x * mul, output_shape=(1,), arguments={'mul': mul}
    )(input_layer)
    model = tf_keras.models.Model([input_layer], output_layer)
    model.compile(
        optimizer=tf_keras.optimizers.Adam(lr=0.001),
        loss=tf_keras.losses.BinaryCrossentropy(),
        metrics=['accuracy'],
    )

    model.fit(x=[[0], [1]], y=[[0], [1]], steps_per_epoch=1)
    model.save(model_dir, save_format='tf')
    return self.createTestEvalSharedModel(
        model_path=model_dir, tags=[tf.saved_model.SERVING]
    )

  def testConvertSlicePlotsToProto(self):
    slice_key = _make_slice_key('fruit', 'apple')
    plot_key = metric_types.PlotKey(
        name='calibration_plot', output_name='output_name')
    calibration_plot = text_format.Parse(
        """
        buckets {
          lower_threshold_inclusive: -inf
          upper_threshold_exclusive: 0.0
          num_weighted_examples { value: 0.0 }
          total_weighted_label { value: 0.0 }
          total_weighted_refined_prediction { value: 0.0 }
        }
        buckets {
          lower_threshold_inclusive: 0.0
          upper_threshold_exclusive: 0.5
          num_weighted_examples { value: 1.0 }
          total_weighted_label { value: 1.0 }
          total_weighted_refined_prediction { value: 0.3 }
        }
        buckets {
          lower_threshold_inclusive: 0.5
          upper_threshold_exclusive: 1.0
          num_weighted_examples { value: 1.0 }
          total_weighted_label { value: 0.0 }
          total_weighted_refined_prediction { value: 0.7 }
        }
        buckets {
          lower_threshold_inclusive: 1.0
          upper_threshold_exclusive: inf
          num_weighted_examples { value: 0.0 }
          total_weighted_label { value: 0.0 }
          total_weighted_refined_prediction { value: 0.0 }
        }
     """, metrics_for_slice_pb2.CalibrationHistogramBuckets())

    expected_plots_for_slice = text_format.Parse(
        """
      slice_key {
        single_slice_keys {
          column: 'fruit'
          bytes_value: 'apple'
        }
      }
      plot_keys_and_values {
        key {
          output_name: "output_name"
          example_weighted { }
        }
        value {
          calibration_histogram_buckets {
            buckets {
              lower_threshold_inclusive: -inf
              upper_threshold_exclusive: 0.0
              num_weighted_examples { value: 0.0 }
              total_weighted_label { value: 0.0 }
              total_weighted_refined_prediction { value: 0.0 }
            }
            buckets {
              lower_threshold_inclusive: 0.0
              upper_threshold_exclusive: 0.5
              num_weighted_examples { value: 1.0 }
              total_weighted_label { value: 1.0 }
              total_weighted_refined_prediction { value: 0.3 }
            }
            buckets {
              lower_threshold_inclusive: 0.5
              upper_threshold_exclusive: 1.0
              num_weighted_examples { value: 1.0 }
              total_weighted_label { value: 0.0 }
              total_weighted_refined_prediction { value: 0.7 }
            }
            buckets {
              lower_threshold_inclusive: 1.0
              upper_threshold_exclusive: inf
              num_weighted_examples { value: 0.0 }
              total_weighted_label { value: 0.0 }
              total_weighted_refined_prediction { value: 0.0 }
            }
          }
        }
      }
    """, metrics_for_slice_pb2.PlotsForSlice())

    got = metrics_plots_and_validations_writer.convert_slice_plots_to_proto(
        (slice_key, {
            plot_key: calibration_plot
        }), None)
    self.assertProtoEquals(expected_plots_for_slice, got)

  def testConvertSliceMetricsToProto(self):
    slice_key = _make_slice_key('age', 5, 'language', 'english', 'price', 0.3)
    slice_metrics = {
        metric_types.MetricKey(name='accuracy', output_name='output_name'): 0.8
    }
    expected_metrics_for_slice = text_format.Parse(
        """
        slice_key {
          single_slice_keys {
            column: 'age'
            int64_value: 5
          }
          single_slice_keys {
            column: 'language'
            bytes_value: 'english'
          }
          single_slice_keys {
            column: 'price'
            float_value: 0.3
          }
        }
        metric_keys_and_values {
          key {
            name: "accuracy"
            output_name: "output_name"
            example_weighted { }
          }
          value {
            double_value {
              value: 0.8
            }
          }
        }""", metrics_for_slice_pb2.MetricsForSlice())

    got = metrics_plots_and_validations_writer.convert_slice_metrics_to_proto(
        (slice_key, slice_metrics), None)
    self.assertProtoEquals(expected_metrics_for_slice, got)

  def testConvertSliceMetricsToProtoConfusionMatrices(self):
    slice_key = _make_slice_key()
    slice_metrics = {
        metric_types.MetricKey(name='confusion_matrix_at_thresholds'):
            binary_confusion_matrices.Matrices(
                thresholds=[0.25, 0.75, 1.00],
                fn=[0.0, 1.0, 2.0],
                tn=[1.0, 1.0, 1.0],
                fp=[0.0, 0.0, 0.0],
                tp=[2.0, 1.0, 0.0])
    }
    expected_metrics_for_slice = text_format.Parse(
        """
        slice_key {}
        metric_keys_and_values {
          key: {
            name: "confusion_matrix_at_thresholds"
            example_weighted { }
          }
          value {
            confusion_matrix_at_thresholds {
              matrices {
                threshold: 0.25
                false_negatives: 0.0
                true_negatives: 1.0
                false_positives: 0.0
                true_positives: 2.0
                precision: 1.0
                recall: 1.0
                f1: 1.0
                accuracy: 1.0
                false_omission_rate: 0.0
                false_positive_rate: 0.0
              }
              matrices {
                threshold: 0.75
                false_negatives: 1.0
                true_negatives: 1.0
                false_positives: 0.0
                true_positives: 1.0
                precision: 1.0
                recall: 0.5
                f1: 0.6666667
                accuracy: 0.6666667
                false_omission_rate: 0.5
                false_positive_rate: 0.0
              }
              matrices {
                threshold: 1.00
                false_negatives: 2.0
                true_negatives: 1.0
                false_positives: 0.0
                true_positives: 0.0
                precision: 1.0
                recall: 0.0
                f1: 0.0
                accuracy: 0.3333333
                false_omission_rate: 0.6666667
                false_positive_rate: 0.0
              }
            }
          }
        }
        """,
        metrics_for_slice_pb2.MetricsForSlice(),
    )

    got = metrics_plots_and_validations_writer.convert_slice_metrics_to_proto(
        (slice_key, slice_metrics), add_metrics_callbacks=[])
    self.assertProtoEquals(expected_metrics_for_slice, got)

  def testConvertSliceMetricsToProtoBoundedValue(self):
    slice_key = _make_slice_key()
    slice_metrics = {
        metric_types.MetricKey(name='bounded_metrics'):
            text_format.Parse(
                """
                lower_bound {
                  value: 0.123
                }
                upper_bound {
                  value: 0.456
                }
                value {
                  value: 0.234
                }
                """, metrics_for_slice_pb2.BoundedValue())
    }
    expected_metrics_for_slice = text_format.Parse(
        """
        slice_key {}
        metric_keys_and_values {
          key {
            name: "bounded_metrics"
            example_weighted {}
          }
          value {
            double_value {
              value: 0.234
            }
          }
          confidence_interval {
            upper_bound {
              double_value {
                value: 0.456
              }
            }
            lower_bound {
              double_value {
                value: 0.123
              }
            }
          }
        }
        """, metrics_for_slice_pb2.MetricsForSlice())

    got = metrics_plots_and_validations_writer.convert_slice_metrics_to_proto(
        (slice_key, slice_metrics), add_metrics_callbacks=[])
    self.assertProtoEquals(expected_metrics_for_slice, got)

  def testConvertSliceMetricsToProtoStringMetrics(self):
    slice_key = _make_slice_key()
    slice_metrics = {
        'valid_ascii': b'test string',
        'valid_unicode': b'\xF0\x9F\x90\x84',  # U+1F404, Cow
        'invalid_unicode': b'\xE2\x28\xA1',
    }
    expected_metrics_for_slice = metrics_for_slice_pb2.MetricsForSlice()
    expected_metrics_for_slice.slice_key.SetInParent()
    expected_metrics_for_slice.metrics[
        'valid_ascii'].bytes_value = slice_metrics['valid_ascii']
    expected_metrics_for_slice.metrics[
        'valid_unicode'].bytes_value = slice_metrics['valid_unicode']
    expected_metrics_for_slice.metrics[
        'invalid_unicode'].bytes_value = slice_metrics['invalid_unicode']

    got = metrics_plots_and_validations_writer.convert_slice_metrics_to_proto(
        (slice_key, slice_metrics), [])
    self.assertProtoEquals(expected_metrics_for_slice, got)

  def testConvertSliceMetricsToProtoMetricKeyWithConfidenceIntervals(self):
    slice_key = _make_slice_key()
    slice_metrics = {
        metric_types.MetricKey('metric_with_ci'):
            types.ValueWithTDistribution(
                unsampled_value=10,
                sample_mean=11,
                sample_standard_deviation=1,
                sample_degrees_of_freedom=20)
    }
    expected_metrics_for_slice = text_format.Parse(
        """
        slice_key {}
        metric_keys_and_values {
          key {
            name: "metric_with_ci"
            example_weighted { }
          }
          value {
            double_value { value: 10 }
          }
          confidence_interval {
            lower_bound: { double_value { value: 8.9140366 } }
            upper_bound: { double_value { value: 13.0859634 } }
            standard_error { double_value { value: 1.0 } }
            degrees_of_freedom { value: 20 }
          }
        }
        """, metrics_for_slice_pb2.MetricsForSlice())
    got = metrics_plots_and_validations_writer.convert_slice_metrics_to_proto(
        (slice_key, slice_metrics), [])
    self.assertProtoEquals(expected_metrics_for_slice, got)

  def testConvertSliceMetricsToProtoDeprecatedMetricsMapWithConfidenceIntervals(
      self):
    slice_key = _make_slice_key()
    slice_metrics = {
        'metric_with_ci':
            types.ValueWithTDistribution(
                unsampled_value=10,
                sample_mean=11,
                sample_standard_deviation=1,
                sample_degrees_of_freedom=20)
    }
    expected_metrics_for_slice = text_format.Parse(
        """
        slice_key {}
        metrics {
          key: "metric_with_ci"
          value {
            bounded_value {
              value: { value: 10 }
              lower_bound: { value: 8.9140366 }
              upper_bound: { value: 13.0859634 }
              methodology: POISSON_BOOTSTRAP
            }
          }
        }
        """, metrics_for_slice_pb2.MetricsForSlice())
    got = metrics_plots_and_validations_writer.convert_slice_metrics_to_proto(
        (slice_key, slice_metrics), [])
    self.assertProtoEquals(expected_metrics_for_slice, got)

  def testCombineValidationsValidationOk(self):
    input_validations = [
        text_format.Parse(
            """
            validation_ok: true
            metric_validations_per_slice {
              slice_key  {
                single_slice_keys {
                  column: "x"
                  bytes_value: "x1"
                }
              }
            }
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
            }""", validation_result_pb2.ValidationResult()),
        text_format.Parse(
            """
            validation_ok: true
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
    ]

    eval_config = config_pb2.EvalConfig(
        model_specs=[
            config_pb2.ModelSpec(name='candidate'),
            config_pb2.ModelSpec(name='baseline', is_baseline=True)
        ],
        slicing_specs=[config_pb2.SlicingSpec()],
        metrics_specs=[
            config_pb2.MetricsSpec(
                metrics=[
                    config_pb2.MetricConfig(
                        class_name='AUC',
                        per_slice_thresholds=[
                            config_pb2.PerSliceMetricThreshold(
                                slicing_specs=[config_pb2.SlicingSpec()],
                                threshold=config_pb2.MetricThreshold(
                                    value_threshold=config_pb2
                                    .GenericValueThreshold(
                                        lower_bound={'value': 0.7})))
                        ]),
                ],
                model_names=['candidate', 'baseline']),
        ])

    expected_validation = text_format.Parse(
        """
        validation_ok: true
        metric_validations_per_slice {
          slice_key  {
            single_slice_keys {
              column: "x"
              bytes_value: "x1"
            }
          }
        }
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

    def verify_fn(result):
      self.assertLen(result, 1)
      self.assertProtoEquals(expected_validation, result[0])

    with beam.Pipeline() as pipeline:
      result = (
          pipeline
          | 'Create' >> beam.Create(input_validations)
          | 'CombineValidations' >> beam.CombineGlobally(
              metrics_plots_and_validations_writer.CombineValidations(
                  eval_config)))
      util.assert_that(result, verify_fn)

  def testCombineValidationsMissingSlices(self):
    input_validations = [
        text_format.Parse(
            """
            validation_ok: false
            metric_validations_per_slice {
              slice_key  {
                single_slice_keys {
                  column: "x"
                  bytes_value: "x1"
                }
              }
              failures {
                metric_key {
                  name: "auc"
                  model_name: "candidate"
                  is_diff: true
                }
                metric_threshold {
                  value_threshold {
                    lower_bound { value: 0.7 }
                  }
                }
                metric_value {
                  double_value { value: 0.6 }
                }
              }
            }
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
            }""", validation_result_pb2.ValidationResult()),
        text_format.Parse(
            """
            validation_ok: true
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
    ]

    slicing_specs = [
        config_pb2.SlicingSpec(),
        config_pb2.SlicingSpec(feature_keys=['x']),
        config_pb2.SlicingSpec(feature_keys=['x', 'y']),
        config_pb2.SlicingSpec(feature_keys=['z']),
    ]
    eval_config = config_pb2.EvalConfig(
        model_specs=[
            config_pb2.ModelSpec(name='candidate'),
            config_pb2.ModelSpec(name='baseline', is_baseline=True)
        ],
        slicing_specs=slicing_specs,
        metrics_specs=[
            config_pb2.MetricsSpec(
                metrics=[
                    config_pb2.MetricConfig(
                        class_name='AUC',
                        per_slice_thresholds=[
                            config_pb2.PerSliceMetricThreshold(
                                slicing_specs=slicing_specs,
                                threshold=config_pb2.MetricThreshold(
                                    value_threshold=config_pb2
                                    .GenericValueThreshold(
                                        lower_bound={'value': 0.7})))
                        ]),
                ],
                model_names=['candidate', 'baseline']),
        ])

    expected_validation = text_format.Parse(
        """
        validation_ok: false
        metric_validations_per_slice {
          slice_key  {
            single_slice_keys {
              column: "x"
              bytes_value: "x1"
            }
          }
          failures {
            metric_key {
              name: "auc"
              model_name: "candidate"
              is_diff: true
            }
            metric_threshold {
              value_threshold {
                lower_bound { value: 0.7 }
              }
            }
            metric_value {
              double_value { value: 0.6 }
            }
          }
        }
        missing_slices {
          feature_keys: "z"
        }
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

    def verify_fn(result):
      self.assertLen(result, 1)
      self.assertProtoEquals(expected_validation, result[0])

    with beam.Pipeline() as pipeline:
      result = (
          pipeline
          | 'Create' >> beam.Create(input_validations)
          | 'CombineValidations' >> beam.CombineGlobally(
              metrics_plots_and_validations_writer.CombineValidations(
                  eval_config)))
      util.assert_that(result, verify_fn)

  def testUncertaintyValuedMetrics(self):
    slice_key = _make_slice_key()
    slice_metrics = {
        'one_dim':
            types.ValueWithTDistribution(2.0, 1.0, 3, 2.0),
        'nans':
            types.ValueWithTDistribution(
                float('nan'), float('nan'), -1, float('nan')),
    }
    expected_metrics_for_slice = text_format.Parse(
        """
        slice_key {}
        metrics {
          key: "one_dim"
          value {
            bounded_value {
              value {
                value: 2.0
              }
              lower_bound {
                value: -1.1824463
              }
              upper_bound {
                value: 5.1824463
              }
              methodology: POISSON_BOOTSTRAP
            }
          }
        }
        metrics {
          key: "nans"
          value {
            bounded_value {
              value {
                value: nan
              }
              lower_bound {
                value: nan
              }
              upper_bound {
                value: nan
              }
              methodology: POISSON_BOOTSTRAP
            }
          }
        }
        """, metrics_for_slice_pb2.MetricsForSlice())
    got = metrics_plots_and_validations_writer.convert_slice_metrics_to_proto(
        (slice_key, slice_metrics), [])
    self.assertProtoEquals(expected_metrics_for_slice, got)

  def testConvertSliceMetricsToProtoTensorValuedMetrics(self):
    slice_key = _make_slice_key()
    slice_metrics = {
        'one_dim':
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        'two_dims':
            np.array([['two', 'dims', 'test'], ['TWO', 'DIMS', 'TEST']]),
        'three_dims':
            np.array([[[100, 200, 300]], [[500, 600, 700]]], dtype=np.int64),
    }
    expected_metrics_for_slice = text_format.Parse(
        """
        slice_key {}
        metrics {
          key: "one_dim"
          value {
            array_value {
              data_type: FLOAT32
              shape: 4
              float32_values: [1.0, 2.0, 3.0, 4.0]
            }
          }
        }
        metrics {
          key: "two_dims"
          value {
            array_value {
              data_type: BYTES
              shape: [2, 3]
              bytes_values: ["two", "dims", "test", "TWO", "DIMS", "TEST"]
            }
          }
        }
        metrics {
          key: "three_dims"
          value {
            array_value {
              data_type: INT64
              shape: [2, 1, 3]
              int64_values: [100, 200, 300, 500, 600, 700]
            }
          }
        }
        """, metrics_for_slice_pb2.MetricsForSlice())
    got = metrics_plots_and_validations_writer.convert_slice_metrics_to_proto(
        (slice_key, slice_metrics), [])
    self.assertProtoEquals(expected_metrics_for_slice, got)

  def testConvertSliceAttributionsToProto(self):
    slice_key = _make_slice_key('language', 'english', 'price', 0.3)
    slice_attributions = {
        metric_types.AttributionsKey(name='mean', output_name='output_name'): {
            'age': 0.8,
            'language': 1.2,
            'price': 2.3,
        },
    }
    expected_attributions_for_slice = text_format.Parse(
        """
        slice_key {
          single_slice_keys {
            column: 'language'
            bytes_value: 'english'
          }
          single_slice_keys {
            column: 'price'
            float_value: 0.3
          }
        }
        attributions_keys_and_values {
          key {
            name: "mean"
            output_name: "output_name"
            example_weighted { }
          }
          values {
            key: "age"
            value: {
              double_value {
                value: 0.8
              }
            }
          }
          values {
            key: "language"
            value: {
              double_value {
                value: 1.2
              }
            }
          }
          values {
            key: "price"
            value: {
              double_value {
                value: 2.3
              }
            }
          }
        }""", metrics_for_slice_pb2.AttributionsForSlice())

    got = metrics_plots_and_validations_writer.convert_slice_attributions_to_proto(
        (slice_key, slice_attributions))
    self.assertProtoEquals(expected_attributions_for_slice, got)

  _OUTPUT_FORMAT_PARAMS = [('tfrecord_file_format', 'tfrecord'),
                           ('parquet_file_format', 'parquet')]

  @parameterized.named_parameters(_OUTPUT_FORMAT_PARAMS)
  def testWriteValidationResults(self, output_file_format):
    model_dir, baseline_dir = self._getExportDir(), self._getBaselineDir()
    eval_shared_model = self._build_keras_model(model_dir, mul=0)
    baseline_eval_shared_model = self._build_keras_model(baseline_dir, mul=1)
    validations_file = os.path.join(self._getTempDir(),
                                    constants.VALIDATIONS_KEY)
    schema = text_format.Parse(
        """
        tensor_representation_group {
          key: ""
          value {
            tensor_representation {
              key: "input_1"
              value {
                dense_tensor {
                  column_name: "input_1"
                  shape { dim { size: 1 } }
                }
              }
            }
          }
        }
        feature {
          name: "input_1"
          type: FLOAT
        }
        feature {
          name: "label"
          type: FLOAT
        }
        feature {
          name: "example_weight"
          type: FLOAT
        }
        feature {
          name: "extra_feature"
          type: BYTES
        }
        """, schema_pb2.Schema())
    tfx_io = test_util.InMemoryTFExampleRecord(
        schema=schema, raw_record_column_name=constants.ARROW_INPUT_COLUMN)
    tensor_adapter_config = tensor_adapter.TensorAdapterConfig(
        arrow_schema=tfx_io.ArrowSchema(),
        tensor_representations=tfx_io.TensorRepresentations())
    examples = [
        self._makeExample(
            input_1=0.0,
            label=1.0,
            example_weight=1.0,
            extra_feature='non_model_feature'),
        self._makeExample(
            input_1=1.0,
            label=0.0,
            example_weight=0.5,
            extra_feature='non_model_feature'),
    ]

    slicing_specs = [
        config_pb2.SlicingSpec(),
        config_pb2.SlicingSpec(feature_keys=['slice_does_not_exist'])
    ]
    cross_slicing_specs = [
        config_pb2.CrossSlicingSpec(
            baseline_spec=config_pb2.SlicingSpec(
                feature_keys=['slice_does_not_exist']),
            slicing_specs=[
                config_pb2.SlicingSpec(feature_keys=['slice_does_not_exist'])
            ])
    ]
    eval_config = config_pb2.EvalConfig(
        model_specs=[
            config_pb2.ModelSpec(
                name='candidate',
                label_key='label',
                example_weight_key='example_weight'),
            config_pb2.ModelSpec(
                name='baseline',
                label_key='label',
                example_weight_key='example_weight',
                is_baseline=True)
        ],
        slicing_specs=slicing_specs,
        cross_slicing_specs=cross_slicing_specs,
        metrics_specs=[
            config_pb2.MetricsSpec(
                metrics=[
                    config_pb2.MetricConfig(
                        class_name='ExampleCount',
                        # 2 > 10, NOT OK.
                        threshold=config_pb2.MetricThreshold(
                            value_threshold=config_pb2.GenericValueThreshold(
                                lower_bound={'value': 10}))),
                ],
                model_names=['candidate', 'baseline'],
                example_weights=config_pb2.ExampleWeightOptions(
                    unweighted=True)),
            config_pb2.MetricsSpec(
                metrics=[
                    config_pb2.MetricConfig(
                        class_name='WeightedExampleCount',
                        per_slice_thresholds=[
                            config_pb2.PerSliceMetricThreshold(
                                slicing_specs=slicing_specs,
                                # 1.5 < 1, NOT OK.
                                threshold=config_pb2.MetricThreshold(
                                    value_threshold=config_pb2
                                    .GenericValueThreshold(
                                        upper_bound={'value': 1})))
                        ],
                        # missing cross slice
                        cross_slice_thresholds=[
                            config_pb2.CrossSliceMetricThreshold(
                                cross_slicing_specs=cross_slicing_specs,
                                threshold=config_pb2.MetricThreshold(
                                    value_threshold=config_pb2
                                    .GenericValueThreshold(
                                        upper_bound={'value': 1})))
                        ]),
                ],
                model_names=['candidate', 'baseline'],
                example_weights=config_pb2.ExampleWeightOptions(weighted=True)),
            config_pb2.MetricsSpec(
                metrics=[
                    config_pb2.MetricConfig(
                        class_name='MeanLabel',
                        # 0.5 > 1 and 0.5 > 1?: NOT OK.
                        threshold=config_pb2.MetricThreshold(
                            change_threshold=config_pb2.GenericChangeThreshold(
                                direction=config_pb2.MetricDirection
                                .HIGHER_IS_BETTER,
                                relative={'value': 1},
                                absolute={'value': 1}))),
                    config_pb2.MetricConfig(
                        # MeanPrediction = (0+0)/(1+0.5) = 0
                        class_name='MeanPrediction',
                        # -.01 < 0 < .01, OK.
                        # Diff% = -.333/.333 = -100% < -99%, OK.
                        # Diff = 0 - .333 = -.333 < 0, OK.
                        threshold=config_pb2.MetricThreshold(
                            value_threshold=config_pb2.GenericValueThreshold(
                                upper_bound={'value': .01},
                                lower_bound={'value': -.01}),
                            change_threshold=config_pb2.GenericChangeThreshold(
                                direction=config_pb2.MetricDirection
                                .LOWER_IS_BETTER,
                                relative={'value': -.99},
                                absolute={'value': 0})))
                ],
                model_names=['candidate', 'baseline']),
        ],
        options=config_pb2.Options(
            disabled_outputs={'values': ['eval_config_pb2.json']}),
    )
    slice_spec = [
        slicer.SingleSliceSpec(spec=s) for s in eval_config.slicing_specs
    ]
    eval_shared_models = {
        'candidate': eval_shared_model,
        'baseline': baseline_eval_shared_model
    }
    extractors = [
        features_extractor.FeaturesExtractor(
            eval_config=eval_config,
            tensor_representations=tensor_adapter_config.tensor_representations
        ),
        labels_extractor.LabelsExtractor(eval_config),
        example_weights_extractor.ExampleWeightsExtractor(eval_config),
        predictions_extractor.PredictionsExtractor(
            eval_shared_model=eval_shared_models, eval_config=eval_config),
        unbatch_extractor.UnbatchExtractor(),
        slice_key_extractor.SliceKeyExtractor(slice_spec=slice_spec)
    ]
    evaluators = [
        metrics_plots_and_validations_evaluator
        .MetricsPlotsAndValidationsEvaluator(
            eval_config=eval_config, eval_shared_model=eval_shared_models)
    ]
    output_paths = {
        constants.VALIDATIONS_KEY: validations_file,
    }
    writers = [
        metrics_plots_and_validations_writer.MetricsPlotsAndValidationsWriter(
            output_paths,
            eval_config=eval_config,
            add_metrics_callbacks=[],
            output_file_format=output_file_format)
    ]

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      _ = (
          pipeline
          | 'Create' >> beam.Create([e.SerializeToString() for e in examples])
          | 'BatchExamples' >> tfx_io.BeamSource()
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | 'ExtractEvaluate' >> model_eval_lib.ExtractAndEvaluate(
              extractors=extractors, evaluators=evaluators)
          | 'WriteResults' >> model_eval_lib.WriteResults(writers=writers))
      # pylint: enable=no-value-for-parameter

    validation_result = (
        metrics_plots_and_validations_writer
        .load_and_deserialize_validation_result(
            os.path.dirname(validations_file), output_file_format))

    expected_validations = [
        text_format.Parse(
            """
            metric_key {
              name: "example_count"
              model_name: "candidate"
              example_weighted { }
            }
            metric_threshold {
              value_threshold {
                lower_bound {
                  value: 10.0
                }
              }
            }
            metric_value {
              double_value {
                value: 2.0
              }
            }
            """, validation_result_pb2.ValidationFailure()),
        text_format.Parse(
            """
            metric_key {
              name: "weighted_example_count"
              model_name: "candidate"
              example_weighted { value: true }
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
            """, validation_result_pb2.ValidationFailure()),
        text_format.Parse(
            """
            metric_key {
              name: "mean_label"
              model_name: "candidate"
              is_diff: true
              example_weighted { value: true }
            }
            metric_threshold {
              change_threshold {
                absolute {
                  value: 1.0
                }
                relative {
                  value: 1.0
                }
                direction: HIGHER_IS_BETTER
              }
            }
            metric_value {
              double_value {
                value: 0.0
              }
            }
            """, validation_result_pb2.ValidationFailure()),
    ]
    self.assertFalse(validation_result.validation_ok)
    self.assertFalse(validation_result.missing_thresholds)
    self.assertLen(validation_result.metric_validations_per_slice, 1)
    self.assertCountEqual(
        expected_validations,
        validation_result.metric_validations_per_slice[0].failures)

    expected_missing_slices = [
        config_pb2.SlicingSpec(feature_keys=['slice_does_not_exist'])
    ]
    self.assertLen(validation_result.missing_slices, 1)
    self.assertCountEqual(expected_missing_slices,
                          validation_result.missing_slices)
    expected_missing_cross_slices = [
        config_pb2.CrossSlicingSpec(
            baseline_spec=config_pb2.SlicingSpec(
                feature_keys=['slice_does_not_exist']),
            slicing_specs=[
                config_pb2.SlicingSpec(feature_keys=['slice_does_not_exist'])
            ])
    ]
    self.assertLen(validation_result.missing_cross_slices, 1)
    self.assertCountEqual(expected_missing_cross_slices,
                          validation_result.missing_cross_slices)

    expected_slicing_details = [
        text_format.Parse(
            """
            slicing_spec {
            }
            num_matching_slices: 1
            """, validation_result_pb2.SlicingDetails()),
    ]
    self.assertLen(validation_result.validation_details.slicing_details, 1)
    self.assertCountEqual(expected_slicing_details,
                          validation_result.validation_details.slicing_details)

  @parameterized.named_parameters(_OUTPUT_FORMAT_PARAMS)
  def testWriteValidationResultsNoThresholds(self, output_file_format):
    model_dir, baseline_dir = self._getExportDir(), self._getBaselineDir()
    eval_shared_model = self._build_keras_model(model_dir, mul=0)
    baseline_eval_shared_model = self._build_keras_model(baseline_dir, mul=1)
    validations_file = os.path.join(self._getTempDir(),
                                    constants.VALIDATIONS_KEY)
    schema = text_format.Parse(
        """
        tensor_representation_group {
          key: ""
          value {
            tensor_representation {
              key: "input_1"
              value {
                dense_tensor {
                  column_name: "input_1"
                  shape { dim { size: 1 } }
                }
              }
            }
          }
        }
        feature {
          name: "input_1"
          type: FLOAT
        }
        feature {
          name: "label"
          type: FLOAT
        }
        feature {
          name: "example_weight"
          type: FLOAT
        }
        feature {
          name: "extra_feature"
          type: BYTES
        }
        """, schema_pb2.Schema())
    tfx_io = test_util.InMemoryTFExampleRecord(
        schema=schema, raw_record_column_name=constants.ARROW_INPUT_COLUMN)
    tensor_adapter_config = tensor_adapter.TensorAdapterConfig(
        arrow_schema=tfx_io.ArrowSchema(),
        tensor_representations=tfx_io.TensorRepresentations())
    examples = [
        self._makeExample(
            input_1=0.0,
            label=1.0,
            example_weight=1.0,
            extra_feature='non_model_feature'),
        self._makeExample(
            input_1=1.0,
            label=0.0,
            example_weight=0.5,
            extra_feature='non_model_feature'),
    ]

    slicing_specs = [
        config_pb2.SlicingSpec(),
    ]
    eval_config = config_pb2.EvalConfig(
        model_specs=[
            config_pb2.ModelSpec(
                name='candidate',
                label_key='label',
                example_weight_key='example_weight'),
            config_pb2.ModelSpec(
                name='baseline',
                label_key='label',
                example_weight_key='example_weight',
                is_baseline=True)
        ],
        slicing_specs=slicing_specs,
        metrics_specs=[
            config_pb2.MetricsSpec(
                metrics=[
                    config_pb2.MetricConfig(class_name='ExampleCount'),
                    config_pb2.MetricConfig(class_name='MeanLabel')
                ],
                model_names=['candidate', 'baseline']),
        ],
        options=config_pb2.Options(
            disabled_outputs={'values': ['eval_config_pb2.json']}),
    )
    slice_spec = [
        slicer.SingleSliceSpec(spec=s) for s in eval_config.slicing_specs
    ]
    eval_shared_models = {
        'candidate': eval_shared_model,
        'baseline': baseline_eval_shared_model
    }
    extractors = [
        features_extractor.FeaturesExtractor(
            eval_config=eval_config,
            tensor_representations=tensor_adapter_config.tensor_representations
        ),
        labels_extractor.LabelsExtractor(eval_config),
        example_weights_extractor.ExampleWeightsExtractor(eval_config),
        predictions_extractor.PredictionsExtractor(
            eval_shared_model=eval_shared_models, eval_config=eval_config),
        unbatch_extractor.UnbatchExtractor(),
        slice_key_extractor.SliceKeyExtractor(slice_spec=slice_spec)
    ]
    evaluators = [
        metrics_plots_and_validations_evaluator
        .MetricsPlotsAndValidationsEvaluator(
            eval_config=eval_config, eval_shared_model=eval_shared_models)
    ]
    output_paths = {
        constants.VALIDATIONS_KEY: validations_file,
    }
    writers = [
        metrics_plots_and_validations_writer.MetricsPlotsAndValidationsWriter(
            output_paths,
            eval_config=eval_config,
            add_metrics_callbacks=[],
            output_file_format=output_file_format)
    ]

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      _ = (
          pipeline
          | 'Create' >> beam.Create([e.SerializeToString() for e in examples])
          | 'BatchExamples' >> tfx_io.BeamSource()
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | 'ExtractEvaluate' >> model_eval_lib.ExtractAndEvaluate(
              extractors=extractors, evaluators=evaluators)
          | 'WriteResults' >> model_eval_lib.WriteResults(writers=writers))
      # pylint: enable=no-value-for-parameter

    validation_result = (
        metrics_plots_and_validations_writer
        .load_and_deserialize_validation_result(
            os.path.dirname(validations_file), output_file_format))

    self.assertFalse(validation_result.validation_ok)
    self.assertTrue(validation_result.missing_thresholds)
    self.assertEmpty(validation_result.metric_validations_per_slice)

    # Add rubber stamp would make validation ok.
    writers = [
        metrics_plots_and_validations_writer.MetricsPlotsAndValidationsWriter(
            output_paths,
            eval_config=eval_config,
            add_metrics_callbacks=[],
            output_file_format=output_file_format,
            rubber_stamp=True)
    ]

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      _ = (
          pipeline
          | 'Create' >> beam.Create([e.SerializeToString() for e in examples])
          | 'BatchExamples' >> tfx_io.BeamSource()
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | 'ExtractEvaluate' >> model_eval_lib.ExtractAndEvaluate(
              extractors=extractors, evaluators=evaluators)
          | 'WriteResults' >> model_eval_lib.WriteResults(writers=writers))
      # pylint: enable=no-value-for-parameter

    validation_result = (
        metrics_plots_and_validations_writer
        .load_and_deserialize_validation_result(
            os.path.dirname(validations_file), output_file_format))

    self.assertTrue(validation_result.validation_ok)
    self.assertFalse(validation_result.missing_thresholds)
    self.assertEmpty(validation_result.metric_validations_per_slice)
    self.assertTrue(validation_result.rubber_stamp)

  @parameterized.named_parameters(('load_from_dir', ''),
                                  ('load_metrics', 'metrics'))
  def testLoadAndDeserializeMetricsNoSuffix(self, load_path_suffix):
    metrics_for_slice = text_format.Parse(
        """
        slice_key {
          single_slice_keys {
            column: "x"
            float_value: 0
          }
        }
        metric_keys_and_values {
          key {
            name: "example_count"
            example_weighted: { value: false }
          }
          value {
            double_value {
              value: 1.0
            }
          }
        }
        """, metrics_for_slice_pb2.MetricsForSlice())

    output_dir = self._getTempDir()
    # TFMA previously wrote metrics in tfrecord format but without suffix.
    legacy_metrics_path = os.path.join(
        output_dir, constants.METRICS_KEY + '-00000-of-00001')
    with tf.io.TFRecordWriter(legacy_metrics_path) as writer:
      writer.write(metrics_for_slice.SerializeToString())

    metric_records = list(
        metrics_plots_and_validations_writer.load_and_deserialize_metrics(
            os.path.join(output_dir, load_path_suffix), 'tfrecord'))
    self.assertLen(metric_records, 1, 'metrics: %s' % metric_records)
    # verify proto roud trips unchanged.
    self.assertProtoEquals(metrics_for_slice, metric_records[0])

  @parameterized.named_parameters(('load_from_dir', ''),
                                  ('load_plots', 'plots'))
  def testLoadAndDeserializePlotsNoSuffix(self, load_path_suffix):
    plots_for_slice = text_format.Parse(
        """
        slice_key {
          single_slice_keys {
            column: "x"
            float_value: 0
          }
        }
        plot_keys_and_values {
          key {}
          value {
            calibration_histogram_buckets {
              buckets {
                lower_threshold_inclusive: 0.0
                upper_threshold_exclusive: 0.1
              }
            }
          }
        }
        """, metrics_for_slice_pb2.PlotsForSlice())

    output_dir = self._getTempDir()
    # TFMA previously wrote plots in tfrecord format but without suffix.
    legacy_plots_path = os.path.join(output_dir,
                                     constants.PLOTS_KEY + '-00000-of-00001')
    with tf.io.TFRecordWriter(legacy_plots_path) as writer:
      writer.write(plots_for_slice.SerializeToString())

    plot_records = list(
        metrics_plots_and_validations_writer.load_and_deserialize_plots(
            os.path.join(output_dir, load_path_suffix), 'tfrecord'))
    self.assertLen(plot_records, 1, 'plots: %s' % plot_records)
    # verify proto roud trips unchanged.
    self.assertProtoEquals(plots_for_slice, plot_records[0])

  @parameterized.named_parameters(('load_from_dir', ''),
                                  ('load_attributions', 'attributions'))
  def testLoadAndDeserializeAttributionsNoSuffix(self, load_path_suffix):
    attributions_for_slice = text_format.Parse(
        """
        slice_key { }
        attributions_keys_and_values {
          key {
            name: "attrbution_name"
          }
          values {
            key: 'f1'
            value {
              double_value { value: 1.0 }
            }
          }
        }
        """, metrics_for_slice_pb2.AttributionsForSlice())

    output_dir = self._getTempDir()
    # TFMA previously wrote attributions in tfrecord format but without suffix.
    legacy_attributions_path = os.path.join(
        output_dir, constants.ATTRIBUTIONS_KEY + '-00000-of-00001')
    with tf.io.TFRecordWriter(legacy_attributions_path) as writer:
      writer.write(attributions_for_slice.SerializeToString())

    attribution_records = list(
        metrics_plots_and_validations_writer.load_and_deserialize_attributions(
            os.path.join(output_dir, load_path_suffix), 'tfrecord'))
    self.assertLen(attribution_records, 1,
                   'attributions: %s' % attribution_records)
    # verify proto roud trips unchanged.
    self.assertProtoEquals(attributions_for_slice, attribution_records[0])

  @parameterized.named_parameters(_OUTPUT_FORMAT_PARAMS)
  def testWriteAttributions(self, output_file_format):
    attributions_file = os.path.join(self._getTempDir(), 'attributions')
    eval_config = config_pb2.EvalConfig(
        model_specs=[config_pb2.ModelSpec()],
        metrics_specs=[
            config_pb2.MetricsSpec(metrics=[
                config_pb2.MetricConfig(class_name=attributions
                                        .TotalAttributions().__class__.__name__)
            ])
        ],
        options=config_pb2.Options(
            disabled_outputs={'values': ['eval_config.json']}))
    extractors = [slice_key_extractor.SliceKeyExtractor()]
    evaluators = [
        metrics_plots_and_validations_evaluator
        .MetricsPlotsAndValidationsEvaluator(eval_config=eval_config)
    ]
    output_paths = {
        constants.ATTRIBUTIONS_KEY: attributions_file,
    }
    writers = [
        metrics_plots_and_validations_writer.MetricsPlotsAndValidationsWriter(
            output_paths,
            eval_config=eval_config,
            output_file_format=output_file_format)
    ]

    example1 = {
        'labels': None,
        'predictions': None,
        'example_weights': None,
        'features': {},
        'attributions': {
            'feature1': 1.1,
            'feature2': 1.2
        }
    }
    example2 = {
        'labels': None,
        'predictions': None,
        'example_weights': None,
        'features': {},
        'attributions': {
            'feature1': 2.1,
            'feature2': 2.2
        }
    }
    example3 = {
        'labels': None,
        'predictions': None,
        'example_weights': None,
        'features': {},
        'attributions': {
            'feature1': np.array([3.1]),
            'feature2': np.array([3.2])
        }
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      _ = (
          pipeline
          | 'Create' >> beam.Create([example1, example2, example3])
          | 'ExtractEvaluate' >> model_eval_lib.ExtractAndEvaluate(
              extractors=extractors, evaluators=evaluators)
          | 'WriteResults' >> model_eval_lib.WriteResults(writers=writers))
      # pylint: enable=no-value-for-parameter

    expected_attributions_for_slice = text_format.Parse(
        """
        slice_key {}
        attributions_keys_and_values {
          key {
            name: "total_attributions"
            example_weighted { }
          }
          values {
            key: "feature1"
            value: {
              double_value {
                value: 6.3
              }
            }
          }
          values {
            key: "feature2"
            value: {
              double_value {
                value: 6.6
              }
            }
          }
        }""", metrics_for_slice_pb2.AttributionsForSlice())

    attribution_records = list(
        metrics_plots_and_validations_writer.load_and_deserialize_attributions(
            attributions_file, output_file_format))
    self.assertLen(attribution_records, 1)
    self.assertProtoEquals(expected_attributions_for_slice,
                           attribution_records[0])


