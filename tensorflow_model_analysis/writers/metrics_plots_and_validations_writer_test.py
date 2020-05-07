# Lint as: python3
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import apache_beam as beam
import tensorflow as tf
from tensorflow_model_analysis import config
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.eval_saved_model.example_trainers import fixed_prediction_estimator
from tensorflow_model_analysis.evaluators import metrics_and_plots_evaluator
from tensorflow_model_analysis.evaluators import metrics_and_plots_evaluator_v2
from tensorflow_model_analysis.extractors import batched_input_extractor
from tensorflow_model_analysis.extractors import batched_predict_extractor_v2
from tensorflow_model_analysis.extractors import predict_extractor
from tensorflow_model_analysis.extractors import slice_key_extractor
from tensorflow_model_analysis.extractors import unbatch_extractor
from tensorflow_model_analysis.post_export_metrics import post_export_metrics
from tensorflow_model_analysis.proto import metrics_for_slice_pb2
from tensorflow_model_analysis.proto import validation_result_pb2
from tensorflow_model_analysis.slicer import slicer_lib as slicer
from tensorflow_model_analysis.writers import metrics_plots_and_validations_writer
from tfx_bsl.tfxio import tensor_adapter
from tfx_bsl.tfxio import test_util
from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2


class MetricsPlotsAndValidationsWriterTest(testutil.TensorflowModelAnalysisTest
                                          ):

  def setUp(self):
    super(MetricsPlotsAndValidationsWriterTest, self).setUp()
    self.longMessage = True  # pylint: disable=invalid-name

  def _getTempDir(self):
    return tempfile.mkdtemp()

  def _getExportDir(self):
    return os.path.join(self._getTempDir(), 'export_dir')

  def _getBaselineDir(self):
    return os.path.join(self._getTempDir(), 'baseline_export_dir')

  def _build_keras_model(self, model_dir, mul):
    input_layer = tf.keras.layers.Input(shape=(1,), name='input')
    output_layer = tf.keras.layers.Lambda(
        lambda x, mul: x * mul, output_shape=(1,), arguments={'mul': mul})(
            input_layer)
    model = tf.keras.models.Model([input_layer], output_layer)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=.001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy'])

    model.fit(x=[[0], [1]], y=[[0], [1]], steps_per_epoch=1)
    model.save(model_dir, save_format='tf')
    return self.createTestEvalSharedModel(
        eval_saved_model_path=model_dir, tags=[tf.saved_model.SERVING])

  def testWriteValidationResults(self):
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
              key: "input"
              value {
                dense_tensor {
                  column_name: "input"
                  shape { dim { size: 1 } }
                }
              }
            }
          }
        }
        feature {
          name: "input"
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
            input=0.0,
            label=1.0,
            example_weight=1.0,
            extra_feature='non_model_feature'),
        self._makeExample(
            input=1.0,
            label=0.0,
            example_weight=0.5,
            extra_feature='non_model_feature'),
    ]

    eval_config = config.EvalConfig(
        model_specs=[
            config.ModelSpec(
                name='candidate',
                label_key='label',
                example_weight_key='example_weight'),
            config.ModelSpec(
                name='baseline',
                label_key='label',
                example_weight_key='example_weight',
                is_baseline=True)
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
                    config.MetricConfig(
                        class_name='ExampleCount',
                        # 2 > 10, NOT OK.
                        threshold=config.MetricThreshold(
                            value_threshold=config.GenericValueThreshold(
                                lower_bound={'value': 10}))),
                    config.MetricConfig(
                        class_name='MeanLabel',
                        # 0 > 0 and 0 > 0%?: NOT OK.
                        threshold=config.MetricThreshold(
                            change_threshold=config.GenericChangeThreshold(
                                direction=config.MetricDirection
                                .HIGHER_IS_BETTER,
                                relative={'value': 0},
                                absolute={'value': 0}))),
                    config.MetricConfig(
                        # MeanPrediction = (0+0)/(1+0.5) = 0
                        class_name='MeanPrediction',
                        # -.01 < 0 < .01, OK.
                        # Diff% = -.333/.333 = -100% < -99%, OK.
                        # Diff = 0 - .333 = -.333 < 0, OK.
                        threshold=config.MetricThreshold(
                            value_threshold=config.GenericValueThreshold(
                                upper_bound={'value': .01},
                                lower_bound={'value': -.01}),
                            change_threshold=config.GenericChangeThreshold(
                                direction=config.MetricDirection
                                .LOWER_IS_BETTER,
                                relative={'value': -.99},
                                absolute={'value': 0})))
                ],
                model_names=['candidate', 'baseline']),
        ],
        options=config.Options(
            disabled_outputs={'values': ['eval_config.json']}),
    )
    slice_spec = [
        slicer.SingleSliceSpec(spec=s) for s in eval_config.slicing_specs
    ]
    eval_shared_models = {
        'candidate': eval_shared_model,
        'baseline': baseline_eval_shared_model
    }
    extractors = [
        batched_input_extractor.BatchedInputExtractor(eval_config),
        batched_predict_extractor_v2.BatchedPredictExtractor(
            eval_shared_model=eval_shared_models,
            eval_config=eval_config,
            tensor_adapter_config=tensor_adapter_config),
        unbatch_extractor.UnbatchExtractor(),
        slice_key_extractor.SliceKeyExtractor(slice_spec=slice_spec)
    ]
    evaluators = [
        metrics_and_plots_evaluator_v2.MetricsAndPlotsEvaluator(
            eval_config=eval_config, eval_shared_model=eval_shared_models)
    ]
    output_paths = {
        constants.VALIDATIONS_KEY: validations_file,
    }
    writers = [
        metrics_plots_and_validations_writer.MetricsPlotsAndValidationsWriter(
            output_paths, add_metrics_callbacks=[])
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

    validation_result = model_eval_lib.load_validation_result(
        os.path.dirname(validations_file))

    expected_validations = [
        text_format.Parse(
            """
            metric_key {
              name: "weighted_example_count"
              model_name: "candidate"
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
              name: "example_count"
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
              name: "mean_label"
              model_name: "candidate"
              is_diff: true
            }
            metric_threshold {
              change_threshold {
                absolute {
                  value: 0.0
                }
                relative {
                  value: 0.0
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
    self.assertLen(validation_result.metric_validations_per_slice, 1)
    self.assertCountEqual(
        expected_validations,
        validation_result.metric_validations_per_slice[0].failures)

  def testWriteMetricsAndPlots(self):
    metrics_file = os.path.join(self._getTempDir(), 'metrics')
    plots_file = os.path.join(self._getTempDir(), 'plots')
    temp_eval_export_dir = os.path.join(self._getTempDir(), 'eval_export_dir')

    _, eval_export_dir = (
        fixed_prediction_estimator.simple_fixed_prediction_estimator(
            None, temp_eval_export_dir))
    eval_config = config.EvalConfig(
        model_specs=[config.ModelSpec()],
        options=config.Options(
            disabled_outputs={'values': ['eval_config.json']}))
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
        metrics_plots_and_validations_writer.MetricsPlotsAndValidationsWriter(
            output_paths, eval_shared_model.add_metrics_callbacks)
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
              eval_shared_model=eval_shared_model,
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
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
