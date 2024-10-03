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
"""Test for MetricsPlotsAndValidationsEvaluator."""


import pytest
import os

from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.addons.fairness.metrics import lift
from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.eval_saved_model.example_trainers import dnn_classifier
from tensorflow_model_analysis.eval_saved_model.example_trainers import fixed_prediction_estimator_extra_fields
from tensorflow_model_analysis.eval_saved_model.example_trainers import linear_classifier
from tensorflow_model_analysis.eval_saved_model.example_trainers import multi_head
from tensorflow_model_analysis.evaluators import metrics_plots_and_validations_evaluator
from tensorflow_model_analysis.extractors import example_weights_extractor
from tensorflow_model_analysis.extractors import features_extractor
from tensorflow_model_analysis.extractors import labels_extractor
from tensorflow_model_analysis.extractors import legacy_predict_extractor
from tensorflow_model_analysis.extractors import materialized_predictions_extractor
from tensorflow_model_analysis.extractors import predictions_extractor
from tensorflow_model_analysis.extractors import slice_key_extractor
from tensorflow_model_analysis.extractors import unbatch_extractor
from tensorflow_model_analysis.metrics import attributions
from tensorflow_model_analysis.metrics import binary_confusion_matrices
from tensorflow_model_analysis.metrics import calibration
from tensorflow_model_analysis.metrics import calibration_plot
from tensorflow_model_analysis.metrics import confusion_matrix_plot
from tensorflow_model_analysis.metrics import metric_specs
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import ndcg
from tensorflow_model_analysis.post_export_metrics import metrics as metric_fns
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.proto import validation_result_pb2
from tensorflow_model_analysis.utils.keras_lib import tf_keras
from tfx_bsl.tfxio import raw_tf_record
from tfx_bsl.tfxio import tensor_adapter
from tfx_bsl.tfxio import test_util

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2



_TF_MAJOR_VERSION = int(tf.version.VERSION.split('.')[0])


def _addExampleCountMetricCallback(  # pylint: disable=invalid-name
    features_dict, predictions_dict, labels_dict
):
  del features_dict
  del labels_dict
  metric_ops = {}
  value_op, update_op = metric_fns.total(
      tf.shape(input=predictions_dict['logits'])[0]
  )
  metric_ops['added_example_count'] = (value_op, update_op)
  return metric_ops


@pytest.mark.xfail(run=False, reason="PR 183 This class contains tests that fail and needs to be fixed. "
"If all tests pass, please remove this mark.")
class MetricsPlotsAndValidationsEvaluatorTest(
    testutil.TensorflowModelAnalysisTest, parameterized.TestCase
):

  def _getExportDir(self):
    return os.path.join(self._getTempDir(), 'export_dir')

  def _getBaselineDir(self):
    return os.path.join(self._getTempDir(), 'baseline_export_dir')

  def _build_keras_model(self, model_name, model_dir, mul):
    input_layer = tf_keras.layers.Input(shape=(1,), name='input_1')
    output_layer = tf_keras.layers.Lambda(
        lambda x, mul: x * mul, output_shape=(1,), arguments={'mul': mul}
    )(input_layer)
    model = tf_keras.models.Model([input_layer], output_layer)
    model.compile(
        optimizer=tf_keras.optimizers.Adam(lr=0.001),
        loss=tf_keras.losses.BinaryCrossentropy(name='loss'),
        metrics=['accuracy'],
    )
    model.save(model_dir, save_format='tf')
    return self.createTestEvalSharedModel(
        model_name=model_name, eval_saved_model_path=model_dir
    )

  def testFilterAndSeparateComputations(self):
    eval_config = config_pb2.EvalConfig(
        model_specs=[
            config_pb2.ModelSpec(name='candidate', label_key='tips'),
            config_pb2.ModelSpec(
                name='baseline', label_key='tips', is_baseline=True
            ),
        ],
        cross_slicing_specs=[config_pb2.CrossSlicingSpec()],
    )
    metrics_specs = metric_specs.specs_from_metrics(
        [
            tf_keras.metrics.BinaryAccuracy(name='accuracy'),
            tf_keras.metrics.AUC(name='auc', num_thresholds=10000),
            tf_keras.metrics.AUC(
                name='auc_precison_recall', curve='PR', num_thresholds=10000
            ),
            tf_keras.metrics.Precision(name='precision'),
            tf_keras.metrics.Recall(name='recall'),
            calibration.MeanLabel(name='mean_label'),
            calibration.MeanPrediction(name='mean_prediction'),
            calibration.Calibration(name='calibration'),
            confusion_matrix_plot.ConfusionMatrixPlot(
                name='confusion_matrix_plot'
            ),
            calibration_plot.CalibrationPlot(name='calibration_plot'),
            lift.Lift(name='lift'),
        ],
        model_names=['candidate', 'baseline'],
        binarize=config_pb2.BinarizationOptions(class_ids={'values': [0, 5]}),
    )
    computations = metric_specs.to_computations(
        metrics_specs, eval_config=eval_config
    )
    non_derived, derived, cross_slice, ci_derived = (
        metrics_plots_and_validations_evaluator._filter_and_separate_computations(
            computations
        )
    )
    # 2 models x 2 classes x _binary_confusion_matrix_[0.5]_100,
    # 2 models x 2 classes x _CalibrationHistogramCombiner
    # 2 models x 2 classes x _calibration_historgram_27
    # 2 models x 2 classes x _CompilableMetricsCombiner,
    # 2 models x 2 classes x _WeightedLabelsPredictionsExamplesCombiner,
    # 4 models x _ExampleCountCombiner
    self.assertLen(non_derived, 20)
    # 2 models x 2 classes x _binary_confusion_matrices_[0.5],
    # 2 models x 2 classes x _binary_confusion_matrices_10000
    # 2 models x 2 classes x _binary_confusion_matrices_confusion_matrix_plot
    # 2 models x 2 classes x precision
    # 2 models x 2 classes x recall
    # 2 models x 2 classes x calibration
    # 2 models x 2 classes x auc_precision_recall
    # 2 models x 2 classes x mean_prediction
    # 2 models x 2 classes x mean_label
    # 2 models x 2 classes x confusion_matrix_plot
    # 2 models x 2 classes x calibration_plot
    # 2 models x 2 classes x auc
    # 2 models x 2 classes x accuracy
    self.assertLen(derived, 52)
    # 2 models x 2 classes x lift
    self.assertLen(cross_slice, 4)
    # None of the metric has CIDerivedMetricComputation.
    self.assertEmpty(ci_derived)

  def testFilterAndSeparateComputationsWithCIDerivedMetrics(self):

    def derived_metric_fn():
      pass

    def ci_derived_fn():
      pass

    computations = [
        metric_types.DerivedMetricComputation(
            [metric_types.MetricKey('key1')], derived_metric_fn
        ),
        metric_types.CIDerivedMetricComputation(
            [metric_types.MetricKey('key1')], ci_derived_fn
        ),
        metric_types.CIDerivedMetricComputation(
            [metric_types.MetricKey('key1')], ci_derived_fn
        ),
    ]
    _, derived, _, ci_derived = (
        metrics_plots_and_validations_evaluator._filter_and_separate_computations(
            computations
        )
    )

    self.assertLen(derived, 1)
    self.assertLen(ci_derived, 1)

  def testEvaluateWithKerasAndValidateMetrics(self):
    model_dir, baseline_dir = self._getExportDir(), self._getBaselineDir()
    eval_shared_model = self._build_keras_model('candidate', model_dir, mul=0)
    baseline_eval_shared_model = self._build_keras_model(
        'baseline', baseline_dir, mul=1
    )

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
        """,
        schema_pb2.Schema(),
    )
    tfx_io = test_util.InMemoryTFExampleRecord(
        schema=schema, raw_record_column_name=constants.ARROW_INPUT_COLUMN
    )
    tensor_adapter_config = tensor_adapter.TensorAdapterConfig(
        arrow_schema=tfx_io.ArrowSchema(),
        tensor_representations=tfx_io.TensorRepresentations(),
    )

    examples = [
        self._makeExample(
            input_1=0.0,
            label=1.0,
            example_weight=1.0,
            extra_feature='non_model_feature',
        ),
        self._makeExample(
            input_1=1.0,
            label=0.0,
            example_weight=0.5,
            extra_feature='non_model_feature',
        ),
    ]

    eval_config = config_pb2.EvalConfig(
        model_specs=[
            config_pb2.ModelSpec(
                name='candidate',
                label_key='label',
                example_weight_key='example_weight',
            ),
            config_pb2.ModelSpec(
                name='baseline',
                label_key='label',
                example_weight_key='example_weight',
                is_baseline=True,
            ),
        ],
        slicing_specs=[config_pb2.SlicingSpec()],
        metrics_specs=[
            config_pb2.MetricsSpec(
                metrics=[
                    config_pb2.MetricConfig(
                        class_name='ExampleCount',
                        # 2 > 10, NOT OK.
                        threshold=config_pb2.MetricThreshold(
                            value_threshold=config_pb2.GenericValueThreshold(
                                lower_bound={'value': 10}
                            )
                        ),
                    ),
                ],
                model_names=['candidate', 'baseline'],
                example_weights=config_pb2.ExampleWeightOptions(
                    unweighted=True
                ),
            ),
            config_pb2.MetricsSpec(
                metrics=[
                    config_pb2.MetricConfig(
                        class_name='WeightedExampleCount',
                        # 1.5 < 1, NOT OK.
                        threshold=config_pb2.MetricThreshold(
                            value_threshold=config_pb2.GenericValueThreshold(
                                upper_bound={'value': 1}
                            )
                        ),
                    ),
                ],
                model_names=['candidate', 'baseline'],
                example_weights=config_pb2.ExampleWeightOptions(weighted=True),
            ),
            config_pb2.MetricsSpec(
                metrics=[
                    config_pb2.MetricConfig(
                        class_name='MeanLabel',
                        # 0 > 1 and 0 > 1?: NOT OK.
                        threshold=config_pb2.MetricThreshold(
                            change_threshold=config_pb2.GenericChangeThreshold(
                                direction=config_pb2.MetricDirection.HIGHER_IS_BETTER,
                                relative={'value': 1},
                                absolute={'value': 1},
                            )
                        ),
                    ),
                    config_pb2.MetricConfig(
                        # MeanPrediction = (0+0)/(1+0.5) = 0
                        class_name='MeanPrediction',
                        # -.01 < 0 < .01, OK.
                        # Diff% = -.333/.333 = -100% < -99%, OK.
                        # Diff = 0 - .333 = -.333 < 0, OK.
                        threshold=config_pb2.MetricThreshold(
                            value_threshold=config_pb2.GenericValueThreshold(
                                upper_bound={'value': 0.01},
                                lower_bound={'value': -0.01},
                            ),
                            change_threshold=config_pb2.GenericChangeThreshold(
                                direction=config_pb2.MetricDirection.LOWER_IS_BETTER,
                                relative={'value': -0.99},
                                absolute={'value': 0},
                            ),
                        ),
                    ),
                ],
                thresholds={
                    'loss': config_pb2.MetricThreshold(
                        value_threshold=config_pb2.GenericValueThreshold(
                            upper_bound={'value': 0}
                        )
                    )
                },
                model_names=['candidate', 'baseline'],
            ),
        ],
    )
    eval_shared_models = [eval_shared_model, baseline_eval_shared_model]
    extractors = [
        features_extractor.FeaturesExtractor(
            eval_config=eval_config,
            tensor_representations=tensor_adapter_config.tensor_representations,
        ),
        labels_extractor.LabelsExtractor(eval_config),
        example_weights_extractor.ExampleWeightsExtractor(eval_config),
        predictions_extractor.PredictionsExtractor(
            eval_shared_model=eval_shared_models, eval_config=eval_config
        ),
        unbatch_extractor.UnbatchExtractor(),
        slice_key_extractor.SliceKeyExtractor(eval_config=eval_config),
    ]
    evaluators = [
        metrics_plots_and_validations_evaluator.MetricsPlotsAndValidationsEvaluator(
            eval_config=eval_config, eval_shared_model=eval_shared_models
        )
    ]

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      evaluations = (
          pipeline
          | 'Create' >> beam.Create([e.SerializeToString() for e in examples])
          | 'BatchExamples' >> tfx_io.BeamSource()
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | 'ExtractEvaluate'
          >> model_eval_lib.ExtractAndEvaluate(
              extractors=extractors, evaluators=evaluators
          )
      )

      # pylint: enable=no-value-for-parameter

      def check_validations(got):
        try:
          self.assertLen(got, 1)
          got = got[0]
          expected_metric_validations_per_slice = [
              text_format.Parse(
                  """
                  metric_key {
                    name: "loss"
                    model_name: "candidate"
                  }
                  metric_threshold {
                    value_threshold {
                      upper_bound {
                        value: 0.0
                      }
                    }
                  }
                  metric_value {
                    double_value {
                      value: 7.712474346160889
                    }
                  }
                  """,
                  validation_result_pb2.ValidationFailure(),
              ),
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
                  """,
                  validation_result_pb2.ValidationFailure(),
              ),
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
                  """,
                  validation_result_pb2.ValidationFailure(),
              ),
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
                  """,
                  validation_result_pb2.ValidationFailure(),
              ),
          ]
          # Loss not supported in TFv1
          if _TF_MAJOR_VERSION < 2:
            expected_metric_validations_per_slice[0].ClearField('metric_value')
            expected_metric_validations_per_slice[0].message = (
                'Metric not found.'
            )
          self.assertFalse(got.validation_ok)
          self.assertLen(got.metric_validations_per_slice, 1)
          self.assertLen(
              got.metric_validations_per_slice[0].failures,
              len(expected_metric_validations_per_slice),
          )
          self.assertCountEqual(
              got.metric_validations_per_slice[0].failures,
              expected_metric_validations_per_slice,
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(
          evaluations[constants.VALIDATIONS_KEY], check_validations
      )

    metric_filter = beam.metrics.metric.MetricsFilter().with_name(
        'metric_computed_ExampleCount_v2_' + constants.TF_KERAS
    )
    actual_metrics_count = (
        pipeline.run()
        .metrics()
        .query(filter=metric_filter)['counters'][0]
        .committed
    )
    self.assertEqual(actual_metrics_count, 1)

  def testEvaluateWithKerasAndDiffMetrics(self):
    model_dir, baseline_dir = self._getExportDir(), self._getBaselineDir()
    eval_shared_model = self._build_keras_model('candidate', model_dir, mul=0)
    baseline_eval_shared_model = self._build_keras_model(
        'baseline', baseline_dir, mul=1
    )

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
        """,
        schema_pb2.Schema(),
    )
    tfx_io = test_util.InMemoryTFExampleRecord(
        schema=schema, raw_record_column_name=constants.ARROW_INPUT_COLUMN
    )
    tensor_adapter_config = tensor_adapter.TensorAdapterConfig(
        arrow_schema=tfx_io.ArrowSchema(),
        tensor_representations=tfx_io.TensorRepresentations(),
    )

    examples = [
        self._makeExample(
            input_1=0.0,
            label=1.0,
            example_weight=1.0,
            extra_feature='non_model_feature',
        ),
        self._makeExample(
            input_1=1.0,
            label=0.0,
            example_weight=0.5,
            extra_feature='non_model_feature',
        ),
    ]

    eval_config = config_pb2.EvalConfig(
        model_specs=[
            config_pb2.ModelSpec(
                name='candidate',
                label_key='label',
                example_weight_key='example_weight',
            ),
            config_pb2.ModelSpec(
                name='baseline',
                label_key='label',
                example_weight_key='example_weight',
                is_baseline=True,
            ),
        ],
        slicing_specs=[config_pb2.SlicingSpec()],
        metrics_specs=metric_specs.specs_from_metrics(
            [
                calibration.MeanLabel('mean_label'),
                calibration.MeanPrediction('mean_prediction'),
            ],
            model_names=['candidate', 'baseline'],
        ),
    )

    eval_shared_models = [eval_shared_model, baseline_eval_shared_model]
    extractors = [
        features_extractor.FeaturesExtractor(
            eval_config=eval_config,
            tensor_representations=tensor_adapter_config.tensor_representations,
        ),
        labels_extractor.LabelsExtractor(eval_config),
        example_weights_extractor.ExampleWeightsExtractor(eval_config),
        predictions_extractor.PredictionsExtractor(
            eval_shared_model=eval_shared_models, eval_config=eval_config
        ),
        unbatch_extractor.UnbatchExtractor(),
        slice_key_extractor.SliceKeyExtractor(eval_config=eval_config),
    ]
    evaluators = [
        metrics_plots_and_validations_evaluator.MetricsPlotsAndValidationsEvaluator(
            eval_config=eval_config, eval_shared_model=eval_shared_models
        )
    ]

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      metrics = (
          pipeline
          | 'Create' >> beam.Create([e.SerializeToString() for e in examples])
          | 'BatchExamples' >> tfx_io.BeamSource()
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | 'ExtractAndEvaluate'
          >> model_eval_lib.ExtractAndEvaluate(
              extractors=extractors, evaluators=evaluators
          )
      )

      # pylint: enable=no-value-for-parameter

      def check_metrics(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          # check only the diff metrics.
          weighted_example_count_key = metric_types.MetricKey(
              name='weighted_example_count',
              model_name='candidate',
              is_diff=True,
              example_weighted=True,
          )
          prediction_key = metric_types.MetricKey(
              name='mean_prediction',
              model_name='candidate',
              is_diff=True,
              example_weighted=True,
          )
          label_key = metric_types.MetricKey(
              name='mean_label',
              model_name='candidate',
              is_diff=True,
              example_weighted=True,
          )
          self.assertDictElementsAlmostEqual(
              got_metrics,
              {
                  weighted_example_count_key: 0,
                  label_key: 0,
                  prediction_key: 0 - (0 * 1 + 1 * 0.5) / (1 + 0.5),
              },
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(
          metrics[constants.METRICS_KEY], check_metrics, label='metrics'
      )

  def testEvaluateWithSlicing(self):
    temp_export_dir = self._getExportDir()
    _, export_dir = (
        fixed_prediction_estimator_extra_fields.simple_fixed_prediction_estimator_extra_fields(
            None, temp_export_dir
        )
    )
    eval_config = config_pb2.EvalConfig(
        model_specs=[
            config_pb2.ModelSpec(
                label_key='label', example_weight_key='fixed_float'
            )
        ],
        slicing_specs=[
            config_pb2.SlicingSpec(),
            config_pb2.SlicingSpec(feature_keys=['fixed_string']),
        ],
        metrics_specs=metric_specs.specs_from_metrics([
            calibration.MeanLabel('mean_label'),
            calibration.MeanPrediction('mean_prediction'),
        ]),
    )
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir
    )
    extractors = [
        legacy_predict_extractor.PredictExtractor(
            eval_shared_model=eval_shared_model, eval_config=eval_config
        ),
        unbatch_extractor.UnbatchExtractor(),
        slice_key_extractor.SliceKeyExtractor(eval_config=eval_config),
    ]
    evaluators = [
        metrics_plots_and_validations_evaluator.MetricsPlotsAndValidationsEvaluator(
            eval_config=eval_config, eval_shared_model=eval_shared_model
        )
    ]

    # fixed_float used as example_weight key
    examples = [
        self._makeExample(
            prediction=0.2,
            label=1.0,
            fixed_int=1,
            fixed_float=1.0,
            fixed_string='fixed_string1',
        ),
        self._makeExample(
            prediction=0.8,
            label=0.0,
            fixed_int=1,
            fixed_float=1.0,
            fixed_string='fixed_string1',
        ),
        self._makeExample(
            prediction=0.5,
            label=0.0,
            fixed_int=2,
            fixed_float=2.0,
            fixed_string='fixed_string2',
        ),
    ]

    tfx_io = raw_tf_record.RawBeamRecordTFXIO(
        physical_format='inmemory',
        raw_record_column_name=constants.ARROW_INPUT_COLUMN,
        telemetry_descriptors=['TFMATest'],
    )
    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      metrics = (
          pipeline
          | 'Create' >> beam.Create([e.SerializeToString() for e in examples])
          | 'BatchExamples' >> tfx_io.BeamSource()
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | 'ExtractAndEvaluate'
          >> model_eval_lib.ExtractAndEvaluate(
              extractors=extractors, evaluators=evaluators
          )
      )

      # pylint: enable=no-value-for-parameter

      def check_metrics(got):
        try:
          self.assertLen(got, 3)
          slices = {}
          for slice_key, value in got:
            slices[slice_key] = value
          overall_slice = ()
          fixed_string1_slice = (('fixed_string', b'fixed_string1'),)
          fixed_string2_slice = (('fixed_string', b'fixed_string2'),)
          self.asssertCountEqual(
              list(slices),
              [overall_slice, fixed_string1_slice, fixed_string2_slice],
          )
          weighted_example_count_key = metric_types.MetricKey(
              name='weighted_example_count', example_weighted=True
          )
          label_key = metric_types.MetricKey(
              name='mean_label', example_weighted=True
          )
          pred_key = metric_types.MetricKey(
              name='mean_prediction', example_weighted=True
          )
          self.assertDictElementsAlmostEqual(
              slices[overall_slice],
              {
                  weighted_example_count_key: 4.0,
                  label_key: (1.0 + 0.0 + 2 * 0.0) / (1.0 + 1.0 + 2.0),
                  pred_key: (0.2 + 0.8 + 2 * 0.5) / (1.0 + 1.0 + 2.0),
              },
          )
          self.assertDictElementsAlmostEqual(
              slices[fixed_string1_slice],
              {
                  weighted_example_count_key: 2.0,
                  label_key: (1.0 + 0.0) / (1.0 + 1.0),
                  pred_key: (0.2 + 0.8) / (1.0 + 1.0),
              },
          )
          self.assertDictElementsAlmostEqual(
              slices[fixed_string2_slice],
              {
                  weighted_example_count_key: 2.0,
                  label_key: (2 * 0.0) / 2.0,
                  pred_key: (2 * 0.5) / 2.0,
              },
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

        util.assert_that(
            metrics[constants.METRICS_KEY], check_metrics, label='metrics'
        )

  def testEvaluateWithAttributions(self):
    eval_config = config_pb2.EvalConfig(
        model_specs=[config_pb2.ModelSpec()],
        metrics_specs=[
            config_pb2.MetricsSpec(
                metrics=[
                    config_pb2.MetricConfig(
                        class_name=attributions.TotalAttributions().__class__.__name__
                    )
                ]
            )
        ],
        options=config_pb2.Options(
            disabled_outputs={'values': ['eval_config_pb2.json']}
        ),
    )
    extractors = [slice_key_extractor.SliceKeyExtractor()]
    evaluators = [
        metrics_plots_and_validations_evaluator.MetricsPlotsAndValidationsEvaluator(
            eval_config=eval_config
        )
    ]

    example1 = {
        'labels': None,
        'predictions': None,
        'example_weights': np.array(1.0),
        'features': {},
        'attributions': {'feature1': 1.1, 'feature2': 1.2},
    }
    example2 = {
        'labels': None,
        'predictions': None,
        'example_weights': np.array(1.0),
        'features': {},
        'attributions': {'feature1': 2.1, 'feature2': 2.2},
    }
    example3 = {
        'labels': None,
        'predictions': None,
        'example_weights': np.array(1.0),
        'features': {},
        'attributions': {
            'feature1': np.array([3.1]),
            'feature2': np.array([3.2]),
        },
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      results = (
          pipeline
          | 'Create' >> beam.Create([example1, example2, example3])
          | 'ExtractEvaluate'
          >> model_eval_lib.ExtractAndEvaluate(
              extractors=extractors, evaluators=evaluators
          )
      )

      # pylint: enable=no-value-for-parameter

      def check_attributions(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_attributions = got[0]
          self.assertEqual(got_slice_key, ())
          total_attributions_key = metric_types.MetricKey(
              name='total_attributions'
          )
          self.assertIn(total_attributions_key, got_attributions)
          self.assertDictElementsAlmostEqual(
              got_attributions[total_attributions_key],
              {'feature1': 1.1 + 2.1 + 3.1, 'feature2': 1.2 + 2.2 + 3.2},
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(
          results[constants.ATTRIBUTIONS_KEY],
          check_attributions,
          label='attributions',
      )

  def testEvaluateWithConfidenceIntervals(self):
    # NOTE: This test does not actually test that confidence intervals are
    #   accurate it only tests that the proto output by the test is well formed.
    #   This test would pass if the confidence interval implementation did
    #   nothing at all except compute the unsampled value.
    temp_export_dir = self._getExportDir()
    _, export_dir = (
        fixed_prediction_estimator_extra_fields.simple_fixed_prediction_estimator_extra_fields(
            None, temp_export_dir
        )
    )
    options = config_pb2.Options()
    options.compute_confidence_intervals.value = True
    options.confidence_intervals.method = (
        config_pb2.ConfidenceIntervalOptions.POISSON_BOOTSTRAP
    )
    eval_config = config_pb2.EvalConfig(
        model_specs=[
            config_pb2.ModelSpec(
                label_key='label', example_weight_key='fixed_float'
            )
        ],
        slicing_specs=[
            config_pb2.SlicingSpec(),
            config_pb2.SlicingSpec(feature_keys=['fixed_string']),
        ],
        metrics_specs=metric_specs.specs_from_metrics([
            calibration.MeanLabel('mean_label'),
            calibration.MeanPrediction('mean_prediction'),
        ]),
        options=options,
    )
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir, tags=[tf.saved_model.SERVING]
    )
    extractors = [
        features_extractor.FeaturesExtractor(eval_config),
        labels_extractor.LabelsExtractor(eval_config),
        example_weights_extractor.ExampleWeightsExtractor(eval_config),
        predictions_extractor.PredictionsExtractor(
            eval_shared_model=eval_shared_model, eval_config=eval_config
        ),
        unbatch_extractor.UnbatchExtractor(),
        slice_key_extractor.SliceKeyExtractor(eval_config=eval_config),
    ]
    evaluators = [
        metrics_plots_and_validations_evaluator.MetricsPlotsAndValidationsEvaluator(
            eval_config=eval_config, eval_shared_model=eval_shared_model
        )
    ]

    # fixed_float used as example_weight key
    examples = [
        self._makeExample(
            prediction=0.2,
            label=1.0,
            fixed_int=1,
            fixed_float=1.0,
            fixed_string='fixed_string1',
        ),
        self._makeExample(
            prediction=0.8,
            label=0.0,
            fixed_int=1,
            fixed_float=1.0,
            fixed_string='fixed_string1',
        ),
        self._makeExample(
            prediction=0.5,
            label=0.0,
            fixed_int=2,
            fixed_float=2.0,
            fixed_string='fixed_string2',
        ),
    ]

    tfx_io = test_util.InMemoryTFExampleRecord(
        raw_record_column_name=constants.ARROW_INPUT_COLUMN
    )
    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      metrics = (
          pipeline
          | 'Create' >> beam.Create([e.SerializeToString() for e in examples])
          | 'BatchExamples' >> tfx_io.BeamSource()
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | 'ExtractAndEvaluate'
          >> model_eval_lib.ExtractAndEvaluate(
              extractors=extractors, evaluators=evaluators
          )
      )

      # pylint: enable=no-value-for-parameter

      def check_metrics(got):
        try:
          self.assertLen(got, 3)
          slices = {}
          for slice_key, value in got:
            slices[slice_key] = value
          overall_slice = ()
          fixed_string1_slice = (('fixed_string', 'fixed_string1'),)
          fixed_string2_slice = (('fixed_string', 'fixed_string2'),)
          self.assertCountEqual(
              list(slices),
              [overall_slice, fixed_string1_slice, fixed_string2_slice],
          )
          example_count_key = metric_types.MetricKey(name='example_count')
          weighted_example_count_key = metric_types.MetricKey(
              name='weighted_example_count', example_weighted=True
          )
          label_key = metric_types.MetricKey(
              name='mean_label', example_weighted=True
          )
          pred_key = metric_types.MetricKey(
              name='mean_prediction', example_weighted=True
          )
          self.assertDictElementsAlmostEqual(
              slices[overall_slice],
              {
                  example_count_key: 3,
                  weighted_example_count_key: 4.0,
              },
          )
          self.assertDictElementsWithTDistributionAlmostEqual(
              slices[overall_slice],
              {
                  label_key: (1.0 + 0.0 + 2 * 0.0) / (1.0 + 1.0 + 2.0),
                  pred_key: (0.2 + 0.8 + 2 * 0.5) / (1.0 + 1.0 + 2.0),
              },
          )
          self.assertDictElementsAlmostEqual(
              slices[fixed_string1_slice],
              {
                  example_count_key: 2,
                  weighted_example_count_key: 2.0,
              },
          )
          self.assertDictElementsWithTDistributionAlmostEqual(
              slices[fixed_string1_slice],
              {
                  label_key: (1.0 + 0.0) / (1.0 + 1.0),
                  pred_key: (0.2 + 0.8) / (1.0 + 1.0),
              },
          )
          self.assertDictElementsAlmostEqual(
              slices[fixed_string2_slice],
              {
                  example_count_key: 1,
                  weighted_example_count_key: 2.0,
              },
          )
          self.assertDictElementsWithTDistributionAlmostEqual(
              slices[fixed_string2_slice],
              {
                  label_key: (2 * 0.0) / 2.0,
                  pred_key: (2 * 0.5) / 2.0,
              },
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(
          metrics[constants.METRICS_KEY], check_metrics, label='metrics'
      )

  def testEvaluateWithJackknife(self):
    temp_export_dir = self._getExportDir()
    _, export_dir = (
        fixed_prediction_estimator_extra_fields.simple_fixed_prediction_estimator_extra_fields(
            None, temp_export_dir
        )
    )
    options = config_pb2.Options()
    options.include_default_metrics.value = False
    options.compute_confidence_intervals.value = True
    options.confidence_intervals.method = (
        config_pb2.ConfidenceIntervalOptions.JACKKNIFE
    )
    eval_config = config_pb2.EvalConfig(
        model_specs=[
            config_pb2.ModelSpec(
                label_key='label', example_weight_key='fixed_float'
            )
        ],
        slicing_specs=[
            config_pb2.SlicingSpec(),
            config_pb2.SlicingSpec(feature_keys=['fixed_string']),
        ],
        metrics_specs=metric_specs.specs_from_metrics([
            calibration.MeanLabel('mean_label'),
            calibration.MeanPrediction('mean_prediction'),
        ]),
        options=options,
    )
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir, tags=[tf.saved_model.SERVING]
    )
    extractors = [
        features_extractor.FeaturesExtractor(eval_config),
        labels_extractor.LabelsExtractor(eval_config),
        example_weights_extractor.ExampleWeightsExtractor(eval_config),
        predictions_extractor.PredictionsExtractor(
            eval_shared_model=eval_shared_model, eval_config=eval_config
        ),
        unbatch_extractor.UnbatchExtractor(),
        slice_key_extractor.SliceKeyExtractor(eval_config=eval_config),
    ]
    evaluators = [
        metrics_plots_and_validations_evaluator.MetricsPlotsAndValidationsEvaluator(
            eval_config=eval_config, eval_shared_model=eval_shared_model
        )
    ]

    # fixed_float used as example_weight key
    examples = [
        self._makeExample(
            prediction=0.2,
            label=1.0,
            fixed_int=1,
            fixed_float=1.0,
            fixed_string='fixed_string1',
        ),
        self._makeExample(
            prediction=0.8,
            label=0.0,
            fixed_int=1,
            fixed_float=1.0,
            fixed_string='fixed_string1',
        ),
        self._makeExample(
            prediction=0.5,
            label=0.0,
            fixed_int=2,
            fixed_float=2.0,
            fixed_string='fixed_string2',
        ),
    ]

    tfx_io = test_util.InMemoryTFExampleRecord(
        raw_record_column_name=constants.ARROW_INPUT_COLUMN
    )
    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      metrics = (
          pipeline
          | 'Create'
          >> beam.Create([e.SerializeToString() for e in examples * 1000])
          | 'BatchExamples' >> tfx_io.BeamSource()
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | 'ExtractAndEvaluate'
          >> model_eval_lib.ExtractAndEvaluate(
              extractors=extractors, evaluators=evaluators
          )
      )

      # pylint: enable=no-value-for-parameter

      def check_metrics(got):
        try:
          self.assertLen(got, 3)
          slices = {}
          for slice_key, value in got:
            slices[slice_key] = value
          overall_slice = ()
          fixed_string1_slice = (('fixed_string', 'fixed_string1'),)
          fixed_string2_slice = (('fixed_string', 'fixed_string2'),)
          self.assertCountEqual(
              list(slices),
              [overall_slice, fixed_string1_slice, fixed_string2_slice],
          )
          example_count_key = metric_types.MetricKey(name='example_count')
          weighted_example_count_key = metric_types.MetricKey(
              name='weighted_example_count', example_weighted=True
          )
          label_key = metric_types.MetricKey(
              name='mean_label', example_weighted=True
          )
          pred_key = metric_types.MetricKey(
              name='mean_prediction', example_weighted=True
          )
          self.assertLen(slices[overall_slice], 4)
          self.assertDictElementsAlmostEqual(
              slices[overall_slice],
              {example_count_key: 3000, weighted_example_count_key: 4000.0},
          )
          self.assertDictElementsWithTDistributionAlmostEqual(
              slices[overall_slice],
              {
                  label_key: (1.0 + 0.0 + 2 * 0.0) / (1.0 + 1.0 + 2.0),
                  pred_key: (0.2 + 0.8 + 2 * 0.5) / (1.0 + 1.0 + 2.0),
              },
          )
          self.assertDictElementsAlmostEqual(
              slices[fixed_string1_slice], {weighted_example_count_key: 2000.0}
          )
          self.assertDictElementsWithTDistributionAlmostEqual(
              slices[fixed_string1_slice],
              {
                  label_key: (1.0 + 0.0) / (1.0 + 1.0),
                  pred_key: (0.2 + 0.8) / (1.0 + 1.0),
              },
          )
          self.assertDictElementsAlmostEqual(
              slices[fixed_string2_slice], {weighted_example_count_key: 2000.0}
          )
          self.assertDictElementsWithTDistributionAlmostEqual(
              slices[fixed_string2_slice],
              {
                  label_key: (2 * 0.0) / 2.0,
                  pred_key: (2 * 0.5) / 2.0,
              },
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(metrics[constants.METRICS_KEY], check_metrics)

  def testEvaluateWithJackknifeAndDiffMetrics(self):
    model_dir, baseline_dir = self._getExportDir(), self._getBaselineDir()
    eval_shared_model = self._build_keras_model('candidate', model_dir, mul=0)
    baseline_eval_shared_model = self._build_keras_model(
        'baseline', baseline_dir, mul=1
    )

    options = config_pb2.Options()
    options.compute_confidence_intervals.value = True
    options.confidence_intervals.method = (
        config_pb2.ConfidenceIntervalOptions.JACKKNIFE
    )

    eval_config = config_pb2.EvalConfig(
        model_specs=[
            config_pb2.ModelSpec(
                name='candidate',
                label_key='label',
                example_weight_key='example_weight',
            ),
            config_pb2.ModelSpec(
                name='baseline',
                label_key='label',
                example_weight_key='example_weight',
                is_baseline=True,
            ),
        ],
        slicing_specs=[config_pb2.SlicingSpec()],
        metrics_specs=metric_specs.specs_from_metrics(
            [
                calibration.MeanLabel('mean_label'),
                calibration.MeanPrediction('mean_prediction'),
            ],
            model_names=['candidate', 'baseline'],
        ),
        options=options,
    )

    eval_shared_models = {
        'candidate': eval_shared_model,
        'baseline': baseline_eval_shared_model,
    }

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
        """,
        schema_pb2.Schema(),
    )
    tfx_io = test_util.InMemoryTFExampleRecord(
        schema=schema, raw_record_column_name=constants.ARROW_INPUT_COLUMN
    )
    tensor_adapter_config = tensor_adapter.TensorAdapterConfig(
        arrow_schema=tfx_io.ArrowSchema(),
        tensor_representations=tfx_io.TensorRepresentations(),
    )

    examples = [
        self._makeExample(
            input_1=0.0,
            label=1.0,
            example_weight=1.0,
            extra_feature='non_model_feature',
        ),
        self._makeExample(
            input_1=1.0,
            label=0.0,
            example_weight=0.5,
            extra_feature='non_model_feature',
        ),
    ]

    extractors = [
        features_extractor.FeaturesExtractor(
            eval_config=eval_config,
            tensor_representations=tensor_adapter_config.tensor_representations,
        ),
        labels_extractor.LabelsExtractor(eval_config),
        example_weights_extractor.ExampleWeightsExtractor(eval_config),
        predictions_extractor.PredictionsExtractor(
            eval_shared_model=eval_shared_models, eval_config=eval_config
        ),
        unbatch_extractor.UnbatchExtractor(),
        slice_key_extractor.SliceKeyExtractor(eval_config=eval_config),
    ]
    evaluators = [
        metrics_plots_and_validations_evaluator.MetricsPlotsAndValidationsEvaluator(
            eval_config=eval_config, eval_shared_model=eval_shared_models
        )
    ]

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      metrics = (
          pipeline
          | 'Create'
          >> beam.Create([e.SerializeToString() for e in examples * 1000])
          | 'BatchExamples' >> tfx_io.BeamSource()
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | 'ExtractAndEvaluate'
          >> model_eval_lib.ExtractAndEvaluate(
              extractors=extractors, evaluators=evaluators
          )
      )

      # pylint: enable=no-value-for-parameter

      def check_metrics(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          # check only the diff metrics.
          weighted_example_count_key = metric_types.MetricKey(
              name='weighted_example_count',
              model_name='candidate',
              is_diff=True,
              example_weighted=True,
          )
          prediction_key = metric_types.MetricKey(
              name='mean_prediction',
              model_name='candidate',
              is_diff=True,
              example_weighted=True,
          )
          label_key = metric_types.MetricKey(
              name='mean_label',
              model_name='candidate',
              is_diff=True,
              example_weighted=True,
          )
          self.assertDictElementsWithTDistributionAlmostEqual(
              got_metrics,
              {
                  weighted_example_count_key: 0,
                  label_key: 0,
                  prediction_key: 0 - (0 * 1 + 1 * 0.5) / (1 + 0.5),
              },
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(metrics[constants.METRICS_KEY], check_metrics)

  def testEvaluateWithRegressionModel(self):
    temp_export_dir = self._getExportDir()
    _, export_dir = (
        fixed_prediction_estimator_extra_fields.simple_fixed_prediction_estimator_extra_fields(
            None, temp_export_dir
        )
    )
    eval_config = config_pb2.EvalConfig(
        model_specs=[
            config_pb2.ModelSpec(
                label_key='label', example_weight_key='fixed_float'
            )
        ],
        slicing_specs=[config_pb2.SlicingSpec()],
        metrics_specs=metric_specs.specs_from_metrics([
            calibration.MeanLabel('mean_label'),
            calibration.MeanPrediction('mean_prediction'),
        ]),
    )
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir, tags=[tf.saved_model.SERVING]
    )
    extractors = [
        features_extractor.FeaturesExtractor(eval_config),
        labels_extractor.LabelsExtractor(eval_config),
        example_weights_extractor.ExampleWeightsExtractor(eval_config),
        predictions_extractor.PredictionsExtractor(
            eval_shared_model=eval_shared_model, eval_config=eval_config
        ),
        unbatch_extractor.UnbatchExtractor(),
        slice_key_extractor.SliceKeyExtractor(eval_config=eval_config),
    ]
    evaluators = [
        metrics_plots_and_validations_evaluator.MetricsPlotsAndValidationsEvaluator(
            eval_config=eval_config, eval_shared_model=eval_shared_model
        )
    ]

    # fixed_float used as example_weight key
    examples = [
        self._makeExample(
            prediction=0.2,
            label=1.0,
            fixed_int=1,
            fixed_float=1.0,
            fixed_string='fixed_string1',
        ),
        self._makeExample(
            prediction=0.8,
            label=0.0,
            fixed_int=1,
            fixed_float=1.0,
            fixed_string='fixed_string1',
        ),
        self._makeExample(
            prediction=0.5,
            label=0.0,
            fixed_int=2,
            fixed_float=2.0,
            fixed_string='fixed_string2',
        ),
    ]

    tfx_io = test_util.InMemoryTFExampleRecord(
        raw_record_column_name=constants.ARROW_INPUT_COLUMN
    )
    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      metrics = (
          pipeline
          | 'Create' >> beam.Create([e.SerializeToString() for e in examples])
          | 'BatchExamples' >> tfx_io.BeamSource()
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | 'ExtractAndEvaluate'
          >> model_eval_lib.ExtractAndEvaluate(
              extractors=extractors, evaluators=evaluators
          )
      )

      # pylint: enable=no-value-for-parameter

      def check_metrics(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          example_count_key = metric_types.MetricKey(name='example_count')
          weighted_example_count_key = metric_types.MetricKey(
              name='weighted_example_count', example_weighted=True
          )
          label_key = metric_types.MetricKey(
              name='mean_label', example_weighted=True
          )
          pred_key = metric_types.MetricKey(
              name='mean_prediction', example_weighted=True
          )
          self.assertEqual(got_slice_key, ())
          self.assertDictElementsAlmostEqual(
              got_metrics,
              {
                  example_count_key: 3,
                  weighted_example_count_key: 4.0,
                  label_key: (1.0 + 0.0 + 2 * 0.0) / (1.0 + 1.0 + 2.0),
                  pred_key: (0.2 + 0.8 + 2 * 0.5) / (1.0 + 1.0 + 2.0),
              },
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(
          metrics[constants.METRICS_KEY], check_metrics, label='metrics'
      )

  def testEvaluateWithCrossSliceLiftMetric(self):
    temp_export_dir = self._getExportDir()
    _, export_dir = (
        fixed_prediction_estimator_extra_fields.simple_fixed_prediction_estimator_extra_fields(
            None, temp_export_dir
        )
    )
    eval_config = config_pb2.EvalConfig(
        model_specs=[
            config_pb2.ModelSpec(
                label_key='label', example_weight_key='fixed_float'
            )
        ],
        slicing_specs=[
            config_pb2.SlicingSpec(
                feature_values={'fixed_string': 'fixed_string1'}
            ),
            config_pb2.SlicingSpec(
                feature_values={'fixed_string': 'fixed_string2'}
            ),
        ],
        cross_slicing_specs=[
            config_pb2.CrossSlicingSpec(
                baseline_spec=config_pb2.SlicingSpec(
                    feature_values={'fixed_string': 'fixed_string1'}
                ),
                slicing_specs=[
                    config_pb2.SlicingSpec(
                        feature_values={'fixed_string': 'fixed_string2'}
                    )
                ],
            ),
        ],
        metrics_specs=metric_specs.specs_from_metrics(
            [lift.Lift(num_buckets=3)]
        ),
    )
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir, tags=[tf.saved_model.SERVING]
    )
    extractors = [
        features_extractor.FeaturesExtractor(eval_config),
        labels_extractor.LabelsExtractor(eval_config),
        example_weights_extractor.ExampleWeightsExtractor(eval_config),
        predictions_extractor.PredictionsExtractor(
            eval_shared_model=eval_shared_model, eval_config=eval_config
        ),
        unbatch_extractor.UnbatchExtractor(),
        slice_key_extractor.SliceKeyExtractor(eval_config=eval_config),
    ]
    evaluators = [
        metrics_plots_and_validations_evaluator.MetricsPlotsAndValidationsEvaluator(
            eval_config=eval_config, eval_shared_model=eval_shared_model
        )
    ]

    # fixed_float used as example_weight key
    examples = [
        self._makeExample(
            prediction=0.1,
            label=0.0,
            fixed_int=1,
            fixed_float=3.0,
            fixed_string='fixed_string1',
        ),
        self._makeExample(
            prediction=0.5,
            label=0.3,
            fixed_int=1,
            fixed_float=5.0,
            fixed_string='fixed_string1',
        ),
        self._makeExample(
            prediction=0.8,
            label=0.6,
            fixed_int=2,
            fixed_float=2.0,
            fixed_string='fixed_string1',
        ),
        self._makeExample(
            prediction=0.3,
            label=0.9,
            fixed_int=1,
            fixed_float=8.0,
            fixed_string='fixed_string1',
        ),
        self._makeExample(
            prediction=0.9,
            label=1.0,
            fixed_int=1,
            fixed_float=3.0,
            fixed_string='fixed_string1',
        ),
        self._makeExample(
            prediction=0.8,
            label=0.0,
            fixed_int=2,
            fixed_float=1.0,
            fixed_string='fixed_string2',
        ),
        self._makeExample(
            prediction=0.3,
            label=0.2,
            fixed_int=1,
            fixed_float=2.0,
            fixed_string='fixed_string2',
        ),
        self._makeExample(
            prediction=0.5,
            label=0.5,
            fixed_int=1,
            fixed_float=5.0,
            fixed_string='fixed_string2',
        ),
        self._makeExample(
            prediction=0.4,
            label=0.7,
            fixed_int=2,
            fixed_float=2.0,
            fixed_string='fixed_string2',
        ),
        self._makeExample(
            prediction=0.3,
            label=1.0,
            fixed_int=2,
            fixed_float=3.0,
            fixed_string='fixed_string2',
        ),
    ]

    tfx_io = test_util.InMemoryTFExampleRecord(
        raw_record_column_name=constants.ARROW_INPUT_COLUMN
    )
    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      metrics = (
          pipeline
          | 'Create' >> beam.Create([e.SerializeToString() for e in examples])
          | 'BatchExamples' >> tfx_io.BeamSource()
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | 'ExtractAndEvaluate'
          >> model_eval_lib.ExtractAndEvaluate(
              extractors=extractors, evaluators=evaluators
          )
      )

      # pylint: enable=no-value-for-parameter

      def check_metrics(got):
        try:
          self.assertLen(got, 3)
          example_count_key = metric_types.MetricKey(name='example_count')
          weighted_example_count_key = metric_types.MetricKey(
              name='weighted_example_count', example_weighted=True
          )
          lift_key = metric_types.MetricKey(
              name='lift@3', example_weighted=True
          )
          for slice_key, metrics in got:
            if slice_key == (('fixed_string', 'fixed_string1'),):
              self.assertDictElementsAlmostEqual(
                  metrics,
                  {
                      example_count_key: 5.0,
                      weighted_example_count_key: 21.0,
                  },
              )
            elif slice_key == (('fixed_string', 'fixed_string2'),):
              self.assertDictElementsAlmostEqual(
                  metrics,
                  {
                      weighted_example_count_key: 13.0,
                      example_count_key: 5.0,
                  },
              )
            else:
              self.assertDictElementsAlmostEqual(
                  metrics,
                  {
                      example_count_key: 0.0,
                      weighted_example_count_key: 8.0,
                      lift_key: -0.211538456165,
                  },
              )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(
          metrics[constants.METRICS_KEY], check_metrics, label='metrics'
      )

  def testEvaluateWithBinaryClassificationModel(self):
    n_classes = 2
    temp_export_dir = self._getExportDir()
    _, export_dir = dnn_classifier.simple_dnn_classifier(
        None, temp_export_dir, n_classes=n_classes
    )

    # Add mean_label, example_count, weighted_example_count, calibration_plot
    eval_config = config_pb2.EvalConfig(
        model_specs=[
            config_pb2.ModelSpec(label_key='label', example_weight_key='age')
        ],
        slicing_specs=[config_pb2.SlicingSpec()],
        metrics_specs=metric_specs.specs_from_metrics([
            calibration.MeanLabel('mean_label'),
            calibration_plot.CalibrationPlot(
                name='calibration_plot', num_buckets=10
            ),
        ]),
    )
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir, tags=[tf.saved_model.SERVING]
    )
    extractors = [
        features_extractor.FeaturesExtractor(eval_config),
        labels_extractor.LabelsExtractor(eval_config),
        example_weights_extractor.ExampleWeightsExtractor(eval_config),
        predictions_extractor.PredictionsExtractor(
            eval_shared_model=eval_shared_model, eval_config=eval_config
        ),
        unbatch_extractor.UnbatchExtractor(),
        slice_key_extractor.SliceKeyExtractor(eval_config=eval_config),
    ]
    evaluators = [
        metrics_plots_and_validations_evaluator.MetricsPlotsAndValidationsEvaluator(
            eval_config=eval_config, eval_shared_model=eval_shared_model
        )
    ]

    examples = [
        self._makeExample(age=1.0, language='english', label=0.0),
        self._makeExample(age=2.0, language='chinese', label=1.0),
        self._makeExample(age=3.0, language='chinese', label=0.0),
    ]

    tfx_io = test_util.InMemoryTFExampleRecord(
        raw_record_column_name=constants.ARROW_INPUT_COLUMN
    )
    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      metrics_and_plots = (
          pipeline
          | 'Create' >> beam.Create([e.SerializeToString() for e in examples])
          | 'BatchExamples' >> tfx_io.BeamSource()
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | 'ExtractAndEvaluate'
          >> model_eval_lib.ExtractAndEvaluate(
              extractors=extractors, evaluators=evaluators
          )
      )

      # pylint: enable=no-value-for-parameter

      def check_metrics(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          example_count_key = metric_types.MetricKey(name='example_count')
          weighted_example_count_key = metric_types.MetricKey(
              name='weighted_example_count', example_weighted=True
          )
          label_key = metric_types.MetricKey(
              name='mean_label', example_weighted=True
          )
          self.assertDictElementsAlmostEqual(
              got_metrics,
              {
                  example_count_key: 3,
                  weighted_example_count_key: 1.0 + 2.0 + 3.0,
                  label_key: (0 * 1.0 + 1 * 2.0 + 0 * 3.0) / (1.0 + 2.0 + 3.0),
              },
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      def check_plots(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_plots = got[0]
          self.assertEqual(got_slice_key, ())
          plot_key = metric_types.PlotKey(
              'calibration_plot', example_weighted=True
          )
          self.assertIn(plot_key, got_plots)
          # 10 buckets + 2 for edge cases
          self.assertLen(got_plots[plot_key].buckets, 12)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(
          metrics_and_plots[constants.METRICS_KEY],
          check_metrics,
          label='metrics',
      )
      util.assert_that(
          metrics_and_plots[constants.PLOTS_KEY], check_plots, label='plots'
      )

  def testEvaluateWithMultiClassModel(self):
    n_classes = 3
    temp_export_dir = self._getExportDir()
    _, export_dir = dnn_classifier.simple_dnn_classifier(
        None, temp_export_dir, n_classes=n_classes
    )

    # Add example_count and weighted_example_count
    eval_config = config_pb2.EvalConfig(
        model_specs=[
            config_pb2.ModelSpec(label_key='label', example_weight_key='age')
        ],
        slicing_specs=[config_pb2.SlicingSpec()],
        metrics_specs=metric_specs.specs_from_metrics(
            [calibration.MeanLabel('mean_label')],
            binarize=config_pb2.BinarizationOptions(
                class_ids={'values': range(n_classes)}
            ),
        ),
    )
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir, tags=[tf.saved_model.SERVING]
    )
    extractors = [
        features_extractor.FeaturesExtractor(eval_config),
        labels_extractor.LabelsExtractor(eval_config),
        example_weights_extractor.ExampleWeightsExtractor(eval_config),
        predictions_extractor.PredictionsExtractor(
            eval_shared_model=eval_shared_model, eval_config=eval_config
        ),
        unbatch_extractor.UnbatchExtractor(),
        slice_key_extractor.SliceKeyExtractor(eval_config=eval_config),
    ]
    evaluators = [
        metrics_plots_and_validations_evaluator.MetricsPlotsAndValidationsEvaluator(
            eval_config=eval_config, eval_shared_model=eval_shared_model
        )
    ]

    examples = [
        self._makeExample(age=1.0, language='english', label=0),
        self._makeExample(age=2.0, language='chinese', label=1),
        self._makeExample(age=3.0, language='english', label=2),
        self._makeExample(age=4.0, language='chinese', label=1),
    ]

    tfx_io = test_util.InMemoryTFExampleRecord(
        raw_record_column_name=constants.ARROW_INPUT_COLUMN
    )
    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      metrics = (
          pipeline
          | 'Create' >> beam.Create([e.SerializeToString() for e in examples])
          | 'BatchExamples' >> tfx_io.BeamSource()
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | 'ExtractAndEvaluate'
          >> model_eval_lib.ExtractAndEvaluate(
              extractors=extractors, evaluators=evaluators
          )
      )

      # pylint: enable=no-value-for-parameter

      def check_metrics(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          example_count_key = metric_types.MetricKey(name='example_count')
          weighted_example_count_key = metric_types.MetricKey(
              name='weighted_example_count', example_weighted=True
          )
          label_key_class_0 = metric_types.MetricKey(
              name='mean_label',
              sub_key=metric_types.SubKey(class_id=0),
              example_weighted=True,
          )
          label_key_class_1 = metric_types.MetricKey(
              name='mean_label',
              sub_key=metric_types.SubKey(class_id=1),
              example_weighted=True,
          )
          label_key_class_2 = metric_types.MetricKey(
              name='mean_label',
              sub_key=metric_types.SubKey(class_id=2),
              example_weighted=True,
          )
          self.assertEqual(got_slice_key, ())
          self.assertDictElementsAlmostEqual(
              got_metrics,
              {
                  example_count_key: 4,
                  weighted_example_count_key: 1.0 + 2.0 + 3.0 + 4.0,
                  label_key_class_0: (1 * 1.0 + 0 * 2.0 + 0 * 3.0 + 0 * 4.0) / (
                      1.0 + 2.0 + 3.0 + 4.0
                  ),
                  label_key_class_1: (0 * 1.0 + 1 * 2.0 + 0 * 3.0 + 1 * 4.0) / (
                      1.0 + 2.0 + 3.0 + 4.0
                  ),
                  label_key_class_2: (0 * 1.0 + 0 * 2.0 + 1 * 3.0 + 0 * 4.0) / (
                      1.0 + 2.0 + 3.0 + 4.0
                  ),
              },
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(
          metrics[constants.METRICS_KEY], check_metrics, label='metrics'
      )

  def testEvaluateWithMultiOutputModel(self):
    temp_export_dir = self._getExportDir()
    _, export_dir = multi_head.simple_multi_head(None, temp_export_dir)

    eval_config = config_pb2.EvalConfig(
        model_specs=[
            config_pb2.ModelSpec(
                label_keys={
                    'chinese_head': 'chinese_label',
                    'english_head': 'english_label',
                    'other_head': 'other_label',
                },
                example_weight_keys={
                    'chinese_head': 'age',
                    'english_head': 'age',
                    'other_head': 'age',
                },
            )
        ],
        slicing_specs=[config_pb2.SlicingSpec()],
        metrics_specs=metric_specs.specs_from_metrics({
            'chinese_head': [calibration.MeanLabel('mean_label')],
            'english_head': [calibration.MeanLabel('mean_label')],
            'other_head': [calibration.MeanLabel('mean_label')],
        }),
    )
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir, tags=[tf.saved_model.SERVING]
    )
    extractors = [
        features_extractor.FeaturesExtractor(eval_config),
        labels_extractor.LabelsExtractor(eval_config),
        example_weights_extractor.ExampleWeightsExtractor(eval_config),
        predictions_extractor.PredictionsExtractor(
            eval_shared_model=eval_shared_model, eval_config=eval_config
        ),
        unbatch_extractor.UnbatchExtractor(),
        slice_key_extractor.SliceKeyExtractor(eval_config=eval_config),
    ]
    evaluators = [
        metrics_plots_and_validations_evaluator.MetricsPlotsAndValidationsEvaluator(
            eval_config=eval_config, eval_shared_model=eval_shared_model
        )
    ]

    examples = [
        self._makeExample(
            age=1.0,
            language='english',
            english_label=1.0,
            chinese_label=0.0,
            other_label=0.0,
        ),
        self._makeExample(
            age=1.0,
            language='chinese',
            english_label=0.0,
            chinese_label=1.0,
            other_label=0.0,
        ),
        self._makeExample(
            age=2.0,
            language='english',
            english_label=1.0,
            chinese_label=0.0,
            other_label=0.0,
        ),
        self._makeExample(
            age=2.0,
            language='other',
            english_label=0.0,
            chinese_label=1.0,
            other_label=1.0,
        ),
    ]

    tfx_io = test_util.InMemoryTFExampleRecord(
        raw_record_column_name=constants.ARROW_INPUT_COLUMN
    )
    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      metrics = (
          pipeline
          | 'Create' >> beam.Create([e.SerializeToString() for e in examples])
          | 'BatchExamples' >> tfx_io.BeamSource()
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | 'ExtractAndEvaluate'
          >> model_eval_lib.ExtractAndEvaluate(
              extractors=extractors, evaluators=evaluators
          )
      )

      # pylint: enable=no-value-for-parameter

      def check_metrics(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          example_count_key = metric_types.MetricKey(name='example_count')
          chinese_weighted_example_count_key = metric_types.MetricKey(
              name='weighted_example_count',
              output_name='chinese_head',
              example_weighted=True,
          )
          chinese_label_key = metric_types.MetricKey(
              name='mean_label',
              output_name='chinese_head',
              example_weighted=True,
          )
          english_weighted_example_count_key = metric_types.MetricKey(
              name='weighted_example_count',
              output_name='english_head',
              example_weighted=True,
          )
          english_label_key = metric_types.MetricKey(
              name='mean_label',
              output_name='english_head',
              example_weighted=True,
          )
          other_weighted_example_count_key = metric_types.MetricKey(
              name='weighted_example_count',
              output_name='other_head',
              example_weighted=True,
          )
          other_label_key = metric_types.MetricKey(
              name='mean_label', output_name='other_head', example_weighted=True
          )
          self.assertDictElementsAlmostEqual(
              got_metrics,
              {
                  example_count_key: 4,
                  chinese_label_key: (0.0 + 1.0 + 2 * 0.0 + 2 * 1.0) / (
                      1.0 + 1.0 + 2.0 + 2.0
                  ),
                  chinese_weighted_example_count_key: 1.0 + 1.0 + 2.0 + 2.0,
                  english_label_key: (1.0 + 0.0 + 2 * 1.0 + 2 * 0.0) / (
                      1.0 + 1.0 + 2.0 + 2.0
                  ),
                  english_weighted_example_count_key: 1.0 + 1.0 + 2.0 + 2.0,
                  other_label_key: (0.0 + 0.0 + 2 * 0.0 + 2 * 1.0) / (
                      1.0 + 1.0 + 2.0 + 2.0
                  ),
                  other_weighted_example_count_key: 1.0 + 1.0 + 2.0 + 2.0,
              },
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(
          metrics[constants.METRICS_KEY], check_metrics, label='metrics'
      )

  @parameterized.named_parameters(
      ('compiled_metrics', False),
      ('evaluate', True),
  )
  def testEvaluateWithKerasModelWithInGraphMetrics(self, add_custom_metrics):
    # Custom metrics not supported in TFv1
    if _TF_MAJOR_VERSION < 2:
      add_custom_metrics = False

    input1 = tf_keras.layers.Input(shape=(1,), name='input_1')
    input2 = tf_keras.layers.Input(shape=(1,), name='input_2')
    inputs = [input1, input2]
    input_layer = tf_keras.layers.concatenate(inputs)
    output_layer = tf_keras.layers.Dense(
        1, activation=tf.nn.sigmoid, name='output'
    )(input_layer)
    model = tf_keras.models.Model(inputs, output_layer)
    # The model.evaluate API is used when custom metrics are used. Otherwise
    # model.compiled_metrics is used.
    if add_custom_metrics:
      model.add_metric(tf.reduce_sum(input_layer), name='custom')
    model.compile(
        optimizer=tf_keras.optimizers.Adam(lr=0.001),
        loss=tf_keras.losses.BinaryCrossentropy(name='loss'),
        metrics=[tf_keras.metrics.BinaryAccuracy(name='binary_accuracy')],
    )

    export_dir = self._getExportDir()
    model.save(export_dir, save_format='tf')

    eval_config = config_pb2.EvalConfig(
        model_specs=[
            config_pb2.ModelSpec(
                label_key='label', example_weight_key='example_weight'
            )
        ],
        slicing_specs=[config_pb2.SlicingSpec()],
        metrics_specs=metric_specs.specs_from_metrics(
            [calibration.MeanLabel('mean_label')],
            unweighted_metrics=[
                tf_keras.metrics.BinaryAccuracy(name='binary_accuracy'),
                calibration.MeanLabel('mean_label'),
            ],
        ),
    )
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir
    )

    examples = [
        self._makeExample(
            input_1=0.0,
            input_2=1.0,
            label=1.0,
            example_weight=1.0,
            extra_feature='non_model_feature',
        ),
        self._makeExample(
            input_1=1.0,
            input_2=0.0,
            label=0.0,
            example_weight=0.5,
            extra_feature='non_model_feature',
        ),
    ]

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
            tensor_representation {
              key: "input_2"
              value {
                dense_tensor {
                  column_name: "input_2"
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
          name: "input_2"
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
        """,
        schema_pb2.Schema(),
    )
    tfx_io = test_util.InMemoryTFExampleRecord(
        schema=schema, raw_record_column_name=constants.ARROW_INPUT_COLUMN
    )
    tensor_adapter_config = tensor_adapter.TensorAdapterConfig(
        arrow_schema=tfx_io.ArrowSchema(),
        tensor_representations=tfx_io.TensorRepresentations(),
    )
    extractors = [
        features_extractor.FeaturesExtractor(
            eval_config=eval_config,
            tensor_representations=tensor_adapter_config.tensor_representations,
        ),
        labels_extractor.LabelsExtractor(eval_config),
        example_weights_extractor.ExampleWeightsExtractor(eval_config),
        predictions_extractor.PredictionsExtractor(
            eval_shared_model=eval_shared_model, eval_config=eval_config
        ),
        unbatch_extractor.UnbatchExtractor(),
        slice_key_extractor.SliceKeyExtractor(eval_config=eval_config),
    ]
    evaluators = [
        metrics_plots_and_validations_evaluator.MetricsPlotsAndValidationsEvaluator(
            eval_config=eval_config, eval_shared_model=eval_shared_model
        )
    ]
    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      metrics = (
          pipeline
          | 'Create' >> beam.Create([e.SerializeToString() for e in examples])
          | 'BatchExamples' >> tfx_io.BeamSource()
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | 'ExtractAndEvaluate'
          >> model_eval_lib.ExtractAndEvaluate(
              extractors=extractors, evaluators=evaluators
          )
      )

      # pylint: enable=no-value-for-parameter

      def check_metrics(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          example_count_key = metric_types.MetricKey(name='example_count')
          weighted_example_count_key = metric_types.MetricKey(
              name='weighted_example_count', example_weighted=True
          )
          label_key = metric_types.MetricKey(
              name='mean_label', example_weighted=True
          )
          label_unweighted_key = metric_types.MetricKey(
              name='mean_label', example_weighted=False
          )
          binary_accuracy_key = metric_types.MetricKey(
              name='binary_accuracy', example_weighted=None
          )
          self.assertIn(binary_accuracy_key, got_metrics)
          binary_accuracy_unweighted_key = metric_types.MetricKey(
              name='binary_accuracy', example_weighted=False
          )
          self.assertIn(binary_accuracy_unweighted_key, got_metrics)
          # Loss not supported in TFv1
          if _TF_MAJOR_VERSION > 1:
            loss_key = metric_types.MetricKey(
                name='loss', example_weighted=None
            )
            self.assertIn(loss_key, got_metrics)
          expected_values = {
              example_count_key: 2,
              weighted_example_count_key: 1.0 + 0.5,
              label_key: (1.0 * 1.0 + 0.0 * 0.5) / (1.0 + 0.5),
              label_unweighted_key: (1.0 + 0.0) / (1.0 + 1.0),
          }
          if add_custom_metrics:
            custom_key = metric_types.MetricKey(
                name='custom', example_weighted=None
            )
            self.assertIn(custom_key, got_metrics)
            expected_values[custom_key] = 0.0 + 1.0 + 0.0 + 1.0
          self.assertDictElementsAlmostEqual(got_metrics, expected_values)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(
          metrics[constants.METRICS_KEY], check_metrics, label='metrics'
      )

  def testEvaluateWithQueryBasedMetrics(self):
    temp_export_dir = self._getExportDir()
    _, export_dir = (
        fixed_prediction_estimator_extra_fields.simple_fixed_prediction_estimator_extra_fields(
            None, temp_export_dir
        )
    )
    eval_config = config_pb2.EvalConfig(
        model_specs=[
            config_pb2.ModelSpec(
                label_key='label', example_weight_key='fixed_int'
            )
        ],
        slicing_specs=[
            config_pb2.SlicingSpec(),
            config_pb2.SlicingSpec(feature_keys=['fixed_string']),
        ],
        metrics_specs=metric_specs.specs_from_metrics(
            [ndcg.NDCG(gain_key='fixed_float', name='ndcg')],
            binarize=config_pb2.BinarizationOptions(
                top_k_list={'values': [1, 2]}
            ),
            query_key='fixed_string',
        ),
    )
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir, tags=[tf.saved_model.SERVING]
    )
    extractors = [
        features_extractor.FeaturesExtractor(eval_config),
        labels_extractor.LabelsExtractor(eval_config),
        example_weights_extractor.ExampleWeightsExtractor(eval_config),
        predictions_extractor.PredictionsExtractor(
            eval_shared_model=eval_shared_model, eval_config=eval_config
        ),
        unbatch_extractor.UnbatchExtractor(),
        slice_key_extractor.SliceKeyExtractor(eval_config=eval_config),
    ]
    evaluators = [
        metrics_plots_and_validations_evaluator.MetricsPlotsAndValidationsEvaluator(
            eval_config=eval_config, eval_shared_model=eval_shared_model
        )
    ]

    # fixed_string used as query_key
    # fixed_float used as gain_key for NDCG
    # fixed_int used as example_weight_key for NDCG
    examples = [
        self._makeExample(
            prediction=0.2,
            label=1.0,
            fixed_float=1.0,
            fixed_string='query1',
            fixed_int=1,
        ),
        self._makeExample(
            prediction=0.8,
            label=0.0,
            fixed_float=0.5,
            fixed_string='query1',
            fixed_int=1,
        ),
        self._makeExample(
            prediction=0.5,
            label=0.0,
            fixed_float=0.5,
            fixed_string='query2',
            fixed_int=2,
        ),
        self._makeExample(
            prediction=0.9,
            label=1.0,
            fixed_float=1.0,
            fixed_string='query2',
            fixed_int=2,
        ),
        self._makeExample(
            prediction=0.1,
            label=0.0,
            fixed_float=0.1,
            fixed_string='query2',
            fixed_int=2,
        ),
        self._makeExample(
            prediction=0.9,
            label=1.0,
            fixed_float=1.0,
            fixed_string='query3',
            fixed_int=3,
        ),
    ]

    tfx_io = test_util.InMemoryTFExampleRecord(
        raw_record_column_name=constants.ARROW_INPUT_COLUMN
    )
    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      metrics = (
          pipeline
          | 'Create' >> beam.Create([e.SerializeToString() for e in examples])
          | 'BatchExamples' >> tfx_io.BeamSource()
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | 'ExtractAndEvaluate'
          >> model_eval_lib.ExtractAndEvaluate(
              extractors=extractors, evaluators=evaluators
          )
      )

      # pylint: enable=no-value-for-parameter

      def check_metrics(got):
        try:
          self.assertLen(got, 4)
          slices = {}
          for slice_key, value in got:
            slices[slice_key] = value
          overall_slice = ()
          query1_slice = (('fixed_string', 'query1'),)
          query2_slice = (('fixed_string', 'query2'),)
          query3_slice = (('fixed_string', 'query3'),)
          self.assertCountEqual(
              list(slices),
              [overall_slice, query1_slice, query2_slice, query3_slice],
          )
          example_count_key = metric_types.MetricKey(name='example_count')
          weighted_example_count_key = metric_types.MetricKey(
              name='weighted_example_count', example_weighted=True
          )
          ndcg1_key = metric_types.MetricKey(
              name='ndcg',
              sub_key=metric_types.SubKey(top_k=1),
              example_weighted=True,
          )
          ndcg2_key = metric_types.MetricKey(
              name='ndcg',
              sub_key=metric_types.SubKey(top_k=2),
              example_weighted=True,
          )
          # Query1 (weight=1): (p=0.8, g=0.5) (p=0.2, g=1.0)
          # Query2 (weight=2): (p=0.9, g=1.0) (p=0.5, g=0.5) (p=0.1, g=0.1)
          # Query3 (weight=3): (p=0.9, g=1.0)
          #
          # DCG@1:  0.5, 1.0, 1.0
          # NDCG@1: 0.5, 1.0, 1.0
          # Average NDCG@1: (1 * 0.5 + 2 * 1.0 + 3 * 1.0) / (1 + 2 + 3) ~ 0.92
          #
          # DCG@2: (0.5 + 1.0/log(3) ~ 0.630930
          #        (1.0 + 0.5/log(3) ~ 1.315465
          #        1.0
          # NDCG@2: (0.5 + 1.0/log(3)) / (1.0 + 0.5/log(3)) ~ 0.85972
          #         (1.0 + 0.5/log(3)) / (1.0 + 0.5/log(3)) = 1.0
          #         1.0
          # Average NDCG@2: (1 * 0.860 + 2 * 1.0 + 3 * 1.0) / (1 + 2 + 3) ~ 0.97
          self.assertDictElementsAlmostEqual(
              slices[overall_slice],
              {
                  example_count_key: 6,
                  weighted_example_count_key: 11.0,
                  ndcg1_key: 0.9166667,
                  ndcg2_key: 0.9766198,
              },
          )
          self.assertDictElementsAlmostEqual(
              slices[query1_slice],
              {
                  example_count_key: 2,
                  weighted_example_count_key: 2.0,
                  ndcg1_key: 0.5,
                  ndcg2_key: 0.85972,
              },
          )
          self.assertDictElementsAlmostEqual(
              slices[query2_slice],
              {
                  example_count_key: 3,
                  weighted_example_count_key: 6.0,
                  ndcg1_key: 1.0,
                  ndcg2_key: 1.0,
              },
          )
          self.assertDictElementsAlmostEqual(
              slices[query3_slice],
              {
                  example_count_key: 1,
                  weighted_example_count_key: 3.0,
                  ndcg1_key: 1.0,
                  ndcg2_key: 1.0,
              },
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(
          metrics[constants.METRICS_KEY], check_metrics, label='metrics'
      )

  def testEvaluateWithEvalSavedModel(self):
    temp_export_dir = self._getExportDir()
    _, export_dir = linear_classifier.simple_linear_classifier(
        None, temp_export_dir
    )
    eval_config = config_pb2.EvalConfig(
        model_specs=[config_pb2.ModelSpec(signature_name='eval')],
        slicing_specs=[
            config_pb2.SlicingSpec(),
            config_pb2.SlicingSpec(feature_keys=['slice_key']),
        ],
    )
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir,
        add_metrics_callbacks=[_addExampleCountMetricCallback],
    )
    extractors = [
        legacy_predict_extractor.PredictExtractor(
            eval_shared_model, eval_config=eval_config
        ),
        unbatch_extractor.UnbatchExtractor(),
        slice_key_extractor.SliceKeyExtractor(eval_config=eval_config),
    ]
    evaluators = [
        metrics_plots_and_validations_evaluator.MetricsPlotsAndValidationsEvaluator(
            eval_config=eval_config, eval_shared_model=eval_shared_model
        )
    ]

    examples = [
        self._makeExample(
            age=3.0, language='english', label=1.0, slice_key='first_slice'
        ),
        self._makeExample(
            age=3.0, language='chinese', label=0.0, slice_key='first_slice'
        ),
        self._makeExample(
            age=4.0, language='english', label=0.0, slice_key='second_slice'
        ),
        self._makeExample(
            age=5.0, language='chinese', label=1.0, slice_key='second_slice'
        ),
        self._makeExample(
            age=5.0, language='chinese', label=1.0, slice_key='second_slice'
        ),
    ]

    tfx_io = raw_tf_record.RawBeamRecordTFXIO(
        physical_format='inmemory',
        raw_record_column_name=constants.ARROW_INPUT_COLUMN,
        telemetry_descriptors=['TFMATest'],
    )
    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      metrics = (
          pipeline
          | 'Create' >> beam.Create([e.SerializeToString() for e in examples])
          | 'BatchExamples' >> tfx_io.BeamSource()
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | 'ExtractAndEvaluate'
          >> model_eval_lib.ExtractAndEvaluate(
              extractors=extractors, evaluators=evaluators
          )
      )

      # pylint: enable=no-value-for-parameter

      def check_metrics(got):
        try:
          self.assertLen(got, 3)
          slices = {}
          for slice_key, value in got:
            slices[slice_key] = value
          overall_slice = ()
          first_slice = (('slice_key', 'first_slice'),)
          second_slice = (('slice_key', 'second_slice'),)
          self.assertCountEqual(
              list(slices), [overall_slice, first_slice, second_slice]
          )
          self.assertDictElementsAlmostEqual(
              slices[overall_slice],
              {
                  metric_types.MetricKey(
                      name='accuracy', example_weighted=None
                  ): 0.4,
                  metric_types.MetricKey(
                      name='label/mean', example_weighted=None
                  ): 0.6,
                  metric_types.MetricKey(
                      name='my_mean_age', example_weighted=None
                  ): 4.0,
                  metric_types.MetricKey(
                      name='my_mean_age_times_label', example_weighted=None
                  ): 2.6,
                  metric_types.MetricKey(
                      name='added_example_count', example_weighted=None
                  ): 5.0,
              },
          )
          self.assertDictElementsAlmostEqual(
              slices[first_slice],
              {
                  metric_types.MetricKey(
                      name='accuracy', example_weighted=None
                  ): 1.0,
                  metric_types.MetricKey(
                      name='label/mean', example_weighted=None
                  ): 0.5,
                  metric_types.MetricKey(
                      name='my_mean_age', example_weighted=None
                  ): 3.0,
                  metric_types.MetricKey(
                      name='my_mean_age_times_label', example_weighted=None
                  ): 1.5,
                  metric_types.MetricKey(
                      name='added_example_count', example_weighted=None
                  ): 2.0,
              },
          )
          self.assertDictElementsAlmostEqual(
              slices[second_slice],
              {
                  metric_types.MetricKey(
                      name='accuracy', example_weighted=None
                  ): 0.0,
                  metric_types.MetricKey(
                      name='label/mean', example_weighted=None
                  ): (2.0 / 3.0),
                  metric_types.MetricKey(
                      name='my_mean_age', example_weighted=None
                  ): (14.0 / 3.0),
                  metric_types.MetricKey(
                      name='my_mean_age_times_label', example_weighted=None
                  ): (10.0 / 3.0),
                  metric_types.MetricKey(
                      name='added_example_count', example_weighted=None
                  ): 3.0,
              },
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(
          metrics[constants.METRICS_KEY], check_metrics, label='metrics'
      )

  def testValidateWithEvalSavedModel(self):
    temp_export_dir = self._getExportDir()
    _, export_dir = linear_classifier.simple_linear_classifier(
        None, temp_export_dir
    )
    eval_config = config_pb2.EvalConfig(
        model_specs=[config_pb2.ModelSpec(signature_name='eval')],
        metrics_specs=[
            config_pb2.MetricsSpec(
                thresholds={
                    'accuracy': config_pb2.MetricThreshold(
                        value_threshold=config_pb2.GenericValueThreshold(
                            lower_bound={'value': 0.9}
                        )
                    ),
                    'nonexistent_metrics': config_pb2.MetricThreshold(
                        value_threshold=config_pb2.GenericValueThreshold(
                            lower_bound={'value': 0.1}
                        )
                    ),
                }
            )
        ],
        slicing_specs=[
            config_pb2.SlicingSpec(),
        ],
    )
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir,
        add_metrics_callbacks=[_addExampleCountMetricCallback],
    )
    extractors = [
        legacy_predict_extractor.PredictExtractor(
            eval_shared_model, eval_config=eval_config
        ),
        unbatch_extractor.UnbatchExtractor(),
        slice_key_extractor.SliceKeyExtractor(eval_config=eval_config),
    ]
    evaluators = [
        metrics_plots_and_validations_evaluator.MetricsPlotsAndValidationsEvaluator(
            eval_config=eval_config, eval_shared_model=eval_shared_model
        )
    ]

    examples = [
        self._makeExample(
            age=3.0, language='english', label=1.0, slice_key='first_slice'
        ),
        self._makeExample(
            age=3.0, language='chinese', label=0.0, slice_key='first_slice'
        ),
        self._makeExample(
            age=4.0, language='english', label=0.0, slice_key='second_slice'
        ),
        self._makeExample(
            age=5.0, language='chinese', label=1.0, slice_key='second_slice'
        ),
        self._makeExample(
            age=5.0, language='chinese', label=1.0, slice_key='second_slice'
        ),
    ]

    tfx_io = raw_tf_record.RawBeamRecordTFXIO(
        physical_format='inmemory',
        raw_record_column_name=constants.ARROW_INPUT_COLUMN,
        telemetry_descriptors=['TFMATest'],
    )
    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      evaluations = (
          pipeline
          | 'Create' >> beam.Create([e.SerializeToString() for e in examples])
          | 'BatchExamples' >> tfx_io.BeamSource()
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | 'ExtractAndEvaluate'
          >> model_eval_lib.ExtractAndEvaluate(
              extractors=extractors, evaluators=evaluators
          )
      )

      # pylint: enable=no-value-for-parameter

      def check_validations(got):
        try:
          self.assertLen(got, 1)
          got = got[0]
          expected_failures = [
              text_format.Parse(
                  """
                 metric_key {
                    name: "accuracy"
                  }
                  metric_threshold {
                    value_threshold {
                      lower_bound {
                        value: 0.9
                      }
                    }
                  }
                  """,
                  validation_result_pb2.ValidationFailure(),
              ),
              text_format.Parse(
                  """
                 metric_key {
                    name: "nonexistent_metrics"
                  }
                  metric_threshold {
                    value_threshold {
                      lower_bound {
                        value: 0.1
                      }
                    }
                  }
                  message: "Metric not found."
                  """,
                  validation_result_pb2.ValidationFailure(),
              ),
          ]
          self.assertFalse(got.validation_ok)
          self.assertLen(got.metric_validations_per_slice, 1)
          # Ignore the metric_value to avoid fragility of float rounding issue.
          # The correctness of the metric_value has been tested in other tests.
          failures = got.metric_validations_per_slice[0].failures
          self.assertLen(failures, len(expected_failures))
          for failure, expected_failure in zip(failures, expected_failures):
            failure.ClearField('metric_value')
            self.assertProtoEquals(expected_failure, failure)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(
          evaluations[constants.VALIDATIONS_KEY], check_validations
      )

  def testEvaluateWithCrossSlicing(self):
    temp_export_dir1 = self._getExportDir()
    _, export_dir1 = (
        fixed_prediction_estimator_extra_fields.simple_fixed_prediction_estimator_extra_fields(
            None, temp_export_dir1
        )
    )
    temp_export_dir2 = self._getExportDir()
    _, export_dir2 = (
        fixed_prediction_estimator_extra_fields.simple_fixed_prediction_estimator_extra_fields(
            None, temp_export_dir2
        )
    )
    example_count_metric = config_pb2.MetricConfig(
        class_name='ExampleCount',
        # 5 >= 3, OK for overall slice
        # 3 >= 3, OK for ('fixed_string', 'fixed_string1')
        # 2 < 3, NOT OK for ('fixed_string', 'fixed_string2')
        # Keep this for verifying cross slice thresholds and single slice
        # thresholds are working together.
        threshold=config_pb2.MetricThreshold(
            value_threshold=config_pb2.GenericValueThreshold(
                lower_bound={'value': 3}
            )
        ),
        cross_slice_thresholds=[
            config_pb2.CrossSliceMetricThreshold(
                # 5-2 >= 2, OK for ((), (('fixed_string', 'fixed_string1'),))
                # 5-3 < 3, NOT OK for ((), (('fixed_string', 'fixed_string2'),))
                threshold=config_pb2.MetricThreshold(
                    value_threshold=config_pb2.GenericValueThreshold(
                        lower_bound={'value': 3}
                    )
                ),
                cross_slicing_specs=[
                    config_pb2.CrossSlicingSpec(
                        baseline_spec=config_pb2.SlicingSpec(),
                        slicing_specs=[
                            config_pb2.SlicingSpec(
                                feature_keys=['fixed_string']
                            )
                        ],
                    )
                ],
            )
        ],
    )
    mean_prediction_metric = config_pb2.MetricConfig(
        class_name='MeanPrediction',
        cross_slice_thresholds=[
            config_pb2.CrossSliceMetricThreshold(
                # MeanPrediction values for slices:
                # (0.2*2+0.9*2)/(2+2)=0.55
                #     for (('fixed_string', 'fixed_string1'),)
                # (0.5*2+0.5*2+0.5*2)/(2+2+2)=0.5
                #     for (('fixed_string', 'fixed_string2'),)
                threshold=config_pb2.MetricThreshold(
                    value_threshold=config_pb2.GenericValueThreshold(
                        # This config should give value threshold error because
                        # (0.55-0.5)=0.05 not inside the bound [0.1, 0.5].
                        upper_bound={'value': 0.5},
                        lower_bound={'value': 0.1},
                    ),
                    change_threshold=config_pb2.GenericChangeThreshold(
                        # This config should give change threshold error because
                        # baseline model and candidate model have same
                        # difference as 0.05 between cross slices. Cross slice
                        # difference value is not changed.
                        direction=config_pb2.MetricDirection.LOWER_IS_BETTER,
                        relative={'value': -0.99},
                        absolute={'value': 0},
                    ),
                ),
                cross_slicing_specs=[
                    config_pb2.CrossSlicingSpec(
                        baseline_spec=config_pb2.SlicingSpec(
                            feature_values={'fixed_string': 'fixed_string1'}
                        ),
                        slicing_specs=[
                            config_pb2.SlicingSpec(
                                feature_values={'fixed_string': 'fixed_string2'}
                            )
                        ],
                    )
                ],
            )
        ],
    )
    eval_config = config_pb2.EvalConfig(
        model_specs=[
            config_pb2.ModelSpec(
                name='candidate',
                label_key='label',
                example_weight_key='fixed_float',
            ),
            config_pb2.ModelSpec(
                name='baseline',
                label_key='label',
                example_weight_key='fixed_float',
                is_baseline=True,
            ),
        ],
        slicing_specs=[
            config_pb2.SlicingSpec(),
            config_pb2.SlicingSpec(feature_keys=['fixed_string']),
        ],
        cross_slicing_specs=[
            config_pb2.CrossSlicingSpec(
                baseline_spec=config_pb2.SlicingSpec(),
                slicing_specs=[
                    config_pb2.SlicingSpec(feature_keys=['fixed_string'])
                ],
            ),
            config_pb2.CrossSlicingSpec(
                baseline_spec=config_pb2.SlicingSpec(
                    feature_values={'fixed_string': 'fixed_string1'}
                ),
                slicing_specs=[
                    config_pb2.SlicingSpec(
                        feature_values={'fixed_string': 'fixed_string2'}
                    )
                ],
            ),
        ],
        metrics_specs=[
            config_pb2.MetricsSpec(
                metrics=[example_count_metric],
                model_names=['candidate', 'baseline'],
                example_weights=config_pb2.ExampleWeightOptions(
                    unweighted=True
                ),
            ),
            config_pb2.MetricsSpec(
                metrics=[mean_prediction_metric],
                model_names=['candidate', 'baseline'],
            ),
        ],
    )
    eval_shared_model = [
        self.createTestEvalSharedModel(
            model_name='candidate', eval_saved_model_path=export_dir1
        ),
        self.createTestEvalSharedModel(
            model_name='baseline', eval_saved_model_path=export_dir2
        ),
    ]
    extractors = [
        legacy_predict_extractor.PredictExtractor(
            eval_shared_model=eval_shared_model, eval_config=eval_config
        ),
        unbatch_extractor.UnbatchExtractor(),
        slice_key_extractor.SliceKeyExtractor(eval_config=eval_config),
    ]
    evaluators = [
        metrics_plots_and_validations_evaluator.MetricsPlotsAndValidationsEvaluator(
            eval_config=eval_config, eval_shared_model=eval_shared_model
        )
    ]

    # fixed_float used as example_weight key
    examples = [
        self._makeExample(
            prediction=0.2,
            label=1.0,
            fixed_int=1,
            fixed_float=1.0,
            fixed_string='fixed_string1',
        ),
        self._makeExample(
            prediction=0.9,
            label=0.0,
            fixed_int=1,
            fixed_float=1.0,
            fixed_string='fixed_string1',
        ),
        self._makeExample(
            prediction=0.5,
            label=0.0,
            fixed_int=2,
            fixed_float=2.0,
            fixed_string='fixed_string2',
        ),
        self._makeExample(
            prediction=0.5,
            label=0.0,
            fixed_int=2,
            fixed_float=2.0,
            fixed_string='fixed_string2',
        ),
        self._makeExample(
            prediction=0.5,
            label=0.0,
            fixed_int=2,
            fixed_float=2.0,
            fixed_string='fixed_string2',
        ),
    ]

    tfx_io = raw_tf_record.RawBeamRecordTFXIO(
        physical_format='inmemory',
        raw_record_column_name=constants.ARROW_INPUT_COLUMN,
        telemetry_descriptors=['TFMATest'],
    )
    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      evaluations = (
          pipeline
          | 'Create' >> beam.Create([e.SerializeToString() for e in examples])
          | 'BatchExamples' >> tfx_io.BeamSource()
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | 'ExtractAndEvaluate'
          >> model_eval_lib.ExtractAndEvaluate(
              extractors=extractors, evaluators=evaluators
          )
      )

      # pylint: enable=no-value-for-parameter

      def check_validations(got):
        try:
          self.assertLen(got, 6)

          def get_slice_keys_hash(metric_validation):
            slice_key, cross_slice_key = None, None
            if metric_validation.slice_key:
              slice_key = metric_validation.slice_key.SerializeToString()
            if metric_validation.cross_slice_key:
              cross_slice_key = (
                  metric_validation.cross_slice_key.SerializeToString()
              )
            return (slice_key, cross_slice_key)

          successful_validations = []
          failed_validations = {}
          for validation_result in got:
            if validation_result.validation_ok:
              successful_validations.append(validation_result)
            else:
              self.assertLen(validation_result.metric_validations_per_slice, 1)
              validation_result = (
                  validation_result.metric_validations_per_slice[0]
              )
              slice_keys_hash = get_slice_keys_hash(validation_result)
              failed_validations[slice_keys_hash] = {}
              for failure in validation_result.failures:
                failure.ClearField('metric_value')
                failed_validations[slice_keys_hash][
                    failure.metric_key.SerializeToString()
                ] = failure
          self.assertLen(successful_validations, 3)
          self.assertLen(failed_validations, 3)

          expected_validations = [
              text_format.Parse(
                  """
                  slice_key {
                    single_slice_keys {
                      column: "fixed_string"
                      bytes_value: "fixed_string1"
                    }
                  }
                  failures {
                    metric_key {
                      name: "example_count"
                      model_name: "candidate"
                      example_weighted { }
                    }
                    metric_threshold {
                      value_threshold {
                        lower_bound {
                          value: 3.0
                        }
                      }
                    }
                  }
                  """,
                  validation_result_pb2.MetricsValidationForSlice(),
              ),
              text_format.Parse(
                  """
                  failures {
                    metric_key {
                      name: "example_count"
                      model_name: "candidate"
                      example_weighted { }
                    }
                    metric_threshold {
                      value_threshold {
                        lower_bound {
                          value: 3.0
                        }
                      }
                    }
                  }
                  cross_slice_key {
                    baseline_slice_key {
                    }
                    comparison_slice_key {
                      single_slice_keys {
                        column: "fixed_string"
                        bytes_value: "fixed_string2"
                      }
                    }
                  }
                  """,
                  validation_result_pb2.MetricsValidationForSlice(),
              ),
              text_format.Parse(
                  """
                  failures {
                    metric_key {
                      name: "mean_prediction"
                      model_name: "candidate"
                      example_weighted { value: true }
                    }
                    metric_threshold {
                      value_threshold {
                        lower_bound {
                          value: 0.1
                        }
                        upper_bound {
                          value: 0.5
                        }
                      }
                    }
                  }
                  failures {
                    metric_key {
                      name: "mean_prediction"
                      model_name: "candidate"
                      is_diff: true
                      example_weighted { value: true }
                    }
                    metric_threshold {
                      change_threshold {
                        absolute {
                        }
                        relative {
                          value: -0.99
                        }
                        direction: LOWER_IS_BETTER
                      }
                    }
                  }
                  cross_slice_key {
                    baseline_slice_key {
                      single_slice_keys {
                        column: "fixed_string"
                        bytes_value: "fixed_string1"
                      }
                    }
                    comparison_slice_key {
                      single_slice_keys {
                        column: "fixed_string"
                        bytes_value: "fixed_string2"
                      }
                    }
                  }
                  """,
                  validation_result_pb2.MetricsValidationForSlice(),
              ),
          ]

          expected_validations_dict = {}
          for expected_validation in expected_validations:
            slice_keys_hash = get_slice_keys_hash(expected_validation)
            expected_validations_dict[slice_keys_hash] = {}
            for failure in expected_validation.failures:
              expected_validations_dict[slice_keys_hash][
                  failure.metric_key.SerializeToString()
              ] = failure

          self.assertEqual(
              set(failed_validations), set(expected_validations_dict)
          )
          for slice_key, validation in failed_validations.items():
            self.assertEqual(
                set(validation), set(expected_validations_dict[slice_key])
            )
            for metric_key, failure in validation.items():
              self.assertProtoEquals(
                  failure, expected_validations_dict[slice_key][metric_key]
              )
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(
          evaluations[constants.VALIDATIONS_KEY], check_validations
      )

  def testAddCrossSliceMetricsMatchAll(self):
    overall_slice_key = ()
    slice_key1 = (('feature', 1),)
    slice_key2 = (('feature', 2),)
    slice_key3 = (('feature', 3),)
    metrics_dict = {}
    sliced_metrics = [
        (overall_slice_key, metrics_dict),
        (slice_key1, metrics_dict),
        (slice_key2, metrics_dict),
        (slice_key3, metrics_dict),
    ]
    with beam.Pipeline() as pipeline:
      cross_sliced_metrics = (
          pipeline
          | 'CreateSlicedMetrics' >> beam.Create(sliced_metrics)
          | 'AddCrossSliceMetrics'
          >> metrics_plots_and_validations_evaluator._AddCrossSliceMetrics(
              cross_slice_specs=[
                  config_pb2.CrossSlicingSpec(
                      baseline_spec={}, slicing_specs=[]
                  )
              ],
              cross_slice_computations=[],
          )
      )

      def check_result(got_sliced_metrics):
        actual_slice_keys = [k for k, _ in got_sliced_metrics]
        expected_slice_keys = [
            # cross slice keys
            (overall_slice_key, slice_key1),
            (overall_slice_key, slice_key2),
            (overall_slice_key, slice_key3),
            # single slice keys
            overall_slice_key,
            slice_key1,
            slice_key2,
            slice_key3,
        ]
        self.assertCountEqual(expected_slice_keys, actual_slice_keys)

      util.assert_that(cross_sliced_metrics, check_result)

  @parameterized.named_parameters(
      ('IntIsDiffable', 1, True),
      ('FloatIsDiffable', 1.0, True),
      ('NumpyFloatDtypeIsDiffable', np.array([1.0], dtype=np.float64), True),
      ('NumpyIntDtypeIsDiffable', np.array([1], dtype=np.int64), True),
      ('MessageNotDiffable', validation_result_pb2.ValidationResult(), False),
      (
          'TupleNotDiffable',
          binary_confusion_matrices.Matrices(
              thresholds=[-1e-7, 0.5, 1.0 + 1e-7],
              tp=[2.0, 1.0, 0.0],
              fp=[2.0, 0.0, 0.0],
              tn=[0.0, 2.0, 2.0],
              fn=[0.0, 1.0, 2.0],
          ),
          True,
      ),
      ('BytesNotDiffable', b'some bytes', False),
      ('NumpyObjectDtypeNotDiffable', np.array(['obj'], dtype=object), False),
  )
  def testIsMetricDiffable(self, metric_value, expected_is_diffable):
    self.assertEqual(
        expected_is_diffable,
        metrics_plots_and_validations_evaluator._is_metric_diffable(
            metric_value
        ),
    )

  def testMetricsSpecsCountersInModelAgnosticMode(self):
    schema = text_format.Parse(
        """
        feature {
          name: "label"
          type: FLOAT
        }
        feature {
          name: "prediction"
          type: FLOAT
        }
        """,
        schema_pb2.Schema(),
    )

    tfx_io = test_util.InMemoryTFExampleRecord(
        schema=schema, raw_record_column_name=constants.ARROW_INPUT_COLUMN
    )

    examples = [
        self._makeExample(label=1.0, prediction=0.7),
        self._makeExample(label=0.0, prediction=0.3),
    ]

    eval_config = config_pb2.EvalConfig(
        model_specs=[
            config_pb2.ModelSpec(prediction_key='prediction', label_key='label')
        ],
        metrics_specs=[
            config_pb2.MetricsSpec(
                metrics=[config_pb2.MetricConfig(class_name='ExampleCount')]
            )
        ],
        slicing_specs=[config_pb2.SlicingSpec()],
    )

    extractors = [
        features_extractor.FeaturesExtractor(eval_config),
        labels_extractor.LabelsExtractor(eval_config),
        example_weights_extractor.ExampleWeightsExtractor(eval_config),
        materialized_predictions_extractor.MaterializedPredictionsExtractor(
            eval_config
        ),
        unbatch_extractor.UnbatchExtractor(),
        slice_key_extractor.SliceKeyExtractor(eval_config=eval_config),
    ]
    evaluators = [
        metrics_plots_and_validations_evaluator.MetricsPlotsAndValidationsEvaluator(
            eval_config
        )
    ]

    with beam.Pipeline() as pipeline:
      _ = (
          pipeline
          | 'Create' >> beam.Create([e.SerializeToString() for e in examples])
          | 'BatchExamples' >> tfx_io.BeamSource()
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | 'ExtractEvaluate'
          >> model_eval_lib.ExtractAndEvaluate(
              extractors=extractors, evaluators=evaluators
          )
      )

    metric_filter = beam.metrics.metric.MetricsFilter().with_name(
        'metric_computed_ExampleCount_v2_' + constants.MODEL_AGNOSTIC
    )
    actual_metrics_count = (
        pipeline.run()
        .metrics()
        .query(filter=metric_filter)['counters'][0]
        .committed
    )
    self.assertEqual(actual_metrics_count, 1)


