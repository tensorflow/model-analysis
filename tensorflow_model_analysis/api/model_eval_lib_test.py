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
"""Test for using the model_eval_lib API."""


import pytest
import json
import os
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.evaluators import metrics_plots_and_validations_evaluator
from tensorflow_model_analysis.metrics import calibration_plot
from tensorflow_model_analysis.metrics import confusion_matrix_metrics
from tensorflow_model_analysis.metrics import metric_specs
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.metrics import ndcg
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.proto import metrics_for_slice_pb2
from tensorflow_model_analysis.proto import validation_result_pb2
from tensorflow_model_analysis.utils import example_keras_model
from tensorflow_model_analysis.utils import test_util
from tensorflow_model_analysis.utils.keras_lib import tf_keras
from tensorflow_model_analysis.view import view_types
from tfx_bsl.coders import example_coder

from google.protobuf import wrappers_pb2
from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2

try:
  import tensorflow_ranking as tfr  # pylint: disable=g-import-not-at-top

  _TFR_IMPORTED = True
except (ImportError, tf.errors.NotFoundError):
  _TFR_IMPORTED = False

try:
  from tensorflowjs.converters import converter as tfjs_converter  # pylint: disable=g-import-not-at-top

  _TFJS_IMPORTED = True
except ModuleNotFoundError:
  _TFJS_IMPORTED = False

_TEST_SEED = 982735

_TF_MAJOR_VERSION = int(tf.version.VERSION.split('.')[0])


@pytest.mark.xfail(run=False, reason="PR 183 This class contains tests that fail and needs to be fixed. "
"If all tests pass, please remove this mark.")
class EvaluateTest(
    test_util.TensorflowModelAnalysisTest, parameterized.TestCase
):

  def setUp(self):
    super().setUp()
    self.longMessage = True  # pylint: disable=invalid-name

  def _getTempDir(self):
    return tempfile.mkdtemp()

  def _exportEvalSavedModel(self, classifier):
    temp_eval_export_dir = os.path.join(self._getTempDir(), 'eval_export_dir')
    _, eval_export_dir = classifier(None, temp_eval_export_dir)
    return eval_export_dir

  def _exportKerasModel(self, classifier):
    temp_export_dir = os.path.join(self._getTempDir(), 'saved_model_export_dir')
    classifier.export(temp_export_dir)
    return temp_export_dir

  def _writeTFExamplesToTFRecords(self, examples):
    data_location = os.path.join(self._getTempDir(), 'input_data.rio')
    with tf.io.TFRecordWriter(data_location) as writer:
      for example in examples:
        writer.write(example.SerializeToString())
    return data_location

  def _writeCSVToTextFile(self, examples):
    data_location = os.path.join(self._getTempDir(), 'input_data.csv')
    with open(data_location, 'w') as writer:
      for example in examples:
        writer.write(example + '\n')
    return data_location

  def assertMetricsAlmostEqual(
      self,
      got_slicing_metrics,
      expected_slicing_metrics,
      output_name='',
      subkey='',
  ):
    if got_slicing_metrics:
      for s, m in got_slicing_metrics:
        metrics = m[output_name][subkey]
        self.assertIn(s, expected_slicing_metrics)
        for metric_name in expected_slicing_metrics[s]:
          self.assertIn(metric_name, metrics)
          self.assertDictElementsAlmostEqual(
              metrics[metric_name], expected_slicing_metrics[s][metric_name]
          )
    else:
      # Only pass if expected_slicing_metrics also evaluates to False.
      self.assertFalse(
          expected_slicing_metrics, msg='Actual slicing_metrics was empty.'
      )

  def assertSliceMetricsEqual(self, expected_metrics, got_metrics):
    self.assertCountEqual(
        list(expected_metrics),
        list(got_metrics),
        msg='keys do not match. expected_metrics: %s, got_metrics: %s'
        % (expected_metrics, got_metrics),
    )
    for key in expected_metrics:
      self.assertProtoEquals(
          expected_metrics[key],
          got_metrics[key],
          msg='value for key %s does not match' % key,
      )

  def assertSliceListEqual(self, expected_list, got_list, value_assert_fn):
    self.assertEqual(
        len(expected_list),
        len(got_list),
        msg='expected_list: %s, got_list: %s' % (expected_list, got_list),
    )
    for index, (expected, got) in enumerate(zip(expected_list, got_list)):
      (expected_key, expected_value) = expected
      (got_key, got_value) = got
      self.assertEqual(
          expected_key, got_key, msg='key mismatch at index %d' % index
      )
      value_assert_fn(expected_value, got_value)

  def assertSlicePlotsListEqual(self, expected_list, got_list):
    self.assertSliceListEqual(expected_list, got_list, self.assertProtoEquals)

  def assertSliceMetricsListEqual(self, expected_list, got_list):
    self.assertSliceListEqual(
        expected_list, got_list, self.assertSliceMetricsEqual
    )

  @parameterized.named_parameters(
      ('tflite', constants.TF_LITE), ('tfjs', constants.TF_JS)
  )
  def testMixedModelTypes(self, model_type):
    examples = [self._makeExample(age=3.0, language='english', label=1.0)]
    data_location = self._writeTFExamplesToTFRecords(examples)
    eval_config = config_pb2.EvalConfig(
        model_specs=[
            config_pb2.ModelSpec(name='model1'),
            config_pb2.ModelSpec(name='model2', model_type=model_type),
        ]
    )
    eval_shared_models = [
        model_eval_lib.default_eval_shared_model(
            model_name='model1',
            eval_saved_model_path='/model1/path',
            eval_config=eval_config,
        ),
        model_eval_lib.default_eval_shared_model(
            model_name='model2',
            eval_saved_model_path='/model2/path',
            eval_config=eval_config,
        ),
    ]
    with self.assertRaisesRegex(
        NotImplementedError, 'support for mixing .* models is not implemented'
    ):
      model_eval_lib.run_model_analysis(
          eval_config=eval_config,
          eval_shared_model=eval_shared_models,
          data_location=data_location,
          output_path=self._getTempDir(),
      )

  def testRunModelAnalysis(self):
    examples = [
        self._makeExample(age=3.0, language='english', label=1.0),
        self._makeExample(age=3.0, language='chinese', label=0.0),
        self._makeExample(age=4.0, language='english', label=1.0),
        self._makeExample(age=5.0, language='chinese', label=1.0),
        self._makeExample(age=5.0, language='hindi', label=1.0),
    ]
    classifier = example_keras_model.get_example_classifier_model(
        example_keras_model.LANGUAGE
    )
    classifier.compile(optimizer=keras.optimizers.Adam(), loss='mse')
    classifier.fit(
        tf.constant([e.SerializeToString() for e in examples]),
        np.array([
            e.features.feature[example_keras_model.LABEL].float_list.value[:][0]
            for e in examples
        ]),
        batch_size=1,
    )
    model_location = self._exportKerasModel(classifier)
    data_location = self._writeTFExamplesToTFRecords(examples)
    eval_config = config_pb2.EvalConfig(
        model_specs=[
            config_pb2.ModelSpec(
                name='model1', example_weight_key='age', label_key='label'
            )
        ],
        slicing_specs=[config_pb2.SlicingSpec(feature_keys=['language'])],
        metrics_specs=[
            config_pb2.MetricsSpec(
                metrics=[
                    config_pb2.MetricConfig(
                        class_name='ExampleCount',
                    ),
                    config_pb2.MetricConfig(
                        class_name='Accuracy',
                    ),
                ]
            )
        ],
        options=config_pb2.Options(
            min_slice_size=wrappers_pb2.Int32Value(value=2)
        ),
    )
    eval_result = model_eval_lib.run_model_analysis(
        eval_shared_model=model_eval_lib.default_eval_shared_model(
            eval_saved_model_path=model_location, eval_config=eval_config
        ),
        data_location=data_location,
        eval_config=eval_config,
        output_path=self._getTempDir(),
    )
    # We only check some of the metrics to ensure that the end-to-end
    # pipeline works.
    expected = {
        (('language', 'hindi'),): {
            '__ERROR__': {
                'debugMessage': (
                    'Example count for this slice key is lower than the '
                    'minimum required value: 2. No data is aggregated for '
                    'this slice.'
                )
            },
        },
        (('language', 'chinese'),): {
            'accuracy': {'doubleValue': 0.0},
            'example_count': {'doubleValue': 8.0},
        },
        (('language', 'english'),): {
            'accuracy': {'doubleValue': 0.0},
            'example_count': {'doubleValue': 7.0},
        },
    }
    self.assertEqual(eval_result.model_location, model_location)
    self.assertEqual(eval_result.data_location, data_location)
    self.assertEqual(
        eval_result.config.slicing_specs[0],
        config_pb2.SlicingSpec(feature_keys=['language']),
    )
    self.assertMetricsAlmostEqual(eval_result.slicing_metrics, expected)
    for _, plot in eval_result.plots:
      self.assertFalse(plot)

  @parameterized.named_parameters(
      {
          'testcase_name': 'WithHistogram',
          'eval_config': text_format.Parse(
              """
              model_specs {
                label_key: "labels"
                prediction_key: "predictions"
              }
              slicing_specs {}
              metrics_specs {
                aggregate: {
                    micro_average: true
                    class_weights { key: 0 value: 1.0 }
                    class_weights { key: 1 value: 0.0 }
                }
                metrics {
                  class_name: "Recall"
                  config: '"name": "recall_class_0", "num_thresholds": 3'
                }
              }
              metrics_specs {
                aggregate: {
                    micro_average: true
                    class_weights { key: 0 value: 0.0 }
                    class_weights { key: 1 value: 1.0 }
                }
                metrics {
                  class_name: "Recall"
                  config: '"name": "recall_class_1", "num_thresholds": 3'
                }
              }
              """,
              config_pb2.EvalConfig(),
          ),
          'expected_class_0_recall': text_format.Parse(
              """
              array_value {
                  data_type: FLOAT64
                  shape: 3
                  float64_values: 1.0
                  float64_values: 1.0
                  float64_values: 0.0
              }
              """,
              metrics_for_slice_pb2.MetricValue(),
          ),
          'expected_class_1_recall': text_format.Parse(
              """
              array_value {
                  data_type: FLOAT64
                  shape: 3
                  float64_values: 1.0
                  float64_values: 0.0
                  float64_values: 0.0
              }
              """,
              metrics_for_slice_pb2.MetricValue(),
          ),
      },
      {
          'testcase_name': 'NoHistogram',
          'eval_config': text_format.Parse(
              """
              model_specs {
                label_key: "labels"
                prediction_key: "predictions"
              }
              slicing_specs {}
              metrics_specs {
                aggregate: {
                    micro_average: true
                    class_weights { key: 0 value: 1.0 }
                    class_weights { key: 1 value: 0.0 }
                }
                metrics {
                  class_name: "Recall"
                  config: '"name": "recall_class_0"'
                }
              }
              metrics_specs {
                aggregate: {
                    micro_average: true
                    class_weights { key: 0 value: 0.0 }
                    class_weights { key: 1 value: 1.0 }
                }
                metrics {
                  class_name: "Recall"
                  config: '"name": "recall_class_1"'
                }
              }
              """,
              config_pb2.EvalConfig(),
          ),
          'expected_class_0_recall': text_format.Parse(
              'double_value { value: 1.0 }',
              metrics_for_slice_pb2.MetricValue(),
          ),
          'expected_class_1_recall': text_format.Parse(
              'double_value { value: 0.0 }',
              metrics_for_slice_pb2.MetricValue(),
          ),
      },
  )
  def testRunModelAnalysisMultiMicroAggregation(
      self, eval_config, expected_class_0_recall, expected_class_1_recall
  ):
    # class 0 is all TPs so has recall 1.0, class 1 is all FPs so has recall 0.0
    examples = [
        self._makeExample(labels=[1.0, 1.0], predictions=[0.9, 0.1]),
        self._makeExample(labels=[1.0, 1.0], predictions=[0.9, 0.1]),
    ]
    data_location = self._writeTFExamplesToTFRecords(examples)
    output_dir = self._getTempDir()
    model_eval_lib.run_model_analysis(
        eval_config=eval_config,
        data_location=data_location,
        output_path=output_dir,
    )

    metrics_for_slice = list(model_eval_lib.load_metrics(output_dir))
    self.assertLen(metrics_for_slice, 1)
    metric_keys_to_values = {
        metric_types.MetricKey.from_proto(kv.key): kv.value
        for kv in metrics_for_slice[0].metric_keys_and_values
    }
    class_0_key = metric_types.MetricKey(
        name='recall_class_0',
        aggregation_type=metric_types.AggregationType(micro_average=True),
    )
    class_1_key = metric_types.MetricKey(
        name='recall_class_1',
        aggregation_type=metric_types.AggregationType(micro_average=True),
    )
    self.assertIn(class_0_key, metric_keys_to_values)
    self.assertEqual(
        expected_class_0_recall, metric_keys_to_values[class_0_key]
    )
    self.assertIn(class_1_key, metric_keys_to_values)
    self.assertEqual(
        expected_class_1_recall, metric_keys_to_values[class_1_key]
    )

  def testRunModelAnalysisWithExplicitModelAgnosticPredictions(self):
    examples = [
        self._makeExample(
            age=3.0, language='english', label=1.0, prediction=0.9
        ),
        self._makeExample(
            age=3.0, language='chinese', label=0.0, prediction=0.4
        ),
        self._makeExample(
            age=4.0, language='english', label=1.0, prediction=0.7
        ),
        self._makeExample(
            age=5.0, language='chinese', label=1.0, prediction=0.2
        ),
    ]
    metrics_specs = [
        config_pb2.MetricsSpec(
            metrics=[config_pb2.MetricConfig(class_name='ExampleCount')],
            example_weights=config_pb2.ExampleWeightOptions(unweighted=True),
        ),
        config_pb2.MetricsSpec(
            metrics=[
                config_pb2.MetricConfig(class_name='WeightedExampleCount')
            ],
            example_weights=config_pb2.ExampleWeightOptions(weighted=True),
        ),
        config_pb2.MetricsSpec(
            metrics=[config_pb2.MetricConfig(class_name='BinaryAccuracy')],
            example_weights=config_pb2.ExampleWeightOptions(weighted=True),
        ),
    ]
    slicing_specs = [config_pb2.SlicingSpec(feature_keys=['language'])]
    model_spec = config_pb2.ModelSpec(
        prediction_key='prediction',
        label_key='label',
        example_weight_key='age',
    )
    eval_config = config_pb2.EvalConfig(
        model_specs=[model_spec],
        metrics_specs=metrics_specs,
        slicing_specs=slicing_specs,
    )
    data_location = self._writeTFExamplesToTFRecords(examples)
    eval_result = model_eval_lib.run_model_analysis(
        eval_config=eval_config,
        data_location=data_location,
        output_path=self._getTempDir(),
    )
    expected = {
        (('language', 'chinese'),): {
            'binary_accuracy': {'doubleValue': 0.375},
            'weighted_example_count': {'doubleValue': 8.0},
            'example_count': {'doubleValue': 2.0},
        },
        (('language', 'english'),): {
            'binary_accuracy': {'doubleValue': 1.0},
            'weighted_example_count': {'doubleValue': 7.0},
            'example_count': {'doubleValue': 2.0},
        },
    }
    self.assertEqual(eval_result.data_location, data_location)
    self.assertEqual(
        eval_result.config.slicing_specs[0],
        config_pb2.SlicingSpec(feature_keys=['language']),
    )
    self.assertMetricsAlmostEqual(eval_result.slicing_metrics, expected)

  @parameterized.named_parameters(
      ('tf_keras', constants.TF_KERAS),
      ('tf_lite', constants.TF_LITE),
      ('tf_js', constants.TF_JS),
      ('baseline_missing', constants.TF_KERAS, True),
      ('rubber_stamp', constants.TF_KERAS, True, True),
      ('tf_keras_custom_metrics', constants.TF_KERAS, False, False, True),
  )
  def testRunModelAnalysisWithKerasModel(
      self,
      model_type,
      remove_baseline=False,
      rubber_stamp=False,
      add_custom_metrics=False,
  ):
    if model_type == constants.TF_JS and not _TFJS_IMPORTED:
      self.skipTest('This test requires TensorFlow JS.')

    # Custom metrics not supported in TFv1
    if _TF_MAJOR_VERSION < 2:
      add_custom_metrics = False

    def _build_keras_model(
        eval_config, export_name='export_dir', rubber_stamp=False
    ):
      input_layer = tf_keras.layers.Input(shape=(28 * 28,), name='data')
      output_layer = tf_keras.layers.Dense(10, activation=tf.nn.softmax)(
          input_layer
      )
      model = tf_keras.models.Model(input_layer, output_layer)
      model.compile(
          optimizer=tf_keras.optimizers.Adam(lr=0.001),
          loss=tf_keras.losses.categorical_crossentropy,
      )
      if add_custom_metrics:
        model.add_metric(tf.reduce_sum(input_layer), 'custom')
      model_location = os.path.join(self._getTempDir(), export_name)
      if model_type == constants.TF_LITE:
        converter = tf.compat.v2.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        tf.io.gfile.makedirs(model_location)
        with tf.io.gfile.GFile(
            os.path.join(model_location, 'tflite'), 'wb'
        ) as f:
          f.write(tflite_model)
      elif model_type == constants.TF_JS:
        src_model_path = tempfile.mkdtemp()
        model.export(src_model_path)

        tfjs_converter.convert([
            '--input_format=tf_saved_model',
            '--saved_model_tags=serve',
            '--signature_name=serving_default',
            src_model_path,
            model_location,
        ])
      else:
        model.export(model_location)
      return model_eval_lib.default_eval_shared_model(
          eval_saved_model_path=model_location,
          eval_config=eval_config,
          rubber_stamp=rubber_stamp,
      )

    examples = [
        self._makeExample(
            data=[0.0] * 28 * 28,
            label=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        self._makeExample(
            data=[1.0] * 28 * 28,
            label=[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        self._makeExample(
            data=[1.0] * 28 * 28,
            label=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ),
    ]
    data_location = self._writeTFExamplesToTFRecords(examples)

    schema = text_format.Parse(
        """
        tensor_representation_group {
          key: ""
          value {
            tensor_representation {
              key: "data"
              value {
                dense_tensor {
                  column_name: "data"
                  shape { dim { size: 784 } }
                }
              }
            }
          }
        }
        feature {
          name: "data"
          type: FLOAT
        }
        feature {
          name: "label"
          type: FLOAT
          shape: { dim { size: 10 } }
          presence: { min_fraction: 1 }
        }
        """,
        schema_pb2.Schema(),
    )
    # TODO(b/73109633): Remove when field is removed or its default changes to
    # False.
    if hasattr(schema, 'generate_legacy_feature_spec'):
      schema.generate_legacy_feature_spec = False

    metrics_spec = config_pb2.MetricsSpec()
    for metric in (confusion_matrix_metrics.AUC(name='auc'),):
      cfg = metric_util.serialize_keras_object(metric)
      metrics_spec.metrics.append(
          config_pb2.MetricConfig(
              class_name=cfg['class_name'], config=json.dumps(cfg['config'])
          )
      )
    tf_keras.backend.clear_session()
    slicing_specs = [
        config_pb2.SlicingSpec(),
        config_pb2.SlicingSpec(feature_keys=['non_existent_slice']),
    ]
    metrics_spec.metrics.append(
        config_pb2.MetricConfig(
            class_name='ExampleCount',
            per_slice_thresholds=[
                config_pb2.PerSliceMetricThreshold(
                    slicing_specs=slicing_specs,
                    threshold=config_pb2.MetricThreshold(
                        value_threshold=config_pb2.GenericValueThreshold(
                            lower_bound={'value': 1}
                        )
                    ),
                ),
                # Change thresholds would be ignored when rubber stamp is true.
                config_pb2.PerSliceMetricThreshold(
                    slicing_specs=slicing_specs,
                    threshold=config_pb2.MetricThreshold(
                        change_threshold=config_pb2.GenericChangeThreshold(
                            direction=config_pb2.MetricDirection.HIGHER_IS_BETTER,
                            absolute={'value': 1},
                        )
                    ),
                ),
            ],
        )
    )
    for class_id in (0, 5):
      metrics_spec.binarize.class_ids.values.append(class_id)
    eval_config = config_pb2.EvalConfig(
        model_specs=[config_pb2.ModelSpec(label_key='label')],
        metrics_specs=[metrics_spec],
    )
    if model_type != constants.TF_KERAS:
      for s in eval_config.model_specs:
        s.model_type = model_type

    model = _build_keras_model(eval_config, rubber_stamp=rubber_stamp)
    baseline = _build_keras_model(eval_config, 'baseline_export')
    if remove_baseline:
      eval_shared_model = model
    else:
      eval_shared_model = {'candidate': model, 'baseline': baseline}
    output_path = self._getTempDir()
    # Raise RuntimeError for missing baseline with change thresholds.
    if not rubber_stamp and remove_baseline:
      with self.assertRaises(RuntimeError):
        model_eval_lib.run_model_analysis(
            eval_config=eval_config,
            eval_shared_model=eval_shared_model,
            data_location=data_location,
            output_path=output_path,
            schema=schema,
        )
      # Will not have any result since the pipeline didn't run.
      return
    else:
      eval_results = model_eval_lib.run_model_analysis(
          eval_config=eval_config,
          eval_shared_model=eval_shared_model,
          data_location=data_location,
          output_path=output_path,
          schema=schema,
      )

    # Directly check validation file since it is not in EvalResult.
    validations_file = os.path.join(
        output_path, f'{constants.VALIDATIONS_KEY}.tfrecord'
    )
    self.assertTrue(os.path.exists(validations_file))
    validation_records = []
    for record in tf.compat.v1.python_io.tf_record_iterator(validations_file):
      validation_records.append(
          validation_result_pb2.ValidationResult.FromString(record)
      )
    self.assertLen(validation_records, 1)
    # Change thresholds ignored when rubber stamping
    expected_result = text_format.Parse(
        """
        validation_ok: false
        rubber_stamp: %s
        missing_slices: {
          feature_keys: "non_existent_slice"
        }
        validation_details {
          slicing_details {
            slicing_spec {
            }
            num_matching_slices: 1
          }
        }""" % rubber_stamp,
        validation_result_pb2.ValidationResult(),
    )
    # Normal run with change threshold not satisfied.
    if not rubber_stamp and not remove_baseline:
      text_format.Parse(
          """
          metric_validations_per_slice {
            slice_key {}
            failures {
             metric_key {
               name: "example_count"
               sub_key { class_id {} }
               model_name: "candidate"
               is_diff: true
               example_weighted { }
             }
             metric_threshold {
               change_threshold {
                 absolute { value: 1 }
                 direction: HIGHER_IS_BETTER
               }
             }
             metric_value { double_value {} }
            }
            failures {
             metric_key {
               name: "example_count"
               sub_key {
                 class_id {
                   value: 5
                 }
               }
               model_name: "candidate"
               is_diff: true
               example_weighted { }
             }
             metric_threshold {
               change_threshold {
                 absolute { value: 1}
                 direction: HIGHER_IS_BETTER
               }
             }
             metric_value { double_value {} }
            }
          }""",
          expected_result,
      )
    self.assertProtoEquals(expected_result, validation_records[0])

    def check_eval_result(eval_result, model_location):
      self.assertEqual(eval_result.model_location, model_location)
      self.assertEqual(eval_result.data_location, data_location)
      self.assertLen(eval_result.slicing_metrics, 1)
      got_slice_key, got_metrics = eval_result.slicing_metrics[0]
      self.assertEqual(got_slice_key, ())
      self.assertIn('', got_metrics)  # output_name
      got_metrics = got_metrics['']
      expected_metrics = {
          'classId:0': {
              'auc': True,
          },
          'classId:5': {
              'auc': True,
          },
      }
      if (
          model_type
          not in (constants.TF_LITE, constants.TF_JS, constants.TF_KERAS)
          and _TF_MAJOR_VERSION >= 2
      ):
        expected_metrics[''] = {'loss': True}
        if add_custom_metrics:
          expected_metrics['']['custom'] = True
      for class_id in expected_metrics:
        self.assertIn(class_id, got_metrics)
        for k in expected_metrics[class_id]:
          self.assertIn(k, got_metrics[class_id])

    # TODO(b/173657964): assert exception for the missing baseline but non
    # rubber stamping test.
    if rubber_stamp or remove_baseline:
      self.assertIsInstance(eval_results, view_types.EvalResult)
      check_eval_result(eval_results, model.model_path)
    else:
      self.assertLen(eval_results._results, 2)
      eval_result_0, eval_result_1 = eval_results._results
      check_eval_result(eval_result_0, model.model_path)
      check_eval_result(eval_result_1, baseline.model_path)

  def testRunModelAnalysisWithKerasMultiOutputModel(self):

    def _build_keras_model(eval_config, export_name='export_dir'):
      layers_per_output = {}
      for output_name in ('output_1', 'output_2'):
        layers_per_output[output_name] = tf_keras.layers.Input(
            shape=(1,), name=output_name
        )
      model = tf_keras.models.Model(layers_per_output, layers_per_output)
      model.compile(loss=tf_keras.losses.categorical_crossentropy)
      model_location = os.path.join(self._getTempDir(), export_name)
      model.export(model_location)
      return model_eval_lib.default_eval_shared_model(
          eval_saved_model_path=model_location,
          eval_config=eval_config,
          rubber_stamp=False,
      )

    examples = [
        self._makeExample(output_1=1.0, output_2=0.0, label_1=0.0, label_2=0.0),
        self._makeExample(output_1=0.7, output_2=0.3, label_1=1.0, label_2=1.0),
        self._makeExample(output_1=0.5, output_2=0.8, label_1=0.0, label_2=1.0),
    ]
    data_location = self._writeTFExamplesToTFRecords(examples)

    metrics_spec = config_pb2.MetricsSpec(
        output_names=['output_1', 'output_2'],
        output_weights={'output_1': 1.0, 'output_2': 1.0},
    )
    for metric in (confusion_matrix_metrics.AUC(name='auc'),):
      cfg = metric_util.serialize_keras_object(metric)
      metrics_spec.metrics.append(
          config_pb2.MetricConfig(
              class_name=cfg['class_name'], config=json.dumps(cfg['config'])
          )
      )
    slicing_specs = [
        config_pb2.SlicingSpec(),
        config_pb2.SlicingSpec(feature_keys=['non_existent_slice']),
    ]
    metrics_spec.metrics.append(
        config_pb2.MetricConfig(
            class_name='ExampleCount',
            per_slice_thresholds=[
                config_pb2.PerSliceMetricThreshold(
                    slicing_specs=slicing_specs,
                    threshold=config_pb2.MetricThreshold(
                        value_threshold=config_pb2.GenericValueThreshold(
                            lower_bound={'value': 1}
                        )
                    ),
                ),
                # Change thresholds would be ignored when rubber stamp is true.
                config_pb2.PerSliceMetricThreshold(
                    slicing_specs=slicing_specs,
                    threshold=config_pb2.MetricThreshold(
                        change_threshold=config_pb2.GenericChangeThreshold(
                            direction=config_pb2.MetricDirection.HIGHER_IS_BETTER,
                            absolute={'value': 1},
                        )
                    ),
                ),
            ],
        )
    )
    eval_config = config_pb2.EvalConfig(
        model_specs=[
            config_pb2.ModelSpec(
                label_keys={'output_1': 'label_1', 'output_2': 'label_2'}
            )
        ],
        metrics_specs=[metrics_spec],
    )

    model = _build_keras_model(eval_config)
    baseline = _build_keras_model(eval_config, 'baseline_export')
    eval_shared_model = {'candidate': model, 'baseline': baseline}
    output_path = self._getTempDir()
    eval_results = model_eval_lib.run_model_analysis(
        eval_config=eval_config,
        eval_shared_model=eval_shared_model,
        data_location=data_location,
        output_path=output_path,
    )

    # Directly check validation file since it is not in EvalResult.
    validations_file = os.path.join(
        output_path, f'{constants.VALIDATIONS_KEY}.tfrecord'
    )
    self.assertTrue(os.path.exists(validations_file))
    validation_records = []
    for record in tf.compat.v1.python_io.tf_record_iterator(validations_file):
      validation_records.append(
          validation_result_pb2.ValidationResult.FromString(record)
      )
    self.assertLen(validation_records, 1)
    expected_result = text_format.Parse(
        """
          metric_validations_per_slice {
            slice_key {}
            failures {
              metric_key {
                name: "example_count"
                model_name: "candidate"
                output_name: "output_1"
                is_diff: true
                example_weighted { }
              }
              metric_threshold {
                change_threshold {
                  absolute { value: 1 }
                  direction: HIGHER_IS_BETTER
                }
              }
              metric_value { double_value {} }
            }
            failures {
              metric_key {
                name: "example_count"
                model_name: "candidate"
                output_name: "output_2"
                is_diff: true
                example_weighted { }
              }
              metric_threshold {
                change_threshold {
                  absolute { value: 1}
                  direction: HIGHER_IS_BETTER
                }
              }
              metric_value { double_value {} }
            }
          }
          missing_slices {
            feature_keys: "non_existent_slice"
          }
          validation_details {
            slicing_details {
              slicing_spec {}
              num_matching_slices: 1
            }
          }""",
        validation_result_pb2.ValidationResult(),
    )
    self.assertProtoEquals(expected_result, validation_records[0])

    def check_eval_result(eval_result, model_location):
      self.assertEqual(eval_result.model_location, model_location)
      self.assertEqual(eval_result.data_location, data_location)
      self.assertLen(eval_result.slicing_metrics, 1)
      got_slice_key, got_metrics = eval_result.slicing_metrics[0]
      self.assertEqual(got_slice_key, ())
      self.assertIn('output_1', got_metrics)
      self.assertIn('auc', got_metrics['output_1'][''])
      self.assertIn('output_2', got_metrics)
      self.assertIn('auc', got_metrics['output_2'][''])
      # Aggregate metrics
      self.assertIn('', got_metrics)
      self.assertIn('auc', got_metrics[''][''])

    # TODO(b/173657964): assert exception for the missing baseline but non
    # rubber stamping test.
    self.assertLen(eval_results._results, 2)
    eval_result_0, eval_result_1 = eval_results._results
    check_eval_result(eval_result_0, model.model_path)
    check_eval_result(eval_result_1, baseline.model_path)

  def testRunModelAnalysisWithQueryBasedMetrics(self):
    input_layer = tf_keras.layers.Input(shape=(1,), name='age')
    output_layer = tf_keras.layers.Dense(1, activation=tf.nn.sigmoid)(
        input_layer
    )
    model = tf_keras.models.Model(input_layer, output_layer)
    model.compile(
        optimizer=tf_keras.optimizers.Adam(lr=0.001),
        loss=tf_keras.losses.binary_crossentropy,
    )

    features = {'age': [[20.0]]}
    labels = [[1]]
    example_weights = [1.0]
    dataset = tf.data.Dataset.from_tensor_slices(
        (features, labels, example_weights)
    )
    dataset = dataset.shuffle(buffer_size=1).repeat().batch(1)
    model.fit(dataset, steps_per_epoch=1)

    model_location = os.path.join(self._getTempDir(), 'export_dir')
    model.export(model_location)

    schema = text_format.Parse(
        """
        tensor_representation_group {
          key: ""
          value {
            tensor_representation {
              key: "age"
              value {
                dense_tensor {
                  column_name: "age"
                  shape { dim { size: 1 } }
                }
              }
            }
            tensor_representation {
              key: "language"
              value {
                dense_tensor {
                  column_name: "language"
                  shape { dim { size: 1 } }
                }
              }
            }
          }
        }
        feature {
          name: "age"
          type: FLOAT
        }
        feature {
          name: "language"
          type: BYTES
        }
        feature {
          name: "label"
          type: FLOAT
        }
        feature {
          name: "varlen"
          type: INT
        }
        """,
        schema_pb2.Schema(),
    )
    examples = [
        self._makeExample(age=3.0, language='english', label=1.0, varlen=[0]),
        self._makeExample(age=5.0, language='chinese', label=0.0, varlen=[1]),
        self._makeExample(age=3.0, language='english', label=0.0, varlen=[2]),
        self._makeExample(
            age=5.0, language='chinese', label=1.0, varlen=[3, 4]
        ),
    ]
    data_location = self._writeTFExamplesToTFRecords(examples)
    slicing_specs = [config_pb2.SlicingSpec()]
    # Test with both a TFMA metric (NDCG), a keras metric (Recall).
    metrics = [
        ndcg.NDCG(gain_key='age', name='ndcg', top_k_list=[1, 2]),
        tf_keras.metrics.Recall(top_k=1),
    ]
    # If tensorflow-ranking imported add MRRMetric.
    if _TFR_IMPORTED:
      metrics.append(tfr.keras.metrics.MRRMetric())
    metrics_specs = metric_specs.specs_from_metrics(
        metrics, query_key='language', include_weighted_example_count=True
    )
    metrics_specs.append(
        config_pb2.MetricsSpec(
            metrics=[
                config_pb2.MetricConfig(
                    class_name='ExampleCount',
                    threshold=config_pb2.MetricThreshold(
                        value_threshold=config_pb2.GenericValueThreshold(
                            lower_bound={'value': 0}
                        )
                    ),
                )
            ]
        )
    )
    eval_config = config_pb2.EvalConfig(
        model_specs=[config_pb2.ModelSpec(label_key='label')],
        slicing_specs=slicing_specs,
        metrics_specs=metrics_specs,
    )
    eval_shared_model = model_eval_lib.default_eval_shared_model(
        eval_saved_model_path=model_location, eval_config=eval_config
    )
    output_path = self._getTempDir()
    eval_result = model_eval_lib.run_model_analysis(
        eval_config=eval_config,
        eval_shared_model=eval_shared_model,
        data_location=data_location,
        output_path=output_path,
        evaluators=[
            metrics_plots_and_validations_evaluator.MetricsPlotsAndValidationsEvaluator(
                eval_config=eval_config, eval_shared_model=eval_shared_model
            )
        ],
        schema=schema,
    )

    # Directly check validation file since it is not in EvalResult.
    validations_file = os.path.join(
        output_path, f'{constants.VALIDATIONS_KEY}.tfrecord'
    )
    self.assertTrue(os.path.exists(validations_file))
    validation_records = []
    for record in tf.compat.v1.python_io.tf_record_iterator(validations_file):
      validation_records.append(
          validation_result_pb2.ValidationResult.FromString(record)
      )
    self.assertLen(validation_records, 1)
    self.assertTrue(validation_records[0].validation_ok)

    self.assertEqual(eval_result.model_location, model_location)
    self.assertEqual(eval_result.data_location, data_location)
    self.assertLen(eval_result.slicing_metrics, 1)
    got_slice_key, got_metrics = eval_result.slicing_metrics[0]
    self.assertEqual(got_slice_key, ())
    self.assertIn('', got_metrics)  # output_name
    got_metrics = got_metrics['']
    expected_metrics = {
        '': {
            'example_count': True,
            'weighted_example_count': True,
        },
        'topK:1': {
            'ndcg': True,
            'recall': True,
        },
        'topK:2': {
            'ndcg': True,
        },
    }
    if _TFR_IMPORTED:
      expected_metrics['']['mrr_metric'] = True
    for group in expected_metrics:
      self.assertIn(group, got_metrics)
      for k in expected_metrics[group]:
        self.assertIn(k, got_metrics[group])

  def testRunModelAnalysisWithUncertainty(self):
    examples = [
        self._makeExample(age=3.0, language='english', label=1.0),
        self._makeExample(age=3.0, language='chinese', label=0.0),
        self._makeExample(age=4.0, language='english', label=1.0),
        self._makeExample(age=5.0, language='chinese', label=1.0),
        self._makeExample(age=5.0, language='hindi', label=1.0),
    ]
    classifier = example_keras_model.get_example_classifier_model(
        example_keras_model.LANGUAGE
    )
    classifier.compile(optimizer=keras.optimizers.Adam(), loss='mse')
    classifier.fit(
        tf.constant([e.SerializeToString() for e in examples]),
        np.array([
            e.features.feature[example_keras_model.LABEL].float_list.value[:][0]
            for e in examples
        ]),
        batch_size=1,
    )
    eval_config = config_pb2.EvalConfig(
        model_specs=[
            config_pb2.ModelSpec(
                name='model1', example_weight_key='age', label_key='label'
            )
        ],
        slicing_specs=[config_pb2.SlicingSpec(feature_keys=['language'])],
        metrics_specs=[
            config_pb2.MetricsSpec(
                metrics=[
                    config_pb2.MetricConfig(
                        class_name='ExampleCount',
                    ),
                    config_pb2.MetricConfig(
                        class_name='Accuracy',
                    ),
                ]
            )
        ],
        options=config_pb2.Options(
            compute_confidence_intervals=wrappers_pb2.BoolValue(value=True),
            min_slice_size=wrappers_pb2.Int32Value(value=2),
        ),
    )
    model_location = self._exportKerasModel(classifier)
    data_location = self._writeTFExamplesToTFRecords(examples)
    eval_result = model_eval_lib.run_model_analysis(
        eval_shared_model=model_eval_lib.default_eval_shared_model(
            eval_saved_model_path=model_location, eval_config=eval_config
        ),
        data_location=data_location,
        eval_config=eval_config,
        output_path=self._getTempDir(),
    )
    # We only check some of the metrics to ensure that the end-to-end
    # pipeline works.
    expected = {
        (('language', 'hindi'),): {
            '__ERROR__': {
                'debugMessage': (
                    'Example count for this slice key is lower than the '
                    'minimum required value: 2. No data is aggregated for '
                    'this slice.'
                )
            },
        },
        (('language', 'english'),): {
            'accuracy': {
                'boundedValue': {
                    'lowerBound': 0.0,
                    'upperBound': 0.0,
                    'value': 0.0,
                }
            },
            'example_count': {'doubleValue': 7.0},
        },
        (('language', 'chinese'),): {
            'accuracy': {
                'boundedValue': {
                    'lowerBound': 0.0,
                    'upperBound': 0.0,
                    'value': 0.0,
                }
            },
            'example_count': {'doubleValue': 8.0},
        },
    }
    self.assertEqual(eval_result.model_location, model_location)
    self.assertEqual(eval_result.data_location, data_location)
    self.assertEqual(
        eval_result.config.slicing_specs[0],
        config_pb2.SlicingSpec(feature_keys=['language']),
    )
    self.assertMetricsAlmostEqual(eval_result.slicing_metrics, expected)
    for _, plot in eval_result.plots:
      self.assertFalse(plot)

  def testRunModelAnalysisWithDeterministicConfidenceIntervals(self):
    examples = [
        self._makeExample(age=3.0, language='english', label=1.0),
        self._makeExample(age=3.0, language='chinese', label=0.0),
        self._makeExample(age=4.0, language='english', label=1.0),
        self._makeExample(age=5.0, language='chinese', label=1.0),
        self._makeExample(age=5.0, language='hindi', label=1.0),
    ]
    classifier = example_keras_model.get_example_classifier_model(
        example_keras_model.LANGUAGE
    )
    classifier.compile(optimizer=keras.optimizers.Adam(), loss='mse')
    classifier.fit(
        tf.constant([e.SerializeToString() for e in examples]),
        np.array([
            e.features.feature[example_keras_model.LABEL].float_list.value[:][0]
            for e in examples
        ]),
        batch_size=1,
    )
    model_location = self._exportKerasModel(classifier)
    data_location = self._writeTFExamplesToTFRecords(examples)
    eval_config = config_pb2.EvalConfig(
        model_specs=[
            config_pb2.ModelSpec(
                name='model1', example_weight_key='age', label_key='label'
            )
        ],
        slicing_specs=[config_pb2.SlicingSpec(feature_keys=['language'])],
        metrics_specs=[
            config_pb2.MetricsSpec(
                metrics=[
                    config_pb2.MetricConfig(
                        class_name='ExampleCount',
                    ),
                    config_pb2.MetricConfig(
                        class_name='Accuracy',
                    ),
                ]
            )
        ],
        options=config_pb2.Options(
            compute_confidence_intervals=wrappers_pb2.BoolValue(value=True),
            min_slice_size=wrappers_pb2.Int32Value(value=2),
        ),
    )
    eval_result = model_eval_lib.run_model_analysis(
        eval_shared_model=model_eval_lib.default_eval_shared_model(
            eval_saved_model_path=model_location, eval_config=eval_config
        ),
        data_location=data_location,
        output_path=self._getTempDir(),
        eval_config=eval_config,
        random_seed_for_testing=_TEST_SEED,
    )
    # We only check some of the metrics to ensure that the end-to-end
    # pipeline works.
    expected = {
        (('language', 'hindi'),): {
            '__ERROR__': {
                'debugMessage': (
                    'Example count for this slice key is lower than the '
                    'minimum required value: 2. No data is aggregated for '
                    'this slice.'
                )
            },
        },
        (('language', 'english'),): {
            'accuracy': {
                'boundedValue': {
                    'lowerBound': 0.0,
                    'upperBound': 0.0,
                    'value': 0.0,
                }
            },
            'example_count': {'doubleValue': 7.0},
        },
        (('language', 'chinese'),): {
            'accuracy': {
                'boundedValue': {
                    'lowerBound': 0.0,
                    'upperBound': 0.0,
                    'value': 0.0,
                }
            },
            'example_count': {'doubleValue': 8.0},
        },
    }
    self.assertEqual(eval_result.model_location, model_location)
    self.assertEqual(eval_result.data_location, data_location)
    self.assertEqual(
        eval_result.config.slicing_specs[0],
        config_pb2.SlicingSpec(feature_keys=['language']),
    )
    self.assertMetricsAlmostEqual(eval_result.slicing_metrics, expected)

    for key, value in eval_result.slicing_metrics:
      if (('language', 'english'),) == key:
        metric = value['']['']['accuracy']
        self.assertAlmostEqual(0.0, metric['boundedValue']['value'], delta=0.1)

    for _, plot in eval_result.plots:
      self.assertFalse(plot)

  # TODO(b/350996394): Add test for plots and CSVtext with Keras model.

  def testRunModelAnalysisWithSchema(self):
    examples = [
        self._makeExample(age=3.0, language='english', label=2.0),
        self._makeExample(age=3.0, language='chinese', label=1.0),
        self._makeExample(age=4.0, language='english', label=2.0),
        self._makeExample(age=5.0, language='chinese', label=2.0),
        self._makeExample(age=5.0, language='hindi', label=2.0),
    ]
    data_location = self._writeTFExamplesToTFRecords(examples)
    classifier = example_keras_model.get_example_classifier_model(
        example_keras_model.LANGUAGE
    )
    classifier.compile(optimizer=keras.optimizers.Adam(), loss='mse')
    classifier.fit(
        tf.constant([e.SerializeToString() for e in examples]),
        np.array([
            e.features.feature[example_keras_model.LABEL].float_list.value[:][0]
            for e in examples
        ]),
        batch_size=1,
    )
    model_location = self._exportKerasModel(classifier)
    eval_config = config_pb2.EvalConfig(
        model_specs=[config_pb2.ModelSpec(label_key='label')],
        metrics_specs=metric_specs.specs_from_metrics(
            [calibration_plot.CalibrationPlot(num_buckets=4)]
        ),
    )
    schema = text_format.Parse(
        """
        feature {
          name: "label"
          type: FLOAT
          float_domain {
            min: 1
            max: 2
          }
        }
        """,
        schema_pb2.Schema(),
    )
    eval_result = model_eval_lib.run_model_analysis(
        eval_config=eval_config,
        schema=schema,
        eval_shared_model=model_eval_lib.default_eval_shared_model(
            eval_saved_model_path=model_location, eval_config=eval_config
        ),
        data_location=data_location,
        output_path=self._getTempDir(),
    )

    expected_metrics = {
        (): {
            'example_count': {'doubleValue': 5.0},
        }
    }
    self.assertMetricsAlmostEqual(eval_result.slicing_metrics, expected_metrics)
    self.assertLen(eval_result.plots, 1)
    slice_key, plots = eval_result.plots[0]
    self.assertEqual((), slice_key)
    got_buckets = plots['']['']['calibrationHistogramBuckets']['buckets']
    # buckets include (-inf, left) and (right, inf) by default, but we are
    # interested in the values of left and right
    self.assertEqual(1.0, got_buckets[1]['lowerThresholdInclusive'])
    self.assertEqual(2.0, got_buckets[-2]['upperThresholdExclusive'])

  def testLoadValidationResult(self):
    result = validation_result_pb2.ValidationResult(validation_ok=True)
    path = os.path.join(absltest.get_default_test_tmpdir(), 'results.tfrecord')
    with tf.io.TFRecordWriter(path) as writer:
      writer.write(result.SerializeToString())
    loaded_result = model_eval_lib.load_validation_result(path)
    self.assertTrue(loaded_result.validation_ok)

  def testLoadValidationResultDir(self):
    result = validation_result_pb2.ValidationResult(validation_ok=True)
    path = os.path.join(
        absltest.get_default_test_tmpdir(), constants.VALIDATIONS_KEY
    )
    with tf.io.TFRecordWriter(path) as writer:
      writer.write(result.SerializeToString())
    loaded_result = model_eval_lib.load_validation_result(os.path.dirname(path))
    self.assertTrue(loaded_result.validation_ok)

  def testLoadValidationResultEmptyFile(self):
    path = os.path.join(
        absltest.get_default_test_tmpdir(), constants.VALIDATIONS_KEY
    )
    with tf.io.TFRecordWriter(path):
      pass
    with self.assertRaises(AssertionError):
      model_eval_lib.load_validation_result(path)

  def testAnalyzeRawData(self):

    # Data
    # age language  label  prediction
    # 17  english   0      0
    # 30  spanish   1      1
    dict_data = [
        {'age': 17, 'language': 'english', 'prediction': 0, 'label': 0},
        {'age': 30, 'language': 'spanish', 'prediction': 1, 'label': 1},
    ]
    df_data = pd.DataFrame(dict_data)

    # Expected Output
    expected_slicing_metrics = {
        (('language', 'english'),): {
            '': {
                '': {
                    'binary_accuracy': {'doubleValue': 1.0},
                    'example_count': {'doubleValue': 1.0},
                }
            }
        },
        (('language', 'spanish'),): {
            '': {
                '': {
                    'binary_accuracy': {'doubleValue': 1.0},
                    'example_count': {'doubleValue': 1.0},
                }
            }
        },
        (): {
            '': {
                '': {
                    'binary_accuracy': {'doubleValue': 1.0},
                    'example_count': {'doubleValue': 2.0},
                }
            }
        },
    }

    # Actual Output
    eval_config = text_format.Parse(
        """
      model_specs {
        label_key: 'label'
        prediction_key: 'prediction'
      }
      metrics_specs {
        metrics { class_name: "BinaryAccuracy" }
        metrics { class_name: "ExampleCount" }
      }
      slicing_specs {}
      slicing_specs {
        feature_keys: 'language'
      }
    """,
        config_pb2.EvalConfig(),
    )
    eval_result = model_eval_lib.analyze_raw_data(df_data, eval_config)

    # Compare Actual and Expected
    self.assertEqual(
        len(eval_result.slicing_metrics), len(expected_slicing_metrics)
    )
    for slicing_metric in eval_result.slicing_metrics:
      slice_key, slice_val = slicing_metric
      self.assertIn(slice_key, expected_slicing_metrics)
      self.assertDictEqual(slice_val, expected_slicing_metrics[slice_key])

  def testAnalyzeRawDataWithoutPrediction(self):
    model_specs = [
        config_pb2.ModelSpec(prediction_key='nonexistent_prediction_key')
    ]
    metrics_specs = [
        config_pb2.MetricsSpec(
            metrics=[config_pb2.MetricConfig(class_name='Accuracy')]
        )
    ]
    eval_config = config_pb2.EvalConfig(
        model_specs=model_specs, metrics_specs=metrics_specs
    )
    df_data = pd.DataFrame([{
        'prediction': 0,
        'label': 0,
    }])
    with self.assertRaises(KeyError):
      model_eval_lib.analyze_raw_data(df_data, eval_config)

  def testAnalyzeRawDataWithoutLabel(self):
    model_specs = [config_pb2.ModelSpec(prediction_key='nonexistent_label_key')]
    metrics_specs = [
        config_pb2.MetricsSpec(
            metrics=[config_pb2.MetricConfig(class_name='Accuracy')]
        )
    ]
    eval_config = config_pb2.EvalConfig(
        model_specs=model_specs, metrics_specs=metrics_specs
    )
    df_data = pd.DataFrame([{
        'prediction': 0,
        'label': 0,
    }])
    with self.assertRaises(KeyError):
      model_eval_lib.analyze_raw_data(df_data, eval_config)

  def testBytesProcessedCountForSerializedExamples(self):
    examples = [
        self._makeExample(age=3.0, language='english', label=1.0),
        self._makeExample(age=3.0, language='chinese', label=0.0),
        self._makeExample(age=4.0, language='english', label=1.0),
        self._makeExample(age=5.0, language='chinese', label=1.0),
        self._makeExample(age=5.0, language='hindi', label=1.0),
    ]
    serialized_examples = [example.SerializeToString() for example in examples]
    expected_num_bytes = sum([len(se) for se in serialized_examples])
    with beam.Pipeline() as p:
      _ = (
          p
          | beam.Create(serialized_examples)
          | 'InputsToExtracts' >> model_eval_lib.InputsToExtracts()
          | 'ExtractAndEvaluate'
          >> model_eval_lib.ExtractAndEvaluate(extractors=[], evaluators=[])
      )
    pipeline_result = p.run()
    metrics = pipeline_result.metrics()
    actual_counter = metrics.query(
        beam.metrics.metric.MetricsFilter().with_name('extract_input_bytes')
    )['counters']
    self.assertLen(actual_counter, 1)
    self.assertEqual(actual_counter[0].committed, expected_num_bytes)

  def testBytesProcessedCountForRecordBatches(self):
    examples = [
        self._makeExample(age=3.0, language='english', label=1.0),
        self._makeExample(age=3.0, language='chinese', label=0.0),
        self._makeExample(age=4.0, language='english', label=1.0),
        self._makeExample(age=5.0, language='chinese', label=1.0),
        self._makeExample(age=5.0, language='hindi', label=1.0),
    ]
    examples = [example.SerializeToString() for example in examples]
    decoder = example_coder.ExamplesToRecordBatchDecoder()
    record_batch = decoder.DecodeBatch(examples)
    expected_num_bytes = record_batch.nbytes
    with beam.Pipeline() as p:
      _ = (
          p
          | beam.Create(record_batch)
          | 'BatchedInputsToExtracts'
          >> model_eval_lib.BatchedInputsToExtracts()
          | 'ExtractAndEvaluate'
          >> model_eval_lib.ExtractAndEvaluate(extractors=[], evaluators=[])
      )
    pipeline_result = p.run()
    metrics = pipeline_result.metrics()
    actual_counter = metrics.query(
        beam.metrics.metric.MetricsFilter().with_name('extract_input_bytes')
    )[metrics.COUNTERS]
    self.assertLen(actual_counter, 1)
    self.assertEqual(actual_counter[0].committed, expected_num_bytes)


