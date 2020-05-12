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
"""Test for using the model_eval_lib API."""

from __future__ import division
from __future__ import print_function

import json
import os
import tempfile

from absl.testing import parameterized

import tensorflow as tf
from tensorflow_model_analysis import config
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.eval_saved_model.example_trainers import csv_linear_classifier
from tensorflow_model_analysis.eval_saved_model.example_trainers import fixed_prediction_estimator
from tensorflow_model_analysis.eval_saved_model.example_trainers import linear_classifier
from tensorflow_model_analysis.eval_saved_model.example_trainers import linear_regressor
from tensorflow_model_analysis.evaluators import metrics_and_plots_evaluator
from tensorflow_model_analysis.evaluators import metrics_and_plots_evaluator_v2
from tensorflow_model_analysis.evaluators import query_based_metrics_evaluator
from tensorflow_model_analysis.evaluators.query_metrics import ndcg as legacy_ndcg
from tensorflow_model_analysis.evaluators.query_metrics import query_statistics
from tensorflow_model_analysis.extractors import feature_extractor
from tensorflow_model_analysis.extractors import predict_extractor
from tensorflow_model_analysis.extractors import slice_key_extractor
from tensorflow_model_analysis.metrics import calibration_plot
from tensorflow_model_analysis.metrics import metric_specs
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import ndcg
from tensorflow_model_analysis.post_export_metrics import metric_keys
from tensorflow_model_analysis.post_export_metrics import post_export_metrics
from tensorflow_model_analysis.proto import validation_result_pb2
from tensorflow_model_analysis.slicer import slicer_lib as slicer

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2

_TEST_SEED = 982735


class EvaluateTest(testutil.TensorflowModelAnalysisTest,
                   parameterized.TestCase):

  def setUp(self):
    super(EvaluateTest, self).setUp()
    self.longMessage = True  # pylint: disable=invalid-name

  def _getTempDir(self):
    return tempfile.mkdtemp()

  def _exportEvalSavedModel(self, classifier):
    temp_eval_export_dir = os.path.join(self._getTempDir(), 'eval_export_dir')
    _, eval_export_dir = classifier(None, temp_eval_export_dir)
    return eval_export_dir

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

  def assertMetricsAlmostEqual(self, got_value, expected_value):
    if got_value:
      for (s, m) in got_value:
        self.assertIn(s, expected_value)
        for k in expected_value[s]:
          metrics = m['']['']
          self.assertIn(k, metrics)
          self.assertDictElementsAlmostEqual(metrics[k], expected_value[s][k])
    else:
      # Only pass if expected_value also evaluates to False.
      self.assertFalse(expected_value, msg='Actual value was empty.')

  def assertSliceMetricsEqual(self, expected_metrics, got_metrics):
    self.assertCountEqual(
        list(expected_metrics.keys()),
        list(got_metrics.keys()),
        msg='keys do not match. expected_metrics: %s, got_metrics: %s' %
        (expected_metrics, got_metrics))
    for key in expected_metrics.keys():
      self.assertProtoEquals(
          expected_metrics[key],
          got_metrics[key],
          msg='value for key %s does not match' % key)

  def assertSliceListEqual(self, expected_list, got_list, value_assert_fn):
    self.assertEqual(
        len(expected_list),
        len(got_list),
        msg='expected_list: %s, got_list: %s' % (expected_list, got_list))
    for index, (expected, got) in enumerate(zip(expected_list, got_list)):
      (expected_key, expected_value) = expected
      (got_key, got_value) = got
      self.assertEqual(
          expected_key, got_key, msg='key mismatch at index %d' % index)
      value_assert_fn(expected_value, got_value)

  def assertSlicePlotsListEqual(self, expected_list, got_list):
    self.assertSliceListEqual(expected_list, got_list, self.assertProtoEquals)

  def assertSliceMetricsListEqual(self, expected_list, got_list):
    self.assertSliceListEqual(expected_list, got_list,
                              self.assertSliceMetricsEqual)

  def testNoConstructFn(self):
    model_location = self._exportEvalSavedModel(
        linear_classifier.simple_linear_classifier)
    examples = [self._makeExample(age=3.0, language='english', label=1.0)]
    data_location = self._writeTFExamplesToTFRecords(examples)
    eval_config = config.EvalConfig()
    # No construct_fn should fail when Beam attempts to call the construct_fn.
    eval_shared_model = types.EvalSharedModel(model_path=model_location)
    with self.assertRaisesRegex(AttributeError,
                                '\'NoneType\' object has no attribute'):
      model_eval_lib.run_model_analysis(
          eval_config=eval_config,
          eval_shared_model=eval_shared_model,
          data_location=data_location,
          output_path=self._getTempDir())

    # Using the default_eval_shared_model should pass as it has a construct_fn.
    eval_shared_model = model_eval_lib.default_eval_shared_model(
        eval_saved_model_path=model_location)
    model_eval_lib.run_model_analysis(
        eval_config=eval_config,
        eval_shared_model=eval_shared_model,
        data_location=data_location,
        output_path=self._getTempDir())

  def testMixedEvalAndNonEvalSignatures(self):
    examples = [self._makeExample(age=3.0, language='english', label=1.0)]
    data_location = self._writeTFExamplesToTFRecords(examples)
    eval_config = config.EvalConfig(model_specs=[
        config.ModelSpec(name='model1'),
        config.ModelSpec(name='model2', signature_name='eval')
    ])
    eval_shared_models = [
        model_eval_lib.default_eval_shared_model(
            model_name='model1',
            eval_saved_model_path='/model1/path',
            eval_config=eval_config),
        model_eval_lib.default_eval_shared_model(
            model_name='model2',
            eval_saved_model_path='/model2/path',
            eval_config=eval_config),
    ]
    with self.assertRaisesRegex(
        NotImplementedError,
        'support for mixing eval and non-eval estimator models is not '
        'implemented'):
      model_eval_lib.run_model_analysis(
          eval_config=eval_config,
          eval_shared_model=eval_shared_models,
          data_location=data_location,
          output_path=self._getTempDir())

  def testMixedTFLiteAndNotTFLiteFormats(self):
    examples = [self._makeExample(age=3.0, language='english', label=1.0)]
    data_location = self._writeTFExamplesToTFRecords(examples)
    eval_config = config.EvalConfig(model_specs=[
        config.ModelSpec(name='model1'),
        config.ModelSpec(name='model2', model_type=constants.TF_LITE)
    ])
    eval_shared_models = [
        model_eval_lib.default_eval_shared_model(
            model_name='model1',
            eval_saved_model_path='/model1/path',
            eval_config=eval_config),
        model_eval_lib.default_eval_shared_model(
            model_name='model2',
            eval_saved_model_path='/model2/path',
            eval_config=eval_config)
    ]
    with self.assertRaisesRegex(
        NotImplementedError,
        'support for mixing tf_lite and non-tf_lite models is not implemented'):
      model_eval_lib.run_model_analysis(
          eval_config=eval_config,
          eval_shared_model=eval_shared_models,
          data_location=data_location,
          output_path=self._getTempDir())

  def testRunModelAnalysisExtraFieldsPlusFeatureExtraction(self):
    model_location = self._exportEvalSavedModel(
        linear_classifier.simple_linear_classifier)
    examples = [
        self._makeExample(age=3.0, language='english', label=1.0, my_slice='a'),
        self._makeExample(age=3.0, language='chinese', label=0.0, my_slice='a'),
        self._makeExample(age=4.0, language='english', label=1.0, my_slice='b'),
        self._makeExample(age=5.0, language='chinese', label=1.0, my_slice='c'),
        self._makeExample(age=5.0, language='hindi', label=1.0)
    ]
    data_location = self._writeTFExamplesToTFRecords(examples)
    slicing_specs = [config.SlicingSpec(feature_keys=['my_slice'])]
    eval_config = config.EvalConfig(slicing_specs=slicing_specs)
    eval_shared_model = model_eval_lib.default_eval_shared_model(
        eval_saved_model_path=model_location, example_weight_key='age')
    slice_spec = [slicer.SingleSliceSpec(spec=slicing_specs[0])]
    extractors_with_feature_extraction = [
        predict_extractor.PredictExtractor(
            eval_shared_model, desired_batch_size=3, materialize=False),
        feature_extractor.FeatureExtractor(
            extract_source=constants.INPUT_KEY,
            extract_dest=constants.FEATURES_PREDICTIONS_LABELS_KEY),
        slice_key_extractor.SliceKeyExtractor(slice_spec, materialize=False)
    ]
    eval_result = model_eval_lib.run_model_analysis(
        eval_config=eval_config,
        eval_shared_model=model_eval_lib.default_eval_shared_model(
            eval_saved_model_path=model_location, example_weight_key='age'),
        data_location=data_location,
        output_path=self._getTempDir(),
        extractors=extractors_with_feature_extraction)
    # We only check some of the metrics to ensure that the end-to-end
    # pipeline works.
    expected = {
        (('my_slice', 'a'),): {
            'accuracy': {
                'doubleValue': 1.0
            },
            'my_mean_label': {
                'doubleValue': 0.5
            },
            metric_keys.EXAMPLE_WEIGHT: {
                'doubleValue': 6.0
            },
            metric_keys.EXAMPLE_COUNT: {
                'doubleValue': 2.0
            },
        },
        (('my_slice', 'b'),): {
            'accuracy': {
                'doubleValue': 1.0
            },
            'my_mean_label': {
                'doubleValue': 1.0
            },
            metric_keys.EXAMPLE_WEIGHT: {
                'doubleValue': 4.0
            },
            metric_keys.EXAMPLE_COUNT: {
                'doubleValue': 1.0
            },
        },
        (('my_slice', 'c'),): {
            'accuracy': {
                'doubleValue': 0.0
            },
            'my_mean_label': {
                'doubleValue': 1.0
            },
            metric_keys.EXAMPLE_WEIGHT: {
                'doubleValue': 5.0
            },
            metric_keys.EXAMPLE_COUNT: {
                'doubleValue': 1.0
            },
        },
    }
    self.assertEqual(eval_result.model_location, model_location.decode())
    self.assertEqual(eval_result.data_location, data_location)
    self.assertEqual(eval_result.config.slicing_specs[0],
                     config.SlicingSpec(feature_keys=['my_slice']))
    self.assertMetricsAlmostEqual(eval_result.slicing_metrics, expected)
    self.assertFalse(eval_result.plots)

  def testRunModelAnalysis(self):
    model_location = self._exportEvalSavedModel(
        linear_classifier.simple_linear_classifier)
    examples = [
        self._makeExample(age=3.0, language='english', label=1.0),
        self._makeExample(age=3.0, language='chinese', label=0.0),
        self._makeExample(age=4.0, language='english', label=1.0),
        self._makeExample(age=5.0, language='chinese', label=1.0),
        self._makeExample(age=5.0, language='hindi', label=1.0)
    ]
    data_location = self._writeTFExamplesToTFRecords(examples)
    slicing_specs = [config.SlicingSpec(feature_keys=['language'])]
    options = config.Options()
    options.min_slice_size.value = 2
    eval_config = config.EvalConfig(
        slicing_specs=slicing_specs, options=options)
    eval_result = model_eval_lib.run_model_analysis(
        eval_config=eval_config,
        eval_shared_model=model_eval_lib.default_eval_shared_model(
            eval_saved_model_path=model_location, example_weight_key='age'),
        data_location=data_location,
        output_path=self._getTempDir())
    # We only check some of the metrics to ensure that the end-to-end
    # pipeline works.
    expected = {
        (('language', 'hindi'),): {
            u'__ERROR__': {
                'debugMessage':
                    u'Example count for this slice key is lower than the '
                    u'minimum required value: 2. No data is aggregated for '
                    u'this slice.'
            },
        },
        (('language', 'chinese'),): {
            'accuracy': {
                'doubleValue': 0.5
            },
            'my_mean_label': {
                'doubleValue': 0.5
            },
            metric_keys.EXAMPLE_WEIGHT: {
                'doubleValue': 8.0
            },
            metric_keys.EXAMPLE_COUNT: {
                'doubleValue': 2.0
            },
        },
        (('language', 'english'),): {
            'accuracy': {
                'doubleValue': 1.0
            },
            'my_mean_label': {
                'doubleValue': 1.0
            },
            metric_keys.EXAMPLE_WEIGHT: {
                'doubleValue': 7.0
            },
            metric_keys.EXAMPLE_COUNT: {
                'doubleValue': 2.0
            },
        }
    }
    self.assertEqual(eval_result.model_location, model_location.decode())
    self.assertEqual(eval_result.data_location, data_location)
    self.assertEqual(eval_result.config.slicing_specs[0],
                     config.SlicingSpec(feature_keys=['language']))
    self.assertMetricsAlmostEqual(eval_result.slicing_metrics, expected)
    self.assertFalse(eval_result.plots)

  def testRunModelAnalysisMultipleModels(self):
    examples = [
        self._makeExample(age=3.0, language='english', label=1.0),
        self._makeExample(age=3.0, language='chinese', label=0.0),
        self._makeExample(age=4.0, language='english', label=1.0),
        self._makeExample(age=5.0, language='chinese', label=1.0)
    ]
    data_location = self._writeTFExamplesToTFRecords(examples)
    model_specs = [
        config.ModelSpec(
            name='model1', signature_name='eval', example_weight_key='age'),
        config.ModelSpec(
            name='model2', signature_name='eval', example_weight_key='age')
    ]
    metrics_specs = [
        config.MetricsSpec(
            metrics=[
                config.MetricConfig(class_name='ExampleCount'),
                config.MetricConfig(class_name='WeightedExampleCount')
            ],
            model_names=['model1', 'model2'])
    ]
    slicing_specs = [config.SlicingSpec(feature_values={'language': 'english'})]
    options = config.Options()
    eval_config = config.EvalConfig(
        model_specs=model_specs,
        metrics_specs=metrics_specs,
        slicing_specs=slicing_specs,
        options=options)
    model_location1 = self._exportEvalSavedModel(
        linear_classifier.simple_linear_classifier)
    model1 = model_eval_lib.default_eval_shared_model(
        model_name='model1',
        eval_saved_model_path=model_location1,
        eval_config=eval_config)
    model_location2 = self._exportEvalSavedModel(
        linear_classifier.simple_linear_classifier)
    model2 = model_eval_lib.default_eval_shared_model(
        model_name='model2',
        eval_saved_model_path=model_location2,
        eval_config=eval_config)
    eval_shared_models = [model1, model2]
    eval_results = model_eval_lib.run_model_analysis(
        eval_shared_model=eval_shared_models,
        eval_config=eval_config,
        data_location=data_location,
        output_path=self._getTempDir())
    # We only check some of the metrics to ensure that the end-to-end
    # pipeline works.
    expected_result_1 = {
        (('language', 'english'),): {
            'example_count': {
                'doubleValue': 2.0
            },
            'weighted_example_count': {
                'doubleValue': 7.0
            },
            'my_mean_label': {
                'doubleValue': 1.0
            },
            metric_keys.EXAMPLE_COUNT: {
                'doubleValue': 2.0
            },
        }
    }
    expected_result_2 = {
        (('language', 'english'),): {
            'example_count': {
                'doubleValue': 2.0
            },
            'weighted_example_count': {
                'doubleValue': 7.0
            },
            'my_mean_label': {
                'doubleValue': 1.0
            },
            metric_keys.EXAMPLE_COUNT: {
                'doubleValue': 2.0
            },
        }
    }
    self.assertLen(eval_results._results, 2)
    eval_result_1 = eval_results._results[0]
    eval_result_2 = eval_results._results[1]
    self.assertEqual(eval_result_1.model_location, model_location1.decode())
    self.assertEqual(eval_result_2.model_location, model_location2.decode())
    self.assertEqual(eval_result_1.data_location, data_location)
    self.assertEqual(eval_result_2.data_location, data_location)
    self.assertEqual(eval_result_1.config.slicing_specs[0],
                     config.SlicingSpec(feature_values={'language': 'english'}))
    self.assertEqual(eval_result_2.config.slicing_specs[0],
                     config.SlicingSpec(feature_values={'language': 'english'}))
    self.assertMetricsAlmostEqual(eval_result_1.slicing_metrics,
                                  expected_result_1)
    self.assertMetricsAlmostEqual(eval_result_2.slicing_metrics,
                                  expected_result_2)

  def testRunModelAnalysisWithModelAgnosticPredictions(self):
    examples = [
        self._makeExample(
            age=3.0, language='english', label=1.0, prediction=0.9),
        self._makeExample(
            age=3.0, language='chinese', label=0.0, prediction=0.4),
        self._makeExample(
            age=4.0, language='english', label=1.0, prediction=0.7),
        self._makeExample(
            age=5.0, language='chinese', label=1.0, prediction=0.2)
    ]
    data_location = self._writeTFExamplesToTFRecords(examples)
    model_specs = [
        config.ModelSpec(
            prediction_key='prediction',
            label_key='label',
            example_weight_key='age')
    ]
    metrics = [
        config.MetricConfig(class_name='ExampleCount'),
        config.MetricConfig(class_name='WeightedExampleCount'),
        config.MetricConfig(class_name='BinaryAccuracy')
    ]
    slicing_specs = [config.SlicingSpec(feature_keys=['language'])]
    eval_config = config.EvalConfig(
        model_specs=model_specs,
        metrics_specs=[config.MetricsSpec(metrics=metrics)],
        slicing_specs=slicing_specs)
    eval_result = model_eval_lib.run_model_analysis(
        eval_config=eval_config,
        data_location=data_location,
        output_path=self._getTempDir())
    expected = {
        (('language', 'chinese'),): {
            'binary_accuracy': {
                'doubleValue': 0.375
            },
            'weighted_example_count': {
                'doubleValue': 8.0
            },
            'example_count': {
                'doubleValue': 2.0
            },
        },
        (('language', 'english'),): {
            'binary_accuracy': {
                'doubleValue': 1.0
            },
            'weighted_example_count': {
                'doubleValue': 7.0
            },
            'example_count': {
                'doubleValue': 2.0
            },
        }
    }
    self.assertEqual(eval_result.data_location, data_location)
    self.assertEqual(eval_result.config.slicing_specs[0],
                     config.SlicingSpec(feature_keys=['language']))
    self.assertMetricsAlmostEqual(eval_result.slicing_metrics, expected)

  @parameterized.named_parameters(('without_tflite_conversion', False),
                                  ('with_tflite_conversion', True))
  def testRunModelAnalysisWithKerasModel(self, convert_to_tflite):

    def _build_keras_model(eval_config, name='export_dir'):
      input_layer = tf.keras.layers.Input(shape=(28 * 28,), name='data')
      output_layer = tf.keras.layers.Dense(
          10, activation=tf.nn.softmax)(
              input_layer)
      model = tf.keras.models.Model(input_layer, output_layer)
      model.compile(
          optimizer=tf.keras.optimizers.Adam(lr=.001),
          loss=tf.keras.losses.categorical_crossentropy)
      features = {'data': [[0.0] * 28 * 28]}
      labels = [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]
      example_weights = [1.0]
      dataset = tf.data.Dataset.from_tensor_slices(
          (features, labels, example_weights))
      dataset = dataset.shuffle(buffer_size=1).repeat().batch(1)
      model.fit(dataset, steps_per_epoch=1)
      model_location = os.path.join(self._getTempDir(), name)
      if convert_to_tflite:
        converter = tf.compat.v2.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        tf.io.gfile.makedirs(model_location)
        with tf.io.gfile.GFile(os.path.join(model_location, 'tflite'),
                               'wb') as f:
          f.write(tflite_model)
      else:
        model.save(model_location, save_format='tf')
      return model_eval_lib.default_eval_shared_model(
          eval_saved_model_path=model_location,
          eval_config=eval_config), model_location

    examples = [
        self._makeExample(data=[0.0] * 28 * 28, label=1.0),
        self._makeExample(data=[1.0] * 28 * 28, label=5.0),
        self._makeExample(data=[1.0] * 28 * 28, label=9.0),
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
        }
        """, schema_pb2.Schema())

    metrics_spec = config.MetricsSpec()
    for metric in (tf.keras.metrics.AUC(),):
      cfg = tf.keras.utils.serialize_keras_object(metric)
      metrics_spec.metrics.append(
          config.MetricConfig(
              class_name=cfg['class_name'], config=json.dumps(cfg['config'])))
    tf.keras.backend.clear_session()
    metrics_spec.metrics.append(
        config.MetricConfig(
            class_name='WeightedExampleCount',
            # 2 > 10, NOT OK.
            threshold=config.MetricThreshold(
                value_threshold=config.GenericValueThreshold(
                    lower_bound={'value': 0}))))
    for class_id in (0, 5, 9):
      metrics_spec.binarize.class_ids.values.append(class_id)
    eval_config = config.EvalConfig(
        model_specs=[config.ModelSpec(label_key='label')],
        metrics_specs=[metrics_spec])
    if convert_to_tflite:
      for s in eval_config.model_specs:
        s.model_type = constants.TF_LITE

    model, model_location = _build_keras_model(eval_config)
    baseline, baseline_model_location = _build_keras_model(
        eval_config, 'baseline_export')
    output_path = self._getTempDir()
    eval_results = model_eval_lib.run_model_analysis(
        eval_config=eval_config,
        eval_shared_model={
            'candidate': model,
            'baseline': baseline
        },
        data_location=data_location,
        output_path=output_path,
        schema=schema)

    # Directly check validaton file since it is not in EvalResult.
    validations_file = os.path.join(output_path, constants.VALIDATIONS_KEY)
    self.assertTrue(os.path.exists(validations_file))
    validation_records = []
    for record in tf.compat.v1.python_io.tf_record_iterator(validations_file):
      validation_records.append(
          validation_result_pb2.ValidationResult.FromString(record))
    self.assertLen(validation_records, 1)
    self.assertTrue(validation_records[0].validation_ok)

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
          'classId:9': {
              'auc': True,
          },
      }
      for class_id in expected_metrics:
        self.assertIn(class_id, got_metrics)
        for k in expected_metrics[class_id]:
          self.assertIn(k, got_metrics[class_id])

    self.assertLen(eval_results._results, 2)
    eval_result_0, eval_result_1 = eval_results._results
    check_eval_result(eval_result_0, model_location)
    check_eval_result(eval_result_1, baseline_model_location)

  def testRunModelAnalysisWithQueryBasedMetrics(self):
    input_layer = tf.keras.layers.Input(shape=(1,), name='age')
    output_layer = tf.keras.layers.Dense(
        1, activation=tf.nn.sigmoid)(
            input_layer)
    model = tf.keras.models.Model(input_layer, output_layer)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=.001),
        loss=tf.keras.losses.binary_crossentropy)

    features = {'age': [[20.0]]}
    labels = [[1]]
    example_weights = [1.0]
    dataset = tf.data.Dataset.from_tensor_slices(
        (features, labels, example_weights))
    dataset = dataset.shuffle(buffer_size=1).repeat().batch(1)
    model.fit(dataset, steps_per_epoch=1)

    model_location = os.path.join(self._getTempDir(), 'export_dir')
    model.save(model_location, save_format='tf')

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
        """, schema_pb2.Schema())
    examples = [
        self._makeExample(age=3.0, language='english', label=1.0),
        self._makeExample(age=5.0, language='chinese', label=0.0),
        self._makeExample(age=3.0, language='english', label=0.0),
        self._makeExample(age=5.0, language='chinese', label=1.0)
    ]
    data_location = self._writeTFExamplesToTFRecords(examples)
    slicing_specs = [config.SlicingSpec()]
    eval_config = config.EvalConfig(
        model_specs=[config.ModelSpec(label_key='label')],
        slicing_specs=slicing_specs,
        metrics_specs=metric_specs.specs_from_metrics(
            [ndcg.NDCG(gain_key='age', name='ndcg')],
            binarize=config.BinarizationOptions(top_k_list={'values': [1]}),
            query_key='language'))
    eval_shared_model = model_eval_lib.default_eval_shared_model(
        eval_saved_model_path=model_location, eval_config=eval_config)
    eval_result = model_eval_lib.run_model_analysis(
        eval_config=eval_config,
        eval_shared_model=eval_shared_model,
        data_location=data_location,
        output_path=self._getTempDir(),
        evaluators=[
            metrics_and_plots_evaluator_v2.MetricsAndPlotsEvaluator(
                eval_config=eval_config, eval_shared_model=eval_shared_model)
        ],
        schema=schema)

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
        },
    }
    for group in expected_metrics:
      self.assertIn(group, got_metrics)
      for k in expected_metrics[group]:
        self.assertIn(k, got_metrics[group])

  def testRunModelAnalysisWithLegacyQueryExtractor(self):
    model_location = self._exportEvalSavedModel(
        linear_classifier.simple_linear_classifier)
    examples = [
        self._makeExample(age=3.0, language='english', label=1.0),
        self._makeExample(age=3.0, language='chinese', label=0.0),
        self._makeExample(age=4.0, language='english', label=0.0),
        self._makeExample(age=5.0, language='chinese', label=1.0)
    ]
    data_location = self._writeTFExamplesToTFRecords(examples)
    slicing_specs = [config.SlicingSpec()]
    eval_config = config.EvalConfig(slicing_specs=slicing_specs)
    eval_shared_model = model_eval_lib.default_eval_shared_model(
        eval_saved_model_path=model_location, example_weight_key='age')
    eval_result = model_eval_lib.run_model_analysis(
        eval_config=eval_config,
        eval_shared_model=eval_shared_model,
        data_location=data_location,
        output_path=self._getTempDir(),
        evaluators=[
            metrics_and_plots_evaluator.MetricsAndPlotsEvaluator(
                eval_shared_model),
            query_based_metrics_evaluator.QueryBasedMetricsEvaluator(
                query_id='language',
                prediction_key='logistic',
                combine_fns=[
                    query_statistics.QueryStatisticsCombineFn(),
                    legacy_ndcg.NdcgMetricCombineFn(
                        at_vals=[1], gain_key='label', weight_key='')
                ]),
        ])
    # We only check some of the metrics to ensure that the end-to-end
    # pipeline works.
    expected = {
        (): {
            'post_export_metrics/total_queries': {
                'doubleValue': 2.0
            },
            'post_export_metrics/min_documents': {
                'doubleValue': 2.0
            },
            'post_export_metrics/max_documents': {
                'doubleValue': 2.0
            },
            'post_export_metrics/total_documents': {
                'doubleValue': 4.0
            },
            'post_export_metrics/ndcg@1': {
                'doubleValue': 0.5
            },
            'post_export_metrics/example_weight': {
                'doubleValue': 15.0
            },
            'post_export_metrics/example_count': {
                'doubleValue': 4.0
            },
        }
    }
    self.assertEqual(eval_result.model_location, model_location.decode())
    self.assertEqual(eval_result.data_location, data_location)
    self.assertEqual(eval_result.config.slicing_specs[0], config.SlicingSpec())
    self.assertMetricsAlmostEqual(eval_result.slicing_metrics, expected)
    self.assertFalse(eval_result.plots)

  def testRunModelAnalysisWithUncertainty(self):
    model_location = self._exportEvalSavedModel(
        linear_classifier.simple_linear_classifier)
    examples = [
        self._makeExample(age=3.0, language='english', label=1.0),
        self._makeExample(age=3.0, language='chinese', label=0.0),
        self._makeExample(age=4.0, language='english', label=1.0),
        self._makeExample(age=5.0, language='chinese', label=1.0),
        self._makeExample(age=5.0, language='hindi', label=1.0)
    ]
    data_location = self._writeTFExamplesToTFRecords(examples)
    slicing_specs = [config.SlicingSpec(feature_keys=['language'])]
    options = config.Options()
    options.compute_confidence_intervals.value = True
    options.min_slice_size.value = 2
    eval_config = config.EvalConfig(
        slicing_specs=slicing_specs, options=options)
    eval_result = model_eval_lib.run_model_analysis(
        eval_config=eval_config,
        eval_shared_model=model_eval_lib.default_eval_shared_model(
            eval_saved_model_path=model_location, example_weight_key='age'),
        data_location=data_location,
        output_path=self._getTempDir())
    # We only check some of the metrics to ensure that the end-to-end
    # pipeline works.
    expected = {
        (('language', 'hindi'),): {
            u'__ERROR__': {
                'debugMessage':
                    u'Example count for this slice key is lower than the '
                    u'minimum required value: 2. No data is aggregated for '
                    u'this slice.'
            },
        },
        (('language', 'chinese'),): {
            metric_keys.EXAMPLE_WEIGHT: {
                'doubleValue': 8.0
            },
            metric_keys.EXAMPLE_COUNT: {
                'doubleValue': 2.0
            },
        },
        (('language', 'english'),): {
            'accuracy': {
                'boundedValue': {
                    'value': 1.0,
                    'lowerBound': 1.0,
                    'upperBound': 1.0,
                    'methodology': 'POISSON_BOOTSTRAP'
                }
            },
            'my_mean_label': {
                'boundedValue': {
                    'value': 1.0,
                    'lowerBound': 1.0,
                    'upperBound': 1.0,
                    'methodology': 'POISSON_BOOTSTRAP'
                }
            },
            metric_keys.EXAMPLE_WEIGHT: {
                'doubleValue': 7.0
            },
            metric_keys.EXAMPLE_COUNT: {
                'doubleValue': 2.0
            },
        }
    }
    self.assertEqual(eval_result.model_location, model_location.decode())
    self.assertEqual(eval_result.data_location, data_location)
    self.assertEqual(eval_result.config.slicing_specs[0],
                     config.SlicingSpec(feature_keys=['language']))
    self.assertMetricsAlmostEqual(eval_result.slicing_metrics, expected)
    self.assertFalse(eval_result.plots)

  def testRunModelAnalysisWithDeterministicConfidenceIntervals(self):
    model_location = self._exportEvalSavedModel(
        linear_classifier.simple_linear_classifier)
    examples = [
        self._makeExample(age=3.0, language='english', label=1.0),
        self._makeExample(age=3.0, language='chinese', label=0.0),
        self._makeExample(age=4.0, language='english', label=1.0),
        self._makeExample(age=5.0, language='chinese', label=1.0),
        self._makeExample(age=5.0, language='hindi', label=1.0)
    ]
    data_location = self._writeTFExamplesToTFRecords(examples)
    slicing_specs = [config.SlicingSpec(feature_keys=['language'])]
    options = config.Options()
    options.compute_confidence_intervals.value = True
    options.min_slice_size.value = 2
    eval_config = config.EvalConfig(
        slicing_specs=slicing_specs, options=options)
    eval_result = model_eval_lib.run_model_analysis(
        eval_config=eval_config,
        eval_shared_model=model_eval_lib.default_eval_shared_model(
            eval_saved_model_path=model_location, example_weight_key='age'),
        data_location=data_location,
        output_path=self._getTempDir(),
        random_seed_for_testing=_TEST_SEED)
    # We only check some of the metrics to ensure that the end-to-end
    # pipeline works.
    expected = {
        (('language', 'hindi'),): {
            u'__ERROR__': {
                'debugMessage':
                    u'Example count for this slice key is lower than the '
                    u'minimum required value: 2. No data is aggregated for '
                    u'this slice.'
            },
        },
        (('language', 'chinese'),): {
            metric_keys.EXAMPLE_WEIGHT: {
                'doubleValue': 8.0
            },
            metric_keys.EXAMPLE_COUNT: {
                'doubleValue': 2.0
            },
        },
        (('language', 'english'),): {
            'accuracy': {
                'boundedValue': {
                    'value': 1.0,
                    'lowerBound': 1.0,
                    'upperBound': 1.0,
                    'methodology': 'POISSON_BOOTSTRAP'
                }
            },
            'my_mean_label': {
                'boundedValue': {
                    'value': 1.0,
                    'lowerBound': 1.0,
                    'upperBound': 1.0,
                    'methodology': 'POISSON_BOOTSTRAP'
                }
            },
            metric_keys.EXAMPLE_WEIGHT: {
                'doubleValue': 7.0
            },
            metric_keys.EXAMPLE_COUNT: {
                'doubleValue': 2.0
            },
        }
    }
    self.assertEqual(eval_result.model_location, model_location.decode())
    self.assertEqual(eval_result.data_location, data_location)
    self.assertEqual(eval_result.config.slicing_specs[0],
                     config.SlicingSpec(feature_keys=['language']))
    self.assertMetricsAlmostEqual(eval_result.slicing_metrics, expected)

    for key, value in eval_result.slicing_metrics:
      if (('language', 'english'),) == key:
        metric = value['']['']['average_loss']
        self.assertAlmostEqual(
            0.171768754720, metric['boundedValue']['value'], delta=0.1)

        metric = value['']['']['auc_precision_recall']
        self.assertAlmostEqual(
            0.99999940395, metric['boundedValue']['value'], delta=0.1)

    self.assertFalse(eval_result.plots)

  def testRunModelAnalysisWithSchema(self):
    model_location = self._exportEvalSavedModel(
        linear_regressor.simple_linear_regressor)
    examples = [
        self._makeExample(age=3.0, language='english', label=2.0),
        self._makeExample(age=3.0, language='chinese', label=1.0),
        self._makeExample(age=4.0, language='english', label=2.0),
        self._makeExample(age=5.0, language='chinese', label=2.0),
        self._makeExample(age=5.0, language='hindi', label=2.0)
    ]
    data_location = self._writeTFExamplesToTFRecords(examples)
    eval_config = config.EvalConfig(
        model_specs=[config.ModelSpec(label_key='label')],
        metrics_specs=metric_specs.specs_from_metrics(
            [calibration_plot.CalibrationPlot(num_buckets=4)]))
    schema = text_format.Parse(
        """
        feature {
          name: "label"
          type: INT
          int_domain {
            min: 1
            max: 2
          }
        }
        """, schema_pb2.Schema())
    eval_result = model_eval_lib.run_model_analysis(
        eval_config=eval_config,
        schema=schema,
        eval_shared_model=model_eval_lib.default_eval_shared_model(
            eval_saved_model_path=model_location, example_weight_key='age'),
        data_location=data_location,
        output_path=self._getTempDir())

    expected_metrics = {(): {metric_keys.EXAMPLE_COUNT: {'doubleValue': 5.0},}}
    self.assertMetricsAlmostEqual(eval_result.slicing_metrics, expected_metrics)
    self.assertLen(eval_result.plots, 1)
    slice_key, plots = eval_result.plots[0]
    self.assertEqual((), slice_key)
    got_buckets = plots['']['']['calibrationHistogramBuckets']['buckets']
    # buckets include (-inf, left) and (right, inf) by default, but we are
    # interested in the values of left and right
    self.assertEqual(1.0, got_buckets[1]['lowerThresholdInclusive'])
    self.assertEqual(2.0, got_buckets[-2]['upperThresholdExclusive'])

  def testRunModelAnalysisWithPlots(self):
    model_location = self._exportEvalSavedModel(
        fixed_prediction_estimator.simple_fixed_prediction_estimator)
    examples = [
        self._makeExample(prediction=0.0, label=1.0),
        self._makeExample(prediction=0.7, label=0.0),
        self._makeExample(prediction=0.8, label=1.0),
        self._makeExample(prediction=1.0, label=1.0),
        self._makeExample(prediction=1.0, label=1.0)
    ]
    data_location = self._writeTFExamplesToTFRecords(examples)
    eval_config = config.EvalConfig()
    eval_shared_model = model_eval_lib.default_eval_shared_model(
        eval_saved_model_path=model_location,
        add_metrics_callbacks=[post_export_metrics.auc_plots()])
    eval_result = model_eval_lib.run_model_analysis(
        eval_config=eval_config,
        eval_shared_model=eval_shared_model,
        data_location=data_location,
        output_path=self._getTempDir())
    # We only check some of the metrics to ensure that the end-to-end
    # pipeline works.
    expected_metrics = {(): {metric_keys.EXAMPLE_COUNT: {'doubleValue': 5.0},}}
    expected_matrix = {
        'threshold': 0.8,
        'falseNegatives': 2.0,
        'trueNegatives': 1.0,
        'truePositives': 2.0,
        'precision': 1.0,
        'recall': 0.5
    }
    self.assertMetricsAlmostEqual(eval_result.slicing_metrics, expected_metrics)
    self.assertLen(eval_result.plots, 1)
    slice_key, plots = eval_result.plots[0]
    self.assertEqual((), slice_key)
    self.assertDictElementsAlmostEqual(
        plots['']['']['confusionMatrixAtThresholds']['matrices'][8001],
        expected_matrix)

  def testRunModelAnalysisWithMultiplePlots(self):
    model_location = self._exportEvalSavedModel(
        fixed_prediction_estimator.simple_fixed_prediction_estimator)
    examples = [
        self._makeExample(prediction=0.0, label=1.0),
        self._makeExample(prediction=0.7, label=0.0),
        self._makeExample(prediction=0.8, label=1.0),
        self._makeExample(prediction=1.0, label=1.0),
        self._makeExample(prediction=1.0, label=1.0)
    ]
    data_location = self._writeTFExamplesToTFRecords(examples)
    eval_config = config.EvalConfig()
    eval_shared_model = model_eval_lib.default_eval_shared_model(
        eval_saved_model_path=model_location,
        add_metrics_callbacks=[
            post_export_metrics.auc_plots(),
            post_export_metrics.auc_plots(metric_tag='test')
        ])
    eval_result = model_eval_lib.run_model_analysis(
        eval_config=eval_config,
        eval_shared_model=eval_shared_model,
        data_location=data_location,
        output_path=self._getTempDir())

    # pipeline works.
    expected_metrics = {(): {metric_keys.EXAMPLE_COUNT: {'doubleValue': 5.0},}}
    expected_matrix = {
        'threshold': 0.8,
        'falseNegatives': 2.0,
        'trueNegatives': 1.0,
        'truePositives': 2.0,
        'precision': 1.0,
        'recall': 0.5
    }
    self.assertMetricsAlmostEqual(eval_result.slicing_metrics, expected_metrics)
    self.assertLen(eval_result.plots, 1)
    slice_key, plots = eval_result.plots[0]
    self.assertEqual((), slice_key)
    self.assertDictElementsAlmostEqual(
        plots['']['']['post_export_metrics']['confusionMatrixAtThresholds']
        ['matrices'][8001], expected_matrix)
    self.assertDictElementsAlmostEqual(
        plots['']['']['post_export_metrics/test']['confusionMatrixAtThresholds']
        ['matrices'][8001], expected_matrix)

  def testRunModelAnalysisForCSVText(self):
    model_location = self._exportEvalSavedModel(
        csv_linear_classifier.simple_csv_linear_classifier)
    examples = [
        '3.0,english,1.0', '3.0,chinese,0.0', '4.0,english,1.0',
        '5.0,chinese,1.0'
    ]
    data_location = self._writeCSVToTextFile(examples)
    eval_config = config.EvalConfig()
    eval_result = model_eval_lib.run_model_analysis(
        eval_config=eval_config,
        eval_shared_model=model_eval_lib.default_eval_shared_model(
            eval_saved_model_path=model_location),
        data_location=data_location,
        file_format='text',
        output_path=self._getTempDir())
    # We only check some of the metrics to ensure that the end-to-end
    # pipeline works.
    expected = {
        (): {
            'accuracy': {
                'doubleValue': 0.75
            },
            metric_keys.EXAMPLE_COUNT: {
                'doubleValue': 4.0
            }
        }
    }
    self.assertMetricsAlmostEqual(eval_result.slicing_metrics, expected)

  def testMultipleModelAnalysis(self):
    model_location_1 = self._exportEvalSavedModel(
        linear_classifier.simple_linear_classifier)
    model_location_2 = self._exportEvalSavedModel(
        linear_classifier.simple_linear_classifier)
    examples = [
        self._makeExample(age=3.0, language='english', label=1.0),
        self._makeExample(age=3.0, language='chinese', label=0.0),
        self._makeExample(age=4.0, language='english', label=1.0),
        self._makeExample(age=5.0, language='chinese', label=1.0)
    ]
    data_location = self._writeTFExamplesToTFRecords(examples)
    eval_results = model_eval_lib.multiple_model_analysis(
        [model_location_1, model_location_2],
        data_location,
        slice_spec=[slicer.SingleSliceSpec(features=[('language', 'english')])])
    # We only check some of the metrics to ensure that the end-to-end
    # pipeline works.
    self.assertLen(eval_results._results, 2)
    expected_result_1 = {
        (('language', 'english'),): {
            'my_mean_label': {
                'doubleValue': 1.0
            },
            metric_keys.EXAMPLE_COUNT: {
                'doubleValue': 2.0
            },
        }
    }
    expected_result_2 = {
        (('language', 'english'),): {
            'my_mean_label': {
                'doubleValue': 1.0
            },
            metric_keys.EXAMPLE_COUNT: {
                'doubleValue': 2.0
            },
        }
    }
    self.assertMetricsAlmostEqual(eval_results._results[0].slicing_metrics,
                                  expected_result_1)
    self.assertMetricsAlmostEqual(eval_results._results[1].slicing_metrics,
                                  expected_result_2)

  def testMultipleDataAnalysis(self):
    model_location = self._exportEvalSavedModel(
        linear_classifier.simple_linear_classifier)
    data_location_1 = self._writeTFExamplesToTFRecords([
        self._makeExample(age=3.0, language='english', label=1.0),
        self._makeExample(age=3.0, language='english', label=0.0),
        self._makeExample(age=5.0, language='chinese', label=1.0)
    ])
    data_location_2 = self._writeTFExamplesToTFRecords(
        [self._makeExample(age=4.0, language='english', label=1.0)])
    eval_results = model_eval_lib.multiple_data_analysis(
        model_location, [data_location_1, data_location_2],
        slice_spec=[slicer.SingleSliceSpec(features=[('language', 'english')])])
    self.assertLen(eval_results._results, 2)
    # We only check some of the metrics to ensure that the end-to-end
    # pipeline works.
    expected_result_1 = {
        (('language', 'english'),): {
            metric_keys.EXAMPLE_COUNT: {
                'doubleValue': 2.0
            },
        }
    }
    expected_result_2 = {
        (('language', 'english'),): {
            metric_keys.EXAMPLE_COUNT: {
                'doubleValue': 1.0
            },
        }
    }
    self.assertMetricsAlmostEqual(eval_results._results[0].slicing_metrics,
                                  expected_result_1)
    self.assertMetricsAlmostEqual(eval_results._results[1].slicing_metrics,
                                  expected_result_2)

  @parameterized.named_parameters(('class_id', 1, None, None),
                                  ('top_k', None, None, 1),
                                  ('k', None, 1, None))
  def testGetMetrics(self, class_id, k, top_k):

    # Slices
    num_slices = 2
    overall_slice = ()
    male_slice = (('gender', 'male'))

    # Output name
    output_name = ''

    # Metrics for overall slice
    metrics_overall = {
        'accuracy': {
            'doubleValue': 0.5,
        },
        'auc': {
            'doubleValue': 0.8,
        },
    }

    # Metrics for male slice
    metrics_male = {
        'accuracy': {
            'doubleValue': 0.8,
        },
        'auc': {
            'doubleValue': 0.5,
        }
    }

    # EvalResult
    sub_key = metric_types.SubKey(class_id, k, top_k)
    slicing_metrics = [(overall_slice, {
        output_name: {
            str(sub_key): metrics_overall
        }
    }), (male_slice, {
        output_name: {
            str(sub_key): metrics_male
        }
    })]
    eval_result = model_eval_lib.EvalResult(slicing_metrics, None, None, None,
                                            None, None)

    # Test get_metrics_for_all_slices()
    actual_metrics = eval_result.get_metrics_for_all_slices(
        class_id=class_id, k=k, top_k=top_k)

    # Assert there is one metrics entry per slice
    self.assertLen(actual_metrics, num_slices)

    # Assert the metrics match the expected values
    self.assertDictEqual(actual_metrics[overall_slice], metrics_overall)
    self.assertDictEqual(actual_metrics[male_slice], metrics_male)

    # Test get_metrics()
    self.assertDictEqual(
        eval_result.get_metrics(class_id=class_id, k=k, top_k=top_k),
        metrics_overall)
    self.assertDictEqual(
        eval_result.get_metrics(
            slice_name=male_slice, class_id=class_id, k=k, top_k=top_k),
        metrics_male)

    # Test get_slices()
    self.assertListEqual(eval_result.get_slices(), [overall_slice, male_slice])


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
