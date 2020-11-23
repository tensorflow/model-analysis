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

from absl.testing import absltest
from absl.testing import parameterized

import pandas as pd
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
from tensorflow_model_analysis.evaluators import legacy_metrics_and_plots_evaluator
from tensorflow_model_analysis.evaluators import legacy_query_based_metrics_evaluator
from tensorflow_model_analysis.evaluators import metrics_plots_and_validations_evaluator
from tensorflow_model_analysis.evaluators.query_metrics import ndcg as legacy_ndcg
from tensorflow_model_analysis.evaluators.query_metrics import query_statistics
from tensorflow_model_analysis.extractors import legacy_feature_extractor
from tensorflow_model_analysis.extractors import legacy_predict_extractor
from tensorflow_model_analysis.extractors import slice_key_extractor
from tensorflow_model_analysis.metrics import calibration_plot
from tensorflow_model_analysis.metrics import metric_specs
from tensorflow_model_analysis.metrics import ndcg
from tensorflow_model_analysis.post_export_metrics import metric_keys
from tensorflow_model_analysis.post_export_metrics import post_export_metrics
from tensorflow_model_analysis.proto import validation_result_pb2
from tensorflow_model_analysis.slicer import slicer_lib
from tensorflow_model_analysis.view import view_types
from tensorflowjs.converters import converter as tfjs_converter

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2

try:
  import tensorflow_ranking as tfr  # pylint: disable=g-import-not-at-top
  _TFR_IMPORTED = True
except (ImportError, tf.errors.NotFoundError):
  _TFR_IMPORTED = False

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

  def assertMetricsAlmostEqual(self,
                               got_slicing_metrics,
                               expected_slicing_metrics,
                               output_name='',
                               subkey=''):
    if got_slicing_metrics:
      for (s, m) in got_slicing_metrics:
        metrics = m[output_name][subkey]
        self.assertIn(s, expected_slicing_metrics)
        for metric_name in expected_slicing_metrics[s]:
          self.assertIn(metric_name, metrics)
          self.assertDictElementsAlmostEqual(
              metrics[metric_name], expected_slicing_metrics[s][metric_name])
    else:
      # Only pass if expected_slicing_metrics also evaluates to False.
      self.assertFalse(
          expected_slicing_metrics, msg='Actual slicing_metrics was empty.')

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

  @parameterized.named_parameters(('tflite', constants.TF_LITE),
                                  ('tfjs', constants.TF_JS))
  def testMixedModelTypes(self, model_type):
    examples = [self._makeExample(age=3.0, language='english', label=1.0)]
    data_location = self._writeTFExamplesToTFRecords(examples)
    eval_config = config.EvalConfig(model_specs=[
        config.ModelSpec(name='model1'),
        config.ModelSpec(name='model2', model_type=model_type)
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
        NotImplementedError, 'support for mixing .* models is not implemented'):
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
    slicing_specs = [slicer_lib.SingleSliceSpec(columns=['my_slice'])]
    eval_shared_model = model_eval_lib.default_eval_shared_model(
        eval_saved_model_path=model_location, example_weight_key='age')
    extractors_with_feature_extraction = [
        legacy_predict_extractor.PredictExtractor(
            eval_shared_model, desired_batch_size=3, materialize=False),
        legacy_feature_extractor.FeatureExtractor(
            extract_source=constants.INPUT_KEY,
            extract_dest=constants.FEATURES_PREDICTIONS_LABELS_KEY),
        slice_key_extractor.SliceKeyExtractor(
            slice_spec=slicing_specs, materialize=False)
    ]
    eval_result = model_eval_lib.run_model_analysis(
        eval_shared_model=model_eval_lib.default_eval_shared_model(
            eval_saved_model_path=model_location, example_weight_key='age'),
        data_location=data_location,
        output_path=self._getTempDir(),
        extractors=extractors_with_feature_extraction,
        slice_spec=slicing_specs)
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
    slicing_specs = [slicer_lib.SingleSliceSpec(columns=['language'])]
    eval_result = model_eval_lib.run_model_analysis(
        eval_shared_model=model_eval_lib.default_eval_shared_model(
            eval_saved_model_path=model_location, example_weight_key='age'),
        data_location=data_location,
        output_path=self._getTempDir(),
        slice_spec=slicing_specs,
        min_slice_size=2)
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

  def testRunModelAnalysisWithCustomizations(self):
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
        model_specs=[config.ModelSpec(model_type='my_model_type')],
        slicing_specs=slicing_specs,
        options=options)
    # Use default model_loader for testing passing custom_model_loader
    model_loader = model_eval_lib.default_eval_shared_model(
        eval_saved_model_path=model_location,
        example_weight_key='age').model_loader
    eval_shared_model = model_eval_lib.default_eval_shared_model(
        eval_saved_model_path=model_location, custom_model_loader=model_loader)
    # Use PredictExtractor for testing passing custom_predict_extractor
    extractors = model_eval_lib.default_extractors(
        eval_shared_model=eval_shared_model,
        eval_config=eval_config,
        custom_predict_extractor=legacy_predict_extractor.PredictExtractor(
            eval_shared_model=eval_shared_model, eval_config=eval_config))
    eval_result = model_eval_lib.run_model_analysis(
        eval_config=eval_config,
        eval_shared_model=eval_shared_model,
        data_location=data_location,
        output_path=self._getTempDir(),
        extractors=extractors)
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

  @parameterized.named_parameters(
      ('tf_keras', constants.TF_KERAS), ('tf_lite', constants.TF_LITE),
      ('tf_js', constants.TF_JS),
      ('baseline_missing', constants.TF_KERAS, True),
      ('rubber_stamp', constants.TF_KERAS, True, True))
  def testRunModelAnalysisWithKerasModel(self,
                                         model_type,
                                         remove_baseline=False,
                                         rubber_stamp=False):

    def _build_keras_model(eval_config,
                           export_name='export_dir',
                           rubber_stamp=False):
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
      model_location = os.path.join(self._getTempDir(), export_name)
      if model_type == constants.TF_LITE:
        converter = tf.compat.v2.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        tf.io.gfile.makedirs(model_location)
        with tf.io.gfile.GFile(os.path.join(model_location, 'tflite'),
                               'wb') as f:
          f.write(tflite_model)
      elif model_type == constants.TF_JS:
        src_model_path = tempfile.mkdtemp()
        model.save(src_model_path, save_format='tf')

        tfjs_converter.convert([
            '--input_format=tf_saved_model',
            '--saved_model_tags=serve',
            '--signature_name=serving_default',
            src_model_path,
            model_location,
        ])
      else:
        model.save(model_location, save_format='tf')
      return model_eval_lib.default_eval_shared_model(
          eval_saved_model_path=model_location,
          eval_config=eval_config,
          rubber_stamp=rubber_stamp)

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
    slicing_specs = [
        config.SlicingSpec(),
        config.SlicingSpec(feature_keys=['non_existent_slice'])
    ]
    metrics_spec.metrics.append(
        config.MetricConfig(
            class_name='WeightedExampleCount',
            per_slice_thresholds=[
                config.PerSliceMetricThreshold(
                    slicing_specs=slicing_specs,
                    threshold=config.MetricThreshold(
                        value_threshold=config.GenericValueThreshold(
                            lower_bound={'value': 0}))),
                # Change thresholds would be ignored when rubber stamp is true.
                config.PerSliceMetricThreshold(
                    slicing_specs=slicing_specs,
                    threshold=config.MetricThreshold(
                        change_threshold=config.GenericChangeThreshold(
                            direction=config.MetricDirection.HIGHER_IS_BETTER,
                            absolute={'value': 0})))
            ]))
    for class_id in (0, 5):
      metrics_spec.binarize.class_ids.values.append(class_id)
    eval_config = config.EvalConfig(
        model_specs=[config.ModelSpec(label_key='label')],
        metrics_specs=[metrics_spec])
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
    eval_results = model_eval_lib.run_model_analysis(
        eval_config=eval_config,
        eval_shared_model=eval_shared_model,
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
    # Change thresholds ignored when rubber stamping
    expected_result = text_format.Parse(
        """
        validation_ok: false
        missing_slices: {
          feature_keys: "non_existent_slice"
        }
        validation_details {
          slicing_details {
            slicing_spec {
            }
            num_matching_slices: 1
          }
        }""", validation_result_pb2.ValidationResult())
    # Normal run with change threshold not satisfied.
    if not rubber_stamp and not remove_baseline:
      text_format.Parse(
          """
          metric_validations_per_slice {
            slice_key {}
            failures {
             metric_key {
               name: "weighted_example_count"
               sub_key { class_id {} }
               model_name: "candidate"
               is_diff: true
             }
             metric_threshold {
               change_threshold {
                 absolute {}
                 direction: HIGHER_IS_BETTER
               }
             }
             metric_value { double_value {} }
            }
            failures {
             metric_key {
               name: "weighted_example_count"
               sub_key {
                 class_id {
                   value: 5
                 }
               }
               model_name: "candidate"
               is_diff: true
             }
             metric_threshold {
               change_threshold {
                 absolute {}
                 direction: HIGHER_IS_BETTER
               }
             }
             metric_value { double_value {} }
            }
          }""", expected_result)
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
    # Test with both a TFMA metric (NDCG), a keras metric (Recall).
    metrics = [
        ndcg.NDCG(gain_key='age', name='ndcg', top_k_list=[1, 2]),
        tf.keras.metrics.Recall(top_k=1),
    ]
    # If tensorflow-ranking imported add MRRMetric.
    if _TFR_IMPORTED:
      metrics.append(tfr.keras.metrics.MRRMetric())
    metrics_specs = metric_specs.specs_from_metrics(
        metrics, query_key='language', include_weighted_example_count=True)
    metrics_specs.append(
        config.MetricsSpec(metrics=[
            config.MetricConfig(
                class_name='WeightedExampleCount',
                threshold=config.MetricThreshold(
                    value_threshold=config.GenericValueThreshold(
                        lower_bound={'value': 0})))
        ]))
    eval_config = config.EvalConfig(
        model_specs=[config.ModelSpec(label_key='label')],
        slicing_specs=slicing_specs,
        metrics_specs=metrics_specs)
    eval_shared_model = model_eval_lib.default_eval_shared_model(
        eval_saved_model_path=model_location, eval_config=eval_config)
    output_path = self._getTempDir()
    eval_result = model_eval_lib.run_model_analysis(
        eval_config=eval_config,
        eval_shared_model=eval_shared_model,
        data_location=data_location,
        output_path=output_path,
        evaluators=[
            metrics_plots_and_validations_evaluator
            .MetricsPlotsAndValidationsEvaluator(
                eval_config=eval_config, eval_shared_model=eval_shared_model)
        ],
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
    slicing_specs = [slicer_lib.SingleSliceSpec()]
    eval_shared_model = model_eval_lib.default_eval_shared_model(
        eval_saved_model_path=model_location, example_weight_key='age')
    eval_result = model_eval_lib.run_model_analysis(
        eval_shared_model=eval_shared_model,
        data_location=data_location,
        output_path=self._getTempDir(),
        evaluators=[
            legacy_metrics_and_plots_evaluator.MetricsAndPlotsEvaluator(
                eval_shared_model),
            legacy_query_based_metrics_evaluator.QueryBasedMetricsEvaluator(
                query_id='language',
                prediction_key='logistic',
                combine_fns=[
                    query_statistics.QueryStatisticsCombineFn(),
                    legacy_ndcg.NdcgMetricCombineFn(
                        at_vals=[1], gain_key='label', weight_key='')
                ]),
        ],
        slice_spec=slicing_specs)
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
    slicing_specs = [slicer_lib.SingleSliceSpec(columns=['language'])]
    eval_result = model_eval_lib.run_model_analysis(
        eval_shared_model=model_eval_lib.default_eval_shared_model(
            eval_saved_model_path=model_location, example_weight_key='age'),
        data_location=data_location,
        output_path=self._getTempDir(),
        slice_spec=slicing_specs,
        compute_confidence_intervals=True,
        min_slice_size=2)
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
    slicing_specs = [slicer_lib.SingleSliceSpec(columns=['language'])]
    eval_result = model_eval_lib.run_model_analysis(
        eval_shared_model=model_eval_lib.default_eval_shared_model(
            eval_saved_model_path=model_location, example_weight_key='age'),
        data_location=data_location,
        output_path=self._getTempDir(),
        slice_spec=slicing_specs,
        compute_confidence_intervals=True,
        min_slice_size=2,
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
    eval_shared_model = model_eval_lib.default_eval_shared_model(
        eval_saved_model_path=model_location,
        add_metrics_callbacks=[post_export_metrics.auc_plots()])
    eval_result = model_eval_lib.run_model_analysis(
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
    eval_shared_model = model_eval_lib.default_eval_shared_model(
        eval_saved_model_path=model_location,
        add_metrics_callbacks=[
            post_export_metrics.auc_plots(),
            post_export_metrics.auc_plots(metric_tag='test')
        ])
    eval_result = model_eval_lib.run_model_analysis(
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
    eval_config = config.EvalConfig(slicing_specs=[
        config.SlicingSpec(feature_values={'language': 'english'})
    ])
    eval_results = model_eval_lib.multiple_model_analysis(
        [model_location_1, model_location_2],
        data_location,
        eval_config=eval_config)
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
    eval_config = config.EvalConfig(slicing_specs=[
        config.SlicingSpec(feature_values={'language': 'english'})
    ])
    eval_results = model_eval_lib.multiple_data_analysis(
        model_location, [data_location_1, data_location_2],
        eval_config=eval_config)
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

  def testLoadValidationResult(self):
    result = validation_result_pb2.ValidationResult(validation_ok=True)
    path = os.path.join(absltest.get_default_test_tmpdir(), 'results.tfrecord')
    with tf.io.TFRecordWriter(path) as writer:
      writer.write(result.SerializeToString())
    loaded_result = model_eval_lib.load_validation_result(path)
    self.assertTrue(loaded_result.validation_ok)

  def testLoadValidationResultDir(self):
    result = validation_result_pb2.ValidationResult(validation_ok=True)
    path = os.path.join(absltest.get_default_test_tmpdir(),
                        constants.VALIDATIONS_KEY)
    with tf.io.TFRecordWriter(path) as writer:
      writer.write(result.SerializeToString())
    loaded_result = model_eval_lib.load_validation_result(os.path.dirname(path))
    self.assertTrue(loaded_result.validation_ok)

  def testLoadValidationResultEmptyFile(self):
    path = os.path.join(absltest.get_default_test_tmpdir(),
                        constants.VALIDATIONS_KEY)
    with tf.io.TFRecordWriter(path):
      pass
    with self.assertRaises(AssertionError):
      model_eval_lib.load_validation_result(path)

  def testAnalyzeRawData(self):

    # Data
    # age language  label  prediction
    # 17  english   0      0
    # 30  spanish   1      1
    dict_data = [{
        'age': 17,
        'language': 'english',
        'prediction': 0,
        'label': 0
    }, {
        'age': 30,
        'language': 'spanish',
        'prediction': 1,
        'label': 1
    }]
    df_data = pd.DataFrame(dict_data)

    # Expected Output
    expected_slicing_metrics = {
        (('language', 'english'),): {
            '': {
                '': {
                    'accuracy': {
                        'doubleValue': 1.0
                    },
                    'example_count': {
                        'doubleValue': 1.0
                    }
                }
            }
        },
        (('language', 'spanish'),): {
            '': {
                '': {
                    'accuracy': {
                        'doubleValue': 1.0
                    },
                    'example_count': {
                        'doubleValue': 1.0
                    }
                }
            }
        },
        (): {
            '': {
                '': {
                    'accuracy': {
                        'doubleValue': 1.0
                    },
                    'example_count': {
                        'doubleValue': 2.0
                    }
                }
            }
        }
    }

    # Actual Output
    eval_config = text_format.Parse(
        """
      model_specs {
        label_key: 'label'
        prediction_key: 'prediction'
      }
      metrics_specs {
        metrics { class_name: "Accuracy" }
        metrics { class_name: "ExampleCount" }
      }
      slicing_specs {}
      slicing_specs {
        feature_keys: 'language'
      }
    """, config.EvalConfig())
    eval_result = model_eval_lib.analyze_raw_data(df_data, eval_config)

    # Compare Actual and Expected
    self.assertEqual(
        len(eval_result.slicing_metrics), len(expected_slicing_metrics))
    for slicing_metric in eval_result.slicing_metrics:
      slice_key, slice_val = slicing_metric
      self.assertIn(slice_key, expected_slicing_metrics)
      self.assertDictEqual(slice_val, expected_slicing_metrics[slice_key])

  def testAnalyzeRawDataWithoutPrediction(self):
    model_specs = [
        config.ModelSpec(prediction_key='nonexistent_prediction_key')
    ]
    metrics_specs = [
        config.MetricsSpec(metrics=[config.MetricConfig(class_name='Accuracy')])
    ]
    eval_config = config.EvalConfig(
        model_specs=model_specs, metrics_specs=metrics_specs)
    df_data = pd.DataFrame([{
        'prediction': 0,
        'label': 0,
    }])
    with self.assertRaises(KeyError):
      model_eval_lib.analyze_raw_data(df_data, eval_config)

  def testAnalyzeRawDataWithoutLabel(self):
    model_specs = [config.ModelSpec(prediction_key='nonexistent_label_key')]
    metrics_specs = [
        config.MetricsSpec(metrics=[config.MetricConfig(class_name='Accuracy')])
    ]
    eval_config = config.EvalConfig(
        model_specs=model_specs, metrics_specs=metrics_specs)
    df_data = pd.DataFrame([{
        'prediction': 0,
        'label': 0,
    }])
    with self.assertRaises(KeyError):
      model_eval_lib.analyze_raw_data(df_data, eval_config)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
