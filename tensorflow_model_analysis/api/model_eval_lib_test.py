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

import os
import pickle
import tempfile

# Standard Imports

import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.eval_saved_model.example_trainers import csv_linear_classifier
from tensorflow_model_analysis.eval_saved_model.example_trainers import fixed_prediction_estimator
from tensorflow_model_analysis.eval_saved_model.example_trainers import linear_classifier
from tensorflow_model_analysis.evaluators import metrics_and_plots_evaluator
from tensorflow_model_analysis.evaluators import query_based_metrics_evaluator
from tensorflow_model_analysis.evaluators.query_metrics import ndcg
from tensorflow_model_analysis.evaluators.query_metrics import query_statistics
from tensorflow_model_analysis.extractors import feature_extractor
from tensorflow_model_analysis.extractors import predict_extractor
from tensorflow_model_analysis.extractors import slice_key_extractor
from tensorflow_model_analysis.post_export_metrics import metric_keys
from tensorflow_model_analysis.post_export_metrics import post_export_metrics
from tensorflow_model_analysis.slicer import slicer


class EvaluateTest(testutil.TensorflowModelAnalysisTest):

  def setUp(self):
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
          self.assertIn(k, m)
          self.assertDictElementsAlmostEqual(m[k], expected_value[s][k])
    else:
      # Only pass if expected_value also evaluates to False.
      self.assertFalse(expected_value, msg='Actual value was empty.')

  def assertSliceMetricsEqual(self, expected_metrics, got_metrics):
    self.assertItemsEqual(
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
    # No construct_fn should fail when Beam attempts to call the construct_fn.
    eval_shared_model = types.EvalSharedModel(model_path=model_location)
    with self.assertRaisesRegexp(TypeError,
                                 '\'NoneType\' object is not callable'):
      model_eval_lib.run_model_analysis(
          eval_shared_model=eval_shared_model, data_location=data_location)

    # Using the default_eval_shared_model should pass as it has a construct_fn.
    eval_shared_model = model_eval_lib.default_eval_shared_model(
        eval_saved_model_path=model_location)
    model_eval_lib.run_model_analysis(
        eval_shared_model=eval_shared_model, data_location=data_location)

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
    slice_spec = [slicer.SingleSliceSpec(columns=['my_slice'])]
    eval_shared_model = model_eval_lib.default_eval_shared_model(
        eval_saved_model_path=model_location, example_weight_key='age')
    extractors_with_feature_extraction = [
        predict_extractor.PredictExtractor(
            eval_shared_model, desired_batch_size=3, materialize=False),
        feature_extractor.FeatureExtractor(
            extract_source=constants.INPUT_KEY,
            extract_dest=constants.FEATURES_PREDICTIONS_LABELS_KEY),
        slice_key_extractor.SliceKeyExtractor(slice_spec, materialize=False)
    ]
    eval_result = model_eval_lib.run_model_analysis(
        model_eval_lib.default_eval_shared_model(
            eval_saved_model_path=model_location, example_weight_key='age'),
        data_location,
        extractors=extractors_with_feature_extraction,
        slice_spec=slice_spec,
        k_anonymization_count=1)
    # We only check some of the metrics to ensure that the end-to-end
    # pipeline works.
    expected = {
        (('my_slice', b'a'),): {
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
        (('my_slice', b'b'),): {
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
        (('my_slice', b'c'),): {
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
    self.assertEqual(eval_result.config.model_location, model_location)
    self.assertEqual(eval_result.config.data_location, data_location)
    self.assertEqual(eval_result.config.slice_spec, slice_spec)
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
    slice_spec = [slicer.SingleSliceSpec(columns=['language'])]
    eval_result = model_eval_lib.run_model_analysis(
        model_eval_lib.default_eval_shared_model(
            eval_saved_model_path=model_location, example_weight_key='age'),
        data_location,
        slice_spec=slice_spec,
        k_anonymization_count=2)
    # We only check some of the metrics to ensure that the end-to-end
    # pipeline works.
    expected = {
        (('language', b'hindi'),): {
            u'__ERROR__': {
                'debugMessage':
                    u'Example count for this slice key is lower than the '
                    u'minimum required value: 2. No data is aggregated for '
                    u'this slice.'
            },
        },
        (('language', b'chinese'),): {
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
        (('language', b'english'),): {
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
    self.assertEqual(eval_result.config.model_location, model_location)
    self.assertEqual(eval_result.config.data_location, data_location)
    self.assertEqual(eval_result.config.slice_spec, slice_spec)
    self.assertMetricsAlmostEqual(eval_result.slicing_metrics, expected)
    self.assertFalse(eval_result.plots)

  def testRunModelAnalysisWithQueryExtractor(self):
    model_location = self._exportEvalSavedModel(
        linear_classifier.simple_linear_classifier)
    examples = [
        self._makeExample(age=3.0, language='english', label=1.0),
        self._makeExample(age=3.0, language='chinese', label=0.0),
        self._makeExample(age=4.0, language='english', label=0.0),
        self._makeExample(age=5.0, language='chinese', label=1.0)
    ]
    data_location = self._writeTFExamplesToTFRecords(examples)
    slice_spec = [slicer.SingleSliceSpec()]
    eval_shared_model = model_eval_lib.default_eval_shared_model(
        eval_saved_model_path=model_location, example_weight_key='age')
    eval_result = model_eval_lib.run_model_analysis(
        eval_shared_model=eval_shared_model,
        data_location=data_location,
        slice_spec=slice_spec,
        evaluators=[
            metrics_and_plots_evaluator.MetricsAndPlotsEvaluator(
                eval_shared_model),
            query_based_metrics_evaluator.QueryBasedMetricsEvaluator(
                query_id='language',
                prediction_key='logistic',
                combine_fns=[
                    query_statistics.QueryStatisticsCombineFn(),
                    ndcg.NdcgMetricCombineFn(
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
    self.assertEqual(eval_result.config.model_location, model_location)
    self.assertEqual(eval_result.config.data_location, data_location)
    self.assertEqual(eval_result.config.slice_spec, slice_spec)
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
    slice_spec = [slicer.SingleSliceSpec(columns=['language'])]
    eval_result = model_eval_lib.run_model_analysis(
        model_eval_lib.default_eval_shared_model(
            eval_saved_model_path=model_location, example_weight_key='age'),
        data_location,
        slice_spec=slice_spec,
        compute_confidence_intervals=True,
        k_anonymization_count=2)

    # We only check some of the metrics to ensure that the end-to-end
    # pipeline works.
    expected = {
        (('language', b'hindi'),): {
            u'__ERROR__': {
                'debugMessage':
                    u'Example count for this slice key is lower than the '
                    u'minimum required value: 2. No data is aggregated for '
                    u'this slice.'
            },
        },
        (('language', b'chinese'),): {
            metric_keys.EXAMPLE_WEIGHT: {
                'doubleValue': 8.0
            },
            metric_keys.EXAMPLE_COUNT: {
                'doubleValue': 2.0
            },
        },
        (('language', b'english'),): {
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
    self.assertEqual(eval_result.config.model_location, model_location)
    self.assertEqual(eval_result.config.data_location, data_location)
    self.assertEqual(eval_result.config.slice_spec, slice_spec)
    self.assertMetricsAlmostEqual(eval_result.slicing_metrics, expected)
    self.assertFalse(eval_result.plots)

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
    eval_shared_model = model_eval_lib.default_eval_shared_model(
        eval_saved_model_path=model_location,
        add_metrics_callbacks=[post_export_metrics.auc_plots()])
    data_location = self._writeTFExamplesToTFRecords(examples)
    eval_result = model_eval_lib.run_model_analysis(eval_shared_model,
                                                    data_location)
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
    self.assertEqual(len(eval_result.plots), 1)
    slice_key, plots = eval_result.plots[0]
    self.assertEqual((), slice_key)
    self.assertDictElementsAlmostEqual(
        plots['post_export_metrics']['confusionMatrixAtThresholds']['matrices']
        [8001], expected_matrix)

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
    eval_shared_model = model_eval_lib.default_eval_shared_model(
        eval_saved_model_path=model_location,
        add_metrics_callbacks=[
            post_export_metrics.auc_plots(),
            post_export_metrics.auc_plots(metric_tag='test')
        ])
    data_location = self._writeTFExamplesToTFRecords(examples)
    eval_result = model_eval_lib.run_model_analysis(eval_shared_model,
                                                    data_location)
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
    self.assertEqual(len(eval_result.plots), 1)
    slice_key, plots = eval_result.plots[0]
    self.assertEqual((), slice_key)
    tf.compat.v1.logging.info(plots.keys())
    self.assertDictElementsAlmostEqual(
        plots['post_export_metrics']['confusionMatrixAtThresholds']['matrices']
        [8001], expected_matrix)
    self.assertDictElementsAlmostEqual(
        plots['post_export_metrics/test']['confusionMatrixAtThresholds']
        ['matrices'][8001], expected_matrix)

  def testRunModelAnalysisForCSVText(self):
    model_location = self._exportEvalSavedModel(
        csv_linear_classifier.simple_csv_linear_classifier)
    examples = [
        '3.0,english,1.0', '3.0,chinese,0.0', '4.0,english,1.0',
        '5.0,chinese,1.0'
    ]
    data_location = self._writeCSVToTextFile(examples)
    eval_result = model_eval_lib.run_model_analysis(
        model_eval_lib.default_eval_shared_model(
            eval_saved_model_path=model_location),
        data_location,
        file_format='text')
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
    self.assertEqual(2, len(eval_results._results))
    expected_result_1 = {
        (('language', b'english'),): {
            'my_mean_label': {
                'doubleValue': 1.0
            },
            metric_keys.EXAMPLE_COUNT: {
                'doubleValue': 2.0
            },
        }
    }
    expected_result_2 = {
        (('language', b'english'),): {
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
    self.assertEqual(2, len(eval_results._results))
    # We only check some of the metrics to ensure that the end-to-end
    # pipeline works.
    expected_result_1 = {
        (('language', b'english'),): {
            metric_keys.EXAMPLE_COUNT: {
                'doubleValue': 2.0
            },
        }
    }
    expected_result_2 = {
        (('language', b'english'),): {
            metric_keys.EXAMPLE_COUNT: {
                'doubleValue': 1.0
            },
        }
    }
    self.assertMetricsAlmostEqual(eval_results._results[0].slicing_metrics,
                                  expected_result_1)
    self.assertMetricsAlmostEqual(eval_results._results[1].slicing_metrics,
                                  expected_result_2)

  def testSerializeDeserializeEvalConfig(self):
    eval_config = model_eval_lib.EvalConfig(
        model_location='/path/to/model',
        data_location='/path/to/data',
        slice_spec=[
            slicer.SingleSliceSpec(
                features=[('age', 5), ('gender', 'f')], columns=['country']),
            slicer.SingleSliceSpec(
                features=[('age', 6), ('gender', 'm')], columns=['interest'])
        ],
        example_weight_metric_key='key')
    serialized = model_eval_lib._serialize_eval_config(eval_config)
    deserialized = pickle.loads(serialized)
    got_eval_config = deserialized[model_eval_lib._EVAL_CONFIG_KEY]
    self.assertEqual(eval_config, got_eval_config)


if __name__ == '__main__':
  tf.test.main()
