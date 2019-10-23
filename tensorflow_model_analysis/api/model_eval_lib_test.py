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
from tensorflow_model_analysis import config
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis import version as tfma_version
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
from tensorflow_model_analysis.slicer import slicer_lib as slicer
from typing import Dict, List, NamedTuple, Optional, Text, Union

if tf.__version__[0] == '1':
  tf.compat.v1.enable_v2_behavior()

LegacyConfig = NamedTuple(
    'LegacyConfig',
    [('model_location', Text), ('data_location', Text),
     ('slice_spec', Optional[List[slicer.SingleSliceSpec]]),
     ('example_count_metric_key', Text),
     ('example_weight_metric_key', Union[Text, Dict[Text, Text]]),
     ('compute_confidence_intervals', bool), ('k_anonymization_count', int)])


class EvaluateTest(testutil.TensorflowModelAnalysisTest):

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
    eval_config = config.EvalConfig(
        input_data_specs=[config.InputDataSpec(location=data_location)],
        model_specs=[config.ModelSpec(location=model_location)],
        output_data_specs=[
            config.OutputDataSpec(default_location=self._getTempDir())
        ])
    # No construct_fn should fail when Beam attempts to call the construct_fn.
    eval_shared_model = types.EvalSharedModel(model_path=model_location)
    with self.assertRaisesRegexp(AttributeError,
                                 '\'NoneType\' object has no attribute'):
      model_eval_lib.run_model_analysis(
          eval_config=eval_config, eval_shared_models=[eval_shared_model])

    # Using the default_eval_shared_model should pass as it has a construct_fn.
    eval_shared_model = model_eval_lib.default_eval_shared_model(
        eval_saved_model_path=model_location)
    model_eval_lib.run_model_analysis(
        eval_config=eval_config, eval_shared_models=[eval_shared_model])

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
    eval_config = config.EvalConfig(
        input_data_specs=[config.InputDataSpec(location=data_location)],
        model_specs=[config.ModelSpec(location=model_location)],
        output_data_specs=[
            config.OutputDataSpec(default_location=self._getTempDir())
        ],
        slicing_specs=slicing_specs)
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
        eval_shared_models=[
            model_eval_lib.default_eval_shared_model(
                eval_saved_model_path=model_location, example_weight_key='age')
        ],
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
    self.assertEqual(eval_result.config.model_specs[0].location,
                     model_location.decode())
    self.assertEqual(eval_result.config.input_data_specs[0].location,
                     data_location)
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
    options.k_anonymization_count.value = 2
    eval_config = config.EvalConfig(
        input_data_specs=[config.InputDataSpec(location=data_location)],
        model_specs=[config.ModelSpec(location=model_location)],
        output_data_specs=[
            config.OutputDataSpec(default_location=self._getTempDir())
        ],
        slicing_specs=slicing_specs,
        options=options)
    eval_result = model_eval_lib.run_model_analysis(
        eval_config=eval_config,
        eval_shared_models=[
            model_eval_lib.default_eval_shared_model(
                eval_saved_model_path=model_location, example_weight_key='age')
        ])
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
    self.assertEqual(eval_result.config.model_specs[0].location,
                     model_location.decode())
    self.assertEqual(eval_result.config.input_data_specs[0].location,
                     data_location)
    self.assertEqual(eval_result.config.slicing_specs[0],
                     config.SlicingSpec(feature_keys=['language']))
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
    slicing_specs = [config.SlicingSpec()]
    eval_config = config.EvalConfig(
        input_data_specs=[config.InputDataSpec(location=data_location)],
        model_specs=[config.ModelSpec(location=model_location)],
        output_data_specs=[
            config.OutputDataSpec(default_location=self._getTempDir())
        ],
        slicing_specs=slicing_specs)
    eval_shared_model = model_eval_lib.default_eval_shared_model(
        eval_saved_model_path=model_location, example_weight_key='age')
    eval_result = model_eval_lib.run_model_analysis(
        eval_config=eval_config,
        eval_shared_models=[eval_shared_model],
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
    self.assertEqual(eval_result.config.model_specs[0].location,
                     model_location.decode())
    self.assertEqual(eval_result.config.input_data_specs[0].location,
                     data_location)
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
    options.k_anonymization_count.value = 2
    eval_config = config.EvalConfig(
        input_data_specs=[config.InputDataSpec(location=data_location)],
        model_specs=[config.ModelSpec(location=model_location)],
        output_data_specs=[
            config.OutputDataSpec(default_location=self._getTempDir())
        ],
        slicing_specs=slicing_specs,
        options=options)
    eval_result = model_eval_lib.run_model_analysis(
        eval_config=eval_config,
        eval_shared_models=[
            model_eval_lib.default_eval_shared_model(
                eval_saved_model_path=model_location, example_weight_key='age')
        ])
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
    self.assertEqual(eval_result.config.model_specs[0].location,
                     model_location.decode())
    self.assertEqual(eval_result.config.input_data_specs[0].location,
                     data_location)
    self.assertEqual(eval_result.config.slicing_specs[0],
                     config.SlicingSpec(feature_keys=['language']))
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
    data_location = self._writeTFExamplesToTFRecords(examples)
    eval_config = config.EvalConfig(
        input_data_specs=[config.InputDataSpec(location=data_location)],
        model_specs=[config.ModelSpec(location=model_location)],
        output_data_specs=[
            config.OutputDataSpec(default_location=self._getTempDir())
        ])
    eval_shared_model = model_eval_lib.default_eval_shared_model(
        eval_saved_model_path=model_location,
        add_metrics_callbacks=[post_export_metrics.auc_plots()])
    eval_result = model_eval_lib.run_model_analysis(
        eval_config=eval_config, eval_shared_models=[eval_shared_model])
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
    eval_config = config.EvalConfig(
        input_data_specs=[config.InputDataSpec(location=data_location)],
        model_specs=[config.ModelSpec(location=model_location)],
        output_data_specs=[
            config.OutputDataSpec(default_location=self._getTempDir())
        ])
    eval_shared_model = model_eval_lib.default_eval_shared_model(
        eval_saved_model_path=model_location,
        add_metrics_callbacks=[
            post_export_metrics.auc_plots(),
            post_export_metrics.auc_plots(metric_tag='test')
        ])
    eval_result = model_eval_lib.run_model_analysis(
        eval_config=eval_config, eval_shared_models=[eval_shared_model])

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
    eval_config = config.EvalConfig(
        input_data_specs=[
            config.InputDataSpec(location=data_location, file_format='text')
        ],
        model_specs=[config.ModelSpec(location=model_location)],
        output_data_specs=[
            config.OutputDataSpec(default_location=self._getTempDir())
        ])
    eval_result = model_eval_lib.run_model_analysis(
        eval_config=eval_config,
        eval_shared_models=[
            model_eval_lib.default_eval_shared_model(
                eval_saved_model_path=model_location)
        ])
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
    self.assertEqual(2, len(eval_results._results))
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

  def testSerializeDeserializeLegacyEvalConfig(self):
    output_path = self._getTempDir()
    old_config = LegacyConfig(
        model_location='/path/to/model',
        data_location='/path/to/data',
        slice_spec=[
            slicer.SingleSliceSpec(
                columns=['country'], features=[('age', 5), ('gender', 'f')]),
            slicer.SingleSliceSpec(
                columns=['interest'], features=[('age', 6), ('gender', 'm')])
        ],
        example_count_metric_key=None,
        example_weight_metric_key='key',
        compute_confidence_intervals=False,
        k_anonymization_count=1)
    final_dict = {}
    final_dict['tfma_version'] = tfma_version.VERSION_STRING
    final_dict['eval_config'] = old_config
    with tf.io.TFRecordWriter(os.path.join(output_path, 'eval_config')) as w:
      w.write(pickle.dumps(final_dict))
    got_eval_config = model_eval_lib.load_eval_config(output_path)
    options = config.Options()
    options.compute_confidence_intervals.value = (
        old_config.compute_confidence_intervals)
    options.k_anonymization_count.value = old_config.k_anonymization_count
    eval_config = config.EvalConfig(
        input_data_specs=[
            config.InputDataSpec(location=old_config.data_location)
        ],
        model_specs=[config.ModelSpec(location=old_config.model_location)],
        output_data_specs=[config.OutputDataSpec(default_location=output_path)],
        slicing_specs=[
            config.SlicingSpec(
                feature_keys=['country'],
                feature_values={
                    'age': '5',
                    'gender': 'f'
                }),
            config.SlicingSpec(
                feature_keys=['interest'],
                feature_values={
                    'age': '6',
                    'gender': 'm'
                })
        ],
        options=options)
    self.assertEqual(eval_config, got_eval_config)

  def testSerializeDeserializeEvalConfig(self):
    output_path = self._getTempDir()
    options = config.Options()
    options.compute_confidence_intervals.value = False
    options.k_anonymization_count.value = 1
    eval_config = config.EvalConfig(
        input_data_specs=[config.InputDataSpec(location='/path/to/data')],
        model_specs=[config.ModelSpec(location='/path/to/model')],
        output_data_specs=[config.OutputDataSpec(default_location=output_path)],
        slicing_specs=[
            config.SlicingSpec(
                feature_keys=['country'],
                feature_values={
                    'age': '5',
                    'gender': 'f'
                }),
            config.SlicingSpec(
                feature_keys=['interest'],
                feature_values={
                    'age': '6',
                    'gender': 'm'
                })
        ],
        options=options)
    with tf.io.gfile.GFile(os.path.join(output_path, 'eval_config.json'),
                           'w') as f:
      f.write(model_eval_lib._serialize_eval_config(eval_config))
    got_eval_config = model_eval_lib.load_eval_config(output_path)
    self.assertEqual(eval_config, got_eval_config)


if __name__ == '__main__':
  tf.test.main()
