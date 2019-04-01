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

import apache_beam as beam
import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.eval_saved_model.example_trainers import csv_linear_classifier
from tensorflow_model_analysis.eval_saved_model.example_trainers import custom_estimator
from tensorflow_model_analysis.eval_saved_model.example_trainers import fixed_prediction_estimator
from tensorflow_model_analysis.eval_saved_model.example_trainers import linear_classifier
from tensorflow_model_analysis.evaluators import metrics_and_plots_evaluator
from tensorflow_model_analysis.post_export_metrics import metric_keys
from tensorflow_model_analysis.post_export_metrics import post_export_metrics
from tensorflow_model_analysis.proto import metrics_for_slice_pb2
from tensorflow_model_analysis.slicer import slicer

from google.protobuf import text_format


def _make_slice_key(*args):
  if len(args) % 2 != 0:
    raise ValueError('number of arguments should be even')

  result = []
  for i in range(0, len(args), 2):
    result.append((args[i], args[i + 1]))

  return tuple(result)


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
    with tf.python_io.TFRecordWriter(data_location) as writer:
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
    for (s, m) in got_value:
      self.assertIn(s, expected_value)
      for k in expected_value[s]:
        self.assertIn(k, m)
        self.assertDictElementsAlmostEqual(m[k], expected_value[s][k])

  def assertSliceMetricsAlmostEqual(self, expected_metrics, got_metrics):
    self.assertItemsEqual(
        list(expected_metrics.keys()),
        list(got_metrics.keys()),
        msg='keys do not match. expected_metrics: %s, got_metrics: %s' %
        (expected_metrics, got_metrics))
    for key in expected_metrics.keys():
      self.assertAlmostEqual(
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
                              self.assertSliceMetricsAlmostEqual)

  def testRunModelAnalysis(self):
    model_location = self._exportEvalSavedModel(
        linear_classifier.simple_linear_classifier)
    examples = [
        self._makeExample(age=3.0, language='english', label=1.0),
        self._makeExample(age=3.0, language='chinese', label=0.0),
        self._makeExample(age=4.0, language='english', label=1.0),
        self._makeExample(age=5.0, language='chinese', label=1.0)
    ]
    data_location = self._writeTFExamplesToTFRecords(examples)
    slice_spec = [slicer.SingleSliceSpec(columns=['language'])]
    eval_result = model_eval_lib.run_model_analysis(
        model_eval_lib.default_eval_shared_model(
            eval_saved_model_path=model_location, example_weight_key='age'),
        data_location,
        slice_spec=slice_spec)
    # We only check some of the metrics to ensure that the end-to-end
    # pipeline works.
    expected = {
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

  def testRunModelAnalysisWithUncertainty(self):
    model_location = self._exportEvalSavedModel(
        linear_classifier.simple_linear_classifier)
    examples = [
        self._makeExample(age=3.0, language='english', label=1.0),
        self._makeExample(age=3.0, language='chinese', label=0.0),
        self._makeExample(age=4.0, language='english', label=1.0),
        self._makeExample(age=5.0, language='chinese', label=1.0)
    ]
    data_location = self._writeTFExamplesToTFRecords(examples)
    slice_spec = [slicer.SingleSliceSpec(columns=['language'])]
    eval_result = model_eval_lib.run_model_analysis(
        model_eval_lib.default_eval_shared_model(
            eval_saved_model_path=model_location, example_weight_key='age'),
        data_location,
        slice_spec=slice_spec,
        num_bootstrap_samples=20)
    # We only check some of the metrics to ensure that the end-to-end
    # pipeline works.
    expected = {
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
    tf.logging.info(plots.keys())
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
        custom_estimator.simple_custom_estimator)
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
            'mean_label': {
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

  def testSerializeDeserializeToFile(self):
    metrics_slice_key = _make_slice_key('fruit', b'pear', 'animal', b'duck')
    metrics_for_slice = text_format.Parse(
        """
        slice_key {
          single_slice_keys {
            column: "fruit"
            bytes_value: "pear"
          }
          single_slice_keys {
            column: "animal"
            bytes_value: "duck"
          }
        }
        metrics {
          key: "accuracy"
          value {
            double_value {
              value: 0.8
            }
          }
        }
        metrics {
          key: "example_weight"
          value {
            double_value {
              value: 10.0
            }
          }
        }
        metrics {
          key: "auc"
          value {
            bounded_value {
              lower_bound {
                value: 0.1
              }
              upper_bound {
                value: 0.3
              }
              value {
                value: 0.2
              }
            }
          }
        }
        metrics {
          key: "auprc"
          value {
            bounded_value {
              lower_bound {
                value: 0.05
              }
              upper_bound {
                value: 0.17
              }
              value {
                value: 0.1
              }
            }
          }
        }""", metrics_for_slice_pb2.MetricsForSlice())
    plots_for_slice = text_format.Parse(
        """
        slice_key {
          single_slice_keys {
            column: "fruit"
            bytes_value: "peach"
          }
          single_slice_keys {
            column: "animal"
            bytes_value: "cow"
          }
        }
        plots {
          key: ''
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
        }""", metrics_for_slice_pb2.PlotsForSlice())
    plots_slice_key = _make_slice_key('fruit', b'peach', 'animal', b'cow')
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

    output_path = self._getTempDir()
    with beam.Pipeline() as pipeline:
      metrics = (
          pipeline
          | 'CreateMetrics' >> beam.Create(
              [metrics_for_slice.SerializeToString()]))
      plots = (
          pipeline
          | 'CreatePlots' >> beam.Create([plots_for_slice.SerializeToString()]))
      evaluation = {constants.METRICS_KEY: metrics, constants.PLOTS_KEY: plots}
      _ = (
          evaluation
          | 'WriteResults' >> model_eval_lib.WriteResults(
              writers=model_eval_lib.default_writers(output_path=output_path)))
      _ = pipeline | model_eval_lib.WriteEvalConfig(eval_config, output_path)

    metrics = metrics_and_plots_evaluator.load_and_deserialize_metrics(
        path=os.path.join(output_path, model_eval_lib._METRICS_OUTPUT_FILE))
    plots = metrics_and_plots_evaluator.load_and_deserialize_plots(
        path=os.path.join(output_path, model_eval_lib._PLOTS_OUTPUT_FILE))
    self.assertSliceMetricsListEqual(
        [(metrics_slice_key, metrics_for_slice.metrics)], metrics)
    self.assertSlicePlotsListEqual([(plots_slice_key, plots_for_slice.plots)],
                                   plots)
    got_eval_config = model_eval_lib.load_eval_config(output_path)
    self.assertEqual(eval_config, got_eval_config)


if __name__ == '__main__':
  tf.test.main()
