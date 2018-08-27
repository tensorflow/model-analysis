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
"""Test for post export metrics.

Note that we actually train and export models within these tests.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis.api.impl import evaluate
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.eval_saved_model.example_trainers import dnn_classifier
from tensorflow_model_analysis.eval_saved_model.example_trainers import dnn_regressor
from tensorflow_model_analysis.eval_saved_model.example_trainers import fixed_prediction_classifier
from tensorflow_model_analysis.eval_saved_model.example_trainers import fixed_prediction_classifier_extra_fields
from tensorflow_model_analysis.eval_saved_model.example_trainers import fixed_prediction_estimator
from tensorflow_model_analysis.eval_saved_model.example_trainers import fixed_prediction_estimator_extra_fields
from tensorflow_model_analysis.eval_saved_model.example_trainers import linear_classifier
from tensorflow_model_analysis.eval_saved_model.example_trainers import linear_regressor
from tensorflow_model_analysis.eval_saved_model.post_export_metrics import post_export_metrics
import tensorflow_model_analysis.eval_saved_model.post_export_metrics.metric_keys as metric_keys
from tensorflow_model_analysis.proto import metrics_for_slice_pb2


class PostExportMetricsTest(testutil.TensorflowModelAnalysisTest):

  def _getEvalExportDir(self):
    return os.path.join(self._getTempDir(), 'eval_export_dir')

  def _runTestWithCustomCheck(self,
                              examples,
                              eval_export_dir,
                              metrics,
                              custom_metrics_check=None,
                              custom_plots_check=None):
    # make sure we are doing some checks
    self.assertTrue(custom_metrics_check is not None or
                    custom_plots_check is not None)
    serialized_examples = [ex.SerializeToString() for ex in examples]
    with beam.Pipeline() as pipeline:
      metrics, plots = (
          pipeline
          | beam.Create(serialized_examples)
          | evaluate.Evaluate(
              eval_saved_model_path=eval_export_dir,
              add_metrics_callbacks=metrics))
      if custom_metrics_check is not None:
        util.assert_that(metrics, custom_metrics_check, label='metrics')
      if custom_plots_check is not None:
        util.assert_that(plots, custom_plots_check, label='plot')

  def _runTest(self, examples, eval_export_dir, metrics, expected_values_dict):

    def check_result(got):  # pylint: disable=invalid-name
      try:
        self.assertEqual(1, len(got), 'got: %s' % got)
        (slice_key, value) = got[0]
        self.assertEqual((), slice_key)
        self.assertDictElementsAlmostEqual(value, expected_values_dict)
      except AssertionError as err:
        raise util.BeamAssertException(err)

    self._runTestWithCustomCheck(
        examples, eval_export_dir, metrics, custom_metrics_check=check_result)

  def testExampleCountNoStandardKeys(self):
    # Test ExampleCount with a custom Estimator that doesn't have any of the
    # standard PredictionKeys.
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        fixed_prediction_estimator.simple_fixed_prediction_estimator(
            None, temp_eval_export_dir, output_prediction_key='non_standard'))
    examples = [
        self._makeExample(prediction=5.0, label=5.0),
        self._makeExample(prediction=6.0, label=6.0),
        self._makeExample(prediction=7.0, label=7.0),
    ]
    expected_values_dict = {
        metric_keys.EXAMPLE_COUNT: 3.0,
    }
    self._runTest(examples, eval_export_dir, [
        post_export_metrics.example_count(),
    ], expected_values_dict)

  def testExampleCountEmptyPredictionsDict(self):
    # Test ExampleCount with a custom Estimator that has empty predictions dict.
    # This is possible if the Estimator doesn't return the predictions dict
    # in EVAL mode, but computes predictions and feeds them into the metrics
    # internally.
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        fixed_prediction_estimator.simple_fixed_prediction_estimator(
            None, temp_eval_export_dir, output_prediction_key=None))
    examples = [
        self._makeExample(prediction=5.0, label=5.0),
        self._makeExample(prediction=6.0, label=6.0),
        self._makeExample(prediction=7.0, label=7.0),
    ]
    expected_values_dict = {
        metric_keys.EXAMPLE_COUNT: 3.0,
    }
    self._runTest(examples, eval_export_dir, [
        post_export_metrics.example_count(),
    ], expected_values_dict)

  def testPostExportMetricsLinearClassifier(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = linear_classifier.simple_linear_classifier(
        None, temp_eval_export_dir)
    examples = [
        self._makeExample(age=3.0, language='english', label=1.0),
        self._makeExample(age=3.0, language='chinese', label=0.0),
        self._makeExample(age=4.0, language='english', label=1.0),
        self._makeExample(age=5.0, language='chinese', label=0.0)
    ]
    expected_values_dict = {
        metric_keys.EXAMPLE_COUNT: 4.0,
        metric_keys.EXAMPLE_WEIGHT: 15.0,
    }
    self._runTest(examples, eval_export_dir, [
        post_export_metrics.example_count(),
        post_export_metrics.example_weight('age')
    ], expected_values_dict)

  def testPostExportMetricsDNNClassifier(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = dnn_classifier.simple_dnn_classifier(
        None, temp_eval_export_dir)
    examples = [
        self._makeExample(age=3.0, language='english', label=1.0),
        self._makeExample(age=3.0, language='chinese', label=0.0),
        self._makeExample(age=4.0, language='english', label=1.0),
        self._makeExample(age=5.0, language='chinese', label=0.0)
    ]
    expected_values_dict = {
        metric_keys.EXAMPLE_COUNT: 4.0,
        metric_keys.EXAMPLE_WEIGHT: 15.0,
    }
    self._runTest(examples, eval_export_dir, [
        post_export_metrics.example_count(),
        post_export_metrics.example_weight('age')
    ], expected_values_dict)

  def testPostExportMetricsLinearRegressor(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = linear_regressor.simple_linear_regressor(
        None, temp_eval_export_dir)
    examples = [
        self._makeExample(age=3.0, language='english', label=1.0),
        self._makeExample(age=3.0, language='chinese', label=0.0),
        self._makeExample(age=4.0, language='english', label=1.0),
        self._makeExample(age=5.0, language='chinese', label=0.0)
    ]
    expected_values_dict = {
        metric_keys.EXAMPLE_COUNT: 4.0,
        metric_keys.EXAMPLE_WEIGHT: 15.0,
    }
    self._runTest(examples, eval_export_dir, [
        post_export_metrics.example_count(),
        post_export_metrics.example_weight('age')
    ], expected_values_dict)

  def testPostExportMetricsDNNRegressor(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = dnn_regressor.simple_dnn_regressor(
        None, temp_eval_export_dir)
    examples = [
        self._makeExample(age=3.0, language='english', label=1.0),
        self._makeExample(age=3.0, language='chinese', label=0.0),
        self._makeExample(age=4.0, language='english', label=1.0),
        self._makeExample(age=5.0, language='chinese', label=0.0)
    ]
    expected_values_dict = {
        metric_keys.EXAMPLE_COUNT: 4.0,
        metric_keys.EXAMPLE_WEIGHT: 15.0,
    }
    self._runTest(examples, eval_export_dir, [
        post_export_metrics.example_count(),
        post_export_metrics.example_weight('age')
    ], expected_values_dict)

  def testPrecisionRecallAtKUnweighted(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        fixed_prediction_classifier.simple_fixed_prediction_classifier(
            None, temp_eval_export_dir))
    examples = [
        self._makeExample(
            classes=['a', 'b', 'c'],
            scores=[0.9, 0.8, 0.7],
            labels=['a', 'c'],
            fixed_float=1.0,
            fixed_string=''),
        self._makeExample(
            classes=['a', 'b', 'c'],
            scores=[0.9, 0.2, 0.1],
            labels=['a'],
            fixed_float=2.0,
            fixed_string=''),
        self._makeExample(
            classes=['a', 'b', 'c'],
            scores=[0.1, 0.2, 0.9],
            labels=['a'],
            fixed_float=3.0,
            fixed_string=''),
    ]

    precision_recall_metric = post_export_metrics.precision_recall_at_k(
        [0, 1, 2, 3, 5])

    def check_result(got):  # pylint: disable=invalid-name
      try:
        self.assertEqual(1, len(got), 'got: %s' % got)
        (slice_key, value) = got[0]
        self.assertEqual((), slice_key)
        self.assertIn(metric_keys.PRECISION_RECALL_AT_K, value)
        table = value[metric_keys.PRECISION_RECALL_AT_K]
        cutoffs = table[:, 0].tolist()
        precision = table[:, 1].tolist()
        recall = table[:, 2].tolist()

        self.assertEqual(cutoffs, [0, 1, 2, 3, 5])
        self.assertSequenceAlmostEqual(
            precision, [4.0 / 9.0, 2.0 / 3.0, 2.0 / 6.0, 4.0 / 9.0, 4.0 / 9.0])
        self.assertSequenceAlmostEqual(
            recall, [4.0 / 4.0, 2.0 / 4.0, 2.0 / 4.0, 4.0 / 4.0, 4.0 / 4.0])

        # Check serialization too.
        # Note that we can't just make this a dict, since proto maps
        # allow uninitialized key access, i.e. they act like defaultdicts.
        output_metrics = metrics_for_slice_pb2.MetricsForSlice().metrics
        precision_recall_metric.populate_stats_and_pop(value, output_metrics)
        self.assertProtoEquals(
            """
            value_at_cutoffs {
              values {
                cutoff: 0
                value: 0.44444444
              }
            }
            value_at_cutoffs {
              values {
                cutoff: 1
                value: 0.66666666
              }
            }
            value_at_cutoffs {
              values {
                cutoff: 2
                value: 0.33333333
              }
            }
            value_at_cutoffs {
              values {
                cutoff: 3
                value: 0.44444444
              }
            }
            value_at_cutoffs {
              values {
                cutoff: 5
                value: 0.44444444
              }
            }
            """, output_metrics[metric_keys.PRECISION_AT_K])
        self.assertProtoEquals(
            """
            value_at_cutoffs {
              values {
                cutoff: 0
                value: 1.0
              }
            }
            value_at_cutoffs {
              values {
                cutoff: 1
                value: 0.5
              }
            }
            value_at_cutoffs {
              values {
                cutoff: 2
                value: 0.5
              }
            }
            value_at_cutoffs {
              values {
                cutoff: 3
                value: 1.0
              }
            }
            value_at_cutoffs {
              values {
                cutoff: 5
                value: 1.0
              }
            }
            """, output_metrics[metric_keys.RECALL_AT_K])
      except AssertionError as err:
        raise util.BeamAssertException(err)

    self._runTestWithCustomCheck(
        examples,
        eval_export_dir, [precision_recall_metric],
        custom_metrics_check=check_result)

  def testPrecisionRecallAtKWeighted(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        fixed_prediction_classifier_extra_fields.
        simple_fixed_prediction_classifier_extra_fields(None,
                                                        temp_eval_export_dir))
    examples = [
        self._makeExample(
            classes=['a', 'b', 'c'],
            scores=[0.9, 0.8, 0.7],
            labels=['a', 'c'],
            fixed_float=1.0,
            fixed_string='',
            fixed_int=0),
        self._makeExample(
            classes=['a', 'b', 'c'],
            scores=[0.9, 0.2, 0.1],
            labels=['a'],
            fixed_float=2.0,
            fixed_string='',
            fixed_int=0),
        self._makeExample(
            classes=['a', 'b', 'c'],
            scores=[0.1, 0.2, 0.9],
            labels=['a'],
            fixed_float=3.0,
            fixed_string='',
            fixed_int=0),
    ]

    def check_result(got):  # pylint: disable=invalid-name
      try:
        self.assertEqual(1, len(got), 'got: %s' % got)
        (slice_key, value) = got[0]
        self.assertEqual((), slice_key)
        self.assertIn(metric_keys.PRECISION_RECALL_AT_K, value)
        table = value[metric_keys.PRECISION_RECALL_AT_K]
        cutoffs = table[:, 0].tolist()
        precision = table[:, 1].tolist()
        recall = table[:, 2].tolist()

        self.assertEqual(cutoffs, [1, 3])
        self.assertSequenceAlmostEqual(precision, [3.0 / 6.0, 7.0 / 18.0])
        self.assertSequenceAlmostEqual(recall, [3.0 / 7.0, 7.0 / 7.0])
      except AssertionError as err:
        raise util.BeamAssertException(err)

    self._runTestWithCustomCheck(
        examples,
        eval_export_dir, [
            post_export_metrics.precision_recall_at_k(
                [1, 3], example_weight_key='fixed_float')
        ],
        custom_metrics_check=check_result)

  def testPrecisionRecallAtKEmptyCutoffs(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        fixed_prediction_classifier_extra_fields.
        simple_fixed_prediction_classifier_extra_fields(None,
                                                        temp_eval_export_dir))
    examples = [
        self._makeExample(
            classes=['a', 'b', 'c'],
            scores=[0.9, 0.8, 0.7],
            labels=['a', 'c'],
            fixed_float=1.0,
            fixed_string='',
            fixed_int=0),
    ]

    def check_result(got):  # pylint: disable=invalid-name
      try:
        self.assertEqual(1, len(got), 'got: %s' % got)
        (slice_key, value) = got[0]
        self.assertEqual((), slice_key)
        self.assertIn(metric_keys.PRECISION_RECALL_AT_K, value)
        table = value[metric_keys.PRECISION_RECALL_AT_K]
        cutoffs = table[:, 0].tolist()
        precision = table[:, 1].tolist()
        recall = table[:, 2].tolist()

        self.assertEqual(cutoffs, [])
        self.assertSequenceAlmostEqual(precision, [])
        self.assertSequenceAlmostEqual(recall, [])
      except AssertionError as err:
        raise util.BeamAssertException(err)

    self._runTestWithCustomCheck(
        examples,
        eval_export_dir, [post_export_metrics.precision_recall_at_k([])],
        custom_metrics_check=check_result)

  def testCalibrationPlotAndPredictionHistogramUnweighted(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        fixed_prediction_estimator.simple_fixed_prediction_estimator(
            None, temp_eval_export_dir))
    examples = [
        # For each example, we set label to prediction + 1.
        # These two go in bucket 0: (-inf, 0)
        self._makeExample(prediction=-10.0, label=-9.0),
        self._makeExample(prediction=-9.0, label=-8.0),
        # This goes in bucket 1: [0, 0.00100)
        self._makeExample(prediction=0.00000, label=1.00000),
        # These three go in bucket 1: [0.00100, 0.00110)
        self._makeExample(prediction=0.00100, label=1.00100),
        self._makeExample(prediction=0.00101, label=1.00101),
        self._makeExample(prediction=0.00102, label=1.00102),
        # These two go in bucket 10000: [0.99990, 1.00000)
        self._makeExample(prediction=0.99998, label=1.99998),
        self._makeExample(prediction=0.99999, label=1.99999),
        # These four go in bucket 10001: [1.0000, +inf)
        self._makeExample(prediction=1.0, label=2.0),
        self._makeExample(prediction=8.0, label=9.0),
        self._makeExample(prediction=9.0, label=10.0),
        self._makeExample(prediction=10.0, label=11.0),
    ]

    calibration_plot = (
        post_export_metrics.calibration_plot_and_prediction_histogram())

    def check_result(got):  # pylint: disable=invalid-name
      try:
        self.assertEqual(1, len(got), 'got: %s' % got)
        (slice_key, value) = got[0]
        self.assertEqual((), slice_key)
        self.assertIn(metric_keys.CALIBRATION_PLOT_MATRICES, value)
        buckets = value[metric_keys.CALIBRATION_PLOT_MATRICES]
        self.assertSequenceAlmostEqual(buckets[0], [-19.0, -17.0, 2.0])
        self.assertSequenceAlmostEqual(buckets[1], [0.0, 1.0, 1.0])
        self.assertSequenceAlmostEqual(buckets[11], [0.00303, 3.00303, 3.0])
        self.assertSequenceAlmostEqual(buckets[10000], [1.99997, 3.99997, 2.0])
        self.assertSequenceAlmostEqual(buckets[10001], [28.0, 32.0, 4.0])
        self.assertIn(metric_keys.CALIBRATION_PLOT_BOUNDARIES, value)
        boundaries = value[metric_keys.CALIBRATION_PLOT_BOUNDARIES]
        self.assertAlmostEqual(0.0, boundaries[0])
        self.assertAlmostEqual(0.001, boundaries[10])
        self.assertAlmostEqual(0.005, boundaries[50])
        self.assertAlmostEqual(0.010, boundaries[100])
        self.assertAlmostEqual(0.100, boundaries[1000])
        self.assertAlmostEqual(0.800, boundaries[8000])
        self.assertAlmostEqual(1.000, boundaries[10000])
        plot_data = metrics_for_slice_pb2.PlotData()
        calibration_plot.populate_plots_and_pop(value, plot_data)
        self.assertProtoEquals(
            """lower_threshold_inclusive:1.0
            upper_threshold_exclusive: inf
            num_weighted_examples {
              value: 4.0
            }
            total_weighted_label {
              value: 32.0
            }
            total_weighted_refined_prediction {
              value: 28.0
            }""", plot_data.calibration_histogram_buckets.buckets[10001])
      except AssertionError as err:
        raise util.BeamAssertException(err)

    self._runTestWithCustomCheck(
        examples,
        eval_export_dir, [calibration_plot],
        custom_plots_check=check_result)

  def testCalibrationPlotAndPredictionHistogramWeighted(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        fixed_prediction_estimator_extra_fields.
        simple_fixed_prediction_estimator_extra_fields(None,
                                                       temp_eval_export_dir))
    examples = [
        # For each example, we set label to prediction + 1.
        self._makeExample(
            prediction=-10.0,
            label=-9.0,
            fixed_float=1.0,
            fixed_string='',
            fixed_int=0),
        self._makeExample(
            prediction=-9.0,
            label=-8.0,
            fixed_float=2.0,
            fixed_string='',
            fixed_int=0),
        self._makeExample(
            prediction=0.0000,
            label=1.0000,
            fixed_float=0.0,
            fixed_string='',
            fixed_int=0),
        self._makeExample(
            prediction=0.00100,
            label=1.00100,
            fixed_float=1.0,
            fixed_string='',
            fixed_int=0),
        self._makeExample(
            prediction=0.00101,
            label=1.00101,
            fixed_float=2.0,
            fixed_string='',
            fixed_int=0),
        self._makeExample(
            prediction=0.00102,
            label=1.00102,
            fixed_float=3.0,
            fixed_string='',
            fixed_int=0),
        self._makeExample(
            prediction=10.0,
            label=11.0,
            fixed_float=7.0,
            fixed_string='',
            fixed_int=0),
    ]

    def check_result(got):  # pylint: disable=invalid-name
      try:
        self.assertEqual(1, len(got), 'got: %s' % got)
        (slice_key, value) = got[0]
        self.assertEqual((), slice_key)
        self.assertIn(metric_keys.CALIBRATION_PLOT_MATRICES, value)
        buckets = value[metric_keys.CALIBRATION_PLOT_MATRICES]
        self.assertSequenceAlmostEqual(buckets[0], [-28.0, -25.0, 3.0])
        self.assertSequenceAlmostEqual(buckets[1], [0.0, 0.0, 0.0])
        self.assertSequenceAlmostEqual(buckets[11], [0.00608, 6.00608, 6.0])
        self.assertSequenceAlmostEqual(buckets[10001], [70.0, 77.0, 7.0])
      except AssertionError as err:
        raise util.BeamAssertException(err)

    self._runTestWithCustomCheck(
        examples,
        eval_export_dir, [
            post_export_metrics.calibration_plot_and_prediction_histogram(
                example_weight_key='fixed_float')
        ],
        custom_plots_check=check_result)

  def testAucPlotsUnweighted(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        fixed_prediction_estimator.simple_fixed_prediction_estimator(
            None, temp_eval_export_dir))
    examples = [
        self._makeExample(prediction=0.0000, label=0.0000),
        self._makeExample(prediction=0.0000, label=1.0000),
        self._makeExample(prediction=0.7000, label=1.0000),
        self._makeExample(prediction=0.8000, label=0.0000),
        self._makeExample(prediction=1.0000, label=1.0000),
    ]

    auc_plots = post_export_metrics.auc_plots()

    def check_result(got):  # pylint: disable=invalid-name
      try:
        self.assertEqual(1, len(got), 'got: %s' % got)
        (slice_key, value) = got[0]
        self.assertEqual((), slice_key)
        self.assertIn(metric_keys.AUC_PLOTS_MATRICES, value)
        matrices = value[metric_keys.AUC_PLOTS_MATRICES]
        #            |      | --------- Threshold -----------
        # true label | pred | -1e-6 | 0.0 | 0.7 | 0.8 | 1.0
        #     -      | 0.0  | FP    | TN  | TN  | TN  | TN
        #     +      | 0.0  | TP    | FN  | FN  | FN  | FN
        #     +      | 0.7  | TP    | TP  | FN  | FN  | FN
        #     -      | 0.8  | FP    | FP  | FP  | TN  | TN
        #     +      | 1.0  | TP    | TP  | TP  | TP  | FN
        self.assertSequenceAlmostEqual(matrices[0],
                                       [0, 0, 2, 3, 3.0 / 5.0, 1.0])
        self.assertSequenceAlmostEqual(matrices[1],
                                       [1, 1, 1, 2, 2.0 / 3.0, 2.0 / 3.0])
        self.assertSequenceAlmostEqual(matrices[7001],
                                       [2, 1, 1, 1, 1.0 / 2.0, 1.0 / 3.0])
        self.assertSequenceAlmostEqual(matrices[8001],
                                       [2, 2, 0, 1, 1.0 / 1.0, 1.0 / 3.0])
        self.assertSequenceAlmostEqual(
            matrices[10001], [3, 2, 0, 0, float('nan'), 0.0])
        self.assertIn(metric_keys.AUC_PLOTS_THRESHOLDS, value)
        thresholds = value[metric_keys.AUC_PLOTS_THRESHOLDS]
        self.assertAlmostEqual(0.0, thresholds[1])
        self.assertAlmostEqual(0.001, thresholds[11])
        self.assertAlmostEqual(0.005, thresholds[51])
        self.assertAlmostEqual(0.010, thresholds[101])
        self.assertAlmostEqual(0.100, thresholds[1001])
        self.assertAlmostEqual(0.800, thresholds[8001])
        self.assertAlmostEqual(1.000, thresholds[10001])
        plot_data = metrics_for_slice_pb2.PlotData()
        auc_plots.populate_plots_and_pop(value, plot_data)
        self.assertProtoEquals(
            """threshold: 1.0
            false_negatives: 3.0
            true_negatives: 2.0
            precision: nan""",
            plot_data.confusion_matrix_at_thresholds.matrices[10001])
      except AssertionError as err:
        raise util.BeamAssertException(err)

    self._runTestWithCustomCheck(
        examples, eval_export_dir, [auc_plots], custom_plots_check=check_result)

  def testConfusionMatrixAtThresholdsWeighted(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        fixed_prediction_estimator_extra_fields.
        simple_fixed_prediction_estimator_extra_fields(None,
                                                       temp_eval_export_dir))
    examples = [
        self._makeExample(
            prediction=0.0000,
            label=0.0000,
            fixed_float=1.0,
            fixed_string='',
            fixed_int=0),
        self._makeExample(
            prediction=0.0000,
            label=1.0000,
            fixed_float=1.0,
            fixed_string='',
            fixed_int=0),
        self._makeExample(
            prediction=0.7000,
            label=1.0000,
            fixed_float=3.0,
            fixed_string='',
            fixed_int=0),
        self._makeExample(
            prediction=0.8000,
            label=0.0000,
            fixed_float=2.0,
            fixed_string='',
            fixed_int=0),
        self._makeExample(
            prediction=1.0000,
            label=1.0000,
            fixed_float=3.0,
            fixed_string='',
            fixed_int=0),
    ]

    def check_result(got):  # pylint: disable=invalid-name
      try:
        self.assertEqual(1, len(got), 'got: %s' % got)
        (slice_key, value) = got[0]
        self.assertEqual((), slice_key)
        self.assertIn(metric_keys.CONFUSION_MATRIX_AT_THRESHOLDS_MATRICES,
                      value)
        matrices = value[metric_keys.CONFUSION_MATRIX_AT_THRESHOLDS_MATRICES]
        #            |      |       --------- Threshold -----------
        # true label | pred | wt   | -1e-6 | 0.0 | 0.7 | 0.8 | 1.0
        #     -      | 0.0  | 1.0  | FP    | TN  | TN  | TN  | TN
        #     +      | 0.0  | 1.0  | TP    | FN  | FN  | FN  | FN
        #     +      | 0.7  | 3.0  | TP    | TP  | FN  | FN  | FN
        #     -      | 0.8  | 2.0  | FP    | FP  | FP  | TN  | TN
        #     +      | 1.0  | 3.0  | TP    | TP  | TP  | TP  | FN
        self.assertSequenceAlmostEqual(matrices[0],
                                       [0.0, 0.0, 3.0, 7.0, 7.0 / 10.0, 1.0])
        self.assertSequenceAlmostEqual(
            matrices[1], [1.0, 1.0, 2.0, 6.0, 6.0 / 8.0, 6.0 / 7.0])
        self.assertSequenceAlmostEqual(
            matrices[2], [4.0, 1.0, 2.0, 3.0, 3.0 / 5.0, 3.0 / 7.0])
        self.assertSequenceAlmostEqual(matrices[3],
                                       [4.0, 3.0, 0.0, 3.0, 1.0, 3.0 / 7.0])
        self.assertSequenceAlmostEqual(
            matrices[4],
            [7.0, 3.0, 0.0, 0.0, float('nan'), 0.0])
        self.assertIn(metric_keys.CONFUSION_MATRIX_AT_THRESHOLDS_THRESHOLDS,
                      value)
        thresholds = value[
            metric_keys.CONFUSION_MATRIX_AT_THRESHOLDS_THRESHOLDS]
        self.assertAlmostEqual(-1e-6, thresholds[0])
        self.assertAlmostEqual(0.0, thresholds[1])
        self.assertAlmostEqual(0.7, thresholds[2])
        self.assertAlmostEqual(0.8, thresholds[3])
        self.assertAlmostEqual(1.0, thresholds[4])
      except AssertionError as err:
        raise util.BeamAssertException(err)

    self._runTestWithCustomCheck(
        examples,
        eval_export_dir, [
            post_export_metrics.confusion_matrix_at_thresholds(
                example_weight_key='fixed_float',
                thresholds=[-1e-6, 0.0, 0.7, 0.8, 1.0])
        ],
        custom_metrics_check=check_result)

  def testConfusionMatrixAtThresholdsSerialization(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        fixed_prediction_estimator.simple_fixed_prediction_estimator(
            None, temp_eval_export_dir))
    examples = [
        self._makeExample(prediction=0.0000, label=0.0000),
        self._makeExample(prediction=0.5000, label=1.0000),
        self._makeExample(prediction=1.0000, label=1.0000),
    ]

    confusion_matrix_at_thresholds_metric = (
        post_export_metrics.confusion_matrix_at_thresholds(
            thresholds=[0.25, 0.75, 1.00]))

    def check_result(got):  # pylint: disable=invalid-name
      try:
        self.assertEqual(1, len(got), 'got: %s' % got)
        (slice_key, value) = got[0]
        self.assertEqual((), slice_key)
        self.assertIn(metric_keys.CONFUSION_MATRIX_AT_THRESHOLDS_MATRICES,
                      value)
        matrices = value[metric_keys.CONFUSION_MATRIX_AT_THRESHOLDS_MATRICES]
        #            |      | ---- Threshold ----
        # true label | pred | 0.25 | 0.75 | 1.00
        #     -      | 0.0  | TN   | TN   | TN
        #     +      | 0.5  | TP   | FN   | FN
        #     +      | 1.0  | TP   | TP   | FN
        self.assertSequenceAlmostEqual(matrices[0],
                                       [0.0, 1.0, 0.0, 2.0, 1.0, 1.0])
        self.assertSequenceAlmostEqual(matrices[1],
                                       [1.0, 1.0, 0.0, 1.0, 1.0, 0.5])
        self.assertSequenceAlmostEqual(
            matrices[2],
            [2.0, 1.0, 0.0, 0.0, float('nan'), 0.0])
        self.assertIn(metric_keys.CONFUSION_MATRIX_AT_THRESHOLDS_THRESHOLDS,
                      value)
        thresholds = value[
            metric_keys.CONFUSION_MATRIX_AT_THRESHOLDS_THRESHOLDS]
        self.assertAlmostEqual(0.25, thresholds[0])
        self.assertAlmostEqual(0.75, thresholds[1])
        self.assertAlmostEqual(1.00, thresholds[2])

        # Check serialization too.
        # Note that we can't just make this a dict, since proto maps
        # allow uninitialized key access, i.e. they act like defaultdicts.
        output_metrics = metrics_for_slice_pb2.MetricsForSlice().metrics
        confusion_matrix_at_thresholds_metric.populate_stats_and_pop(
            value, output_metrics)
        self.assertProtoEquals(
            """
            confusion_matrix_at_thresholds {
              matrices {
                threshold: 0.25
                false_negatives: 0.0
                true_negatives: 1.0
                false_positives: 0.0
                true_positives: 2.0
                precision: 1.0
                recall: 1.0
              }
              matrices {
                threshold: 0.75
                false_negatives: 1.0
                true_negatives: 1.0
                false_positives: 0.0
                true_positives: 1.0
                precision: 1.0
                recall: 0.5
              }
              matrices {
                threshold: 1.00
                false_negatives: 2.0
                true_negatives: 1.0
                false_positives: 0.0
                true_positives: 0.0
                precision: nan
                recall: 0.0
              }
            }
            """, output_metrics[metric_keys.CONFUSION_MATRIX_AT_THRESHOLDS])
      except AssertionError as err:
        raise util.BeamAssertException(err)

    self._runTestWithCustomCheck(
        examples,
        eval_export_dir, [confusion_matrix_at_thresholds_metric],
        custom_metrics_check=check_result)

  def testCalibrationPlotAndPredictionHistogramLinearClassifier(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        linear_classifier.simple_linear_classifier(None, temp_eval_export_dir))

    examples = [
        self._makeExample(age=3.0, language='english', label=1.0),
        self._makeExample(age=3.0, language='chinese', label=0.0),
        self._makeExample(age=4.0, language='english', label=1.0),
        self._makeExample(age=5.0, language='chinese', label=0.0)
    ]

    def check_result(got):  # pylint: disable=invalid-name
      try:
        self.assertEqual(1, len(got), 'got: %s' % got)
        (slice_key, value) = got[0]
        self.assertEqual((), slice_key)
        self.assertIn(metric_keys.CALIBRATION_PLOT_MATRICES, value)
        # We just check that the bucket sums look sane, since we don't know
        # the exact predictions of the model.
        #
        # Note that the correctness of the bucketing is tested in the other
        # two tests with the fixed prediction estimator. This test is more
        # for ensuring that this metric is compatible with the canned
        # Estimators, for which the prediction Tensor returned for a batch
        # of examples will be a N x 1 Tensor, rather than just an N element
        # vector.
        buckets = value[metric_keys.CALIBRATION_PLOT_MATRICES]
        bucket_sums = np.sum(buckets, axis=0)
        self.assertAlmostEqual(bucket_sums[1], 2.0)  # label sum
        self.assertAlmostEqual(bucket_sums[2], 4.0)  # weight sum
      except AssertionError as err:
        raise util.BeamAssertException(err)

    self._runTestWithCustomCheck(
        examples,
        eval_export_dir,
        [post_export_metrics.calibration_plot_and_prediction_histogram()],
        custom_plots_check=check_result)

  def testCalibrationPlotSerialization(self):
    # Calibration plots for the model
    # {prediction:0.3, true_label:+},
    # {prediction:0.7, true_label:-}
    #
    # These plots were generated by hand. For this test to make sense
    # it must actually match the kind of output the TFMA produces.
    tfma_plots = {
        metric_keys.CALIBRATION_PLOT_MATRICES:
            np.array([
                [0.0, 0.0, 0.0],
                [0.3, 1.0, 1.0],
                [0.7, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ]),
        metric_keys.CALIBRATION_PLOT_BOUNDARIES:
            np.array([0.0, 0.5, 1.0]),
    }
    expected_plot_data = """
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
    """
    plot_data = metrics_for_slice_pb2.PlotData()
    calibration_plot = (
        post_export_metrics.calibration_plot_and_prediction_histogram())
    calibration_plot.populate_plots_and_pop(tfma_plots, plot_data)
    self.assertProtoEquals(expected_plot_data, plot_data)
    self.assertFalse(metric_keys.CALIBRATION_PLOT_MATRICES in tfma_plots)
    self.assertFalse(metric_keys.CALIBRATION_PLOT_BOUNDARIES in tfma_plots)

  def testAucUnweighted(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        fixed_prediction_estimator.simple_fixed_prediction_estimator(
            None, temp_eval_export_dir))
    examples = [
        self._makeExample(prediction=0.0000, label=0.0000),
        self._makeExample(prediction=0.0000, label=1.0000),
        self._makeExample(prediction=0.7000, label=1.0000),
        self._makeExample(prediction=0.8000, label=0.0000),
        self._makeExample(prediction=1.0000, label=1.0000),
    ]

    expected_values_dict = {
        metric_keys.AUC: 0.58333,
        metric_keys.lower_bound(metric_keys.AUC): 0.5,
        metric_keys.upper_bound(metric_keys.AUC): 0.66667,
        metric_keys.lower_bound(metric_keys.AUPRC): 0.74075,
        metric_keys.lower_bound(metric_keys.AUPRC): 0.70000,
        metric_keys.upper_bound(metric_keys.AUPRC): 0.77778,
    }

    self._runTest(
        examples,
        eval_export_dir,
        [post_export_metrics.auc(),
         post_export_metrics.auc(curve='PR')],
        expected_values_dict)

  def testAucUnweightedSerialization(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        fixed_prediction_estimator.simple_fixed_prediction_estimator(
            None, temp_eval_export_dir))
    examples = [
        self._makeExample(prediction=0.0000, label=0.0000),
        self._makeExample(prediction=0.0000, label=1.0000),
        self._makeExample(prediction=0.7000, label=1.0000),
        self._makeExample(prediction=0.8000, label=0.0000),
        self._makeExample(prediction=1.0000, label=1.0000),
    ]

    expected_values_dict = {
        metric_keys.lower_bound(metric_keys.AUPRC): 0.74075,
        metric_keys.lower_bound(metric_keys.AUPRC): 0.70000,
        metric_keys.upper_bound(metric_keys.AUPRC): 0.77778,
    }

    auc_metric = post_export_metrics.auc(curve='PR')

    def check_result(got):  # pylint: disable=invalid-name
      try:
        self.assertEqual(1, len(got), 'got: %s' % got)
        (slice_key, value) = got[0]
        self.assertEqual((), slice_key)
        self.assertDictElementsAlmostEqual(value, expected_values_dict)

        # Check serialization too.
        # Note that we can't just make this a dict, since proto maps
        # allow uninitialized key access, i.e. they act like defaultdicts.
        output_metrics = metrics_for_slice_pb2.MetricsForSlice().metrics
        auc_metric.populate_stats_and_pop(value, output_metrics)
        self.assertProtoEquals(
            """
            bounded_value {
              lower_bound {
                value: 0.6999999
              }
              upper_bound {
                value: 0.7777776
              }
              value {
                value: 0.7407472
              }
            }
            """, output_metrics[metric_keys.AUPRC])
      except AssertionError as err:
        raise util.BeamAssertException(err)

    self._runTestWithCustomCheck(
        examples,
        eval_export_dir, [auc_metric],
        custom_metrics_check=check_result)

  def testAucPlotSerialization(self):
    # Auc for the model
    # {prediction:0.3, true_label:+},
    # {prediction:0.7, true_label:-}
    #
    # These plots were generated by hand. For this test to make sense
    # it must actually match the kind of output the TFMA produces.
    tfma_plots = {
        metric_keys.AUC_PLOTS_MATRICES:
            np.array([
                [0.0, 0.0, 1.0, 1.0, 0.5, 1.0],
                [0.0, 0.0, 1.0, 1.0, 0.5, 1.0],
                [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ]),
        metric_keys.AUC_PLOTS_THRESHOLDS:
            np.array([1e-6, 0, 0.5, 1.0]),
    }
    expected_plot_data = """
      confusion_matrix_at_thresholds {
        matrices {
          threshold: 1e-6
          true_positives: 1.0
          false_positives: 1.0
          true_negatives: 0.0
          false_negatives: 0.0
          precision: 0.5
          recall: 1.0
        }
      }
      confusion_matrix_at_thresholds {
        matrices {
          threshold: 0
          true_positives: 1.0
          false_positives: 1.0
          true_negatives: 0.0
          false_negatives: 0.0
          precision: 0.5
          recall: 1.0
        }
      }
      confusion_matrix_at_thresholds {
        matrices {
          threshold: 0.5
          true_positives: 0.0
          false_positives: 1.0
          true_negatives: 0.0
          false_negatives: 1.0
          precision: 0.0
          recall: 0.0
        }
      }
      confusion_matrix_at_thresholds {
        matrices {
          threshold: 1.0
          true_positives: 0.0
          false_positives: 0.0
          true_negatives: 1.0
          false_negatives: 1.0
          precision: 0.0
          recall: 0.0
        }
      }
    """
    plot_data = metrics_for_slice_pb2.PlotData()
    auc_plots = post_export_metrics.auc_plots()
    auc_plots.populate_plots_and_pop(tfma_plots, plot_data)
    self.assertProtoEquals(expected_plot_data, plot_data)
    self.assertFalse(metric_keys.AUC_PLOTS_MATRICES in tfma_plots)
    self.assertFalse(metric_keys.AUC_PLOTS_THRESHOLDS in tfma_plots)


if __name__ == '__main__':
  tf.test.main()
