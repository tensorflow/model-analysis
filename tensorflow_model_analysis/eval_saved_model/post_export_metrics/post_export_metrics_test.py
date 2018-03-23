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
from tensorflow_model_analysis.eval_saved_model.example_trainers import fixed_prediction_estimator
from tensorflow_model_analysis.eval_saved_model.example_trainers import fixed_prediction_estimator_extra_fields
from tensorflow_model_analysis.eval_saved_model.example_trainers import linear_classifier
from tensorflow_model_analysis.eval_saved_model.example_trainers import linear_regressor
from tensorflow_model_analysis.eval_saved_model.post_export_metrics import post_export_metrics
import tensorflow_model_analysis.eval_saved_model.post_export_metrics.metric_keys as metric_keys


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

  def _runTest(self, examples, eval_export_dir, metrics_to_check):
    metrics = [metric for (_, metric, _) in metrics_to_check]
    expected_values_dict = {
        name: expected
        for (name, _, expected) in metrics_to_check
    }

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
    metrics_to_check = [
        (metric_keys.EXAMPLE_COUNT, post_export_metrics.example_count(), 4.0),
        (metric_keys.EXAMPLE_WEIGHT, post_export_metrics.example_weight('age'),
         15.0),
    ]
    self._runTest(examples, eval_export_dir, metrics_to_check)

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
    metrics_to_check = [
        (metric_keys.EXAMPLE_COUNT, post_export_metrics.example_count(), 4.0),
        (metric_keys.EXAMPLE_WEIGHT, post_export_metrics.example_weight('age'),
         15.0),
    ]
    self._runTest(examples, eval_export_dir, metrics_to_check)

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
      except AssertionError as err:
        raise util.BeamAssertException(err)

    self._runTestWithCustomCheck(
        examples,
        eval_export_dir,
        [post_export_metrics.calibration_plot_and_prediction_histogram()],
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
            prediction=-10.0, label=-9.0, fixed_float=1.0, fixed_string=''),
        self._makeExample(
            prediction=-9.0, label=-8.0, fixed_float=2.0, fixed_string=''),
        self._makeExample(
            prediction=0.0000, label=1.0000, fixed_float=0.0, fixed_string=''),
        self._makeExample(
            prediction=0.00100, label=1.00100, fixed_float=1.0,
            fixed_string=''),
        self._makeExample(
            prediction=0.00101, label=1.00101, fixed_float=2.0,
            fixed_string=''),
        self._makeExample(
            prediction=0.00102, label=1.00102, fixed_float=3.0,
            fixed_string=''),
        self._makeExample(
            prediction=10.0, label=11.0, fixed_float=7.0, fixed_string=''),
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
      except AssertionError as err:
        raise util.BeamAssertException(err)

    self._runTestWithCustomCheck(
        examples,
        eval_export_dir, [post_export_metrics.auc_plots()],
        custom_plots_check=check_result)

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


if __name__ == '__main__':
  tf.test.main()
