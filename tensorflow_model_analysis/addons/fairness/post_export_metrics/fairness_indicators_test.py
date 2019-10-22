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
# Standard Imports

import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import config
from tensorflow_model_analysis.addons.fairness.post_export_metrics import fairness_indicators  # pylint: disable=unused-import
from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.api import tfma_unit
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.eval_saved_model.example_trainers import fixed_prediction_estimator_extra_fields
from tensorflow_model_analysis.eval_saved_model.example_trainers import multi_head
from tensorflow_model_analysis.evaluators import metrics_and_plots_evaluator
from tensorflow_model_analysis.post_export_metrics import post_export_metrics
import tensorflow_model_analysis.post_export_metrics.metric_keys as metric_keys
from tensorflow_model_analysis.proto import metrics_for_slice_pb2
from tensorflow_model_analysis.slicer import slicer_lib as slicer


_TEST_SEED = 857586


class FairnessIndicatorsTest(testutil.TensorflowModelAnalysisTest):

  compute_confidence_intervals = False  # Set to True to test uncertainty.
  deterministic_test_seed = _TEST_SEED

  def _getEvalExportDir(self):
    return os.path.join(self._getTempDir(), 'eval_export_dir')

  def _runTestWithCustomCheck(self,
                              examples,
                              eval_export_dir,
                              metrics_callbacks,
                              slice_spec=None,
                              custom_metrics_check=None,
                              custom_plots_check=None,
                              custom_result_check=None):
    # make sure we are doing some checks
    self.assertTrue(custom_metrics_check is not None or
                    custom_plots_check is not None or
                    custom_result_check is not None)
    serialized_examples = [ex.SerializeToString() for ex in examples]
    slicing_specs = None
    if slice_spec:
      slicing_specs = [s.to_proto() for s in slice_spec]
    eval_config = config.EvalConfig(
        input_data_specs=[config.InputDataSpec()],
        model_specs=[config.ModelSpec(location=eval_export_dir)],
        output_data_specs=[config.OutputDataSpec()],
        slicing_specs=slicing_specs)
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=eval_export_dir,
        add_metrics_callbacks=metrics_callbacks)
    extractors = model_eval_lib.default_extractors(
        eval_config=eval_config, eval_shared_models=[eval_shared_model])
    with beam.Pipeline() as pipeline:
      (metrics, plots), _ = (
          pipeline
          | 'Create' >> beam.Create(serialized_examples)
          | 'InputsToExtracts' >> model_eval_lib.InputsToExtracts()
          | 'Extract' >> tfma_unit.Extract(extractors=extractors)  # pylint: disable=no-value-for-parameter
          | 'ComputeMetricsAndPlots' >>
          metrics_and_plots_evaluator.ComputeMetricsAndPlots(
              eval_shared_model=eval_shared_model,
              compute_confidence_intervals=self.compute_confidence_intervals,
              random_seed_for_testing=self.deterministic_test_seed))
      if custom_metrics_check is not None:
        util.assert_that(metrics, custom_metrics_check, label='metrics')
      if custom_plots_check is not None:
        util.assert_that(plots, custom_plots_check, label='plot')

    result = pipeline.run()
    if custom_result_check is not None:
      custom_result_check(result)

  def _runTest(self, examples, eval_export_dir, metrics, expected_values_dict):

    def check_result(got):  # pylint: disable=invalid-name
      try:
        self.assertEqual(1, len(got), 'got: %s' % got)
        (slice_key, value) = got[0]
        self.assertEqual((), slice_key)
        self.assertDictElementsAlmostEqual(value, expected_values_dict)
      except AssertionError as err:
        tf.compat.v1.logging.error(got)
        raise util.BeamAssertException(err)

    self._runTestWithCustomCheck(
        examples, eval_export_dir, metrics, custom_metrics_check=check_result)

  def makeConfusionMatrixExamples(self):
    """Helper to create a set of examples used by multiple tests."""
    #            |      |       --------- Threshold -----------
    # true label | pred | wt   | -1e-6 | 0.0 | 0.7 | 0.8 | 1.0
    #     -      | 0.0  | 1.0  | FP    | TN  | TN  | TN  | TN
    #     +      | 0.0  | 1.0  | TP    | FN  | FN  | FN  | FN
    #     +      | 0.7  | 3.0  | TP    | TP  | FN  | FN  | FN
    #     -      | 0.8  | 2.0  | FP    | FP  | FP  | TN  | TN
    #     +      | 1.0  | 3.0  | TP    | TP  | TP  | TP  | FN
    return [
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

  def testFairnessIndicatorsDigitsKey(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        fixed_prediction_estimator_extra_fields
        .simple_fixed_prediction_estimator_extra_fields(None,
                                                        temp_eval_export_dir))
    examples = self.makeConfusionMatrixExamples()
    fairness_metrics = post_export_metrics.fairness_indicators(
        example_weight_key='fixed_float', thresholds=[0.5, 0.59, 0.599, 0.5999])

    def check_result(got):  # pylint: disable=invalid-name
      try:
        self.assertEqual(1, len(got), 'got: %s' % got)
        (slice_key, value) = got[0]
        self.assertEqual((), slice_key)
        self.assertIn(metric_keys.base_key('true_positive_rate@0.5000'), value)
        self.assertIn(metric_keys.base_key('false_positive_rate@0.5000'), value)
        self.assertIn(metric_keys.base_key('true_positive_rate@0.5900'), value)
        self.assertIn(metric_keys.base_key('false_positive_rate@0.5900'), value)
        self.assertIn(metric_keys.base_key('true_positive_rate@0.5990'), value)
        self.assertIn(metric_keys.base_key('false_positive_rate@0.5990'), value)
        self.assertIn(metric_keys.base_key('true_positive_rate@0.5999'), value)
        self.assertIn(metric_keys.base_key('false_positive_rate@0.5999'), value)
      except AssertionError as err:
        raise util.BeamAssertException(err)

    self._runTestWithCustomCheck(
        examples,
        eval_export_dir, [fairness_metrics],
        custom_metrics_check=check_result)

  def testFairnessIndicatorsCounters(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        multi_head.simple_multi_head(None, temp_eval_export_dir))

    examples = [
        self._makeExample(
            age=3.0,
            language='english',
            english_label=1.0,
            chinese_label=0.0,
            other_label=0.0),
        self._makeExample(
            age=3.0,
            language='chinese',
            english_label=0.0,
            chinese_label=1.0,
            other_label=0.0),
        self._makeExample(
            age=4.0,
            language='english',
            english_label=1.0,
            chinese_label=0.0,
            other_label=0.0),
        self._makeExample(
            age=5.0,
            language='chinese',
            english_label=0.0,
            chinese_label=1.0,
            other_label=0.0),
        self._makeExample(
            age=6.0,
            language='chinese',
            english_label=0.0,
            chinese_label=1.0,
            other_label=0.0),
    ]
    fairness_english = post_export_metrics.fairness_indicators(
        target_prediction_keys=['english_head/logistic'],
        labels_key='english_head')
    fairness_chinese = post_export_metrics.fairness_indicators(
        target_prediction_keys=['chinese_head/logistic'],
        labels_key='chinese_head')

    def check_metric_counter(result):
      metric_filter = beam.metrics.metric.MetricsFilter().with_name(
          'metric_computed_fairness_indicators')
      actual_metrics_count = result.metrics().query(
          filter=metric_filter)['counters'][0].committed
      self.assertEqual(actual_metrics_count, 2)

    self._runTestWithCustomCheck(
        examples,
        eval_export_dir, [
            fairness_english,
            fairness_chinese,
        ],
        custom_result_check=check_metric_counter)

  def testFairnessIndicatorsAtThresholdsWeighted(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        fixed_prediction_estimator_extra_fields
        .simple_fixed_prediction_estimator_extra_fields(None,
                                                        temp_eval_export_dir))
    examples = self.makeConfusionMatrixExamples()
    fairness_metrics = post_export_metrics.fairness_indicators(
        example_weight_key='fixed_float', thresholds=[0.0, 0.7, 0.8, 1.0])

    def check_result(got):  # pylint: disable=invalid-name
      try:
        self.assertEqual(1, len(got), 'got: %s' % got)
        (slice_key, value) = got[0]
        self.assertEqual((), slice_key)
        expected_values_dict = {
            metric_keys.base_key('true_positive_rate@0.00'): 6.0 / 7.0,
            metric_keys.base_key('false_positive_rate@0.00'): 2.0 / 3.0,
            metric_keys.base_key('positive_rate@0.00'): 0.8,
            metric_keys.base_key('true_negative_rate@0.00'): 1.0 / 3.0,
            metric_keys.base_key('false_negative_rate@0.00'): 1.0 / 7.0,
            metric_keys.base_key('negative_rate@0.00'): 2.0 / 10.0,
            metric_keys.base_key('true_positive_rate@0.70'): 3.0 / 7.0,
            metric_keys.base_key('false_positive_rate@0.70'): 2.0 / 3.0,
            metric_keys.base_key('positive_rate@0.70'): 0.5,
            metric_keys.base_key('true_negative_rate@0.70'): 1.0 / 3.0,
            metric_keys.base_key('false_negative_rate@0.70'): 4.0 / 7.0,
            metric_keys.base_key('negative_rate@0.70'): 5.0 / 10.0,
            metric_keys.base_key('true_positive_rate@0.80'): 3.0 / 7.0,
            metric_keys.base_key('false_positive_rate@0.80'): 0,
            metric_keys.base_key('positive_rate@0.80'): 0.3,
            metric_keys.base_key('true_negative_rate@0.80'): 3.0 / 3.0,
            metric_keys.base_key('false_negative_rate@0.80'): 4.0 / 7.0,
            metric_keys.base_key('negative_rate@0.80'): 7.0 / 10.0,
            metric_keys.base_key('true_positive_rate@1.00'): 0,
            metric_keys.base_key('false_positive_rate@1.00'): 0,
            metric_keys.base_key('positive_rate@1.00'): 0,
            metric_keys.base_key('true_negative_rate@1.00'): 1,
            metric_keys.base_key('false_negative_rate@1.00'): 1,
            metric_keys.base_key('negative_rate@1.00'): 1,
        }
        self.assertDictElementsAlmostEqual(value, expected_values_dict)
      except AssertionError as err:
        raise util.BeamAssertException(err)

    self._runTestWithCustomCheck(
        examples,
        eval_export_dir, [fairness_metrics],
        custom_metrics_check=check_result)

  def testFairnessIndicatorsAtThresholdsWeightedWithUncertainty(self):
    self.compute_confidence_intervals = True
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        fixed_prediction_estimator_extra_fields
        .simple_fixed_prediction_estimator_extra_fields(None,
                                                        temp_eval_export_dir))
    examples = self.makeConfusionMatrixExamples()
    fairness_metrics = post_export_metrics.fairness_indicators(
        example_weight_key='fixed_float', thresholds=[0.0, 0.7, 0.8, 1.0])

    def check_result(got):  # pylint: disable=invalid-name
      try:
        self.assertEqual(1, len(got), 'got: %s' % got)
        (slice_key, value) = got[0]
        self.assertEqual((), slice_key)
        expected_values_dict = {
            metric_keys.base_key('true_positive_rate@0.00'): 6.0 / 7.0,
            metric_keys.base_key('false_positive_rate@0.00'): 2.0 / 3.0,
            metric_keys.base_key('positive_rate@0.00'): 0.8,
            metric_keys.base_key('true_negative_rate@0.00'): 1.0 / 3.0,
            metric_keys.base_key('false_negative_rate@0.00'): 1.0 / 7.0,
            metric_keys.base_key('negative_rate@0.00'): 2.0 / 10.0,
            metric_keys.base_key('true_positive_rate@0.70'): 3.0 / 7.0,
            metric_keys.base_key('false_positive_rate@0.70'): 2.0 / 3.0,
            metric_keys.base_key('positive_rate@0.70'): 0.5,
            metric_keys.base_key('true_negative_rate@0.70'): 1.0 / 3.0,
            metric_keys.base_key('false_negative_rate@0.70'): 4.0 / 7.0,
            metric_keys.base_key('negative_rate@0.70'): 5.0 / 10.0,
            metric_keys.base_key('true_positive_rate@0.80'): 3.0 / 7.0,
            metric_keys.base_key('false_positive_rate@0.80'): 0,
            metric_keys.base_key('positive_rate@0.80'): 0.3,
            metric_keys.base_key('true_negative_rate@0.80'): 3.0 / 3.0,
            metric_keys.base_key('false_negative_rate@0.80'): 4.0 / 7.0,
            metric_keys.base_key('negative_rate@0.80'): 7.0 / 10.0,
            metric_keys.base_key('true_positive_rate@1.00'): 0,
            metric_keys.base_key('false_positive_rate@1.00'): 0,
            metric_keys.base_key('positive_rate@1.00'): 0,
            metric_keys.base_key('true_negative_rate@1.00'): 1,
            metric_keys.base_key('false_negative_rate@1.00'): 1,
            metric_keys.base_key('negative_rate@1.00'): 1,
        }
        self.assertDictElementsWithTDistributionAlmostEqual(
            value, expected_values_dict)
      except AssertionError as err:
        raise util.BeamAssertException(err)

    self._runTestWithCustomCheck(
        examples,
        eval_export_dir, [fairness_metrics],
        custom_metrics_check=check_result)

  def testFairnessIndicatorsAtDefaultThresholds(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        fixed_prediction_estimator_extra_fields
        .simple_fixed_prediction_estimator_extra_fields(None,
                                                        temp_eval_export_dir))
    examples = self.makeConfusionMatrixExamples()
    fairness_metrics = post_export_metrics.fairness_indicators()

    def check_result(got):  # pylint: disable=invalid-name
      try:
        self.assertEqual(1, len(got), 'got: %s' % got)
        (slice_key, value) = got[0]
        self.assertEqual((), slice_key)
        expected_values_dict = {
            metric_keys.base_key('true_positive_rate@0.10'): 2.0 / 3.0,
            metric_keys.base_key('false_positive_rate@0.10'): 1.0 / 2.0,
            metric_keys.base_key('positive_rate@0.10'): 3.0 / 5.0,
            metric_keys.base_key('true_negative_rate@0.10'): 1.0 / 2.0,
            metric_keys.base_key('false_negative_rate@0.10'): 1.0 / 3.0,
            metric_keys.base_key('negative_rate@0.10'): 2.0 / 5.0,
            metric_keys.base_key('true_positive_rate@0.20'): 2.0 / 3.0,
            metric_keys.base_key('true_positive_rate@0.30'): 2.0 / 3.0,
            metric_keys.base_key('true_positive_rate@0.40'): 2.0 / 3.0,
            metric_keys.base_key('true_positive_rate@0.50'): 2.0 / 3.0,
            metric_keys.base_key('true_positive_rate@0.60'): 2.0 / 3.0,
            metric_keys.base_key('true_positive_rate@0.70'): 1.0 / 3.0,
            metric_keys.base_key('false_positive_rate@0.70'): 1.0 / 2.0,
            metric_keys.base_key('positive_rate@0.70'): 2.0 / 5.0,
            metric_keys.base_key('true_negative_rate@0.70'): 1.0 / 2.0,
            metric_keys.base_key('false_negative_rate@0.70'): 2.0 / 3.0,
            metric_keys.base_key('negative_rate@0.70'): 3.0 / 5.0,
            metric_keys.base_key('positive_rate@0.80'): 1.0 / 5.0,
            metric_keys.base_key('true_positive_rate@0.90'): 1.0 / 3.0,
            metric_keys.base_key('false_positive_rate@0.90'): 0.0 / 2.0,
            metric_keys.base_key('positive_rate@0.90'): 1.0 / 5.0,
            metric_keys.base_key('true_negative_rate@0.90'): 2.0 / 2.0,
            metric_keys.base_key('false_negative_rate@0.90'): 2.0 / 3.0,
            metric_keys.base_key('negative_rate@0.90'): 4.0 / 5.0,
        }
        self.assertDictElementsAlmostEqual(value, expected_values_dict)
      except AssertionError as err:
        raise util.BeamAssertException(err)

    self._runTestWithCustomCheck(
        examples,
        eval_export_dir, [fairness_metrics],
        custom_metrics_check=check_result)

  def testFairnessIndicatorsZeroes(self):

    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        fixed_prediction_estimator_extra_fields
        .simple_fixed_prediction_estimator_extra_fields(None,
                                                        temp_eval_export_dir))
    examples = self.makeConfusionMatrixExamples()[0:1]
    fairness_metrics = post_export_metrics.fairness_indicators()

    def check_result(got):  # pylint: disable=invalid-name
      try:
        self.assertEqual(1, len(got), 'got: %s' % got)
        (slice_key, value) = got[0]
        self.assertEqual((), slice_key)
        self.assertTrue(
            np.isnan(value[metric_keys.base_key('true_positive_rate@0.10')]))
      except AssertionError as err:
        raise util.BeamAssertException(err)

    self._runTestWithCustomCheck(
        examples,
        eval_export_dir, [fairness_metrics],
        custom_metrics_check=check_result)

  def testFairnessIndicatorsMultiHead(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        multi_head.simple_multi_head(None, temp_eval_export_dir))

    examples = [
        self._makeExample(
            age=3.0,
            language='english',
            english_label=1.0,
            chinese_label=0.0,
            other_label=0.0),
        self._makeExample(
            age=3.0,
            language='chinese',
            english_label=0.0,
            chinese_label=1.0,
            other_label=0.0),
        self._makeExample(
            age=4.0,
            language='english',
            english_label=1.0,
            chinese_label=0.0,
            other_label=0.0),
        self._makeExample(
            age=5.0,
            language='chinese',
            english_label=0.0,
            chinese_label=1.0,
            other_label=0.0),
        self._makeExample(
            age=6.0,
            language='chinese',
            english_label=0.0,
            chinese_label=1.0,
            other_label=0.0),
    ]
    fairness_english = post_export_metrics.fairness_indicators(
        target_prediction_keys=['english_head/logistic'],
        labels_key='english_head')
    fairness_chinese = post_export_metrics.fairness_indicators(
        target_prediction_keys=['chinese_head/logistic'],
        labels_key='chinese_head')

    def check_metric_result(got):
      try:
        self.assertEqual(1, len(got), 'got: %s' % got)
        (slice_key, value) = got[0]
        self.assertEqual((), slice_key)
        expected_values_dict = {
            metric_keys.base_key(
                'english_head/logistic/true_positive_rate@0.10'):
                1.0,
            metric_keys.base_key(
                'chinese_head/logistic/true_positive_rate@0.10'):
                1.0,
        }
        self.assertDictElementsAlmostEqual(value, expected_values_dict)
      except AssertionError as err:
        raise util.BeamAssertException(err)

    self._runTestWithCustomCheck(
        examples,
        eval_export_dir, [
            fairness_english,
            fairness_chinese,
        ],
        custom_metrics_check=check_metric_result)

  def testFairnessAucs(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        fixed_prediction_estimator_extra_fields
        .simple_fixed_prediction_estimator_extra_fields(None,
                                                        temp_eval_export_dir))
    examples = [
        # Subgroup
        self._makeExample(prediction=0.0000, label=0.0000, fixed_int=1),
        self._makeExample(prediction=0.0000, label=1.0000, fixed_int=1),
        self._makeExample(prediction=1.0000, label=0.0000, fixed_int=1),
        self._makeExample(prediction=1.0000, label=1.0000, fixed_int=1),
        # Background
        self._makeExample(prediction=0.0000, label=0.0000, fixed_int=0),
        self._makeExample(prediction=0.0000, label=1.0000, fixed_int=0),
        self._makeExample(prediction=1.0000, label=0.0000, fixed_int=0),
        self._makeExample(prediction=1.0000, label=1.0000, fixed_int=0),
    ]
    fairness_auc = post_export_metrics.fairness_auc(subgroup_key='fixed_int')

    def check_result(got):
      try:
        self.assertEqual(1, len(got), 'got: %s' % got)
        (slice_key, value) = got[0]
        self.assertEqual((), slice_key)
        expected_value = {
            # Subgroup
            'post_export_metrics/fairness/auc/subgroup_auc/fixed_int':
                0.5,
            'post_export_metrics/fairness/auc/subgroup_auc/fixed_int/lower_bound':
                0.25,
            'post_export_metrics/fairness/auc/subgroup_auc/fixed_int/upper_bound':
                0.75,
            # BNSP
            'post_export_metrics/fairness/auc/bnsp_auc/fixed_int':
                0.5,
            'post_export_metrics/fairness/auc/bnsp_auc/fixed_int/lower_bound':
                0.25,
            'post_export_metrics/fairness/auc/bnsp_auc/fixed_int/upper_bound':
                0.75,
            # BPSN
            'post_export_metrics/fairness/auc/bpsn_auc/fixed_int':
                0.5,
            'post_export_metrics/fairness/auc/bpsn_auc/fixed_int/lower_bound':
                0.25,
            'post_export_metrics/fairness/auc/bpsn_auc/fixed_int/upper_bound':
                0.75,
            'average_loss':
                0.5,
        }
        self.assertDictElementsAlmostEqual(value, expected_value)
      except AssertionError as err:
        raise util.BeamAssertException(err)

    self._runTestWithCustomCheck(
        examples,
        eval_export_dir, [fairness_auc],
        custom_metrics_check=check_result)

  def testFairnessAucsWithFeatureSlices(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        fixed_prediction_estimator_extra_fields
        .simple_fixed_prediction_estimator_extra_fields(None,
                                                        temp_eval_export_dir))
    examples = [
        # Subgroup
        self._makeExample(
            prediction=0.0000, label=0.0000, fixed_int=1, fixed_string='a'),
        self._makeExample(
            prediction=0.0000, label=1.0000, fixed_int=1, fixed_string='a'),
        self._makeExample(
            prediction=1.0000, label=0.0000, fixed_int=1, fixed_string='a'),
        self._makeExample(
            prediction=1.0000, label=1.0000, fixed_int=1, fixed_string='a'),
        self._makeExample(
            prediction=0.0000, label=0.0000, fixed_int=1, fixed_string='b'),
        self._makeExample(
            prediction=0.0000, label=1.0000, fixed_int=1, fixed_string='b'),
        self._makeExample(
            prediction=1.0000, label=0.0000, fixed_int=1, fixed_string='b'),
        self._makeExample(
            prediction=1.0000, label=1.0000, fixed_int=1, fixed_string='b'),
        # Background
        self._makeExample(
            prediction=0.0000, label=0.0000, fixed_int=0, fixed_string='a'),
        self._makeExample(
            prediction=0.0000, label=1.0000, fixed_int=0, fixed_string='a'),
        self._makeExample(
            prediction=1.0000, label=0.0000, fixed_int=0, fixed_string='a'),
        self._makeExample(
            prediction=1.0000, label=1.0000, fixed_int=0, fixed_string='a'),
        self._makeExample(
            prediction=0.0000, label=0.0000, fixed_int=0, fixed_string='b'),
        self._makeExample(
            prediction=0.0000, label=1.0000, fixed_int=0, fixed_string='b'),
        self._makeExample(
            prediction=1.0000, label=0.0000, fixed_int=0, fixed_string='b'),
        self._makeExample(
            prediction=1.0000, label=1.0000, fixed_int=0, fixed_string='b'),
    ]
    fairness_auc = post_export_metrics.fairness_auc(subgroup_key='fixed_int')

    def check_result(got):
      try:
        self.assertEqual(3, len(got), 'got: %s' % got)
        for _, value in got:
          expected_value = {
              # Subgroup
              'post_export_metrics/fairness/auc/subgroup_auc/fixed_int':
                  0.5,
              'post_export_metrics/fairness/auc/subgroup_auc/fixed_int/lower_bound':
                  0.25,
              'post_export_metrics/fairness/auc/subgroup_auc/fixed_int/upper_bound':
                  0.75,
              # BNSP
              'post_export_metrics/fairness/auc/bnsp_auc/fixed_int':
                  0.5,
              'post_export_metrics/fairness/auc/bnsp_auc/fixed_int/lower_bound':
                  0.25,
              'post_export_metrics/fairness/auc/bnsp_auc/fixed_int/upper_bound':
                  0.75,
              # BPSN
              'post_export_metrics/fairness/auc/bpsn_auc/fixed_int':
                  0.5,
              'post_export_metrics/fairness/auc/bpsn_auc/fixed_int/lower_bound':
                  0.25,
              'post_export_metrics/fairness/auc/bpsn_auc/fixed_int/upper_bound':
                  0.75,
              'average_loss':
                  0.5,
          }
          self.assertDictElementsAlmostEqual(value, expected_value)

        # Check serialization too.
        output_metrics = metrics_for_slice_pb2.MetricsForSlice().metrics
        for slice_key, value in got:
          fairness_auc.populate_stats_and_pop(slice_key, value, output_metrics)
        for key in (
            metric_keys.FAIRNESS_AUC + '/subgroup_auc/fixed_int',
            metric_keys.FAIRNESS_AUC + '/bpsn_auc/fixed_int',
            metric_keys.FAIRNESS_AUC + '/bnsp_auc/fixed_int',
        ):
          self.assertProtoEquals(
              """
              bounded_value {
                lower_bound {
                  value: 0.2500001
                }
                upper_bound {
                  value: 0.7499999
                }
                value {
                  value: 0.5
                }
                methodology: RIEMANN_SUM
              }
              """, output_metrics[key])
      except AssertionError as err:
        raise util.BeamAssertException(err)

    self._runTestWithCustomCheck(
        examples,
        eval_export_dir, [fairness_auc],
        slice_spec=[
            slicer.SingleSliceSpec(),
            slicer.SingleSliceSpec(columns=['fixed_string'])
        ],
        custom_metrics_check=check_result)


if __name__ == '__main__':
  tf.test.main()
