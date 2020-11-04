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
"""Test for using the MetricsAndPlotsEvaluator API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.api import tfma_unit
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.eval_saved_model.example_trainers import fixed_prediction_estimator
from tensorflow_model_analysis.eval_saved_model.example_trainers import fixed_prediction_estimator_no_labels
from tensorflow_model_analysis.eval_saved_model.example_trainers import linear_classifier
from tensorflow_model_analysis.evaluators import legacy_metrics_and_plots_evaluator as metrics_and_plots_evaluator
from tensorflow_model_analysis.extractors import legacy_predict_extractor
from tensorflow_model_analysis.extractors import slice_key_extractor
from tensorflow_model_analysis.post_export_metrics import metric_keys
from tensorflow_model_analysis.post_export_metrics import metrics as metric_fns
from tensorflow_model_analysis.post_export_metrics import post_export_metrics
from tensorflow_model_analysis.slicer import slicer_lib as slicer


def _addExampleCountMetricCallback(  # pylint: disable=invalid-name
    features_dict, predictions_dict, labels_dict):
  del features_dict
  del labels_dict
  metric_ops = {}
  value_op, update_op = metric_fns.total(
      tf.shape(input=predictions_dict['logits'])[0])
  metric_ops['added_example_count'] = (value_op, update_op)
  return metric_ops


def _addPyFuncMetricCallback(  # pylint: disable=invalid-name
    features_dict, predictions_dict, labels_dict):
  del features_dict
  del predictions_dict

  total_value = tf.compat.v1.Variable(
      initial_value=0.0,
      dtype=tf.float64,
      trainable=False,
      collections=[
          tf.compat.v1.GraphKeys.METRIC_VARIABLES,
          tf.compat.v1.GraphKeys.LOCAL_VARIABLES
      ],
      validate_shape=True,
      name='total')

  def my_func(x):
    return np.sum(x, dtype=np.float64)

  update_op = tf.compat.v1.assign_add(
      total_value, tf.compat.v1.py_func(my_func, [labels_dict], tf.float64))
  value_op = tf.identity(total_value)
  metric_ops = {}
  metric_ops['py_func_label_sum'] = (value_op, update_op)
  return metric_ops


class EvaluateMetricsAndPlotsTest(testutil.TensorflowModelAnalysisTest):

  def setUp(self):
    super(EvaluateMetricsAndPlotsTest, self).setUp()
    self.longMessage = True  # pylint: disable=invalid-name

  def _getEvalExportDir(self):
    return os.path.join(self._getTempDir(), 'eval_export_dir')

  def testEvaluateNoSlicing(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = linear_classifier.simple_linear_classifier(
        None, temp_eval_export_dir)
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=eval_export_dir,
        add_metrics_callbacks=[_addExampleCountMetricCallback])
    extractors = [
        legacy_predict_extractor.PredictExtractor(eval_shared_model),
        slice_key_extractor.SliceKeyExtractor()
    ]

    with beam.Pipeline() as pipeline:
      example1 = self._makeExample(age=3.0, language='english', label=1.0)
      example2 = self._makeExample(age=3.0, language='chinese', label=0.0)
      example3 = self._makeExample(age=4.0, language='english', label=1.0)
      example4 = self._makeExample(age=5.0, language='chinese', label=0.0)

      (metrics, _), _ = (
          pipeline
          | 'Create' >> beam.Create([
              example1.SerializeToString(),
              example2.SerializeToString(),
              example3.SerializeToString(),
              example4.SerializeToString()
          ])
          | 'InputsToExtracts' >> model_eval_lib.InputsToExtracts()
          | 'Extract' >> tfma_unit.Extract(extractors=extractors)  # pylint: disable=no-value-for-parameter
          | 'ComputeMetricsAndPlots' >> metrics_and_plots_evaluator
          .ComputeMetricsAndPlots(eval_shared_model=eval_shared_model))

      def check_result(got):
        try:
          self.assertEqual(1, len(got), 'got: %s' % got)
          (slice_key, value) = got[0]
          self.assertEqual((), slice_key)
          self.assertDictElementsAlmostEqual(
              value, {
                  'accuracy': 1.0,
                  'label/mean': 0.5,
                  'my_mean_age': 3.75,
                  'my_mean_age_times_label': 1.75,
                  'added_example_count': 4.0
              })
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(metrics, check_result)

  def testEvaluateWithSlicingAndDifferentBatchSizes(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = linear_classifier.simple_linear_classifier(
        None, temp_eval_export_dir)
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=eval_export_dir,
        add_metrics_callbacks=[_addExampleCountMetricCallback])
    extractors = [
        legacy_predict_extractor.PredictExtractor(eval_shared_model),
        slice_key_extractor.SliceKeyExtractor([
            slicer.SingleSliceSpec(),
            slicer.SingleSliceSpec(columns=['slice_key'])
        ])
    ]

    for batch_size in [1, 2, 4, 8]:

      with beam.Pipeline() as pipeline:
        example1 = self._makeExample(
            age=3.0, language='english', label=1.0, slice_key='first_slice')
        example2 = self._makeExample(
            age=3.0, language='chinese', label=0.0, slice_key='first_slice')
        example3 = self._makeExample(
            age=4.0, language='english', label=0.0, slice_key='second_slice')
        example4 = self._makeExample(
            age=5.0, language='chinese', label=1.0, slice_key='second_slice')
        example5 = self._makeExample(
            age=5.0, language='chinese', label=1.0, slice_key='second_slice')

        (metrics, plots), _ = (
            pipeline
            | 'Create' >> beam.Create([
                example1.SerializeToString(),
                example2.SerializeToString(),
                example3.SerializeToString(),
                example4.SerializeToString(),
                example5.SerializeToString(),
            ])
            | 'InputsToExtracts' >> model_eval_lib.InputsToExtracts()
            | 'Extract' >> tfma_unit.Extract(extractors=extractors)  # pylint:disable=no-value-for-parameter
            | 'ComputeMetricsAndPlots' >>
            metrics_and_plots_evaluator.ComputeMetricsAndPlots(
                eval_shared_model=eval_shared_model,
                desired_batch_size=batch_size))

        def check_result(got):
          try:
            self.assertEqual(3, len(got), 'got: %s' % got)
            slices = {}
            for slice_key, value in got:
              slices[slice_key] = value
            overall_slice = ()
            first_slice = (('slice_key', 'first_slice'),)
            second_slice = (('slice_key', 'second_slice'),)
            self.assertCountEqual(
                list(slices.keys()), [overall_slice, first_slice, second_slice])
            self.assertDictElementsAlmostEqual(
                slices[overall_slice], {
                    'accuracy': 0.4,
                    'label/mean': 0.6,
                    'my_mean_age': 4.0,
                    'my_mean_age_times_label': 2.6,
                    'added_example_count': 5.0
                })
            self.assertDictElementsAlmostEqual(
                slices[first_slice], {
                    'accuracy': 1.0,
                    'label/mean': 0.5,
                    'my_mean_age': 3.0,
                    'my_mean_age_times_label': 1.5,
                    'added_example_count': 2.0
                })
            self.assertDictElementsAlmostEqual(
                slices[second_slice], {
                    'accuracy': 0.0,
                    'label/mean': 2.0 / 3.0,
                    'my_mean_age': 14.0 / 3.0,
                    'my_mean_age_times_label': 10.0 / 3.0,
                    'added_example_count': 3.0
                })

          except AssertionError as err:
            # This function is redefined every iteration, so it will have the
            # right value of batch_size.
            raise util.BeamAssertException('batch_size = %d, error: %s' %
                                           (batch_size, err))  # pylint: disable=cell-var-from-loop

        util.assert_that(metrics, check_result, label='metrics')
        util.assert_that(plots, util.is_empty(), label='plots')

  def testEvaluateWithSlicingAndUncertainty(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = linear_classifier.simple_linear_classifier(
        None, temp_eval_export_dir)
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=eval_export_dir,
        add_metrics_callbacks=[_addExampleCountMetricCallback])
    extractors = [
        legacy_predict_extractor.PredictExtractor(eval_shared_model),
        slice_key_extractor.SliceKeyExtractor([
            slicer.SingleSliceSpec(),
            slicer.SingleSliceSpec(columns=['slice_key'])
        ])
    ]

    for batch_size in [1, 2, 4, 8]:

      with beam.Pipeline() as pipeline:
        example1 = self._makeExample(
            age=3.0, language='english', label=1.0, slice_key='first_slice')
        example2 = self._makeExample(
            age=3.0, language='chinese', label=0.0, slice_key='first_slice')
        example3 = self._makeExample(
            age=4.0, language='english', label=0.0, slice_key='second_slice')
        example4 = self._makeExample(
            age=5.0, language='chinese', label=1.0, slice_key='second_slice')
        example5 = self._makeExample(
            age=5.0, language='chinese', label=1.0, slice_key='second_slice')

        (metrics, _), _ = (
            pipeline
            | 'Create' >> beam.Create([
                example1.SerializeToString(),
                example2.SerializeToString(),
                example3.SerializeToString(),
                example4.SerializeToString(),
                example5.SerializeToString(),
            ])
            | 'InputsToExtracts' >> model_eval_lib.InputsToExtracts()
            | 'Extract' >> tfma_unit.Extract(extractors=extractors)  # pylint: disable=no-value-for-parameter
            | 'ComputeMetricsAndPlots' >>
            metrics_and_plots_evaluator.ComputeMetricsAndPlots(
                eval_shared_model=eval_shared_model,
                desired_batch_size=batch_size,
                compute_confidence_intervals=True))

        def check_result(got):
          try:
            self.assertEqual(3, len(got), 'got: %s' % got)
            slices = {}
            for slice_key, value in got:
              slices[slice_key] = value
            overall_slice = ()
            first_slice = (('slice_key', 'first_slice'),)
            second_slice = (('slice_key', 'second_slice'),)
            self.assertCountEqual(
                list(slices.keys()), [overall_slice, first_slice, second_slice])
            self.assertDictElementsWithTDistributionAlmostEqual(
                slices[overall_slice], {
                    'accuracy': 0.4,
                    'label/mean': 0.6,
                    'my_mean_age': 4.0,
                    'my_mean_age_times_label': 2.6,
                    'added_example_count': 5.0
                })
            self.assertDictElementsWithTDistributionAlmostEqual(
                slices[first_slice], {
                    'accuracy': 1.0,
                    'label/mean': 0.5,
                    'my_mean_age': 3.0,
                    'my_mean_age_times_label': 1.5,
                    'added_example_count': 2.0
                })
            self.assertDictElementsWithTDistributionAlmostEqual(
                slices[second_slice], {
                    'accuracy': 0.0,
                    'label/mean': 2.0 / 3.0,
                    'my_mean_age': 14.0 / 3.0,
                    'my_mean_age_times_label': 10.0 / 3.0,
                    'added_example_count': 3.0
                })

          except AssertionError as err:
            # This function is redefined every iteration, so it will have the
            # right value of batch_size.
            raise util.BeamAssertException('batch_size = %d, error: %s' %
                                           (batch_size, err))  # pylint: disable=cell-var-from-loop

        util.assert_that(metrics, check_result, label='metrics')

  def testEvaluateNoSlicingAddPostExportAndCustomMetrics(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = linear_classifier.simple_linear_classifier(
        None, temp_eval_export_dir)
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=eval_export_dir,
        add_metrics_callbacks=[
            _addExampleCountMetricCallback,
            # Note that since everything runs in-process this doesn't
            # actually test that the py_func can be correctly recreated
            # on workers in a distributed context.
            _addPyFuncMetricCallback,
            post_export_metrics.example_count(),
            post_export_metrics.example_weight(example_weight_key='age')
        ])
    extractors = [
        legacy_predict_extractor.PredictExtractor(eval_shared_model),
        slice_key_extractor.SliceKeyExtractor()
    ]

    with beam.Pipeline() as pipeline:
      example1 = self._makeExample(age=3.0, language='english', label=1.0)
      example2 = self._makeExample(age=3.0, language='chinese', label=0.0)
      example3 = self._makeExample(age=4.0, language='english', label=1.0)
      example4 = self._makeExample(age=5.0, language='chinese', label=0.0)

      (metrics, plots), _ = (
          pipeline
          | 'Create' >> beam.Create([
              example1.SerializeToString(),
              example2.SerializeToString(),
              example3.SerializeToString(),
              example4.SerializeToString()
          ])
          | 'InputsToExtracts' >> model_eval_lib.InputsToExtracts()
          | 'Extract' >> tfma_unit.Extract(extractors=extractors)  # pylint: disable=no-value-for-parameter
          | 'ComputeMetricsAndPlots' >> metrics_and_plots_evaluator
          .ComputeMetricsAndPlots(eval_shared_model=eval_shared_model))

      def check_result(got):
        try:
          self.assertEqual(1, len(got), 'got: %s' % got)
          (slice_key, value) = got[0]
          self.assertEqual((), slice_key)
          self.assertDictElementsAlmostEqual(
              got_values_dict=value,
              expected_values_dict={
                  'accuracy': 1.0,
                  'label/mean': 0.5,
                  'my_mean_age': 3.75,
                  'my_mean_age_times_label': 1.75,
                  'added_example_count': 4.0,
                  'py_func_label_sum': 2.0,
                  metric_keys.EXAMPLE_COUNT: 4.0,
                  metric_keys.EXAMPLE_WEIGHT: 15.0
              })
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(metrics, check_result, label='metrics')
      util.assert_that(plots, util.is_empty(), label='plots')

  def testEvaluateNoSlicingAddPostExportAndCustomMetricsUnsupervisedModel(self):
    # Mainly for testing that the ExampleCount post export metric works with
    # unsupervised models.
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        fixed_prediction_estimator_no_labels
        .simple_fixed_prediction_estimator_no_labels(None,
                                                     temp_eval_export_dir))
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=eval_export_dir,
        add_metrics_callbacks=[
            post_export_metrics.example_count(),
            post_export_metrics.example_weight(example_weight_key='prediction')
        ])
    extractors = [
        legacy_predict_extractor.PredictExtractor(eval_shared_model),
        slice_key_extractor.SliceKeyExtractor()
    ]

    with beam.Pipeline() as pipeline:
      example1 = self._makeExample(prediction=1.0)
      example2 = self._makeExample(prediction=2.0)

      (metrics, plots), _ = (
          pipeline
          | 'Create' >> beam.Create([
              example1.SerializeToString(),
              example2.SerializeToString(),
          ])
          | 'InputsToExtracts' >> model_eval_lib.InputsToExtracts()
          | 'Extract' >> tfma_unit.Extract(extractors=extractors)  # pylint: disable=no-value-for-parameter
          | 'ComputeMetricsAndPlots' >> metrics_and_plots_evaluator
          .ComputeMetricsAndPlots(eval_shared_model=eval_shared_model))

      def check_result(got):
        try:
          self.assertEqual(1, len(got), 'got: %s' % got)
          (slice_key, value) = got[0]
          self.assertEqual((), slice_key)
          self.assertDictElementsAlmostEqual(
              got_values_dict=value,
              expected_values_dict={
                  'average_loss': 2.5,
                  metric_keys.EXAMPLE_COUNT: 2.0,
                  metric_keys.EXAMPLE_WEIGHT: 3.0
              })
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(metrics, check_result, label='metrics')
      util.assert_that(plots, util.is_empty(), label='plots')

  def testEvaluateWithPlots(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        fixed_prediction_estimator.simple_fixed_prediction_estimator(
            None, temp_eval_export_dir))
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=eval_export_dir,
        add_metrics_callbacks=[
            post_export_metrics.example_count(),
            post_export_metrics.auc_plots()
        ])
    extractors = [
        legacy_predict_extractor.PredictExtractor(eval_shared_model),
        slice_key_extractor.SliceKeyExtractor()
    ]

    with beam.Pipeline() as pipeline:
      example1 = self._makeExample(prediction=0.0, label=1.0)
      example2 = self._makeExample(prediction=0.7, label=0.0)
      example3 = self._makeExample(prediction=0.8, label=1.0)
      example4 = self._makeExample(prediction=1.0, label=1.0)

      (metrics, plots), _ = (
          pipeline
          | 'Create' >> beam.Create([
              example1.SerializeToString(),
              example2.SerializeToString(),
              example3.SerializeToString(),
              example4.SerializeToString()
          ])
          | 'InputsToExtracts' >> model_eval_lib.InputsToExtracts()
          | 'Extract' >> tfma_unit.Extract(extractors=extractors)  # pylint: disable=no-value-for-parameter
          | 'ComputeMetricsAndPlots' >> metrics_and_plots_evaluator
          .ComputeMetricsAndPlots(eval_shared_model=eval_shared_model))

      def check_metrics(got):
        try:
          self.assertEqual(1, len(got), 'got: %s' % got)
          (slice_key, value) = got[0]
          self.assertEqual((), slice_key)
          self.assertDictElementsAlmostEqual(
              got_values_dict=value,
              expected_values_dict={
                  metric_keys.EXAMPLE_COUNT: 4.0,
              })
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(metrics, check_metrics, label='metrics')

      def check_plots(got):
        try:
          self.assertEqual(1, len(got), 'got: %s' % got)
          (slice_key, value) = got[0]
          self.assertEqual((), slice_key)
          self.assertDictMatrixRowsAlmostEqual(
              got_values_dict=value,
              expected_values_dict={
                  metric_keys.AUC_PLOTS_MATRICES: [
                      (8001, [2, 1, 0, 1, 1.0 / 1.0, 1.0 / 3.0])
                  ],
              })
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(plots, check_plots, label='plots')


if __name__ == '__main__':
  tf.test.main()
