# Copyright 2020 Google LLC
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
"""Tests for flip rate metric."""


import pytest
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis.addons.fairness.metrics.counterfactual_fairness import flip_rate
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util


@pytest.mark.xfail(run=False, reason="PR 183 This class contains tests that fail and needs to be fixed. "
"If all tests pass, please remove this mark.")
class FlipRateTest(testutil.TensorflowModelAnalysisTest):

  def testFlipRate(self):
    computations = flip_rate.FlipRate(
        thresholds=[0.3],
        counterfactual_prediction_key='counterfactual_pred_key',
        example_id_key='example_id_key',
    ).computations(example_weighted=True)
    binary_confusion_matrix = computations[0]
    matrices = computations[1]
    flip_count_metrics = computations[2]
    flip_rate_metrics = computations[3]
    examples = [
        {
            'labels': None,
            'predictions': np.array([0.5]),
            'example_weights': np.array([1.0]),
            'features': {
                'counterfactual_pred_key': np.array([0.7]),
                'example_id_key': np.array(['id_1']),
            },
        },
        {
            'labels': None,
            'predictions': np.array([0.1]),
            'example_weights': np.array([3.0]),
            'features': {
                'counterfactual_pred_key': np.array([1.0]),
                'example_id_key': np.array(['id_2']),
            },
        },
        {
            'labels': None,
            'predictions': np.array([0.5]),
            'example_weights': np.array([2.0]),
            'features': {
                'counterfactual_pred_key': np.array([0.2]),
                'example_id_key': np.array(['id_3']),
            },
        },
        {
            'labels': None,
            'predictions': np.array([0.2]),
            'example_weights': np.array([1.0]),
            'features': {
                'counterfactual_pred_key': np.array([0.4]),
                'example_id_key': np.array(['id_4']),
            },
        },
    ]

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create(examples)
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs, True)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeBinaryConfusionMatrix'
          >> beam.CombinePerKey(binary_confusion_matrix.combiner)
          | 'ComputeMatrices'
          >> beam.Map(
              lambda x: (x[0], matrices.result(x[1]))
          )  # pyformat: ignore
          | 'ComputeFlipCount'
          >> beam.Map(lambda x: (x[0], flip_count_metrics.result(x[1])))
          | 'ComputeFlipRate'
          >> beam.Map(lambda x: (x[0], flip_rate_metrics.result(x[1])))
      )  # pyformat: ignore

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          self.assertLen(got_metrics, 5)
          self.assertDictElementsAlmostEqual(
              got_metrics,
              {
                  metric_types.MetricKey(
                      name='flip_rate/overall@0.3', example_weighted=True
                  ): 0.85714286,
                  metric_types.MetricKey(
                      name='flip_rate/positive_to_negative@0.3',
                      example_weighted=True,
                  ): 0.28571428,
                  metric_types.MetricKey(
                      name='flip_rate/negative_to_positive@0.3',
                      example_weighted=True,
                  ): 0.57142857,
              },
          )
          self.assertAllEqual(
              got_metrics[
                  metric_types.MetricKey(
                      name='flip_rate/positive_to_negative_examples_ids@0.3',
                      example_weighted=True,
                  )
              ],
              np.array([['id_3']]),
          )
          self.assertAllEqual(
              got_metrics[
                  metric_types.MetricKey(
                      name='flip_rate/negative_to_positive_examples_ids@0.3',
                      example_weighted=True,
                  )
              ],
              np.array([['id_2'], ['id_4']]),
          )
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testUnweightedFlipRate(self):
    computations = flip_rate.FlipRate(
        thresholds=[0.3],
        counterfactual_prediction_key='counterfactual_pred_key',
        example_id_key='example_id_key',
    ).computations(example_weighted=False)
    binary_confusion_matrix = computations[0]
    matrices = computations[1]
    flip_count_metrics = computations[2]
    flip_rate_metrics = computations[3]
    examples = [
        {
            'labels': None,
            'predictions': np.array([0.5]),
            'features': {
                'counterfactual_pred_key': np.array([0.7]),
                'example_id_key': np.array(['id_1']),
            },
        },
        {
            'labels': None,
            'predictions': np.array([0.1]),
            'features': {
                'counterfactual_pred_key': np.array([1.0]),
                'example_id_key': np.array(['id_2']),
            },
        },
        {
            'labels': None,
            'predictions': np.array([0.5]),
            'features': {
                'counterfactual_pred_key': np.array([0.2]),
                'example_id_key': np.array(['id_3']),
            },
        },
        {
            'labels': None,
            'predictions': np.array([0.2]),
            'features': {
                'counterfactual_pred_key': np.array([0.4]),
                'example_id_key': np.array(['id_4']),
            },
        },
    ]

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create(examples)
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs, True)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeBinaryConfusionMatrix'
          >> beam.CombinePerKey(binary_confusion_matrix.combiner)
          | 'ComputeMatrices'
          >> beam.Map(
              lambda x: (x[0], matrices.result(x[1]))
          )  # pyformat: ignore
          | 'ComputeFlipCount'
          >> beam.Map(lambda x: (x[0], flip_count_metrics.result(x[1])))
          | 'ComputeFlipRate'
          >> beam.Map(lambda x: (x[0], flip_rate_metrics.result(x[1])))
      )  # pyformat: ignore

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          self.assertLen(got_metrics, 6)
          self.assertSameElements(
              got_metrics[
                  metric_types.MetricKey(
                      name='flip_rate/sample_examples_ids@0.3',
                      example_weighted=False,
                  )
              ],
              np.array([['id_2'], ['id_3'], ['id_4']]),
          )
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')


