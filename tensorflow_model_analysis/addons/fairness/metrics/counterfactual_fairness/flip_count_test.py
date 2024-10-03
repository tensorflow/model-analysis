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
"""Tests for flip count metric."""


import pytest
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis.addons.fairness.metrics.counterfactual_fairness import flip_count
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.proto import config_pb2

from google.protobuf import text_format


@pytest.mark.xfail(run=False, reason="PR 183 This class contains tests that fail and needs to be fixed. "
"If all tests pass, please remove this mark.")
class FlipCountTest(testutil.TensorflowModelAnalysisTest):

  def testFlipCount(self):
    computations = flip_count.FlipCount(
        thresholds=[0.3],
        counterfactual_prediction_key='counterfactual_pred_key',
        example_id_key='example_id_key',
    ).computations(example_weighted=True)
    binary_confusion_matrix = computations[0]
    matrices = computations[1]
    metrics = computations[2]
    # TODO(b/171180441): Handle absence of ground truth labels in counterfactual
    # examples while computing flip count metrics.
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
            'predictions': np.array([0.1, 0.7]),  # to test flattening
            'example_weights': np.array([3.0]),
            'features': {
                'counterfactual_pred_key': np.array([1.0, 0.1]),
                'example_id_key': np.array(['id_2']),
            },
        },
        {
            'labels': None,
            'predictions': np.array([0.5, 0.2]),
            'example_weights': np.array([2.0]),
            'features': {
                'counterfactual_pred_key': np.array([0.2, 0.4]),
                'example_id_key': np.array(['id_3']),
            },
        },
        {
            'labels': None,
            'predictions': np.array([0.2, 0.1]),
            'example_weights': np.array([1.0]),
            'features': {
                'counterfactual_pred_key': np.array([0.4, 0.5]),
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
          | 'ComputeMetrics' >> beam.Map(lambda x: (x[0], metrics.result(x[1])))
      )

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          self.assertLen(got_metrics, 6)
          self.assertDictElementsAlmostEqual(
              got_metrics,
              {
                  metric_types.MetricKey(
                      name='flip_count/positive_to_negative@0.3',
                      example_weighted=True,
                  ): 5.0,
                  metric_types.MetricKey(
                      name='flip_count/negative_to_positive@0.3',
                      example_weighted=True,
                  ): 7.0,
                  metric_types.MetricKey(
                      name='flip_count/positive_examples_count@0.3',
                      example_weighted=True,
                  ): 6.0,
                  metric_types.MetricKey(
                      name='flip_count/negative_examples_count@0.3',
                      example_weighted=True,
                  ): 7.0,
              },
          )
          self.assertAllEqual(
              got_metrics[
                  metric_types.MetricKey(
                      name='flip_count/positive_to_negative_examples_ids@0.3',
                      example_weighted=True,
                  )
              ],
              np.array([['id_2'], ['id_3']]),
          )
          self.assertAllEqual(
              got_metrics[
                  metric_types.MetricKey(
                      name='flip_count/negative_to_positive_examples_ids@0.3',
                      example_weighted=True,
                  )
              ],
              np.array([['id_2'], ['id_3'], ['id_4']]),
          )
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testFlipCountWitEvalConfig(self):
    eval_config = text_format.Parse(
        """
        model_specs: {
          name: "original"
        }
        model_specs: {
          name: "counterfactual"
          is_baseline: true
        }
        """,
        config_pb2.EvalConfig(),
    )
    computations = flip_count.FlipCount(
        thresholds=[0.3], example_id_key='example_id_key'
    ).computations(
        eval_config=eval_config,
        example_weighted=True,
        model_names=['original', 'counterfactual'],
        output_names=[''],
    )
    binary_confusion_matrix = computations[0]
    matrices = computations[1]
    metrics = computations[2]
    original_model_name = 'original'
    counterfactual_model_name = 'counterfactual'
    examples = [
        {
            'labels': None,
            'predictions': {
                original_model_name: np.array([0.5]),
                counterfactual_model_name: np.array([0.7]),
            },
            'example_weights': np.array([1.0]),
            'features': {
                'example_id_key': np.array(['id_1']),
            },
        },
        {
            'labels': None,
            'predictions': {
                original_model_name: np.array([0.1, 0.7]),  # to test flattening
                counterfactual_model_name: np.array([1.0, 0.1]),
            },
            'example_weights': np.array([3.0]),
            'features': {
                'example_id_key': np.array(['id_2']),
            },
        },
        {
            'labels': None,
            'predictions': {
                original_model_name: np.array([0.5, 0.2]),
                counterfactual_model_name: np.array([0.2, 0.4]),
            },
            'example_weights': np.array([2.0]),
            'features': {
                'example_id_key': np.array(['id_3']),
            },
        },
        {
            'labels': None,
            'predictions': {
                original_model_name: np.array([0.2, 0.1]),
                counterfactual_model_name: np.array([0.4, 0.5]),
            },
            'example_weights': np.array([1.0]),
            'features': {
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
          >> beam.Map(lambda x: (x[0], matrices.result(x[1])))
          | 'ComputeMetrics' >> beam.Map(lambda x: (x[0], metrics.result(x[1])))
      )

      # pylint: enable=no-value-for-parameter
      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          self.assertLen(got_metrics, 6)
          self.assertDictElementsAlmostEqual(
              got_metrics,
              {
                  metric_types.MetricKey(
                      name='flip_count/positive_to_negative@0.3',
                      model_name='original',
                      example_weighted=True,
                  ): 5.0,
                  metric_types.MetricKey(
                      name='flip_count/negative_to_positive@0.3',
                      model_name='original',
                      example_weighted=True,
                  ): 7.0,
                  metric_types.MetricKey(
                      name='flip_count/positive_examples_count@0.3',
                      model_name='original',
                      example_weighted=True,
                  ): 6.0,
                  metric_types.MetricKey(
                      name='flip_count/negative_examples_count@0.3',
                      model_name='original',
                      example_weighted=True,
                  ): 7.0,
              },
          )
          self.assertAllEqual(
              got_metrics[
                  metric_types.MetricKey(
                      name='flip_count/positive_to_negative_examples_ids@0.3',
                      model_name='original',
                      example_weighted=True,
                  )
              ],
              np.array([['id_2'], ['id_3']]),
          )
          self.assertAllEqual(
              got_metrics[
                  metric_types.MetricKey(
                      name='flip_count/negative_to_positive_examples_ids@0.3',
                      model_name='original',
                      example_weighted=True,
                  )
              ],
              np.array([['id_2'], ['id_3'], ['id_4']]),
          )
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testFlipCount_cfPredictionKeyMissing_raiseValueError(self):
    computations = flip_count.FlipCount(
        thresholds=[0.3],
        counterfactual_prediction_key='counterfactual_pred_key',
        example_id_key='example_id_key',
    ).computations()
    binary_confusion_matrix = computations[0]
    matrices = computations[1]
    metrics = computations[2]
    examples = [
        {
            'labels': None,
            'predictions': np.array([0.5]),
            'example_weights': np.array([1.0]),
            'features': {
                'example_id_key': np.array(['id_1']),
            },
        },
        {
            'labels': None,
            'predictions': np.array([0.2, 0.1]),
            'example_weights': np.array([1.0]),
            'features': {
                'counterfactual_pred_key': np.array([0.4, 0.5]),
                'example_id_key': np.array(['id_4']),
            },
        },
    ]

    with self.assertRaises(ValueError):
      with beam.Pipeline() as pipeline:
        # pylint: disable=no-value-for-parameter
        _ = (
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
            | 'FlipCount' >> beam.Map(lambda x: (x[0], metrics.result(x[1])))
        )

  def testFlipCount_cfPredictionValueNone_raiseValueError(self):
    computations = flip_count.FlipCount(
        thresholds=[0.3],
        counterfactual_prediction_key='counterfactual_pred_key',
        example_id_key='example_id_key',
    ).computations()
    binary_confusion_matrix = computations[0]
    matrices = computations[1]
    metrics = computations[2]
    examples = [
        {
            'labels': None,
            'predictions': np.array([0.5]),
            'example_weights': np.array([1.0]),
            'features': {
                'counterfactual_pred_key': None,
                'example_id_key': np.array(['id_1']),
            },
        },
        {
            'labels': None,
            'predictions': np.array([0.2, 0.1]),
            'example_weights': np.array([1.0]),
            'features': {
                'counterfactual_pred_key': np.array([0.4, 0.5]),
                'example_id_key': np.array(['id_4']),
            },
        },
    ]

    with self.assertRaises(ValueError):
      with beam.Pipeline() as pipeline:
        # pylint: disable=no-value-for-parameter
        _ = (
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
            | 'FlipCount' >> beam.Map(lambda x: (x[0], metrics.result(x[1])))
        )

  def testFlipCount_predictionKeysSizeMisMatch_raiseValueError(self):
    computations = flip_count.FlipCount(
        thresholds=[0.3],
        counterfactual_prediction_key='counterfactual_pred_key',
        example_id_key='example_id_key',
    ).computations()
    binary_confusion_matrix = computations[0]
    matrices = computations[1]
    metrics = computations[2]
    examples = [
        {
            'labels': None,
            'predictions': np.array([0.5]),
            'example_weights': np.array([1.0]),
            'features': {
                'counterfactual_pred_key': np.array([0.4, 0.5]),
                'example_id_key': np.array(['id_1']),
            },
        },
        {
            'labels': None,
            'predictions': np.array([0.2, 0.1]),
            'example_weights': np.array([1.0]),
            'features': {
                'counterfactual_pred_key': np.array([0.4, 0.5]),
                'example_id_key': np.array(['id_4']),
            },
        },
    ]

    with self.assertRaises(ValueError):
      with beam.Pipeline() as pipeline:
        # pylint: disable=no-value-for-parameter
        _ = (
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
            | 'FlipCount' >> beam.Map(lambda x: (x[0], metrics.result(x[1])))
        )

  def testFlipCount_predictionIsEmpty_raiseValueError(self):
    computations = flip_count.FlipCount(
        thresholds=[0.3],
        counterfactual_prediction_key='counterfactual_pred_key',
        example_id_key='example_id_key',
    ).computations()
    binary_confusion_matrix = computations[0]
    matrices = computations[1]
    metrics = computations[2]
    examples = [
        {
            'labels': None,
            'predictions': np.array([]),
            'example_weights': np.array([1.0]),
            'features': {
                'counterfactual_pred_key': np.array([0.4, 0.5]),
                'example_id_key': np.array(['id_1']),
            },
        },
        {
            'labels': None,
            'predictions': np.array([0.2, 0.1]),
            'example_weights': np.array([1.0]),
            'features': {
                'counterfactual_pred_key': np.array([0.4, 0.5]),
                'example_id_key': np.array(['id_4']),
            },
        },
    ]

    with self.assertRaises(ValueError):
      with beam.Pipeline() as pipeline:
        # pylint: disable=no-value-for-parameter
        _ = (
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
            | 'FlipCount' >> beam.Map(lambda x: (x[0], metrics.result(x[1])))
        )


