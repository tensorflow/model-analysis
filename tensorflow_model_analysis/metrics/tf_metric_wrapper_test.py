# Copyright 2019 Google LLC
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
"""Tests for TF metric wrapper."""


import pytest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.metrics import tf_metric_wrapper
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.utils import test_util
from tensorflow_model_analysis.utils.keras_lib import tf_keras


class _CustomMetric(tf_keras.metrics.Mean):

  def __init__(self, name='custom', dtype=None, update_y_pred=True):
    super().__init__(name=name, dtype=dtype)
    self.update_y_pred = update_y_pred

  def update_state(self, y_true, y_pred, sample_weight):
    return super().update_state(
        y_pred if self.update_y_pred else y_true, sample_weight=sample_weight
    )

  def get_config(self):
    cfg = super().get_config()
    cfg.update({'update_y_pred': self.update_y_pred})
    return cfg


class _CustomConfusionMatrixMetric(tf_keras.metrics.Precision):

  def __init__(self, name='custom', dtype=None):
    super().__init__(name=name, dtype=dtype)

  def update_state(self, y_true, y_pred, sample_weight):
    super().update_state(y_true, y_pred, sample_weight=sample_weight)

  def get_config(self):
    # Remove config items we don't accept or they will be passed to __init__.
    base_config = super().get_config()
    return {'name': base_config['name'], 'dtype': base_config['dtype']}


class _CustomMeanSquaredError(tf_keras.metrics.MeanSquaredError):

  def __init__(self, name, dtype=None):
    super().__init__(name=name, dtype=dtype)

  def result(self):
    mse = super().result()
    return {'mse': mse, 'one_minus_mse': 1 - mse}


@pytest.mark.xfail(run=False, reason="PR 183 This class contains tests that fail and needs to be fixed. "
"If all tests pass, please remove this mark.")
class ConfusionMatrixMetricsTest(
    test_util.TensorflowModelAnalysisTest, parameterized.TestCase
):

  #  This is needed because of pickling errors when using
  #  parameterized.named_parameters with TF metric types.
  def _tf_metric_by_name(self, metric_name):
    """Returns instance of tf_keras.metric with default args given name."""
    if metric_name == 'auc':
      return tf_keras.metrics.AUC(name='auc')
    elif metric_name == 'auc_pr':
      return tf_keras.metrics.AUC(name='auc_pr', curve='PR')
    elif metric_name == 'precision':
      return tf_keras.metrics.Precision(name='precision')
    elif metric_name == 'precision@2':
      return tf_keras.metrics.Precision(name='precision@2', top_k=2)
    elif metric_name == 'precision@3':
      return tf_keras.metrics.Precision(name='precision@3', top_k=3)
    elif metric_name == 'recall':
      return tf_keras.metrics.Recall(name='recall')
    elif metric_name == 'recall@2':
      return tf_keras.metrics.Recall(name='recall@2', top_k=2)
    elif metric_name == 'recall@3':
      return tf_keras.metrics.Recall(name='recall@3', top_k=3)
    elif metric_name == 'true_positives':
      return tf_keras.metrics.TruePositives(name='true_positives')
    elif metric_name == 'false_positives':
      return tf_keras.metrics.FalsePositives(name='false_positives')
    elif metric_name == 'true_negatives':
      return tf_keras.metrics.TrueNegatives(name='true_negatives')
    elif metric_name == 'false_negatives':
      return tf_keras.metrics.FalseNegatives(name='false_negatives')
    elif metric_name == 'specificity_at_sensitivity':
      return tf_keras.metrics.SpecificityAtSensitivity(
          0.5, name='specificity_at_sensitivity'
      )
    elif metric_name == 'sensitivity_at_specificity':
      return tf_keras.metrics.SensitivityAtSpecificity(
          0.5, name='sensitivity_at_specificity'
      )

  @parameterized.named_parameters(
      ('auc', 'auc', 0.75),
      ('auc_pr', 'auc_pr', 0.79727),
      ('precision', 'precision', 1.0),
      ('recall', 'recall', 0.5),
      ('true_positives', 'true_positives', 1.0),
      ('false_positives', 'false_positives', 0.0),
      ('true_negatives', 'true_negatives', 2.0),
      ('false_negatives', 'false_negatives', 1.0),
      ('specificity_at_sensitivity', 'specificity_at_sensitivity', 1.0),
      ('sensitivity_at_specificity', 'sensitivity_at_specificity', 1.0),
  )
  def testMetricsWithoutWeights(self, metric_name, expected_value):
    # TODO (b/151636380): remove when CL/299961405 is propagated through Kokoro.
    if metric_name == 'specificity_at_sensitivity':
      fix_present = hasattr(
          tf_keras.metrics.SpecificityAtSensitivity,
          '_find_max_under_constraint',
      )
      if not fix_present:
        expected_value = 0.5
    computations = tf_metric_wrapper.tf_metric_computations(
        [self._tf_metric_by_name(metric_name)], example_weighted=False
    )
    histogram = computations[0]
    matrix = computations[1]
    metric = computations[2]

    example1 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.0]),
        'example_weights': np.array([0.1]),  # ignored, example_weighted=False
    }
    example2 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.5]),
        'example_weights': np.array([0.2]),  # ignored, example_weighted=False
    }
    example3 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.3]),
        'example_weights': np.array([0.3]),  # ignored, example_weighted=False
    }
    example4 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.9]),
        'example_weights': np.array([0.4]),  # ignored, example_weighted=False
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example1, example2, example3, example4])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeHistogram' >> beam.CombinePerKey(histogram.combiner)
          | 'ComputeConfusionMatrix'
          >> beam.Map(lambda x: (x[0], matrix.result(x[1])))
          | 'ComputeMetric' >> beam.Map(lambda x: (x[0], metric.result(x[1])))
      )

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          key = metric_types.MetricKey(name=metric_name, example_weighted=False)
          self.assertDictElementsAlmostEqual(
              got_metrics, {key: expected_value}, places=5
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  @parameterized.named_parameters(
      ('auc', 'auc', 0.64286),
      ('auc_pr', 'auc_pr', 0.37467),
      ('precision', 'precision', 0.5833333),
      ('recall', 'recall', 1.0),
      ('true_positives', 'true_positives', 0.7),
      ('false_positives', 'false_positives', 0.5),
      ('true_negatives', 'true_negatives', 0.9),
      ('false_negatives', 'false_negatives', 0.0),
      ('specificity_at_sensitivity', 'specificity_at_sensitivity', 0.642857),
      ('sensitivity_at_specificity', 'sensitivity_at_specificity', 1.0),
  )
  def testMetricsWithWeights(self, metric_name, expected_value):
    # TODO (b/151636380): remove when CL/299961405 is propagated through Kokoro.
    if metric_name == 'specificity_at_sensitivity':
      fix_present = hasattr(
          tf_keras.metrics.SpecificityAtSensitivity,
          '_find_max_under_constraint',
      )
      if not fix_present:
        expected_value = 0.0

    computations = tf_metric_wrapper.tf_metric_computations(
        [self._tf_metric_by_name(metric_name)], example_weighted=True
    )
    histogram = computations[0]
    matrix = computations[1]
    metric = computations[2]

    example1 = {
        'labels': np.array([0.0]),
        'predictions': np.array([1.0]),
        'example_weights': np.array([0.5]),
    }
    example2 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.7]),
        'example_weights': np.array([0.7]),
    }
    example3 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.5]),
        'example_weights': np.array([0.9]),
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example1, example2, example3])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeHistogram' >> beam.CombinePerKey(histogram.combiner)
          | 'ComputeConfusionMatrix'
          >> beam.Map(lambda x: (x[0], matrix.result(x[1])))
          | 'ComputeMetric' >> beam.Map(lambda x: (x[0], metric.result(x[1])))
      )

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          key = metric_types.MetricKey(name=metric_name, example_weighted=True)
          self.assertDictElementsAlmostEqual(
              got_metrics, {key: expected_value}, places=5
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  @parameterized.named_parameters(
      ('auc', 'auc', 0.8571428),
      ('auc_pr', 'auc_pr', 0.77369833),
      ('true_positives', 'true_positives', 1.4),
      ('false_positives', 'false_positives', 0.6),
      ('true_negatives', 'true_negatives', 1.0),
      ('false_negatives', 'false_negatives', 0.0),
  )
  def testMetricsWithFractionalLabels(self, metric_name, expected_value):
    computations = tf_metric_wrapper.tf_metric_computations(
        [self._tf_metric_by_name(metric_name)]
    )
    histogram = computations[0]
    matrix = computations[1]
    metric = computations[2]

    # The following examples will be expanded to:
    #
    # prediction | label | weight
    #     0.0    |   -   |  1.0
    #     0.7    |   -   |  0.4
    #     0.7    |   +   |  0.6
    #     1.0    |   -   |  0.2
    #     1.0    |   +   |  0.8
    example1 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.0]),
        'example_weights': np.array([1.0]),
    }
    example2 = {
        'labels': np.array([0.6]),
        'predictions': np.array([0.7]),
        'example_weights': np.array([1.0]),
    }
    example3 = {
        'labels': np.array([0.8]),
        'predictions': np.array([1.0]),
        'example_weights': np.array([1.0]),
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example1, example2, example3])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeHistogram' >> beam.CombinePerKey(histogram.combiner)
          | 'ComputeConfusionMatrix'
          >> beam.Map(lambda x: (x[0], matrix.result(x[1])))
          | 'ComputeMetric' >> beam.Map(lambda x: (x[0], metric.result(x[1])))
      )

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          key = metric_types.MetricKey(name=metric_name)
          self.assertDictElementsAlmostEqual(
              got_metrics, {key: expected_value}, places=5
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  @parameterized.named_parameters(
      ('precision@2', 'precision', 2, 1.6 / (1.6 + 3.2)),
      ('recall@2', 'recall', 2, 1.6 / (1.6 + 0.8)),
      ('precision@3', 'precision', 3, 1.9 / (1.9 + 5.3)),
      ('recall@3', 'recall', 3, 1.9 / (1.9 + 0.5)),
  )
  def testMultiClassMetricsUsingConfusionMatrix(
      self, metric_name, top_k, expected_value
  ):
    computations = tf_metric_wrapper.tf_metric_computations(
        [self._tf_metric_by_name(metric_name)],
        sub_key=metric_types.SubKey(top_k=top_k),
        example_weighted=True,
    )
    histogram = computations[0]
    matrix = computations[1]
    metric = computations[2]

    # top_k = 2
    #   TP = 0.5*0 + 0.7*1 + 0.9*1 + 0.3*0 = 1.6
    #   FP = 0.5*2 + 0.7*1 + 0.9*1 + 0.3*2 = 3.2
    #   FN = 0.5*1 + 0.7*0 + 0.9*0 + 0.3*1 = 0.8
    #
    # top_k = 3
    #   TP = 0.5*0 + 0.7*1 + 0.9*1 + 0.3*1 = 1.9
    #   FP = 0.5*3 + 0.7*2 + 0.9*2 + 0.3*2 = 5.3
    #   FN = 0.5*1 + 0.7*0 + 0.9*0 + 0.3*0 = 0.5
    example1 = {
        'labels': np.array([2]),
        'predictions': np.array([0.1, 0.2, 0.1, 0.25, 0.35]),
        'example_weights': np.array([0.5]),
    }
    example2 = {
        'labels': np.array([1]),
        'predictions': np.array([0.2, 0.3, 0.05, 0.15, 0.3]),
        'example_weights': np.array([0.7]),
    }
    example3 = {
        'labels': np.array([3]),
        'predictions': np.array([0.01, 0.2, 0.09, 0.5, 0.2]),
        'example_weights': np.array([0.9]),
    }
    example4 = {
        'labels': np.array([1]),
        'predictions': np.array([0.3, 0.2, 0.05, 0.4, 0.05]),
        # This tests that multi-dimensional weights are allowed.
        'example_weights': np.array([0.3, 0.3, 0.3, 0.3, 0.3]),
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example1, example2, example3, example4])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeHistogram' >> beam.CombinePerKey(histogram.combiner)
          | 'ComputeConfusionMatrix'
          >> beam.Map(lambda x: (x[0], matrix.result(x[1])))
          | 'ComputeMetric' >> beam.Map(lambda x: (x[0], metric.result(x[1])))
      )

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          key = metric_types.MetricKey(
              name=metric_name,
              sub_key=metric_types.SubKey(top_k=top_k),
              example_weighted=True,
          )
          self.assertDictElementsAlmostEqual(
              got_metrics, {key: expected_value}, places=5
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  @parameterized.named_parameters(
      ('precision@2', 'precision@2', 1.6 / (1.6 + 3.2)),
      ('recall@2', 'recall@2', 1.6 / (1.6 + 0.8)),
      ('precision@3', 'precision@3', 1.9 / (1.9 + 5.3)),
      ('recall@3', 'recall@3', 1.9 / (1.9 + 0.5)),
  )
  def testMultiClassMetricsUsingKerasConfig(self, metric_name, expected_value):
    metric = tf_metric_wrapper.tf_metric_computations(
        [self._tf_metric_by_name(metric_name)], example_weighted=True
    )[0]

    # top_k = 2
    #   TP = 0.5*0 + 0.7*1 + 0.9*1 + 0.3*0 = 1.6
    #   FP = 0.5*2 + 0.7*1 + 0.9*1 + 0.3*2 = 3.2
    #   FN = 0.5*1 + 0.7*0 + 0.9*0 + 0.3*1 = 0.8
    #
    # top_k = 3
    #   TP = 0.5*0 + 0.7*1 + 0.9*1 + 0.3*1 = 1.9
    #   FP = 0.5*3 + 0.7*2 + 0.9*2 + 0.3*2 = 5.3
    #   FN = 0.5*1 + 0.7*0 + 0.9*0 + 0.3*0 = 0.5
    example1 = {
        'labels': np.array([2]),
        'predictions': np.array([0.1, 0.2, 0.1, 0.25, 0.35]),
        'example_weights': np.array([0.5]),
    }
    example2 = {
        'labels': np.array([1]),
        'predictions': np.array([0.2, 0.3, 0.05, 0.15, 0.3]),
        'example_weights': np.array([0.7]),
    }
    example3 = {
        'labels': np.array([3]),
        'predictions': np.array([0.01, 0.2, 0.09, 0.5, 0.2]),
        'example_weights': np.array([0.9]),
    }
    example4 = {
        'labels': np.array([1]),
        'predictions': np.array([0.3, 0.2, 0.05, 0.4, 0.05]),
        'example_weights': np.array([0.3]),
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example1, example2, example3, example4])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'Combine' >> beam.CombinePerKey(metric.combiner)
      )

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          top_k = int(metric_name.split('@')[1])
          key = metric_types.MetricKey(
              name=metric_name,
              sub_key=metric_types.SubKey(top_k=top_k),
              example_weighted=True,
          )
          self.assertDictElementsAlmostEqual(
              got_metrics, {key: expected_value}, places=5
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')


@pytest.mark.xfail(run=False, reason="PR 183 This class contains tests that fail and needs to be fixed. "
"If all tests pass, please remove this mark.")
class NonConfusionMatrixMetricsTest(
    test_util.TensorflowModelAnalysisTest, parameterized.TestCase
):

  def testSimpleMetric(self):
    computation = tf_metric_wrapper.tf_metric_computations(
        [tf_keras.metrics.MeanSquaredError(name='mse')]
    )[0]

    example = {
        'labels': [0, 0, 1, 1],
        'predictions': [0, 0.5, 0.3, 0.9],
        'example_weights': [1.0],
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'Combine' >> beam.CombinePerKey(computation.combiner)
      )

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          mse_key = metric_types.MetricKey(name='mse')
          self.assertDictElementsAlmostEqual(got_metrics, {mse_key: 0.1875})

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testSparseMetric(self):
    computation = tf_metric_wrapper.tf_metric_computations([
        tf_keras.metrics.SparseCategoricalCrossentropy(
            name='sparse_categorical_crossentropy'
        )
    ])[0]

    # Simulate a multi-class problem with 3 labels.
    example = {
        'labels': [1],
        'predictions': [0.3, 0.6, 0.1],
        'example_weights': [1.0],
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'Combine' >> beam.CombinePerKey(computation.combiner)
      )

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          key = metric_types.MetricKey(name='sparse_categorical_crossentropy')
          # 0*log(.3) -1*log(0.6)-0*log(.1) = 0.51
          self.assertDictElementsAlmostEqual(got_metrics, {key: 0.51083})

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testRaisesErrorForInvalidNonSparseSettings(self):
    with self.assertRaises(ValueError):
      tf_metric_wrapper.tf_metric_computations(
          [
              tf_keras.metrics.SparseCategoricalCrossentropy(
                  name='sparse_categorical_crossentropy'
              )
          ],
          aggregation_type=metric_types.AggregationType(micro_average=True),
      )

  def testMetricWithClassWeights(self):
    computation = tf_metric_wrapper.tf_metric_computations(
        [tf_keras.metrics.MeanSquaredError(name='mse')],
        aggregation_type=metric_types.AggregationType(micro_average=True),
        class_weights={0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4},
    )[0]

    # Simulate a multi-class problem with 4 labels. The use of class weights
    # implies micro averaging which only makes sense for multi-class metrics.
    example = {
        'labels': [0, 0, 1, 0],
        'predictions': [0, 0.5, 0.3, 0.9],
        'example_weights': [1.0],
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'Combine' >> beam.CombinePerKey(computation.combiner)
      )

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          mse_key = metric_types.MetricKey(name='mse')
          # numerator = (0.1*0**2 + 0.2*0.5**2 + 0.3*0.7**2 + 0.4*0.9**2)
          # denominator = (.1 + .2 + 0.3 + 0.4)
          # numerator / denominator = 0.521
          self.assertDictElementsAlmostEqual(got_metrics, {mse_key: 0.521})

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testCustomTFMetric(self):
    metric = tf_metric_wrapper.tf_metric_computations(
        [_CustomMetric()], example_weighted=True
    )[0]

    example1 = {'labels': [0.0], 'predictions': [0.2], 'example_weights': [1.0]}
    example2 = {'labels': [0.0], 'predictions': [0.8], 'example_weights': [1.0]}
    example3 = {'labels': [0.0], 'predictions': [0.5], 'example_weights': [2.0]}

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example1, example2, example3])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'Combine' >> beam.CombinePerKey(metric.combiner)
      )

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())

          custom_key = metric_types.MetricKey(
              name='custom', example_weighted=True
          )
          self.assertDictElementsAlmostEqual(
              got_metrics,
              {custom_key: (0.2 + 0.8 + 2 * 0.5) / (1.0 + 1.0 + 2.0)},
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testCustomConfusionMatrixTFMetric(self):
    metric = tf_metric_wrapper.tf_metric_computations(
        [_CustomConfusionMatrixMetric()]
    )[0]

    # tp = 1
    # fp = 1
    example1 = {'labels': [0.0], 'predictions': [0.7], 'example_weights': [1.0]}
    example2 = {'labels': [1.0], 'predictions': [0.8], 'example_weights': [1.0]}

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example1, example2])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'Combine' >> beam.CombinePerKey(metric.combiner)
      )

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())

          custom_key = metric_types.MetricKey(name='custom')
          self.assertDictElementsAlmostEqual(
              got_metrics, {custom_key: 1.0 / (1.0 + 1.0)}
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  @parameterized.named_parameters(*[
      dict(
          testcase_name='within_example',
          example_indices=[0],
          # label_sum = (1 - 1 - 1 - 1) * 1.0 = -2.0
          # pred_sum = (0.1 + 0.2 + 0.3 + 0.0) = 0.6
          # weights_total = 1.0 * 4 = 4.0
          expected={
              metric_types.MetricKey(
                  name='custom_label', example_weighted=True
              ): (-2.0 / 4.0),
              metric_types.MetricKey(
                  name='custom_pred', example_weighted=True
              ): (0.6 / 4.0),
          },
      ),
      dict(
          testcase_name='across_examples',
          # label_sum = (1 - 1 - 1 - 1) * 1.0 +
          #             (1 + 2 - 1.0 - 1) * 1.0 +
          #             (1 + 2 + 3 - 1) * 2.0
          #           = 9.0
          #
          # pred_sum = (0.1 + 0.2 + 0.3 + 0.0) * 1.0 +
          #            (0.1 + 0.2 + 0.0 - 1.0) * 1.0 +
          #            (0.1 + 0.2 + 0.3 - 1.0) * 2.0
          #           = -0.9
          #
          # weights_total = (1.0 * 4 + 1.0 * 4 + 2.0 * 4) = 16.0
          example_indices=[0, 1, 2],
          expected={
              metric_types.MetricKey(
                  name='custom_label', example_weighted=True
              ): (9.0 / 16.0),
              metric_types.MetricKey(
                  name='custom_pred', example_weighted=True
              ): (-0.9 / 16.0),
          },
      ),
  ])
  def testCustomTFMetricWithPadding(self, example_indices, expected):
    computation = tf_metric_wrapper.tf_metric_computations(
        [
            _CustomMetric(name='custom_label', update_y_pred=False),
            _CustomMetric(name='custom_pred', update_y_pred=True),
        ],
        eval_config=config_pb2.EvalConfig(
            model_specs=[
                config_pb2.ModelSpec(
                    padding_options=config_pb2.PaddingOptions(
                        label_int_padding=-1,
                        prediction_float_padding=-1.0,
                    )
                )
            ]
        ),
        example_weighted=True,
    )[0]

    examples = [
        {
            'labels': np.array([1], dtype=np.int64),
            'predictions': np.array([0.1, 0.2, 0.3, 0.0]),
            'example_weights': np.array([1.0]),
        },
        {
            'labels': np.array([1, 2], dtype=np.int64),
            'predictions': np.array([0.1, 0.2, 0.0]),
            'example_weights': np.array([1.0]),
        },
        {
            'labels': np.array([1, 2, 3], dtype=np.int64),
            'predictions': np.array([0.1, 0.2, 0.3]),
            'example_weights': np.array([2.0]),
        },
    ]

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([examples[i] for i in example_indices])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'Combine' >> beam.CombinePerKey(computation.combiner)
      )

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          self.assertDictElementsAlmostEqual(got_metrics, expected)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testMultiOutputTFMetric(self):
    computation = tf_metric_wrapper.tf_metric_computations({
        'output_name': [tf_keras.metrics.MeanSquaredError(name='mse')],
    })[0]

    extracts = {
        'labels': {
            'output_name': [0, 0, 1, 1],
        },
        'predictions': {
            'output_name': [0, 0.5, 0.3, 0.9],
        },
        'example_weights': {'output_name': [1.0]},
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([extracts])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'Combine' >> beam.CombinePerKey(computation.combiner)
      )

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          mse_key = metric_types.MetricKey(
              name='mse', output_name='output_name'
          )
          self.assertDictElementsAlmostEqual(
              got_metrics,
              {
                  mse_key: 0.1875,
              },
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testTFMetricWithDictResult(self):
    computation = tf_metric_wrapper.tf_metric_computations({
        'output_name': [_CustomMeanSquaredError(name='mse')],
    })[0]

    extracts = {
        'labels': {
            'output_name': [0, 0, 1, 1],
        },
        'predictions': {
            'output_name': [0, 0.5, 0.3, 0.9],
        },
        'example_weights': {'output_name': [1.0]},
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([extracts])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'Combine' >> beam.CombinePerKey(computation.combiner)
      )

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          mse_key = metric_types.MetricKey(
              name='mse/mse', output_name='output_name'
          )
          one_minus_mse_key = metric_types.MetricKey(
              name='mse/one_minus_mse', output_name='output_name'
          )
          self.assertDictElementsAlmostEqual(
              got_metrics, {mse_key: 0.1875, one_minus_mse_key: 0.8125}
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testTFMetricWithClassID(self):
    computation = tf_metric_wrapper.tf_metric_computations(
        [tf_keras.metrics.MeanSquaredError(name='mse')],
        sub_key=metric_types.SubKey(class_id=1),
        example_weighted=False,
    )[0]

    example1 = {
        'labels': [2],
        'predictions': [0.5, 0.0, 0.5],
        'example_weights': [0.1],  # ignored, example_weighted=False
    }
    example2 = {
        'labels': [0],
        'predictions': [0.2, 0.5, 0.3],
        'example_weights': [0.2],  # ignored, example_weighted=False
    }
    example3 = {
        'labels': [1],
        'predictions': [0.2, 0.3, 0.5],
        'example_weights': [0.3],  # ignored, example_weighted=False
    }
    example4 = {
        'labels': [1],
        'predictions': [0.0, 0.9, 0.1],
        'example_weights': [0.4],  # ignored, example_weighted=False
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example1, example2, example3, example4])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'Combine' >> beam.CombinePerKey(computation.combiner)
      )

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          mse_key = metric_types.MetricKey(
              name='mse',
              sub_key=metric_types.SubKey(class_id=1),
              example_weighted=False,
          )
          self.assertDictElementsAlmostEqual(
              got_metrics,
              {
                  mse_key: 0.1875,
              },
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testBatching(self):
    computation = tf_metric_wrapper.tf_metric_computations(
        [_CustomMetric(), tf_keras.metrics.MeanSquaredError(name='mse')],
        desired_batch_size=2,
        example_weighted=True,
    )[0]

    example1 = {'labels': [0.0], 'predictions': [0.0], 'example_weights': [1.0]}
    example2 = {'labels': [0.0], 'predictions': [0.5], 'example_weights': [1.0]}
    example3 = {'labels': [1.0], 'predictions': [0.3], 'example_weights': [1.0]}
    example4 = {'labels': [1.0], 'predictions': [0.9], 'example_weights': [1.0]}
    example5 = {'labels': [1.0], 'predictions': [0.5], 'example_weights': [0.0]}

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create'
          >> beam.Create([example1, example2, example3, example4, example5])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'Combine' >> beam.CombinePerKey(computation.combiner)
      )

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1, 'got: %s' % got)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())

          custom_key = metric_types.MetricKey(
              name='custom', example_weighted=True
          )
          mse_key = metric_types.MetricKey(name='mse', example_weighted=True)
          self.assertDictElementsAlmostEqual(
              got_metrics,
              {
                  custom_key: (0.0 + 0.5 + 0.3 + 0.9 + 0.0) / (
                      1.0 + 1.0 + 1.0 + 1.0 + 0.0
                  ),
                  mse_key: 0.1875,
              },
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testMergeAccumulators(self):
    computation = tf_metric_wrapper.tf_metric_computations(
        [tf_keras.metrics.MeanSquaredError(name='mse')],
        desired_batch_size=2,
        example_weighted=True,
    )[0]

    example1 = {'labels': [0.0], 'predictions': [0.0], 'example_weights': [1.0]}
    example2 = {'labels': [0.0], 'predictions': [0.5], 'example_weights': [1.0]}
    example3 = {'labels': [1.0], 'predictions': [0.3], 'example_weights': [1.0]}
    example4 = {'labels': [1.0], 'predictions': [0.9], 'example_weights': [1.0]}
    example5 = {'labels': [1.0], 'predictions': [0.5], 'example_weights': [0.0]}

    computation.combiner.setup()
    combiner_inputs = []
    for e in (example1, example2, example3, example4, example5):
      combiner_inputs.append(metric_util.to_standard_metric_inputs(e))
    acc1 = computation.combiner.create_accumulator()
    acc1 = computation.combiner.add_input(acc1, combiner_inputs[0])
    acc1 = computation.combiner.add_input(acc1, combiner_inputs[1])
    acc1 = computation.combiner.add_input(acc1, combiner_inputs[2])
    acc2 = computation.combiner.create_accumulator()
    acc2 = computation.combiner.add_input(acc2, combiner_inputs[3])
    acc2 = computation.combiner.add_input(acc2, combiner_inputs[4])
    acc = computation.combiner.merge_accumulators([acc1, acc2])

    got_metrics = computation.combiner.extract_output(acc)
    mse_key = metric_types.MetricKey(name='mse', example_weighted=True)
    self.assertDictElementsAlmostEqual(got_metrics, {mse_key: 0.1875})


@pytest.mark.xfail(run=False, reason="PR 183 This class contains tests that fail and needs to be fixed. "
"If all tests pass, please remove this mark.")
class MixedMetricsTest(test_util.TensorflowModelAnalysisTest):

  def testWithMixedMetrics(self):
    computations = tf_metric_wrapper.tf_metric_computations([
        tf_keras.metrics.AUC(name='auc'),
        tf_keras.losses.BinaryCrossentropy(name='binary_crossentropy'),
        tf_keras.metrics.MeanSquaredError(name='mse'),
    ])

    confusion_histogram = computations[0]
    confusion_matrix = computations[1].result
    confusion_metrics = computations[2].result
    non_confusion_metrics = computations[3]

    example1 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.0]),
        'example_weights': np.array([1.0]),
    }
    example2 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.5]),
        'example_weights': np.array([1.0]),
    }
    example3 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.3]),
        'example_weights': np.array([1.0]),
    }
    example4 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.9]),
        'example_weights': np.array([1.0]),
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      sliced_examples = (
          pipeline
          | 'Create' >> beam.Create([example1, example2, example3, example4])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
      )

      confusion_result = (
          sliced_examples
          | 'ComputeHistogram'
          >> beam.CombinePerKey(confusion_histogram.combiner)
          | 'ComputeConfusionMatrix'
          >> beam.Map(lambda x: (x[0], confusion_matrix(x[1])))
          | 'ComputeMetric'
          >> beam.Map(lambda x: (x[0], confusion_metrics(x[1])))
      )

      non_confusion_result = sliced_examples | 'Combine' >> beam.CombinePerKey(
          non_confusion_metrics.combiner
      )

      # pylint: enable=no-value-for-parameter

      def check_confusion_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          auc_key = metric_types.MetricKey(name='auc')
          self.assertDictElementsAlmostEqual(
              got_metrics, {auc_key: 0.75}, places=5
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      def check_non_confusion_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          mse_key = metric_types.MetricKey(name='mse')
          binary_crossentropy_key = metric_types.MetricKey(
              name='binary_crossentropy'
          )
          self.assertDictElementsAlmostEqual(
              got_metrics,
              {mse_key: 0.1875, binary_crossentropy_key: 0.50061995},
              places=5,
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(
          confusion_result, check_confusion_result, label='confusion'
      )
      util.assert_that(
          non_confusion_result,
          check_non_confusion_result,
          label='non_confusion',
      )


