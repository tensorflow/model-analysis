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
"""Tests for utils for evaluations using keras_util."""


import pytest
import tempfile
import unittest

from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.evaluators import keras_util
from tensorflow_model_analysis.metrics import metric_types

_TF_MAJOR_VERSION = int(tf.version.VERSION.split('.')[0])


@pytest.mark.xfail(run=False, reason="PR 183 This class contains tests that fail and needs to be fixed. "
"If all tests pass, please remove this mark.")
class KerasSavedModelUtilTest(
    testutil.TensorflowModelAnalysisTest, parameterized.TestCase
):

  def _createBinaryClassificationMetrics(self):
    return [
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.AUC(name='auc_pr', curve='PR'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.TruePositives(name='true_positives'),
        tf.keras.metrics.FalsePositives(name='false_positives'),
        tf.keras.metrics.TrueNegatives(name='true_negatives'),
        tf.keras.metrics.FalseNegatives(name='false_negatives'),
        tf.keras.metrics.SpecificityAtSensitivity(
            0.5, name='specificity_at_sensitivity'
        ),
        tf.keras.metrics.SensitivityAtSpecificity(
            0.5, name='sensitivity_at_specificity'
        ),
    ]

  def _createBinaryClassificationLosses(self):
    return [tf.keras.losses.BinaryCrossentropy()]

  def _createBinaryClassificationModel(
      self, sequential=True, output_names=None, add_custom_metrics=False
  ):
    if not output_names:
      layer = tf.keras.layers.Input(shape=(1,), name='output')
      if sequential:
        model = tf.keras.models.Sequential([layer, layer])
      else:
        model = tf.keras.models.Model(layer, layer)
      if add_custom_metrics:
        model.add_metric(tf.reduce_sum(layer), name='custom')
      model.compile(
          loss=self._createBinaryClassificationLosses(),
          metrics=self._createBinaryClassificationMetrics(),
          weighted_metrics=self._createBinaryClassificationMetrics(),
      )
      model.fit(np.array([[1]]), np.array([[1]]))
    else:
      layers_per_output = {}
      metrics_per_output = {}
      weighted_metrics_per_output = {}
      losses_per_output = {}
      for output_name in output_names:
        layers_per_output[output_name] = tf.keras.layers.Input(
            shape=(1,), name=output_name
        )
        metrics_per_output[output_name] = (
            self._createBinaryClassificationMetrics()
        )
        weighted_metrics_per_output[output_name] = (
            self._createBinaryClassificationMetrics()
        )
        losses_per_output[output_name] = (
            self._createBinaryClassificationLosses()
        )
      if sequential:
        raise ValueError('Sequential not supported with multi-output models')
      else:
        model = tf.keras.models.Model(layers_per_output, layers_per_output)
      if add_custom_metrics:
        for output_name in output_names:
          model.add_metric(
              tf.reduce_sum(layers_per_output[output_name]),
              name='custom_' + output_name,
          )
      model.compile(
          loss=losses_per_output,
          metrics=metrics_per_output,
          weighted_metrics=weighted_metrics_per_output,
      )
      model.fit(
          {n: np.array([[1]]) for n in output_names},
          {n: np.array([[1]]) for n in output_names},
      )

    export_path = tempfile.mkdtemp()
    model.save(export_path, save_format='tf')
    return export_path

  def _createMultiClassClassificationMetrics(self):
    return [
        tf.keras.metrics.Precision(name='precision@2', top_k=2),
        tf.keras.metrics.Precision(name='precision@3', top_k=3),
        tf.keras.metrics.Recall(name='recall@2', top_k=2),
        tf.keras.metrics.Recall(name='recall@3', top_k=3),
    ]

  def _createMultiClassClassificationLosses(self):
    # Note cannot use SparseCategorialCrossentropy since we are using Precision
    # and Recall for the metrics which require dense labels.
    return [tf.keras.losses.CategoricalCrossentropy()]

  def _createMultiClassClassificationModel(
      self, sequential=True, output_names=None, add_custom_metrics=False
  ):
    if not output_names:
      layer = tf.keras.layers.Input(shape=(5,), name='output')
      if sequential:
        model = tf.keras.models.Sequential([layer, layer])
      else:
        model = tf.keras.models.Model(layer, layer)
      if add_custom_metrics:
        model.add_metric(tf.reduce_sum(layer), name='custom')
      model.compile(
          loss=self._createMultiClassClassificationLosses(),
          metrics=self._createMultiClassClassificationMetrics(),
          weighted_metrics=self._createMultiClassClassificationMetrics(),
      )
      model.fit(np.array([[1, 0, 0, 0, 0]]), np.array([[1, 0, 0, 0, 0]]))
    else:
      layers_per_output = {}
      metrics_per_output = {}
      weighted_metrics_per_output = {}
      losses_per_output = {}
      for output_name in output_names:
        layers_per_output[output_name] = tf.keras.layers.Input(
            shape=(5,), name=output_name
        )
        metrics_per_output[output_name] = (
            self._createMultiClassClassificationMetrics()
        )
        weighted_metrics_per_output[output_name] = (
            self._createMultiClassClassificationMetrics()
        )
        losses_per_output[output_name] = (
            self._createMultiClassClassificationLosses()
        )
      if sequential:
        raise ValueError('Sequential not supported with multi-output models')
      else:
        model = tf.keras.models.Model(layers_per_output, layers_per_output)
      if add_custom_metrics:
        for output_name in output_names:
          model.add_metric(
              tf.reduce_sum(layers_per_output[output_name]),
              name='custom_' + output_name,
          )
      model.compile(
          loss=losses_per_output,
          metrics=metrics_per_output,
          weighted_metrics=weighted_metrics_per_output,
      )
      model.fit(
          {n: np.array([[1, 0, 0, 0, 0]]) for n in output_names},
          {n: np.array([[1, 0, 0, 0, 0]]) for n in output_names},
      )

    export_path = tempfile.mkdtemp()
    model.save(export_path, save_format='tf')
    return export_path

  @parameterized.named_parameters(
      ('compiled_metrics_sequential_model', True, False),
      ('compiled_metrics_functional_model', False, False),
      ('evaluate', False, True),
  )
  @unittest.skipIf(_TF_MAJOR_VERSION < 2, 'not all options supported in TFv1')
  def testWithBinaryClassification(self, sequential_model, add_custom_metrics):
    # If custom metrics are used, then model.evaluate is called.
    export_dir = self._createBinaryClassificationModel(
        sequential=sequential_model, add_custom_metrics=add_custom_metrics
    )
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir
    )
    computation = keras_util.metric_computations_using_keras_saved_model(
        '', eval_shared_model.model_loader, None
    )[0]

    inputs = [
        metric_types.StandardMetricInputs(
            labels=np.array([0.0]),
            predictions=np.array([1.0]),
            example_weights=np.array([0.5]),
            features={'output': np.array([1.0])},
        ),
        metric_types.StandardMetricInputs(
            labels=np.array([1.0]),
            predictions=np.array([0.7]),
            example_weights=np.array([0.7]),
            features={'output': np.array([0.7])},
        ),
        metric_types.StandardMetricInputs(
            labels=np.array([0.0]),
            predictions=np.array([0.5]),
            example_weights=np.array([0.9]),
            features={'output': np.array([0.5])},
        ),
    ]

    expected_values = {
        'auc': 0.5,
        'auc_pr': 0.30685,
        'precision': 0.5,
        'recall': 1.0,
        'true_positives': 1.0,
        'false_positives': 1.0,
        'true_negatives': 1.0,
        'false_negatives': 0.0,
        'specificity_at_sensitivity': 0.5,
        'sensitivity_at_specificity': 1.0,
        'weighted_auc': 0.64286,
        'weighted_auc_pr': 0.37467,
        'weighted_precision': 0.5833333,
        'weighted_recall': 1.0,
        'weighted_true_positives': 0.7,
        'weighted_false_positives': 0.5,
        'weighted_true_negatives': 0.9,
        'weighted_false_negatives': 0.0,
        'weighted_specificity_at_sensitivity': 0.642857,
        'weighted_sensitivity_at_specificity': 1.0,
        'loss': 2.861993,
    }
    if add_custom_metrics:
      # Loss is different due to rounding errors from tf.Example conversion.
      expected_values['loss'] = 2.8327076
      expected_values['custom'] = 1.0 + 0.7 + 0.5

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create(inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeMetric' >> beam.CombinePerKey(computation.combiner)
      )

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          expected = {
              metric_types.MetricKey(name=name, example_weighted=None): value
              for name, value in expected_values.items()
          }
          self.assertDictElementsAlmostEqual(got_metrics, expected)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  @parameterized.named_parameters(
      ('compiled_metrics', False),
      ('evaluate', True),
  )
  @unittest.skipIf(_TF_MAJOR_VERSION < 2, 'not all options supported in TFv1')
  def testWithBinaryClassificationMultiOutput(self, add_custom_metrics):
    # If custom metrics are used, then model.evaluate is called.
    export_dir = self._createBinaryClassificationModel(
        sequential=False,
        output_names=('output_1', 'output_2'),
        add_custom_metrics=add_custom_metrics,
    )
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir
    )
    computation = keras_util.metric_computations_using_keras_saved_model(
        '', eval_shared_model.model_loader, None
    )[0]

    inputs = [
        metric_types.StandardMetricInputs(
            labels={
                'output_1': np.array([0.0]),
                'output_2': np.array([0.0]),
            },
            predictions={
                'output_1': np.array([1.0]),
                'output_2': np.array([0.0]),
            },
            example_weights={
                'output_1': np.array([0.5]),
                'output_2': np.array([0.1]),
            },
            features={'output_1': np.array([1.0]), 'output_2': np.array([0.0])},
        ),
        metric_types.StandardMetricInputs(
            labels={
                'output_1': np.array([1.0]),
                'output_2': np.array([1.0]),
            },
            predictions={
                'output_1': np.array([0.7]),
                'output_2': np.array([0.3]),
            },
            example_weights={
                'output_1': np.array([0.7]),
                'output_2': np.array([0.4]),
            },
            features={'output_1': np.array([0.7]), 'output_2': np.array([0.3])},
        ),
        metric_types.StandardMetricInputs(
            labels={
                'output_1': np.array([0.0]),
                'output_2': np.array([1.0]),
            },
            predictions={
                'output_1': np.array([0.5]),
                'output_2': np.array([0.8]),
            },
            example_weights={
                'output_1': np.array([0.9]),
                'output_2': np.array([0.7]),
            },
            features={'output_1': np.array([0.5]), 'output_2': np.array([0.8])},
        ),
    ]

    expected_values = {
        'output_1': {
            'auc': 0.5,
            'auc_pr': 0.30685,
            'precision': 0.5,
            'recall': 1.0,
            'true_positives': 1.0,
            'false_positives': 1.0,
            'true_negatives': 1.0,
            'false_negatives': 0.0,
            'specificity_at_sensitivity': 0.5,
            'sensitivity_at_specificity': 1.0,
            'weighted_auc': 0.64286,
            'weighted_auc_pr': 0.37467,
            'weighted_precision': 0.5833333,
            'weighted_recall': 1.0,
            'weighted_true_positives': 0.7,
            'weighted_false_positives': 0.5,
            'weighted_true_negatives': 0.9,
            'weighted_false_negatives': 0.0,
            'weighted_specificity_at_sensitivity': 0.642857,
            'weighted_sensitivity_at_specificity': 1.0,
            'loss': 2.861993,
        },
        'output_2': {
            'auc': 1.0,
            'auc_pr': 1.0,
            'precision': 1.0,
            'recall': 0.5,
            'true_positives': 1.0,
            'false_positives': 0.0,
            'true_negatives': 1.0,
            'false_negatives': 1.0,
            'specificity_at_sensitivity': 1.0,
            'sensitivity_at_specificity': 1.0,
            'weighted_auc': 1.0,
            'weighted_auc_pr': 1.0,
            'weighted_precision': 1.0,
            'weighted_recall': 0.6363636,
            'weighted_true_positives': 0.7,
            'weighted_false_positives': 0.0,
            'weighted_true_negatives': 0.1,
            'weighted_false_negatives': 0.4,
            'weighted_specificity_at_sensitivity': 1.0,
            'weighted_sensitivity_at_specificity': 1.0,
            'loss': 0.21259646,
        },
        '': {'loss': 2.861993 + 0.21259646},
    }
    if add_custom_metrics:
      # Loss is different due to rounding errors from tf.Example conversion.
      expected_values['output_1']['loss'] = 2.8327076
      expected_values['output_2']['loss'] = 0.21259646
      expected_values['']['loss'] = 2.8327076 + 0.21259646
      expected_values['']['custom_output_1'] = 1.0 + 0.7 + 0.5
      expected_values['']['custom_output_2'] = 0.0 + 0.3 + 0.8

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create(inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeMetric' >> beam.CombinePerKey(computation.combiner)
      )

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          expected = {}
          for output_name, per_output_values in expected_values.items():
            for name, value in per_output_values.items():
              key = metric_types.MetricKey(
                  name=name, output_name=output_name, example_weighted=None
              )
              expected[key] = value
          self.assertDictElementsAlmostEqual(got_metrics, expected)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  @parameterized.named_parameters(
      ('compiled_metrics_sequential_model', True, False),
      ('compiled_metrics_functional_model', False, False),
      ('evaluate', False, True),
  )
  @unittest.skipIf(_TF_MAJOR_VERSION < 2, 'not all options supported in TFv1')
  def testWithMultiClassClassification(
      self, sequential_model, add_custom_metrics
  ):
    export_dir = self._createMultiClassClassificationModel(
        sequential=sequential_model, add_custom_metrics=add_custom_metrics
    )
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir
    )
    computation = keras_util.metric_computations_using_keras_saved_model(
        '', eval_shared_model.model_loader, None
    )[0]

    inputs = [
        metric_types.StandardMetricInputs(
            labels=np.array([0, 0, 1, 0, 0]),
            predictions=np.array([0.1, 0.2, 0.1, 0.25, 0.35]),
            example_weights=np.array([0.5]),
            features={
                'output': np.array([0.1, 0.2, 0.1, 0.25, 0.35]),
            },
        ),
        metric_types.StandardMetricInputs(
            labels=np.array([0, 1, 0, 0, 0]),
            predictions=np.array([0.2, 0.3, 0.05, 0.15, 0.3]),
            example_weights=np.array([0.7]),
            features={
                'output': np.array([0.2, 0.3, 0.05, 0.15, 0.3]),
            },
        ),
        metric_types.StandardMetricInputs(
            labels=np.array([0, 0, 0, 1, 0]),
            predictions=np.array([0.01, 0.2, 0.09, 0.5, 0.2]),
            example_weights=np.array([0.9]),
            features={
                'output': np.array([0.01, 0.2, 0.09, 0.5, 0.2]),
            },
        ),
        metric_types.StandardMetricInputs(
            labels=np.array([0, 1, 0, 0, 0]),
            predictions=np.array([0.3, 0.2, 0.05, 0.4, 0.05]),
            example_weights=np.array([0.3]),
            features={
                'output': np.array([0.3, 0.2, 0.05, 0.4, 0.05]),
            },
        ),
    ]

    # Unweighted:
    #   top_k = 2
    #     TP = 0 + 1 + 1 + 0 = 2
    #     FP = 2 + 1 + 1 + 2 = 6
    #     FN = 1 + 0 + 0 + 1 = 2
    #
    #   top_k = 3
    #     TP = 0 + 1 + 1 + 1 = 3
    #     FP = 3 + 2 + 2 + 2 = 9
    #     FN = 1 + 0 + 0 + 0 = 1
    #
    # Weighted:
    #   top_k = 2
    #     TP = 0.5*0 + 0.7*1 + 0.9*1 + 0.3*0 = 1.6
    #     FP = 0.5*2 + 0.7*1 + 0.9*1 + 0.3*2 = 3.2
    #     FN = 0.5*1 + 0.7*0 + 0.9*0 + 0.3*1 = 0.8
    #
    #   top_k = 3
    #     TP = 0.5*0 + 0.7*1 + 0.9*1 + 0.3*1 = 1.9
    #     FP = 0.5*3 + 0.7*2 + 0.9*2 + 0.3*2 = 5.3
    #     FN = 0.5*1 + 0.7*0 + 0.9*0 + 0.3*0 = 0.5
    expected_values = {
        'precision@2': 2 / (2 + 6),
        'precision@3': 3 / (3 + 9),
        'recall@2': 2 / (2 + 2),
        'recall@3': 3 / (3 + 1),
        'weighted_precision@2': 1.6 / (1.6 + 3.2),
        'weighted_precision@3': 1.9 / (1.9 + 5.3),
        'weighted_recall@2': 1.6 / (1.6 + 0.8),
        'weighted_recall@3': 1.9 / (1.9 + 0.5),
        'loss': 0.77518,
    }
    if add_custom_metrics:
      expected_values['custom'] = 4.0

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create(inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeMetric' >> beam.CombinePerKey(computation.combiner)
      )

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          expected = {}
          for name, value in expected_values.items():
            sub_key = None
            if '@' in name:
              sub_key = metric_types.SubKey(top_k=int(name.split('@')[1]))
            key = metric_types.MetricKey(
                name=name, sub_key=sub_key, example_weighted=None
            )
            expected[key] = value
          self.assertDictElementsAlmostEqual(got_metrics, expected)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  @parameterized.named_parameters(
      ('compiled_metrics', False), ('evaluate', True)
  )
  @unittest.skipIf(_TF_MAJOR_VERSION < 2, 'not all options supported in TFv1')
  def testWithMultiClassClassificationMultiOutput(self, add_custom_metrics):
    export_dir = self._createMultiClassClassificationModel(
        sequential=False,
        output_names=('output_1', 'output_2'),
        add_custom_metrics=add_custom_metrics,
    )
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir
    )
    computation = keras_util.metric_computations_using_keras_saved_model(
        '', eval_shared_model.model_loader, None
    )[0]

    inputs = [
        metric_types.StandardMetricInputs(
            labels={
                'output_1': np.array([0, 0, 1, 0, 0]),
                'output_2': np.array([0, 0, 1, 0, 0]),
            },
            predictions={
                'output_1': np.array([0.1, 0.2, 0.1, 0.25, 0.35]),
                'output_2': np.array([0.1, 0.2, 0.1, 0.25, 0.35]),
            },
            example_weights={
                'output_1': np.array([0.5]),
                'output_2': np.array([0.5]),
            },
            features={
                'output_1': np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
                'output_2': np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            },
            transformed_features={
                'output_1': np.array([0.1, 0.2, 0.1, 0.25, 0.35]),
                'output_2': np.array([0.1, 0.2, 0.1, 0.25, 0.35]),
            },
        ),
        metric_types.StandardMetricInputs(
            labels={
                'output_1': np.array([0, 1, 0, 0, 0]),
                'output_2': np.array([0, 1, 0, 0, 0]),
            },
            predictions={
                'output_1': np.array([0.2, 0.3, 0.05, 0.15, 0.3]),
                'output_2': np.array([0.2, 0.3, 0.05, 0.15, 0.3]),
            },
            example_weights={
                'output_1': np.array([0.7]),
                'output_2': np.array([0.7]),
            },
            features={
                'output_1': np.array([0.2, 0.3, 0.05, 0.15, 0.3]),
                'output_2': np.array([0.2, 0.3, 0.05, 0.15, 0.3]),
            },
        ),
        metric_types.StandardMetricInputs(
            labels={
                'output_1': np.array([0, 0, 0, 1, 0]),
                'output_2': np.array([0, 0, 0, 1, 0]),
            },
            predictions={
                'output_1': np.array([0.01, 0.2, 0.09, 0.5, 0.2]),
                'output_2': np.array([0.01, 0.2, 0.09, 0.5, 0.2]),
            },
            example_weights={
                'output_1': np.array([0.9]),
                'output_2': np.array([0.9]),
            },
            features={
                'output_1': np.array([0.01, 0.2, 0.09, 0.5, 0.2]),
                'output_2': np.array([0.01, 0.2, 0.09, 0.5, 0.2]),
            },
        ),
        metric_types.StandardMetricInputs(
            labels={
                'output_1': np.array([0, 1, 0, 0, 0]),
                'output_2': np.array([0, 1, 0, 0, 0]),
            },
            predictions={
                'output_1': np.array([0.3, 0.2, 0.05, 0.4, 0.05]),
                'output_2': np.array([0.3, 0.2, 0.05, 0.4, 0.05]),
            },
            example_weights={
                'output_1': np.array([0.3]),
                'output_2': np.array([0.3]),
            },
            features={
                'output_1': np.array([0.3, 0.2, 0.05, 0.4, 0.05]),
                'output_2': np.array([0.3, 0.2, 0.05, 0.4, 0.05]),
            },
        ),
    ]

    # Unweighted:
    #   top_k = 2
    #     TP = 0 + 1 + 1 + 0 = 2
    #     FP = 2 + 1 + 1 + 2 = 6
    #     FN = 1 + 0 + 0 + 1 = 2
    #
    #   top_k = 3
    #     TP = 0 + 1 + 1 + 1 = 3
    #     FP = 3 + 2 + 2 + 2 = 9
    #     FN = 1 + 0 + 0 + 0 = 1
    #
    # Weighted:
    #   top_k = 2
    #     TP = 0.5*0 + 0.7*1 + 0.9*1 + 0.3*0 = 1.6
    #     FP = 0.5*2 + 0.7*1 + 0.9*1 + 0.3*2 = 3.2
    #     FN = 0.5*1 + 0.7*0 + 0.9*0 + 0.3*1 = 0.8
    #
    #   top_k = 3
    #     TP = 0.5*0 + 0.7*1 + 0.9*1 + 0.3*1 = 1.9
    #     FP = 0.5*3 + 0.7*2 + 0.9*2 + 0.3*2 = 5.3
    #     FN = 0.5*1 + 0.7*0 + 0.9*0 + 0.3*0 = 0.5
    expected_values = {
        'output_1': {
            'precision@2': 2 / (2 + 6),
            'precision@3': 3 / (3 + 9),
            'recall@2': 2 / (2 + 2),
            'recall@3': 3 / (3 + 1),
            'weighted_precision@2': 1.6 / (1.6 + 3.2),
            'weighted_precision@3': 1.9 / (1.9 + 5.3),
            'weighted_recall@2': 1.6 / (1.6 + 0.8),
            'weighted_recall@3': 1.9 / (1.9 + 0.5),
            'loss': 0.77518433,
        },
        'output_2': {
            'precision@2': 2 / (2 + 6),
            'precision@3': 3 / (3 + 9),
            'recall@2': 2 / (2 + 2),
            'recall@3': 3 / (3 + 1),
            'weighted_precision@2': 1.6 / (1.6 + 3.2),
            'weighted_precision@3': 1.9 / (1.9 + 5.3),
            'weighted_recall@2': 1.6 / (1.6 + 0.8),
            'weighted_recall@3': 1.9 / (1.9 + 0.5),
            'loss': 0.77518433,
        },
        '': {'loss': 0.77518433 + 0.77518433},
    }
    if add_custom_metrics:
      expected_values['']['custom_output_1'] = 4.0
      expected_values['']['custom_output_2'] = 4.0

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create(inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeMetric' >> beam.CombinePerKey(computation.combiner)
      )

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          expected = {}
          for output_name, per_output_values in expected_values.items():
            for name, value in per_output_values.items():
              sub_key = None
              if '@' in name:
                sub_key = metric_types.SubKey(top_k=int(name.split('@')[1]))
              key = metric_types.MetricKey(
                  name=name,
                  output_name=output_name,
                  sub_key=sub_key,
                  example_weighted=None,
              )
              expected[key] = value
          self.assertDictElementsAlmostEqual(got_metrics, expected)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')


