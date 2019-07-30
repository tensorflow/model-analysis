# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test Model Agnostic graph handler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Standard Imports
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import types
from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.api import tfma_unit
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.evaluators import metrics_and_plots_evaluator
from tensorflow_model_analysis.extractors import slice_key_extractor
from tensorflow_model_analysis.model_agnostic_eval import model_agnostic_evaluate_graph
from tensorflow_model_analysis.model_agnostic_eval import model_agnostic_extractor
from tensorflow_model_analysis.model_agnostic_eval import model_agnostic_predict as agnostic_predict
from tensorflow_model_analysis.slicer import slicer


# Create a dummy metric callback that adds two metrics:
#   1) tf_metric_mean : tf.metric_mean which generates the mean of all
#      predictions and labels.
#   2) py_func_total_label: custom metric which generates the sum of all
#      predictions and labels.
def add_mean_callback(features_dict, predictions_dict, labels_dict):
  """Callback to add our custom post-export metrics."""
  del features_dict

  metric_ops = {}

  # Adding a tf.metrics metric.
  all_values = list(labels_dict.values()) + list(predictions_dict.values())
  metric_ops['tf_metric_mean'] = tf.compat.v1.metrics.mean(all_values)

  # Defining and adding a py_func metric
  # Note that for py_func metrics, you must still store the metric state in
  # tf.Variables.
  total_label = tf.compat.v1.Variable(
      initial_value=0.0,
      dtype=tf.float64,
      trainable=False,
      collections=[
          tf.compat.v1.GraphKeys.METRIC_VARIABLES,
          tf.compat.v1.GraphKeys.LOCAL_VARIABLES
      ],
      validate_shape=True,
      name='total_label')

  def my_func(x):
    return np.sum(x, dtype=np.float64)

  value_op = tf.identity(total_label)
  update_op = tf.compat.v1.assign_add(
      total_label, tf.compat.v1.py_func(my_func, [all_values], tf.float64))

  metric_ops['py_func_total_label'] = value_op, update_op

  return metric_ops


class ModelAgnosticEvaluateGraphTest(testutil.TensorflowModelAnalysisTest):

  def testEvaluateGraph(self):
    # Have 3 labels of values 3, 23, 16 and predictions of values 2, 2, 2.
    # This should give sum = 48 and mean = 8.
    examples = [
        self._makeExample(
            age=3.0, language='english', predictions=2.0, labels=3.0),
        self._makeExample(
            age=3.0, language='chinese', predictions=2.0, labels=23.0),
        self._makeExample(
            age=4.0, language='english', predictions=2.0, labels=16.0),
    ]
    serialized_examples = [e.SerializeToString() for e in examples]

    # Set up a model agnostic config so we can get the FPLConfig.
    feature_map = {
        'age': tf.io.FixedLenFeature([], tf.float32),
        'language': tf.io.VarLenFeature(tf.string),
        'predictions': tf.io.FixedLenFeature([], tf.float32),
        'labels': tf.io.FixedLenFeature([], tf.float32)
    }

    model_agnostic_config = agnostic_predict.ModelAgnosticConfig(
        label_keys=['labels'],
        prediction_keys=['predictions'],
        feature_spec=feature_map)

    # Create a Model Anostic Evaluate graph handler and feed in the FPL list.
    evaluate_graph = model_agnostic_evaluate_graph.ModelAgnosticEvaluateGraph(
        [add_mean_callback], model_agnostic_config)
    evaluate_graph.metrics_reset_update_get_list(serialized_examples)
    outputs = evaluate_graph.get_metric_values()

    # Verify that we got the right metrics out.
    self.assertEqual(2, len(outputs))
    self.assertEqual(outputs['tf_metric_mean'], 8.0)
    self.assertEqual(outputs['py_func_total_label'], 48.0)

  def testEvaluateMultiLabelsPredictions(self):
    # Test case where we have multiple labels/predictions
    # Have 6 labels of values 3, 5, 23, 12, 16, 31 and
    # 6 predictions of values 2, 2, 2, 4, 4, 4
    # This should give sum = 108 and mean = 9.

    examples = [
        self._makeExample(
            age=1.0, prediction=2, prediction_2=4, label=3, label_2=5),
        self._makeExample(
            age=1.0, prediction=2, prediction_2=4, label=23, label_2=12),
        self._makeExample(
            age=1.0, prediction=2, prediction_2=4, label=16, label_2=31),
    ]
    serialized_examples = [e.SerializeToString() for e in examples]

    # Set up a model agnostic config so we can get the FPLConfig.
    feature_map = {
        'age': tf.io.FixedLenFeature([], tf.float32),
        'prediction': tf.io.FixedLenFeature([], tf.int64),
        'prediction_2': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'label_2': tf.io.FixedLenFeature([], tf.int64)
    }

    model_agnostic_config = agnostic_predict.ModelAgnosticConfig(
        label_keys=['label', 'label_2'],
        prediction_keys=['prediction', 'prediction_2'],
        feature_spec=feature_map)

    # Create a Model Anostic Evaluate graph handler and feed in the FPL list.
    evaluate_graph = model_agnostic_evaluate_graph.ModelAgnosticEvaluateGraph(
        [add_mean_callback], model_agnostic_config)
    evaluate_graph.metrics_reset_update_get_list(serialized_examples)
    outputs = evaluate_graph.get_metric_values()

    # Verify that we got the right metrics out.
    self.assertEqual(2, len(outputs))
    self.assertEqual(outputs['tf_metric_mean'], 9.0)
    self.assertEqual(outputs['py_func_total_label'], 108.0)

  def testModelAgnosticConstructFn(self):
    # End to end test for the entire flow going from tf.Examples -> metrics
    # with slicing.
    with beam.Pipeline() as pipeline:
      # Set up the inputs. All we need is are tf.Examples and an example parsing
      # spec with explicit mapping for key to (Features, Predictions, Labels).
      # TODO(b/119788402): Add a fairness data examples/callbacks as another
      # test.
      examples = [
          self._makeExample(
              age=3.0, language='english', probabilities=1.0, labels=1.0),
          self._makeExample(
              age=3.0, language='chinese', probabilities=3.0, labels=0.0),
          self._makeExample(
              age=4.0, language='english', probabilities=2.0, labels=1.0),
          self._makeExample(
              age=5.0, language='chinese', probabilities=3.0, labels=0.0),
          # Add some examples with no language.
          self._makeExample(age=5.0, probabilities=2.0, labels=10.0),
          self._makeExample(age=6.0, probabilities=1.0, labels=0.0)
      ]
      serialized_examples = [e.SerializeToString() for e in examples]

      # Set up a config to bucket our example keys.
      feature_map = {
          'age': tf.io.FixedLenFeature([], tf.float32),
          'language': tf.io.VarLenFeature(tf.string),
          'probabilities': tf.io.FixedLenFeature([], tf.float32),
          'labels': tf.io.FixedLenFeature([], tf.float32)
      }

      model_agnostic_config = agnostic_predict.ModelAgnosticConfig(
          label_keys=['labels'],
          prediction_keys=['probabilities'],
          feature_spec=feature_map)

      # Set up the Model Agnostic Extractor
      extractors = [
          model_agnostic_extractor.ModelAgnosticExtractor(
              model_agnostic_config=model_agnostic_config,
              desired_batch_size=3),
          slice_key_extractor.SliceKeyExtractor([
              slicer.SingleSliceSpec(),
              slicer.SingleSliceSpec(columns=['language'])
          ])
      ]

      # Set up the metrics we wish to calculate via a metric callback. In
      # particular, this metric calculates the mean and sum of all labels.
      eval_shared_model = types.EvalSharedModel(
          add_metrics_callbacks=[add_mean_callback],
          construct_fn=model_agnostic_evaluate_graph.make_construct_fn(
              add_metrics_callbacks=[add_mean_callback],
              config=model_agnostic_config))

      # Run our pipeline doing Extract -> Slice -> Fanout -> Calculate Metrics.
      (metrics, _), _ = (
          pipeline
          | 'Create Examples' >> beam.Create(serialized_examples)
          | 'InputsToExtracts' >> model_eval_lib.InputsToExtracts()
          | 'Extract' >> tfma_unit.Extract(extractors=extractors)  # pylint: disable=no-value-for-parameter
          | 'ComputeMetricsAndPlots' >> metrics_and_plots_evaluator
          .ComputeMetricsAndPlots(eval_shared_model=eval_shared_model))

      # Verify our metrics are properly generated per slice.
      def check_result(got):
        self.assertEqual(3, len(got), 'got: %s' % got)
        slices = {}
        for slice_key, metrics in got:
          slices[slice_key] = metrics
        overall_slice = ()
        english_slice = (('language', b'english'),)
        chinese_slice = (('language', b'chinese'),)

        self.assertItemsEqual(
            list(slices.keys()), [overall_slice, english_slice, chinese_slice])
        # Overall slice has label/predictions sum = 24 and 12 elements.
        self.assertDictElementsAlmostEqual(slices[overall_slice], {
            'tf_metric_mean': 2.0,
            'py_func_total_label': 24.0,
        })
        # English slice has label/predictions sum = 5 and 4 elements.
        self.assertDictElementsAlmostEqual(slices[english_slice], {
            'tf_metric_mean': 1.25,
            'py_func_total_label': 5.0,
        })
        # Chinese slice has label/predictions sum = 6 and 4 elements.
        self.assertDictElementsAlmostEqual(slices[chinese_slice], {
            'tf_metric_mean': 1.5,
            'py_func_total_label': 6.0,
        })

      util.assert_that(metrics, check_result)


if __name__ == '__main__':
  tf.test.main()
