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
"""Unit test library for testing your TFMA models / metrics.

This is publicly accessible as the tfma.test module. Example usage:

class MyModelTFMATest(tfma.test.TestCase):

  def testWithoutBeam(self):
    path = train_and_export_my_model(...)
    examples = [self.makeExample(age=5, label=1.0),
                self.makeExample(age=10, label=0.0)]
    expected_metrics={
      'average_loss': tfma.test.BoundedValue(upper_bound=1.0),
      'auc': tfma.test.BoundedValue(lower_bound=0.5),
      'example_count': 3.0,
    }
    self.assertMetricsComputedWithoutBeamAre(
      eval_saved_model_path=path,
      serialized_examples=examples,
      expected_metrics=expected_metrics)

  def testWithBeam(self):
    path = train_and_export_my_model(...)
    examples = [self.makeExample(age=5, label=1.0),
                self.makeExample(age=10, label=0.0)]
    expected_metrics={
      'average_loss': tfma.test.BoundedValue(upper_bound=1.0),
      'auc': tfma.test.BoundedValue(lower_bound=0.5),
      'example_count': 3.0,
    }
    self.assertMetricsComputedWithBeamAre(
      eval_saved_model_path=path,
      serialized_examples=examples,
      expected_metrics=expected_metrics)

We recommend that you actually train and export your model with the test, as
opposed to training and exporting the model once and saving it alongside the
test. This is so that the model is always exported using the latest code and is
of the latest format.

Note that if you are retraining a new model for each test, your model may have
different weights each time and have different metric values. As such, we
recommend that you use BoundedValue with loose bounds to avoid flaky tests.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function


import apache_beam as beam
from apache_beam.testing import util as beam_util

from tensorflow_model_analysis.api.impl import evaluate
from tensorflow_model_analysis.eval_saved_model import load
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.slicer import slicer
from tensorflow_model_analysis.types_compat import Any, List, Dict, Union


class BoundedValue(object):
  """Represents a bounded value for a metric for the TFMA unit test."""

  def __init__(self,
               lower_bound = float('-inf'),
               upper_bound = float('inf')):
    self.lower_bound = lower_bound
    self.upper_bound = upper_bound


class TestCase(testutil.TensorflowModelAnalysisTest):
  """Test class with extra methods for unit-testing TFMA models / metrics."""

  def makeExample(self, **kwargs):
    """Returns a serialized TF.Example with the fields set accordingly.

    Example usage:
      makeExample(name="Tom", age=20, weight=132.5, colors=["red", "green"],
                  label=1.0)

    Note that field types will be inferred from the type of the arguments, so
    1.0 goes into a float_list, 1 goes into an int64_list, and "one" goes into
    a bytes_list. As the example illustrates, both singleton values and lists
    of values are accepted.

    Args:
     **kwargs: key=value pairs specifying the field values for the example.

    Returns:
      Serialized TF.Example with the fields set accordingly.
    """
    return self._makeExample(**kwargs).SerializeToString()

  def assertDictElementsWithinBounds(
      self, got_values_dict,
      expected_values_dict):
    for key, value in expected_values_dict.items():
      self.assertIn(key, got_values_dict)
      got_value = got_values_dict[key]
      if isinstance(value, BoundedValue):
        if got_value < value.lower_bound or got_value > value.upper_bound:
          self.fail('expecting key %s to have value between %f and %f '
                    '(both ends inclusive), but value was %f instead' %
                    (key, value.lower_bound, value.upper_bound, got_value))
      else:
        self.assertAlmostEqual(got_value, value, msg='key = %s' % key)

  def assertMetricsComputedWithoutBeamAre(self, eval_saved_model_path,
                                          serialized_examples,
                                          expected_metrics):
    """Checks metrics in-memory using the low-level APIs without Beam.

    Example usage:
      self.assertMetricsComputedWithoutBeamAre(
        eval_saved_model_path=path,
        serialized_examples=[self.makeExample(age=5, label=1.0),
                             self.makeExample(age=10, label=0.0)],
        expected_metrics={'average_loss': 0.1})

    Args:
      eval_saved_model_path: Path to the directory containing the
        EvalSavedModel.
      serialized_examples: List of serialized example bytes.
      expected_metrics: Dictionary of expected metric values.
    """
    self.assertDictElementsWithinBounds(
        got_values_dict=self._computeMetricsWithoutBeam(eval_saved_model_path,
                                                        serialized_examples),
        expected_values_dict=expected_metrics)

  def assertMetricsComputedWithoutBeamNoBatchingAre(
      self, eval_saved_model_path, serialized_examples,
      expected_metrics):
    """Checks metrics in-memory using the low-level APIs without Beam.

    This is the non-batched version of assertMetricsComputedWithoutBeamAre.
    This can be useful for debugging batching issues with TFMA or with your
    model (e.g. your model or metrics only works with a fixed-batch size - TFMA
    requires that your model can accept batches of any size).

    Args:
      eval_saved_model_path: Path to the directory containing the
        EvalSavedModel.
      serialized_examples: List of serialized example bytes.
      expected_metrics: Dictionary of expected metric values.
    """
    self.assertDictElementsWithinBounds(
        got_values_dict=self._computeMetricsWithoutBeamNoBatching(
            eval_saved_model_path, serialized_examples),
        expected_values_dict=expected_metrics)

  def _computeMetricsWithoutBeam(
      self, eval_saved_model_path,
      serialized_examples):
    """Computes metrics in-memory using the low-level APIs without Beam.

    Args:
      eval_saved_model_path: Path to the directory containing the
        EvalSavedModel.
      serialized_examples: List of serialized example bytes.

    Returns:
      Metrics computed by TFMA using your model on the given examples.
    """
    eval_saved_model = load.EvalSavedModel(eval_saved_model_path)
    features_predictions_labels_list = eval_saved_model.predict_list(
        serialized_examples)
    eval_saved_model.metrics_reset_update_get_list(
        features_predictions_labels_list)
    return eval_saved_model.get_metric_values()

  def _computeMetricsWithoutBeamNoBatching(
      self, eval_saved_model_path,
      serialized_examples):
    """Computes metrics in-memory using the low-level APIs without Beam.

    This is the non-batched version of computeMetricsWithoutBeam. This can be
    useful for debugging batching issues with TFMA or with your model
    (e.g. your model or metrics only works with a fixed-batch size - TFMA
    requires that your model can accept batches of any size)

    Args:
      eval_saved_model_path: Path to the directory containing the
        EvalSavedModel.
      serialized_examples: List of serialized example bytes.

    Returns:
      Metrics computed by TFMA using your model on the given examples.
    """
    eval_saved_model = load.EvalSavedModel(eval_saved_model_path)

    for example in serialized_examples:
      features_predictions_labels = eval_saved_model.predict(example)
      eval_saved_model.perform_metrics_update(features_predictions_labels)
    return eval_saved_model.get_metric_values()

  def assertMetricsComputedWithBeamAre(self, eval_saved_model_path,
                                       serialized_examples,
                                       expected_metrics):
    """Checks metrics computed using Beam.

    Metrics will be computed over all examples, without any slicing. If you
    want to provide your own PCollection (e.g. read a large number of examples
    from a file), if you want to check metrics over certain slices, or if you
    want to add additional post-export metrics, use the more general
    assertGeneralMetricsComputedWithBeamAre.

    Example usage:
      self.assertMetricsComputedWithBeamAre(
        eval_saved_model_path=path,
        serialized_examples=[self.makeExample(age=5, label=1.0),
                             self.makeExample(age=10, label=0.0)],
        expected_metrics={'average_loss': 0.1})

    Args:
      eval_saved_model_path: Path to the directory containing the
        EvalSavedModel.
      serialized_examples: List of serialized example bytes.
      expected_metrics: Dictionary of expected metric values.
    """

    def check_metrics(got):
      """Check metrics callback."""
      try:
        self.assertEqual(
            1, len(got), 'expecting metrics for exactly one slice, but got %d '
            'slices instead. metrics were: %s' % (len(got), got))
        (slice_key, value) = got[0]
        self.assertEqual((), slice_key)
        self.assertDictElementsWithinBounds(
            got_values_dict=value, expected_values_dict=expected_metrics)
      except AssertionError as err:
        raise beam_util.BeamAssertException(err)

    with beam.Pipeline() as pipeline:
      metrics, _ = (
          pipeline
          | 'CreateExamples' >> beam.Create(serialized_examples)
          | 'Evaluate' >>
          evaluate.Evaluate(eval_saved_model_path=eval_saved_model_path))

      beam_util.assert_that(metrics, check_metrics)

  def assertGeneralMetricsComputedWithBeamAre(
      self, eval_saved_model_path,
      examples_pcollection,
      slice_spec,
      add_metrics_callbacks,
      expected_slice_metrics):
    """Checks metrics computed using Beam.

    A more general version of assertMetricsComputedWithBeamAre. Note that the
    caller is responsible for setting up and running the Beam pipeline.

    Example usage:
      def add_metrics(features, predictions, labels):
       metric_ops = {
         'mse': tf.metrics.mean_squared_error(labels, predictions['logits']),
         'mae': tf.metrics.mean_absolute_error(labels, predictions['logits']),
      }
      return metric_ops

      with beam.Pipeline() as pipeline:
        expected_slice_metrics = {
            (): {
              'mae': 0.1,
              'mse': 0.2,
              tfma.post_export_metrics.metric_keys.AUC:
                tfma.test.BoundedValue(lower_bound=0.5)
            },
            (('age', 10),): {
              'mae': 0.2,
              'mse': 0.3,
              tfma.post_export_metrics.metric_keys.AUC:
                tfma.test.BoundedValue(lower_bound=0.5)
            },
        }
        examples = pipeline | 'ReadExamples' >> beam.io.ReadFromTFRecord(path)
        self.assertGeneralMetricsComputedWithBeamAre(
          eval_saved_model_path=path,
          examples_pcollection=examples,
          slice_spec=[tfma.SingleSliceSpec(),
                      tfma.SingleSliceSpec(columns=['age'])],
          add_metrics_callbacks=[
            add_metrics, tfma.post_export_metrics.post_export_metrics.auc()],
          expected_slice_metrics=expected_slice_metrics)

    Args:
      eval_saved_model_path: Path to the directory containing the
        EvalSavedModel.
      examples_pcollection: A PCollection of serialized example bytes.
      slice_spec: List of slice specifications.
      add_metrics_callbacks: Callbacks for adding additional metrics.
      expected_slice_metrics: Dictionary of dictionaries describing the expected
        metrics for each slice. The outer dictionary map slice keys to the
        expected metrics for that slice.
    """

    def check_metrics(got):
      """Check metrics callback."""
      try:
        slices = {}
        for slice_key, value in got:
          slices[slice_key] = value
        self.assertItemsEqual(slices.keys(), expected_slice_metrics.keys())
        for slice_key, expected_metrics in expected_slice_metrics.items():
          self.assertDictElementsWithinBounds(
              got_values_dict=slices[slice_key],
              expected_values_dict=expected_metrics)
      except AssertionError as err:
        raise beam_util.BeamAssertException(err)

    metrics, _ = (
        examples_pcollection
        | 'Evaluate' >> evaluate.Evaluate(
            eval_saved_model_path=eval_saved_model_path,
            slice_spec=slice_spec,
            add_metrics_callbacks=add_metrics_callbacks))

    beam_util.assert_that(metrics, check_metrics)
