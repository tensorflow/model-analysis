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
# Standard __future__ imports
from __future__ import print_function

from typing import Any, List, Dict, Text, Optional

import apache_beam as beam
from apache_beam.testing import util as beam_util

from tensorflow_model_analysis import config
from tensorflow_model_analysis import types
from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.eval_saved_model import load
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.evaluators import legacy_metrics_and_plots_evaluator
from tensorflow_model_analysis.extractors import extractor
from tensorflow_model_analysis.slicer import slicer_lib as slicer


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(types.Extracts)
def Extract(  # pylint: disable=invalid-name
    extracts: beam.pvalue.PCollection, extractors: List[extractor.Extractor]):
  for x in extractors:
    extracts = (extracts | x.stage_name >> x.ptransform)
  return extracts


class BoundedValue(object):
  """Represents a bounded value for a metric for the TFMA unit test."""

  def __init__(self,
               lower_bound: float = float('-inf'),
               upper_bound: float = float('inf')):
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

  def assertDictElementsWithinBounds(self, got_values_dict: Dict[Text, Any],
                                     expected_values_dict: Dict[Text, Any]):
    """Checks the elements for two dictionaries.

    It asserts all values in `expected_values_dict` are close to values with the
    same key in `got_values_dict`.

    Args:
      got_values_dict: The actual dictionary.
      expected_values_dict: The expected dictionary. The values in can be either
        `BoundedValue` or any type accepted by
        `tf.test.TestCase.assertAllClose()`. When the type is `BoundedValue`, it
        expects the corresponding value from `got_values_dict` falls into the
        boundaries provided in the `BoundedValue`.
    """
    for key, value in expected_values_dict.items():
      self.assertIn(key, got_values_dict)
      got_value = got_values_dict[key]
      if isinstance(value, BoundedValue):
        if got_value < value.lower_bound or got_value > value.upper_bound:
          self.fail('expecting key %s to have value between %f and %f '
                    '(both ends inclusive), but value was %f instead' %
                    (key, value.lower_bound, value.upper_bound, got_value))
      else:
        self.assertAllClose(got_value, value, msg='key = %s' % key)

  def assertMetricsComputedWithoutBeamAre(self, eval_saved_model_path: Text,
                                          serialized_examples: List[bytes],
                                          expected_metrics: Dict[Text, Any]):
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
      self, eval_saved_model_path: Text, serialized_examples: List[bytes],
      expected_metrics: Dict[Text, Any]):
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
      self, eval_saved_model_path: Text,
      serialized_examples: List[bytes]) -> Dict[Text, Any]:
    """Computes metrics in-memory using the low-level APIs without Beam.

    Args:
      eval_saved_model_path: Path to the directory containing the
        EvalSavedModel.
      serialized_examples: List of serialized example bytes.

    Returns:
      Metrics computed by TFMA using your model on the given examples.
    """
    eval_saved_model = load.EvalSavedModel(eval_saved_model_path)
    eval_saved_model.metrics_reset_update_get_list(serialized_examples)
    return eval_saved_model.get_metric_values()

  def _computeMetricsWithoutBeamNoBatching(
      self, eval_saved_model_path: Text,
      serialized_examples: List[bytes]) -> Dict[Text, Any]:
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
      eval_saved_model.metrics_reset_update_get_list([example])
    return eval_saved_model.get_metric_values()

  def assertMetricsComputedWithBeamAre(
      self,
      eval_saved_model_path: Text,
      serialized_examples: List[bytes],
      expected_metrics: Dict[Text, Any],
      add_metrics_callbacks: Optional[List[
          types.AddMetricsCallbackType]] = None):
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
      add_metrics_callbacks: Optional. Callbacks for adding additional metrics.
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

    eval_config = config.EvalConfig()
    eval_shared_model = model_eval_lib.default_eval_shared_model(
        eval_saved_model_path=eval_saved_model_path,
        add_metrics_callbacks=add_metrics_callbacks)
    extractors = model_eval_lib.default_extractors(
        eval_config=eval_config, eval_shared_model=eval_shared_model)

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      (metrics, _), _ = (
          pipeline
          | 'CreateExamples' >> beam.Create(serialized_examples)
          | 'InputsToExtracts' >> model_eval_lib.InputsToExtracts()
          | 'Extract' >> Extract(extractors=extractors)
          | 'ComputeMetricsAndPlots' >> legacy_metrics_and_plots_evaluator
          .ComputeMetricsAndPlots(eval_shared_model=eval_shared_model))
      # pylint: enable=no-value-for-parameter

      beam_util.assert_that(metrics, check_metrics)

  def assertGeneralMetricsComputedWithBeamAre(
      self, eval_saved_model_path: Text,
      examples_pcollection: beam.pvalue.PCollection,
      slice_spec: List[slicer.SingleSliceSpec],
      add_metrics_callbacks: List[types.AddMetricsCallbackType],
      expected_slice_metrics: Dict[Any, Dict[Text, Any]]):
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
          slice_spec=[tfma.slicer.SingleSliceSpec(),
                      tfma.slicer.SingleSliceSpec(columns=['age'])],
          add_metrics_callbacks=[
            add_metrics, tfma.post_export_metrics.auc()],
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
        self.assertItemsEqual(
            list(slices.keys()), list(expected_slice_metrics.keys()))
        for slice_key, expected_metrics in expected_slice_metrics.items():
          self.assertDictElementsWithinBounds(
              got_values_dict=slices[slice_key],
              expected_values_dict=expected_metrics)
      except AssertionError as err:
        raise beam_util.BeamAssertException(err)

    slicing_specs = None
    if slice_spec:
      slicing_specs = [s.to_proto() for s in slice_spec]
    eval_config = config.EvalConfig(slicing_specs=slicing_specs)
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=eval_saved_model_path,
        add_metrics_callbacks=add_metrics_callbacks)
    extractors = model_eval_lib.default_extractors(
        eval_config=eval_config, eval_shared_model=eval_shared_model)

    # pylint: disable=no-value-for-parameter
    (metrics, _), _ = (
        examples_pcollection
        | 'InputsToExtracts' >> model_eval_lib.InputsToExtracts()
        | 'Extract' >> Extract(extractors=extractors)
        | 'ComputeMetricsAndPlots' >> legacy_metrics_and_plots_evaluator
        .ComputeMetricsAndPlots(eval_shared_model=eval_shared_model))
    # pylint: enable=no-value-for-parameter

    beam_util.assert_that(metrics, check_metrics)
