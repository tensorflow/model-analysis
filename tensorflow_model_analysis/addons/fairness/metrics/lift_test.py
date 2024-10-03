# Copyright 2021 Google LLC
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
"""Tests for lift metrics."""


import pytest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis.addons.fairness.metrics import lift
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.proto import config_pb2


@pytest.mark.xfail(run=False, reason="PR 183 This class contains tests that fail and needs to be fixed. "
"If all tests pass, please remove this mark.")
class LiftTest(testutil.TensorflowModelAnalysisTest, parameterized.TestCase):

  def _assert_test(
      self,
      num_buckets,
      baseline_examples,
      comparison_examples,
      lift_metric_value,
      ignore_out_of_bound_examples=False,
  ):
    eval_config = config_pb2.EvalConfig(
        cross_slicing_specs=[config_pb2.CrossSlicingSpec()]
    )
    computations = lift.Lift(
        num_buckets=num_buckets,
        ignore_out_of_bound_examples=ignore_out_of_bound_examples,
    ).computations(eval_config=eval_config, example_weighted=True)
    histogram = computations[0]
    lift_metrics = computations[1]

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      baseline_result = (
          pipeline
          | 'CreateB' >> beam.Create(baseline_examples)
          | 'ProcessB' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSliceB' >> beam.Map(lambda x: ((), x))
          | 'ComputeHistogramB' >> beam.CombinePerKey(histogram.combiner)
      )  # pyformat: ignore

      comparison_result = (
          pipeline
          | 'CreateC' >> beam.Create(comparison_examples)
          | 'ProcessC' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSliceC' >> beam.Map(lambda x: ('slice', x))
          | 'ComputeHistogramC' >> beam.CombinePerKey(histogram.combiner)
      )  # pyformat: ignore

      # pylint: enable=no-value-for-parameter

      merged_result = (
          baseline_result,
          comparison_result,
      ) | 'MergePCollections' >> beam.Flatten()

      def check_result(got):
        try:
          self.assertLen(got, 2)
          slice_1, metric_1 = got[0]
          _, metric_2 = got[1]
          if not slice_1:
            lift_value = lift_metrics.cross_slice_comparison(metric_1, metric_2)
          else:
            lift_value = lift_metrics.cross_slice_comparison(metric_2, metric_1)

          self.assertDictElementsAlmostEqual(
              lift_value,
              {
                  metric_types.MetricKey(
                      name=f'lift@{num_buckets}', example_weighted=True
                  ): lift_metric_value,
              },
          )
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(merged_result, check_result, label='result')

  def testLift_continuousLabelsAndPredictions(self):
    baseline_examples = [
        {
            'labels': np.array([0.0]),
            'predictions': np.array([0.1]),
            'example_weights': np.array([3.0]),
        },
        {
            'labels': np.array([0.3]),
            'predictions': np.array([0.5]),
            'example_weights': np.array([5.0]),
        },
        {
            'labels': np.array([0.6]),
            'predictions': np.array([0.8]),
            'example_weights': np.array([2.0]),
        },
        {
            'labels': np.array([0.9]),
            'predictions': np.array([0.3]),
            'example_weights': np.array([8.0]),
        },
        {
            'labels': np.array([0.9]),
            'predictions': np.array([0.9]),
            'example_weights': np.array([3.0]),
        },
    ]

    comparison_examples = [
        {
            'labels': np.array([0.0]),
            'predictions': np.array([0.8]),
            'example_weights': np.array([1.0]),
        },
        {
            'labels': np.array([0.2]),
            'predictions': np.array([0.3]),
            'example_weights': np.array([2.0]),
        },
        {
            'labels': np.array([0.5]),
            'predictions': np.array([0.5]),
            'example_weights': np.array([5.0]),
        },
        {
            'labels': np.array([0.7]),
            'predictions': np.array([0.4]),
            'example_weights': np.array([2.0]),
        },
        {
            'labels': np.array([0.9]),
            'predictions': np.array([0.3]),
            'example_weights': np.array([3.0]),
        },
    ]

    self._assert_test(3, baseline_examples, comparison_examples, -0.136013986)

  def testLift_baselineAndComparisonAreSame(self):
    baseline_examples = [
        {
            'labels': np.array([0.0]),
            'predictions': np.array([0.1]),
            'example_weights': np.array([3.0]),
        },
        {
            'labels': np.array([0.3]),
            'predictions': np.array([0.5]),
            'example_weights': np.array([5.0]),
        },
        {
            'labels': np.array([0.6]),
            'predictions': np.array([0.8]),
            'example_weights': np.array([2.0]),
        },
        {
            'labels': np.array([0.9]),
            'predictions': np.array([0.3]),
            'example_weights': np.array([8.0]),
        },
        {
            'labels': np.array([1.0]),
            'predictions': np.array([0.9]),
            'example_weights': np.array([3.0]),
        },
    ]

    self._assert_test(3, baseline_examples, baseline_examples, 0.0)

  def testLift_ignoringOutOfBoundExamples(self):
    baseline_examples = [
        {
            'labels': np.array([0.0]),
            'predictions': np.array([0.1]),
            'example_weights': np.array([3.0]),
        },
        {
            'labels': np.array([0.3]),
            'predictions': np.array([0.5]),
            'example_weights': np.array([5.0]),
        },
        {
            'labels': np.array([0.6]),
            'predictions': np.array([0.8]),
            'example_weights': np.array([2.0]),
        },
        {
            'labels': np.array([0.9]),
            'predictions': np.array([0.3]),
            'example_weights': np.array([8.0]),
        },
        {
            'labels': np.array([-0.9]),  # Ignore this example
            'predictions': np.array([0.3]),
            'example_weights': np.array([8.0]),
        },
        {
            'labels': np.array([0.9]),
            'predictions': np.array([0.9]),
            'example_weights': np.array([3.0]),
        },
    ]

    comparison_examples = [
        {
            'labels': np.array([0.0]),
            'predictions': np.array([0.8]),
            'example_weights': np.array([1.0]),
        },
        {
            'labels': np.array([0.2]),
            'predictions': np.array([0.3]),
            'example_weights': np.array([2.0]),
        },
        {
            'labels': np.array([0.5]),
            'predictions': np.array([0.5]),
            'example_weights': np.array([5.0]),
        },
        {
            'labels': np.array([0.7]),
            'predictions': np.array([0.4]),
            'example_weights': np.array([2.0]),
        },
        {
            'labels': np.array([1.9]),  # Ignore this example
            'predictions': np.array([0.3]),
            'example_weights': np.array([8.0]),
        },
        {
            'labels': np.array([0.9]),
            'predictions': np.array([0.3]),
            'example_weights': np.array([3.0]),
        },
    ]

    self._assert_test(
        3,
        baseline_examples,
        comparison_examples,
        -0.136013986,
        ignore_out_of_bound_examples=True,
    )

  def testLift_binaryLabelsAndContinuousPredictions(self):
    baseline_examples = [
        {
            'labels': np.array([0.0]),
            'predictions': np.array([0.1]),
            'example_weights': np.array([3.0]),
        },
        {
            'labels': np.array([0.0]),
            'predictions': np.array([0.5]),
            'example_weights': np.array([5.0]),
        },
        {
            'labels': np.array([1.0]),
            'predictions': np.array([0.8]),
            'example_weights': np.array([2.0]),
        },
        {
            'labels': np.array([1.0]),
            'predictions': np.array([0.3]),
            'example_weights': np.array([8.0]),
        },
        {
            'labels': np.array([1.0]),
            'predictions': np.array([0.9]),
            'example_weights': np.array([3.0]),
        },
    ]

    comparison_examples = [
        {
            'labels': np.array([0.0]),
            'predictions': np.array([0.8]),
            'example_weights': np.array([1.0]),
        },
        {
            'labels': np.array([0.0]),
            'predictions': np.array([0.3]),
            'example_weights': np.array([2.0]),
        },
        {
            'labels': np.array([0.0]),
            'predictions': np.array([0.5]),
            'example_weights': np.array([5.0]),
        },
        {
            'labels': np.array([1.0]),
            'predictions': np.array([0.4]),
            'example_weights': np.array([2.0]),
        },
        {
            'labels': np.array([1.0]),
            'predictions': np.array([0.3]),
            'example_weights': np.array([3.0]),
        },
    ]

    self._assert_test(2, baseline_examples, comparison_examples, 0.01715976331)

  def testLift_binaryLabelsAndPredictions(self):
    baseline_examples = [
        {
            'labels': np.array([0.0]),
            'predictions': np.array([1.0]),
            'example_weights': np.array([3.0]),
        },
        {
            'labels': np.array([0.0]),
            'predictions': np.array([0.0]),
            'example_weights': np.array([5.0]),
        },
        {
            'labels': np.array([1.0]),
            'predictions': np.array([0.0]),
            'example_weights': np.array([2.0]),
        },
        {
            'labels': np.array([1.0]),
            'predictions': np.array([1.0]),
            'example_weights': np.array([8.0]),
        },
        {
            'labels': np.array([1.0]),
            'predictions': np.array([0.0]),
            'example_weights': np.array([3.0]),
        },
    ]

    comparison_examples = [
        {
            'labels': np.array([0.0]),
            'predictions': np.array([1.0]),
            'example_weights': np.array([1.0]),
        },
        {
            'labels': np.array([0.0]),
            'predictions': np.array([0.0]),
            'example_weights': np.array([2.0]),
        },
        {
            'labels': np.array([0.0]),
            'predictions': np.array([1.0]),
            'example_weights': np.array([5.0]),
        },
        {
            'labels': np.array([1.0]),
            'predictions': np.array([0.0]),
            'example_weights': np.array([2.0]),
        },
        {
            'labels': np.array([1.0]),
            'predictions': np.array([1.0]),
            'example_weights': np.array([3.0]),
        },
    ]

    self._assert_test(2, baseline_examples, comparison_examples, 0.224852071)

  def testLift_raisesExceptionWhenEvalConfigIsNone(self):
    with self.assertRaises(ValueError):
      _ = lift.Lift(num_buckets=3).computations()

  def testLift_raisesExceptionWhenCrossSlicingSpecIsAbsent(self):
    with self.assertRaises(ValueError):
      _ = lift.Lift(num_buckets=3).computations(
          eval_config=config_pb2.EvalConfig()
      )


