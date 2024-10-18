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
"""Tests for evaluators.jackknife."""

import functools

from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing import util
from tensorflow_model_analysis.api import types
from tensorflow_model_analysis.evaluators import confidence_intervals_util
from tensorflow_model_analysis.evaluators import jackknife
from tensorflow_model_analysis.metrics import binary_confusion_matrices
from tensorflow_model_analysis.metrics import metric_types


class ListCombineFn(beam.CombineFn):

  def __init__(self, extract_output_append=None):
    self._extract_output_append = extract_output_append

  def create_accumulator(self):
    return []

  def add_input(self, accumulator, element):
    return accumulator + [element]

  def merge_accumulators(self, accumulators):
    return functools.reduce(list.__add__, accumulators)

  def extract_output(self, accumulator):
    if self._extract_output_append:
      return accumulator + [self._extract_output_append]
    else:
      return accumulator


class ListCombineFnExtractOutputNotImplemented(ListCombineFn):

  def extract_output(self, accumulator):
    raise NotImplementedError(
        'extract_output intentionally not implement to verify behavior. We '
        'would like to be able to mock a combine_fn and then call '
        'combine_fn.extract_output.assert_not_called().'
    )


class ListCombineFnAddInputNotImplemented(ListCombineFn):

  def add_input(self, accumulator, element):
    raise NotImplementedError(
        'add_input intentionally not implement to verify behavior. We would '
        'like to be able to mock a combine_fn and then call '
        'combine_fn.add_input.assert_not_called().'
    )


class JackknifeTest(absltest.TestCase):

  def test_accumulate_only_combiner(self):
    with beam.Pipeline() as pipeline:
      result = (
          pipeline
          | 'Create' >> beam.Create([1, 2])
          | 'AccumulateOnlyCombine'
          >> beam.CombineGlobally(
              jackknife._AccumulateOnlyCombineFn(
                  ListCombineFnExtractOutputNotImplemented(
                      extract_output_append=3
                  )
              )
          )
      )

      def check_result(got_pcoll):
        self.assertLen(got_pcoll, 1)
        self.assertEqual(got_pcoll[0], [1, 2])

      util.assert_that(result, check_result)

  def test_accumulator_combiner(self):
    with beam.Pipeline() as pipeline:
      result = (
          pipeline
          | 'Create' >> beam.Create([[1], [2]])
          | 'AccumulatorCombine'
          >> beam.CombineGlobally(
              jackknife._AccumulatorCombineFn(
                  ListCombineFnAddInputNotImplemented(extract_output_append=3)
              )
          )
      )

      def check_result(got_pcoll):
        self.assertLen(got_pcoll, 1)
        self.assertEqual(got_pcoll[0], [1, 2, 3])

      util.assert_that(result, check_result)

  def test_jackknife_sample_combine_fn(self):
    x_key = metric_types.MetricKey('x')
    y_key = metric_types.MetricKey('y')
    cm_key = metric_types.MetricKey('confusion_matrix')
    cm_metric = binary_confusion_matrices.Matrices(
        thresholds=[0.5], tp=[0], fp=[1], tn=[2], fn=[3]
    )
    slice_key1 = (('slice_feature', 1),)
    slice_key2 = (('slice_feature', 2),)
    samples = [
        # point estimate for slice 1
        (
            slice_key1,
            confidence_intervals_util.SampleMetrics(
                sample_id=jackknife._FULL_SAMPLE_ID,
                metrics={
                    x_key: 1.6,
                    y_key: 16,
                    cm_key: cm_metric,
                },
            ),
        ),
        # sample values 1 of 2 for slice 1
        (
            slice_key1,
            confidence_intervals_util.SampleMetrics(
                sample_id=0,
                metrics={
                    x_key: 1,
                    y_key: 10,
                    cm_key: cm_metric - 1,
                },
            ),
        ),
        # sample values 2 of 2 for slice 1
        (
            slice_key1,
            confidence_intervals_util.SampleMetrics(
                sample_id=1,
                metrics={
                    x_key: 2,
                    y_key: 20,
                    cm_key: cm_metric + 1,
                },
            ),
        ),
        # point estimate for slice 2
        (
            slice_key2,
            confidence_intervals_util.SampleMetrics(
                sample_id=jackknife._FULL_SAMPLE_ID,
                metrics={
                    x_key: 3.3,
                    y_key: 33,
                    cm_key: cm_metric,
                },
            ),
        ),
        # sample values 1 of 2 for slice 2
        (
            slice_key2,
            confidence_intervals_util.SampleMetrics(
                sample_id=0,
                metrics={
                    x_key: 2,
                    y_key: 20,
                    cm_key: cm_metric - 10,
                },
            ),
        ),
        # sample values 2 of 2 for slice 2
        (
            slice_key2,
            confidence_intervals_util.SampleMetrics(
                sample_id=1,
                metrics={
                    x_key: 4,
                    y_key: 40,
                    cm_key: cm_metric + 10,
                },
            ),
        ),
    ]

    with beam.Pipeline() as pipeline:
      result = (
          pipeline
          | 'Create' >> beam.Create(samples, reshuffle=False)
          | 'CombineJackknifeSamplesPerKey'
          >> beam.CombinePerKey(
              jackknife._JackknifeSampleCombineFn(num_jackknife_samples=2)
          )
      )

      # WARNING: Do not change this test without carefully considering the
      # impact on clients due to changed CI bounds. The current implementation
      # follows jackknife cookie bucket method described in:
      # go/rasta-confidence-intervals
      def check_result(got_pcoll):
        expected_pcoll = [
            (
                slice_key1,
                {
                    x_key: types.ValueWithTDistribution(
                        sample_mean=1.5,
                        sample_standard_deviation=0.5,
                        sample_degrees_of_freedom=1,
                        unsampled_value=1.6,
                    ),
                    y_key: types.ValueWithTDistribution(
                        sample_mean=15.0,
                        sample_standard_deviation=5,
                        sample_degrees_of_freedom=1,
                        unsampled_value=16,
                    ),
                    cm_key: types.ValueWithTDistribution(
                        sample_mean=cm_metric,
                        sample_standard_deviation=(
                            binary_confusion_matrices.Matrices(
                                thresholds=[0.5], tp=[1], fp=[1], tn=[1], fn=[1]
                            )
                        ),
                        sample_degrees_of_freedom=1,
                        unsampled_value=cm_metric,
                    ),
                },
            ),
            (
                slice_key2,
                {
                    x_key: types.ValueWithTDistribution(
                        sample_mean=3.0,
                        sample_standard_deviation=1,
                        sample_degrees_of_freedom=1,
                        unsampled_value=3.3,
                    ),
                    y_key: types.ValueWithTDistribution(
                        sample_mean=30.0,
                        sample_standard_deviation=10,
                        sample_degrees_of_freedom=1,
                        unsampled_value=33,
                    ),
                    cm_key: types.ValueWithTDistribution(
                        sample_mean=cm_metric,
                        sample_standard_deviation=(
                            binary_confusion_matrices.Matrices(
                                thresholds=[0.5],
                                tp=[10],
                                fp=[10],
                                tn=[10],
                                fn=[10],
                            )
                        ),
                        sample_degrees_of_freedom=1,
                        unsampled_value=cm_metric,
                    ),
                },
            ),
        ]
        self.assertCountEqual(expected_pcoll, got_pcoll)

      util.assert_that(result, check_result)


