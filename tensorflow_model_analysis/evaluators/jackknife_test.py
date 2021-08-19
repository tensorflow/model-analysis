# Lint as: python3
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
import numpy as np

from tensorflow_model_analysis import types
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
        'combine_fn.extract_output.assert_not_called().')


class ListCombineFnAddInputNotImplemented(ListCombineFn):

  def add_input(self, accumulator, element):
    raise NotImplementedError(
        'add_input intentionally not implement to verify behavior. We would '
        'like to be able to mock a combine_fn and then call '
        'combine_fn.add_input.assert_not_called().')


class JackknifeTest(absltest.TestCase):

  def test_accumulate_only_combiner(self):
    with beam.Pipeline() as pipeline:
      result = (
          pipeline
          | 'Create' >> beam.Create([1, 2])
          | 'AccumulateOnlyCombine' >> beam.CombineGlobally(
              jackknife._AccumulateOnlyCombineFn(
                  ListCombineFnExtractOutputNotImplemented(
                      extract_output_append=3))))

      def check_result(got_pcoll):
        self.assertLen(got_pcoll, 1)
        self.assertEqual(got_pcoll[0], [1, 2])

      util.assert_that(result, check_result)

  def test_accumulator_combiner(self):
    with beam.Pipeline() as pipeline:
      result = (
          pipeline
          | 'Create' >> beam.Create([[1], [2]])
          | 'AccumulatorCombine' >> beam.CombineGlobally(
              jackknife._AccumulatorCombineFn(
                  ListCombineFnAddInputNotImplemented(extract_output_append=3)))
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
        thresholds=[0.5], tp=[0], fp=[1], tn=[2], fn=[3])
    slice_key1 = (('slice_feature', 1),)
    slice_key2 = (('slice_feature', 2),)
    samples = [
        # unsampled value for slice 1
        (slice_key1,
         confidence_intervals_util.SampleMetrics(
             sample_id=jackknife._FULL_SAMPLE_ID,
             metrics={
                 x_key: 1.6,
                 y_key: 16,
                 cm_key: cm_metric,
                 jackknife._JACKKNIFE_EXAMPLE_COUNT_METRIC_KEY: 100,
             })),
        # sample values 1 of 2 for slice 1
        (slice_key1,
         confidence_intervals_util.SampleMetrics(
             sample_id=0, metrics={
                 x_key: 1,
                 y_key: 10,
                 cm_key: cm_metric - 1,
             })),
        # sample values 2 of 2 for slice 1
        (slice_key1,
         confidence_intervals_util.SampleMetrics(
             sample_id=1, metrics={
                 x_key: 2,
                 y_key: 20,
                 cm_key: cm_metric + 1,
             })),
        # unsampled value for slice 2
        (slice_key2,
         confidence_intervals_util.SampleMetrics(
             sample_id=jackknife._FULL_SAMPLE_ID,
             metrics={
                 x_key: 3.3,
                 y_key: 33,
                 cm_key: cm_metric,
                 jackknife._JACKKNIFE_EXAMPLE_COUNT_METRIC_KEY: 1000,
             })),
        # sample values 1 of 2 for slice 2
        (slice_key2,
         confidence_intervals_util.SampleMetrics(
             sample_id=0,
             metrics={
                 x_key: 2,
                 y_key: 20,
                 cm_key: cm_metric - 10,
             })),
        # sample values 2 of 2 for slice 2
        (slice_key2,
         confidence_intervals_util.SampleMetrics(
             sample_id=1,
             metrics={
                 x_key: 4,
                 y_key: 40,
                 cm_key: cm_metric + 10,
             })),
    ]

    with beam.Pipeline() as pipeline:
      result = (
          pipeline
          | 'Create' >> beam.Create(samples, reshuffle=False)
          | 'CombineJackknifeSamplesPerKey' >> beam.CombinePerKey(
              jackknife._JackknifeSampleCombineFn(num_jackknife_samples=2)))

      # For standard error calculations, see delete-d jackknife formula in:
      # https://www.stat.berkeley.edu/~hhuang/STAT152/Jackknife-Bootstrap.pdf
      # Rather than normalize by all possible n-choose-d samples, we normalize
      # by the actual number of samples (2).
      def check_result(got_pcoll):
        slice1_jackknife_factor = (100 - 100 / 2) / (100 / 2)
        slice2_jackknife_factor = (1000 - 1000 / 2) / (1000 / 2)
        expected_pcoll = [
            (
                slice_key1,
                {
                    x_key:
                        types.ValueWithTDistribution(
                            sample_mean=1.5,
                            # sample_standard_deviation=0.5
                            sample_standard_deviation=(
                                slice1_jackknife_factor *
                                np.var([1, 2], ddof=1))**0.5,
                            sample_degrees_of_freedom=1,
                            unsampled_value=1.6),
                    y_key:
                        types.ValueWithTDistribution(
                            sample_mean=15.,
                            # sample_standard_deviation=5,
                            sample_standard_deviation=(
                                slice1_jackknife_factor *
                                np.var([10, 20], ddof=1))**0.5,
                            sample_degrees_of_freedom=1,
                            unsampled_value=16),
                    cm_key:
                        types.ValueWithTDistribution(
                            sample_mean=cm_metric,
                            sample_standard_deviation=(
                                cm_metric * 0 + slice1_jackknife_factor *
                                np.var([-1, 1], ddof=1))**0.5,
                            sample_degrees_of_freedom=1,
                            unsampled_value=cm_metric),
                    jackknife._JACKKNIFE_EXAMPLE_COUNT_METRIC_KEY:
                        100,
                }),
            (
                slice_key2,
                {
                    x_key:
                        types.ValueWithTDistribution(
                            sample_mean=3.,
                            # sample_standard_deviation=1,
                            sample_standard_deviation=(
                                slice2_jackknife_factor *
                                np.var([2, 4], ddof=1))**0.5,
                            sample_degrees_of_freedom=1,
                            unsampled_value=3.3),
                    y_key:
                        types.ValueWithTDistribution(
                            sample_mean=30.,
                            # sample_standard_deviation=10,
                            sample_standard_deviation=(
                                slice2_jackknife_factor *
                                np.var([20, 40], ddof=1))**0.5,
                            sample_degrees_of_freedom=1,
                            unsampled_value=33),
                    cm_key:
                        types.ValueWithTDistribution(
                            sample_mean=cm_metric,
                            sample_standard_deviation=(
                                cm_metric * 0 + slice2_jackknife_factor *
                                np.var([-10, 10], ddof=1))**0.5,
                            sample_degrees_of_freedom=1,
                            unsampled_value=cm_metric),
                    jackknife._JACKKNIFE_EXAMPLE_COUNT_METRIC_KEY:
                        1000,
                }),
        ]
        self.assertCountEqual(expected_pcoll, got_pcoll)

      util.assert_that(result, check_result)


if __name__ == '__main__':
  absltest.main()
