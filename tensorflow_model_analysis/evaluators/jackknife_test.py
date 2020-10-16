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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing import util
import numpy as np

from tensorflow_model_analysis import types
from tensorflow_model_analysis.evaluators import jackknife
from tensorflow_model_analysis.metrics import binary_confusion_matrices
from tensorflow_model_analysis.metrics import example_count
from tensorflow_model_analysis.metrics import metric_types


class JackknifePTransformTest(absltest.TestCase):

  def test_jackknife_combine_per_key(self):

    def dict_value_sum(dict_elements):
      """Toy combiner which sums dict values."""
      result = collections.defaultdict(int)
      for dict_element in dict_elements:
        for k, v in dict_element.items():
          result[k] += v
      return result

    sliced_extracts = [
        (((u'slice_feature1', 1),), {
            u'label': 0
        }),
        (((u'slice_feature1', 1),), {
            u'label': 2
        }),
        (((u'slice_feature1', 2),), {
            u'label': 2
        }),
        (((u'slice_feature1', 2),), {
            u'label': 4
        }),
    ]
    with beam.Pipeline() as pipeline:
      result = (
          pipeline
          | 'Create' >> beam.Create(sliced_extracts, reshuffle=False)
          | 'JackknifeCombinePerKey' >> jackknife.JackknifeCombinePerKey(
              beam.combiners.SingleInputTupleCombineFn(dict_value_sum),
              num_jackknife_samples=2,
              random_seed=0))

      def check_result(got_pcoll):
        expected_pcoll = [(((u'slice_feature1', 1), (u'_sample_id', -1)), ({
            'label': 2
        }, {
            jackknife._JACKKNIFE_EXAMPLE_COUNT_METRIC_KEY: 2
        })), (((u'slice_feature1', 1), (u'_sample_id', 0)), ({
            'label': 2
        },)), (((u'slice_feature1', 1), (u'_sample_id', 1)), ({
            'label': 0
        },)),
                          (((u'slice_feature1', 2), (u'_sample_id', -1)), ({
                              'label': 6
                          }, {
                              jackknife._JACKKNIFE_EXAMPLE_COUNT_METRIC_KEY: 2
                          })),
                          (((u'slice_feature1', 2), (u'_sample_id', 0)), ({
                              'label': 2
                          },)),
                          (((u'slice_feature1', 2), (u'_sample_id', 1)), ({
                              'label': 4
                          },))]
        self.assertCountEqual(expected_pcoll, got_pcoll)

      util.assert_that(result, check_result)

  def test_jackknife_merge_jackknife_samples(self):
    x_key = metric_types.MetricKey(u'x')
    y_key = metric_types.MetricKey(u'y')
    cm_key = metric_types.MetricKey(u'confusion_matrix')
    cm_metric = binary_confusion_matrices.Matrices(
        thresholds=[0.5], tp=[0], fp=[1], tn=[2], fn=[3])
    example_count_key = metric_types.MetricKey(example_count.EXAMPLE_COUNT_NAME)
    slice_key1 = (u'slice_feature', 1)
    slice_key2 = (u'slice_feature', 2)
    sliced_derived_metrics = [
        # unsampled value for slice 1
        ((slice_key1, (jackknife._JACKKNIFE_SAMPLE_ID_KEY,
                       jackknife._JACKKNIFE_FULL_SAMPLE_ID)), {
                           x_key: 1.6,
                           y_key: 16,
                           cm_key: cm_metric,
                           example_count_key: 100,
                           jackknife._JACKKNIFE_EXAMPLE_COUNT_METRIC_KEY: 100
                       }),
        # sample values 1 of 2 for slice 1
        ((slice_key1, (jackknife._JACKKNIFE_SAMPLE_ID_KEY, 0)), {
            x_key: 1,
            y_key: 10,
            cm_key: cm_metric,
            example_count_key: 45,
        }),
        # sample values 2 of 2 for slice 1
        ((slice_key1, (jackknife._JACKKNIFE_SAMPLE_ID_KEY, 1)), {
            x_key: 2,
            y_key: 20,
            cm_key: cm_metric,
            example_count_key: 55,
        }),
        # unsampled value for slice 2
        ((slice_key2, (jackknife._JACKKNIFE_SAMPLE_ID_KEY,
                       jackknife._JACKKNIFE_FULL_SAMPLE_ID)), {
                           x_key: 3.3,
                           y_key: 33,
                           cm_key: cm_metric,
                           example_count_key: 1000,
                           jackknife._JACKKNIFE_EXAMPLE_COUNT_METRIC_KEY: 1000
                       }),
        # sample values 1 of 2 for slice 2
        ((slice_key2, (jackknife._JACKKNIFE_SAMPLE_ID_KEY, 0)), {
            x_key: 2,
            y_key: 20,
            cm_key: cm_metric,
            example_count_key: 450,
        }),
        # sample values 2 of 2 for slice 2
        ((slice_key2, (jackknife._JACKKNIFE_SAMPLE_ID_KEY, 1)), {
            x_key: 4,
            y_key: 40,
            cm_key: cm_metric,
            example_count_key: 550,
        }),
    ]

    with beam.Pipeline() as pipeline:
      result = (
          pipeline
          | 'Create' >> beam.Create(sliced_derived_metrics, reshuffle=False)
          | 'JackknifeCombinePerKey' >> jackknife.MergeJackknifeSamples(
              num_jackknife_samples=2, skip_ci_metric_keys=[example_count_key]))

      # For standard error calculations, see delete-d jackknife formula in:
      # https://www.stat.berkeley.edu/~hhuang/STAT152/Jackknife-Bootstrap.pdf
      # Rather than normalize by all possible n-choose-d samples, we normalize
      # by the actual number of samples (2).
      def check_result(got_pcoll):
        expected_pcoll = [
            (
                (slice_key1,),
                {
                    x_key:
                        types.ValueWithTDistribution(
                            sample_mean=1.5,
                            # (((100 - 100/2)/(100/2))*np.var([1, 2]))**0.5
                            sample_standard_deviation=.5,
                            sample_degrees_of_freedom=1,
                            unsampled_value=1.6),
                    y_key:
                        types.ValueWithTDistribution(
                            sample_mean=15,
                            # (((100 - 100/2)/(100/2))*np.var([10, 20]))**0.5
                            sample_standard_deviation=5,
                            sample_degrees_of_freedom=1,
                            unsampled_value=16),
                    cm_key:
                        cm_metric,
                    example_count_key:
                        100,
                }),
            (
                (slice_key2,),
                {
                    x_key:
                        types.ValueWithTDistribution(
                            sample_mean=3,
                            # (((1000 - 1000/2)/(1000/2))*np.var([2, 4]))**0.5
                            sample_standard_deviation=1,
                            sample_degrees_of_freedom=1,
                            unsampled_value=3.3),
                    y_key:
                        types.ValueWithTDistribution(
                            sample_mean=30,
                            # (((1000 - 1000/2)/(1000/2))*np.var([20, 40]))**0.5
                            sample_standard_deviation=10,
                            sample_degrees_of_freedom=1,
                            unsampled_value=33),
                    cm_key:
                        cm_metric,
                    example_count_key:
                        1000,
                }),
        ]
        self.assertCountEqual(expected_pcoll, got_pcoll)

      util.assert_that(result, check_result)

  def test_jackknife_merge_jackknife_samples_small_samples(self):
    metric_key = metric_types.MetricKey(u'metric')
    slice_key1 = (u'slice_feature', 1)
    slice_key2 = (u'slice_feature', 2)
    # the sample value is irrelevant for this test as we only verify counters.
    sample_value = {metric_key: 42}
    sliced_derived_metrics = [
        # unsampled value for slice 1
        ((slice_key1, (jackknife._JACKKNIFE_SAMPLE_ID_KEY,
                       jackknife._JACKKNIFE_FULL_SAMPLE_ID)), {
                           metric_key: 2.1,
                           jackknife._JACKKNIFE_EXAMPLE_COUNT_METRIC_KEY: 16
                       }),
        # 5 sample values for slice 1
        ((slice_key1, (jackknife._JACKKNIFE_SAMPLE_ID_KEY, 0)), sample_value),
        ((slice_key1, (jackknife._JACKKNIFE_SAMPLE_ID_KEY, 1)), sample_value),
        ((slice_key1, (jackknife._JACKKNIFE_SAMPLE_ID_KEY, 2)), sample_value),
        ((slice_key1, (jackknife._JACKKNIFE_SAMPLE_ID_KEY, 3)), sample_value),
        ((slice_key1, (jackknife._JACKKNIFE_SAMPLE_ID_KEY, 4)), sample_value),
        # unsampled value for slice 2
        ((slice_key2, (jackknife._JACKKNIFE_SAMPLE_ID_KEY,
                       jackknife._JACKKNIFE_FULL_SAMPLE_ID)), {
                           metric_key: 6.3,
                           jackknife._JACKKNIFE_EXAMPLE_COUNT_METRIC_KEY: 10000
                       }),
        # 5 sample values for slice 2
        ((slice_key2, (jackknife._JACKKNIFE_SAMPLE_ID_KEY, 0)), sample_value),
        ((slice_key2, (jackknife._JACKKNIFE_SAMPLE_ID_KEY, 1)), sample_value),
        ((slice_key2, (jackknife._JACKKNIFE_SAMPLE_ID_KEY, 2)), sample_value),
        ((slice_key2, (jackknife._JACKKNIFE_SAMPLE_ID_KEY, 3)), sample_value),
        ((slice_key2, (jackknife._JACKKNIFE_SAMPLE_ID_KEY, 4)), sample_value),
    ]

    with beam.Pipeline() as pipeline:
      _ = (
          pipeline
          | 'Create' >> beam.Create(sliced_derived_metrics, reshuffle=False)
          | 'MergeJackknifeSamples' >>
          jackknife.MergeJackknifeSamples(num_jackknife_samples=5))

      result = pipeline.run()
      # we expect one bad jackknife samples counter increment for slice1.
      # slice1: num_samples=5, n=16, d=3.2, sqrt(n)=4, d < sqrt(n) = True
      # slice2: num_samples=5, n=10000, d=2000, sqrt(n)=100, d < sqrt(n) = False
      metric_filter = beam.metrics.metric.MetricsFilter().with_name(
          'num_slices_with_small_jackknife_samples')
      counters = result.metrics().query(filter=metric_filter)['counters']
      self.assertLen(counters, 1)
      self.assertEqual(1, counters[0].committed)

      # verify total slice counter
      metric_filter = beam.metrics.metric.MetricsFilter().with_name(
          'num_slices')
      counters = result.metrics().query(filter=metric_filter)['counters']
      self.assertLen(counters, 1)
      self.assertEqual(2, counters[0].committed)

  def test_jackknife_merge_jackknife_samples_missing_samples(self):
    metric_key = metric_types.MetricKey(u'metric')
    slice_key1 = (u'slice_feature', 1)
    slice_key2 = (u'slice_feature', 2)
    # the sample value is irrelevant for this test as we only verify counters.
    sample_value = {metric_key: 42}
    sliced_derived_metrics = [
        # unsampled value for slice 1
        ((slice_key1, (jackknife._JACKKNIFE_SAMPLE_ID_KEY,
                       jackknife._JACKKNIFE_FULL_SAMPLE_ID)), {
                           metric_key: 2.1,
                           jackknife._JACKKNIFE_EXAMPLE_COUNT_METRIC_KEY: 16
                       }),
        # 2 sample values for slice 1
        ((slice_key1, (jackknife._JACKKNIFE_SAMPLE_ID_KEY, 0)), sample_value),
        ((slice_key1, (jackknife._JACKKNIFE_SAMPLE_ID_KEY, 1)), sample_value),
        # unsampled value for slice 2
        ((slice_key2, (jackknife._JACKKNIFE_SAMPLE_ID_KEY,
                       jackknife._JACKKNIFE_FULL_SAMPLE_ID)), {
                           metric_key: 6.3,
                           jackknife._JACKKNIFE_EXAMPLE_COUNT_METRIC_KEY: 10000
                       }),
        # Only 1 sample value (missing sample ID 1) for slice 2
        ((slice_key2, (jackknife._JACKKNIFE_SAMPLE_ID_KEY, 0)), sample_value),
    ]

    with beam.Pipeline() as pipeline:
      _ = (
          pipeline
          | 'Create' >> beam.Create(sliced_derived_metrics, reshuffle=False)
          | 'MergeJackknifeSamples' >>
          jackknife.MergeJackknifeSamples(num_jackknife_samples=2))

      result = pipeline.run()
      # we expect one missing samples counter increment for slice2, since we
      # expected 2 samples, but only saw 1.
      metric_filter = beam.metrics.metric.MetricsFilter().with_name(
          'num_slices_missing_jackknife_samples')
      counters = result.metrics().query(filter=metric_filter)['counters']
      self.assertLen(counters, 1)
      self.assertEqual(1, counters[0].committed)

      metric_filter = beam.metrics.metric.MetricsFilter().with_name(
          'zero_variance_metric_slice_size_dist')
      counters = result.metrics().query(filter=metric_filter)['distributions']
      self.assertLen(counters, 1)
      self.assertEqual(1, counters[0].committed.count)
      self.assertEqual(16, counters[0].committed.sum)
      self.assertEqual(16, counters[0].committed.min)
      self.assertEqual(16, counters[0].committed.max)

      # verify total slice counter
      metric_filter = beam.metrics.metric.MetricsFilter().with_name(
          'num_slices')
      counters = result.metrics().query(filter=metric_filter)['counters']
      self.assertLen(counters, 1)
      self.assertEqual(2, counters[0].committed)

  def test_jackknife_merge_jackknife_samples_numpy_overflow(self):
    sample_values = np.random.RandomState(seed=0).randint(0, 1e10, 20)
    slice_key = (u'slice_feature', 1)
    metric_key = metric_types.MetricKey(u'metric')
    sliced_derived_metrics = [
        ((slice_key, (jackknife._JACKKNIFE_SAMPLE_ID_KEY,
                      jackknife._JACKKNIFE_FULL_SAMPLE_ID)), {
                          metric_key: 1,
                          jackknife._JACKKNIFE_EXAMPLE_COUNT_METRIC_KEY: 200,
                      })
    ]
    for sample_id, value in enumerate(sample_values):
      sliced_derived_metrics.append(
          ((slice_key, (jackknife._JACKKNIFE_SAMPLE_ID_KEY, sample_id)), {
              metric_key: value,
          }))
    with beam.Pipeline() as pipeline:
      result = (
          pipeline
          | 'Create' >> beam.Create(sliced_derived_metrics, reshuffle=False)
          | 'JackknifeCombinePerKey' >>
          jackknife.MergeJackknifeSamples(num_jackknife_samples=20))

      def check_result(got_pcoll):
        expected_pcoll = [
            ((slice_key,), {
                metric_key:
                    types.ValueWithTDistribution(
                        sample_mean=5293977041.15,
                        sample_standard_deviation=12845957824.018991,
                        sample_degrees_of_freedom=19,
                        unsampled_value=1),
            }),
        ]
        self.assertCountEqual(expected_pcoll, got_pcoll)

      util.assert_that(result, check_result)


if __name__ == '__main__':
  absltest.main()
