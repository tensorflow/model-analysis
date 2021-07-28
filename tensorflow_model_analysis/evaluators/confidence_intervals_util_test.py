# Lint as: python3
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
"""Tests for confidence_intervals_util."""

from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing import util
import numpy as np

from tensorflow_model_analysis import types
from tensorflow_model_analysis.evaluators import confidence_intervals_util
from tensorflow_model_analysis.metrics import binary_confusion_matrices
from tensorflow_model_analysis.metrics import example_count
from tensorflow_model_analysis.metrics import metric_types

_FULL_SAMPLE_ID = -1


class ConfidenceIntervalsUtilTest(absltest.TestCase):

  def test_sample_combine_fn_per_slice(self):
    x_key = metric_types.MetricKey('x')
    y_key = metric_types.MetricKey('y')
    cm_key = metric_types.MetricKey('confusion_matrix')
    cm_metric = binary_confusion_matrices.Matrices(
        thresholds=[0.5], tp=[0], fp=[1], tn=[2], fn=[3])
    example_count_key = metric_types.MetricKey(example_count.EXAMPLE_COUNT_NAME)
    slice_key1 = (('slice_feature', 1),)
    slice_key2 = (('slice_feature', 2),)
    samples = [
        # unsampled value for slice 1
        (slice_key1,
         confidence_intervals_util.SampleMetrics(
             sample_id=_FULL_SAMPLE_ID,
             metrics={
                 x_key: 1.6,
                 y_key: 16,
                 cm_key: cm_metric,
                 example_count_key: 100,
             })),
        # sample values 1 of 2 for slice 1
        (slice_key1,
         confidence_intervals_util.SampleMetrics(
             sample_id=0,
             metrics={
                 x_key: 1,
                 y_key: 10,
                 cm_key: cm_metric,
                 example_count_key: 45,
             })),
        # sample values 2 of 2 for slice 1
        (slice_key1,
         confidence_intervals_util.SampleMetrics(
             sample_id=1,
             metrics={
                 x_key: 2,
                 y_key: 20,
                 cm_key: cm_metric,
                 example_count_key: 55,
             })),
        # unsampled value for slice 2
        (slice_key2,
         confidence_intervals_util.SampleMetrics(
             sample_id=_FULL_SAMPLE_ID,
             metrics={
                 x_key: 3.3,
                 y_key: 33,
                 cm_key: cm_metric,
                 example_count_key: 1000,
             })),
        # sample values 1 of 2 for slice 2
        (slice_key2,
         confidence_intervals_util.SampleMetrics(
             sample_id=0,
             metrics={
                 x_key: 2,
                 y_key: 20,
                 cm_key: cm_metric,
                 example_count_key: 450,
             })),
        # sample values 2 of 2 for slice 2
        (slice_key2,
         confidence_intervals_util.SampleMetrics(
             sample_id=1,
             metrics={
                 x_key: 4,
                 y_key: 40,
                 cm_key: cm_metric,
                 example_count_key: 550,
             })),
    ]

    with beam.Pipeline() as pipeline:
      result = (
          pipeline
          | 'Create' >> beam.Create(samples, reshuffle=False)
          | 'CombineSamplesPerKey' >> beam.CombinePerKey(
              confidence_intervals_util.SampleCombineFn(
                  num_samples=2,
                  full_sample_id=_FULL_SAMPLE_ID,
                  skip_ci_metric_keys=[example_count_key])))

      def check_result(got_pcoll):
        expected_pcoll = [
            (
                slice_key1,
                {
                    x_key:
                        types.ValueWithTDistribution(
                            sample_mean=1.5,
                            # sample_standard_deviation=0.5
                            sample_standard_deviation=np.std([1, 2], ddof=1),
                            sample_degrees_of_freedom=1,
                            unsampled_value=1.6),
                    y_key:
                        types.ValueWithTDistribution(
                            sample_mean=15,
                            # sample_standard_deviation=5,
                            sample_standard_deviation=np.std([10, 20], ddof=1),
                            sample_degrees_of_freedom=1,
                            unsampled_value=16),
                    cm_key:
                        cm_metric,
                    example_count_key:
                        100,
                }),
            (
                slice_key2,
                {
                    x_key:
                        types.ValueWithTDistribution(
                            sample_mean=3,
                            # sample_standard_deviation=1,
                            sample_standard_deviation=np.std([2, 4], ddof=1),
                            sample_degrees_of_freedom=1,
                            unsampled_value=3.3),
                    y_key:
                        types.ValueWithTDistribution(
                            sample_mean=30,
                            # sample_standard_deviation=10,
                            sample_standard_deviation=np.std([20, 40], ddof=1),
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

  def test_sample_combine_fn(self):
    metric_key = metric_types.MetricKey(name='metric')
    samples = [
        confidence_intervals_util.SampleMetrics(
            sample_id=0, metrics={metric_key: 0}),
        confidence_intervals_util.SampleMetrics(
            sample_id=1, metrics={metric_key: 7}),
        confidence_intervals_util.SampleMetrics(
            sample_id=_FULL_SAMPLE_ID, metrics={metric_key: 4})
    ]

    with beam.Pipeline() as pipeline:
      result = (
          pipeline
          | 'Create' >> beam.Create(samples, reshuffle=False)
          | 'CombineSamples' >> beam.CombineGlobally(
              confidence_intervals_util.SampleCombineFn(
                  num_samples=2, full_sample_id=_FULL_SAMPLE_ID)))

      def check_result(got_pcoll):
        self.assertLen(got_pcoll, 1)
        metrics = got_pcoll[0]

        self.assertIn(metric_key, metrics)
        self.assertAlmostEqual(metrics[metric_key].sample_mean, 3.5, delta=0.1)
        self.assertAlmostEqual(
            metrics[metric_key].sample_standard_deviation, 4.94, delta=0.1)
        self.assertEqual(metrics[metric_key].sample_degrees_of_freedom, 1)
        self.assertEqual(metrics[metric_key].unsampled_value, 4.0)

      util.assert_that(result, check_result)

  def test_sample_combine_fn_missing_samples(self):
    metric_key = metric_types.MetricKey('metric')
    example_count_key = metric_types.MetricKey(example_count.EXAMPLE_COUNT_NAME)
    slice_key1 = (('slice_feature', 1),)
    slice_key2 = (('slice_feature', 2),)
    # the sample value is irrelevant for this test as we only verify counters.
    sample_value = {metric_key: 42}
    samples = [
        # unsampled value for slice 1
        (slice_key1,
         confidence_intervals_util.SampleMetrics(
             sample_id=_FULL_SAMPLE_ID,
             metrics={
                 metric_key: 2.1,
                 example_count_key: 16
             })),
        # 2 sample values for slice 1
        (slice_key1,
         confidence_intervals_util.SampleMetrics(
             sample_id=0, metrics=sample_value)),
        (slice_key1,
         confidence_intervals_util.SampleMetrics(
             sample_id=1, metrics=sample_value)),
        # unsampled value for slice 2
        (slice_key2,
         confidence_intervals_util.SampleMetrics(
             sample_id=_FULL_SAMPLE_ID,
             metrics={
                 metric_key: 6.3,
                 example_count_key: 10000
             })),
        # Only 1 sample value (missing sample ID 1) for slice 2
        (slice_key2,
         confidence_intervals_util.SampleMetrics(
             sample_id=0, metrics=sample_value)),
    ]

    with beam.Pipeline() as pipeline:
      result = (
          pipeline
          | 'Create' >> beam.Create(samples, reshuffle=False)
          | 'CombineSamplesPerKey' >> beam.CombinePerKey(
              confidence_intervals_util.SampleCombineFn(
                  num_samples=2,
                  full_sample_id=_FULL_SAMPLE_ID,
                  skip_ci_metric_keys=[example_count_key])))

      def check_result(got_pcoll):
        self.assertLen(got_pcoll, 2)
        slice2_metrics = None
        for slice_key, metrics in got_pcoll:
          if slice_key == slice_key2:
            slice2_metrics = metrics
            break
        self.assertIsNotNone(slice2_metrics)
        self.assertIn(metric_types.MetricKey('__ERROR__'), slice2_metrics)

      util.assert_that(result, check_result)

      runner_result = pipeline.run()
      # we expect one missing samples counter increment for slice2, since we
      # expected 2 samples, but only saw 1.
      metric_filter = beam.metrics.metric.MetricsFilter().with_name(
          'num_slices_missing_samples')
      counters = runner_result.metrics().query(filter=metric_filter)['counters']
      self.assertLen(counters, 1)
      self.assertEqual(1, counters[0].committed)

      # verify total slice counter
      metric_filter = beam.metrics.metric.MetricsFilter().with_name(
          'num_slices')
      counters = runner_result.metrics().query(filter=metric_filter)['counters']
      self.assertLen(counters, 1)
      self.assertEqual(2, counters[0].committed)

  def test_sample_combine_fn_sample_is_nan(self):
    metric_key = metric_types.MetricKey('metric')
    # the sample value is irrelevant for this test as we only verify counters.
    samples = [
        # unsampled value
        (confidence_intervals_util.SampleMetrics(
            sample_id=_FULL_SAMPLE_ID, metrics={
                metric_key: 2,
            })),
        (confidence_intervals_util.SampleMetrics(
            sample_id=0, metrics={metric_key: 2})),
        (confidence_intervals_util.SampleMetrics(
            sample_id=1, metrics={metric_key: float('nan')})),
    ]

    with beam.Pipeline() as pipeline:
      result = (
          pipeline
          | 'Create' >> beam.Create(samples, reshuffle=False)
          | 'CombineSamplesPerKey' >> beam.CombineGlobally(
              confidence_intervals_util.SampleCombineFn(
                  num_samples=2, full_sample_id=_FULL_SAMPLE_ID)))

      def check_result(got_pcoll):
        self.assertLen(got_pcoll, 1)
        metrics = got_pcoll[0]

        self.assertIn(metric_key, metrics)
        self.assertTrue(np.isnan(metrics[metric_key].sample_mean))
        self.assertTrue(np.isnan(metrics[metric_key].sample_standard_deviation))
        self.assertEqual(metrics[metric_key].sample_degrees_of_freedom, 1)
        self.assertEqual(metrics[metric_key].unsampled_value, 2.0)

      util.assert_that(result, check_result)

  def test_sample_combine_fn_numpy_overflow(self):
    sample_values = np.random.RandomState(seed=0).randint(0, 1e10, 20)
    metric_key = metric_types.MetricKey('metric')
    samples = [
        confidence_intervals_util.SampleMetrics(
            sample_id=_FULL_SAMPLE_ID, metrics={
                metric_key: 1,
            })
    ]
    for sample_id, value in enumerate(sample_values):
      samples.append(
          confidence_intervals_util.SampleMetrics(
              sample_id=sample_id, metrics={
                  metric_key: value,
              }))
    with beam.Pipeline() as pipeline:
      result = (
          pipeline
          | 'Create' >> beam.Create(samples, reshuffle=False)
          | 'CombineSamples' >> beam.CombineGlobally(
              confidence_intervals_util.SampleCombineFn(
                  num_samples=20, full_sample_id=_FULL_SAMPLE_ID)))

      def check_result(got_pcoll):
        expected_pcoll = [
            {
                metric_key:
                    types.ValueWithTDistribution(
                        sample_mean=5293977041.15,
                        sample_standard_deviation=3023624729.537026,
                        sample_degrees_of_freedom=19,
                        unsampled_value=1),
            },
        ]
        self.assertCountEqual(expected_pcoll, got_pcoll)

      util.assert_that(result, check_result)


if __name__ == '__main__':
  absltest.main()
