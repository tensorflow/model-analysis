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


import pytest
from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
from numpy import testing
from tensorflow_model_analysis.evaluators import confidence_intervals_util
from tensorflow_model_analysis.metrics import binary_confusion_matrices
from tensorflow_model_analysis.metrics import metric_types

_FULL_SAMPLE_ID = -1


class _ValidateSampleCombineFn(confidence_intervals_util.SampleCombineFn):

  def extract_output(
      self,
      accumulator: confidence_intervals_util.SampleCombineFn.SampleAccumulator,
  ) -> confidence_intervals_util.SampleCombineFn.SampleAccumulator:
    return self._validate_accumulator(accumulator)


@pytest.mark.xfail(run=False, reason="PR 183 This class contains tests that fail and needs to be fixed. "
"If all tests pass, please remove this mark.")
class ConfidenceIntervalsUtilTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': '_ints',
          'values': [0, 1, 2],
          'ddof': 1,
          'expected_mean': 1,
          'expected_std': np.std([0, 1, 2], ddof=1),
      },
      {
          'testcase_name': '_ndarrays',
          'values': [np.array([0]), np.array([1]), np.array([2])],
          'ddof': 1,
          'expected_mean': np.array([1]),
          'expected_std': np.array([np.std([0, 1, 2], ddof=1)]),
      },
      {
          'testcase_name': '_confusion_matrices',
          'values': [
              binary_confusion_matrices.Matrices(
                  thresholds=[0.5], tp=[0], fp=[1], tn=[2], fn=[3]
              ),
              binary_confusion_matrices.Matrices(
                  thresholds=[0.5], tp=[4], fp=[5], tn=[6], fn=[7]
              ),
              binary_confusion_matrices.Matrices(
                  thresholds=[0.5], tp=[8], fp=[9], tn=[10], fn=[11]
              ),
          ],
          'ddof': 1,
          'expected_mean': binary_confusion_matrices.Matrices(
              thresholds=[0.5],
              tp=np.mean([0, 4, 8]),
              fp=np.mean([1, 5, 9]),
              tn=np.mean([2, 6, 10]),
              fn=np.mean([3, 7, 11]),
          ),
          'expected_std': binary_confusion_matrices.Matrices(
              thresholds=[0.5],
              tp=np.std([0, 4, 8], ddof=1),
              fp=np.std([1, 5, 9], ddof=1),
              tn=np.std([2, 6, 10], ddof=1),
              fn=np.std([3, 7, 11], ddof=1),
          ),
      },
  )
  def test_mean_and_std(self, values, ddof, expected_mean, expected_std):
    actual_mean, actual_std = confidence_intervals_util.mean_and_std(
        values, ddof
    )
    self.assertEqual(expected_mean, actual_mean)
    self.assertEqual(expected_std, actual_std)

  def test_sample_combine_fn(self):
    metric_key = metric_types.MetricKey('metric')
    array_metric_key = metric_types.MetricKey('array_metric')
    missing_sample_metric_key = metric_types.MetricKey('missing_metric')
    non_numeric_metric_key = metric_types.MetricKey('non_numeric_metric')
    non_numeric_array_metric_key = metric_types.MetricKey('non_numeric_array')
    mixed_type_array_metric_key = metric_types.MetricKey('mixed_type_array')
    skipped_metric_key = metric_types.MetricKey('skipped_metric')
    slice_key1 = (('slice_feature', 1),)
    slice_key2 = (('slice_feature', 2),)
    # the sample value is irrelevant for this test as we only verify counters.
    samples = [
        # unsampled value for slice 1
        (
            slice_key1,
            confidence_intervals_util.SampleMetrics(
                sample_id=_FULL_SAMPLE_ID,
                metrics={
                    metric_key: 2.1,
                    array_metric_key: np.array([1, 2]),
                    missing_sample_metric_key: 3,
                    non_numeric_metric_key: 'a',
                    non_numeric_array_metric_key: np.array(['a', 'aaa']),
                    mixed_type_array_metric_key: np.array(['a']),
                    skipped_metric_key: 16,
                },
            ),
        ),
        # sample values for slice 1
        (
            slice_key1,
            confidence_intervals_util.SampleMetrics(
                sample_id=0,
                metrics={
                    metric_key: 1,
                    array_metric_key: np.array([2, 3]),
                    missing_sample_metric_key: 2,
                    non_numeric_metric_key: 'b',
                    non_numeric_array_metric_key: np.array(['a', 'aaa']),
                    # one sample is an empty float array
                    mixed_type_array_metric_key: np.array([], dtype=float),
                    skipped_metric_key: 7,
                },
            ),
        ),
        # sample values for slice 1 missing missing_sample_metric_key
        (
            slice_key1,
            confidence_intervals_util.SampleMetrics(
                sample_id=1,
                metrics={
                    metric_key: 2,
                    array_metric_key: np.array([0, 1]),
                    non_numeric_metric_key: 'c',
                    non_numeric_array_metric_key: np.array(['a', 'aaa']),
                    # one sample is a unicode array
                    mixed_type_array_metric_key: np.array(['a']),
                    skipped_metric_key: 8,
                },
            ),
        ),
        # unsampled value for slice 2
        (
            slice_key2,
            confidence_intervals_util.SampleMetrics(
                sample_id=_FULL_SAMPLE_ID,
                metrics={
                    metric_key: 6.3,
                    array_metric_key: np.array([10, 20]),
                    missing_sample_metric_key: 6,
                    non_numeric_metric_key: 'd',
                    non_numeric_array_metric_key: np.array(['a', 'aaa']),
                    mixed_type_array_metric_key: np.array(['a']),
                    skipped_metric_key: 10000,
                },
            ),
        ),
        # Only 1 sample value (missing sample ID 1) for slice 2
        (
            slice_key2,
            confidence_intervals_util.SampleMetrics(
                sample_id=0,
                metrics={
                    metric_key: 3,
                    array_metric_key: np.array([20, 30]),
                    missing_sample_metric_key: 12,
                    non_numeric_metric_key: 'd',
                    non_numeric_array_metric_key: np.array(['a', 'aaa']),
                    mixed_type_array_metric_key: np.array(['a']),
                    skipped_metric_key: 5000,
                },
            ),
        ),
    ]

    with beam.Pipeline() as pipeline:
      result = (
          pipeline
          | 'Create' >> beam.Create(samples, reshuffle=False)
          | 'CombineSamplesPerKey'
          >> beam.CombinePerKey(
              _ValidateSampleCombineFn(
                  num_samples=2,
                  full_sample_id=_FULL_SAMPLE_ID,
                  skip_ci_metric_keys=[skipped_metric_key],
              )
          )
      )

      def check_result(got_pcoll):
        self.assertLen(got_pcoll, 2)
        accumulators_by_slice = dict(got_pcoll)

        self.assertIn(slice_key1, accumulators_by_slice)
        slice1_accumulator = accumulators_by_slice[slice_key1]
        # check unsampled value
        self.assertIn(metric_key, slice1_accumulator.point_estimates)
        self.assertEqual(2.1, slice1_accumulator.point_estimates[metric_key])
        # check numeric case sample_values
        self.assertIn(metric_key, slice1_accumulator.metric_samples)
        self.assertEqual([1, 2], slice1_accumulator.metric_samples[metric_key])
        # check numeric array in sample_values
        self.assertIn(array_metric_key, slice1_accumulator.metric_samples)
        array_metric_samples = slice1_accumulator.metric_samples[
            array_metric_key
        ]
        self.assertLen(array_metric_samples, 2)
        testing.assert_array_equal(np.array([2, 3]), array_metric_samples[0])
        testing.assert_array_equal(np.array([0, 1]), array_metric_samples[1])
        # check that non-numeric metric sample_values are not present
        self.assertIn(
            non_numeric_metric_key, slice1_accumulator.point_estimates
        )
        self.assertNotIn(
            non_numeric_metric_key, slice1_accumulator.metric_samples
        )
        self.assertIn(
            non_numeric_array_metric_key, slice1_accumulator.point_estimates
        )
        self.assertNotIn(
            non_numeric_array_metric_key, slice1_accumulator.metric_samples
        )
        self.assertIn(
            mixed_type_array_metric_key, slice1_accumulator.point_estimates
        )
        self.assertNotIn(
            mixed_type_array_metric_key, slice1_accumulator.metric_samples
        )
        # check that single metric missing samples generates error
        error_key = metric_types.MetricKey('__ERROR__')
        self.assertIn(error_key, slice1_accumulator.point_estimates)
        self.assertRegex(
            slice1_accumulator.point_estimates[error_key],
            'CI not computed for.*missing_metric.*',
        )
        # check that skipped metrics have no samples
        self.assertNotIn(skipped_metric_key, slice1_accumulator.metric_samples)

        self.assertIn(slice_key2, accumulators_by_slice)
        slice2_accumulator = accumulators_by_slice[slice_key2]
        # check unsampled value
        self.assertIn(metric_key, slice2_accumulator.point_estimates)
        self.assertEqual(6.3, slice2_accumulator.point_estimates[metric_key])
        # check that entirely missing sample generates error
        self.assertIn(
            metric_types.MetricKey('__ERROR__'),
            slice2_accumulator.point_estimates,
        )
        self.assertRegex(
            slice2_accumulator.point_estimates[error_key],
            'CI not computed because only 1.*Expected 2.*',
        )

      util.assert_that(result, check_result)

      runner_result = pipeline.run()
      # we expect one missing samples counter increment for slice2, since we
      # expected 2 samples, but only saw 1.
      metric_filter = beam.metrics.metric.MetricsFilter().with_name(
          'num_slices_missing_samples'
      )
      counters = runner_result.metrics().query(filter=metric_filter)['counters']
      self.assertLen(counters, 1)
      self.assertEqual(1, counters[0].committed)

      # verify total slice counter
      metric_filter = beam.metrics.metric.MetricsFilter().with_name(
          'num_slices'
      )
      counters = runner_result.metrics().query(filter=metric_filter)['counters']
      self.assertLen(counters, 1)
      self.assertEqual(2, counters[0].committed)

  def test_sample_combine_fn_no_input(self):
    slice_key = (('slice_feature', 1),)
    samples = [
        (
            slice_key,
            confidence_intervals_util.SampleMetrics(
                sample_id=_FULL_SAMPLE_ID, metrics={}
            ),
        ),
        (
            slice_key,
            confidence_intervals_util.SampleMetrics(sample_id=0, metrics={}),
        ),
        (
            slice_key,
            confidence_intervals_util.SampleMetrics(sample_id=1, metrics={}),
        ),
    ]

    with beam.Pipeline() as pipeline:
      result = (
          pipeline
          | 'Create' >> beam.Create(samples)
          | 'CombineSamplesPerKey'
          >> beam.CombinePerKey(
              _ValidateSampleCombineFn(
                  num_samples=2, full_sample_id=_FULL_SAMPLE_ID
              )
          )
      )

      def check_result(got_pcoll):
        self.assertLen(got_pcoll, 1)
        accumulators_by_slice = dict(got_pcoll)
        self.assertIn(slice_key, accumulators_by_slice)
        accumulator = accumulators_by_slice[slice_key]
        self.assertEqual(2, accumulator.num_samples)
        self.assertIsInstance(accumulator.point_estimates, dict)
        self.assertIsInstance(accumulator.metric_samples, dict)

      util.assert_that(result, check_result)


