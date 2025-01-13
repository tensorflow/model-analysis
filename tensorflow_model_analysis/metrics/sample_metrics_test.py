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
"""Tests for sample_metrics."""


import pytest
from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.metrics import sample_metrics


@pytest.mark.xfail(run=False, reason="PR 183 This class contains tests that fail and needs to be fixed. "
"If all tests pass, please remove this mark.")
class SampleTest(absltest.TestCase):

  def testFixedSizeSample(self):
    metric = sample_metrics.FixedSizeSample(
        sampled_key='sampled_key', size=2, random_seed=0).computations()[0]

    examples = []
    for i in range(5):
      examples.append({
          constants.LABELS_KEY: np.array([0]),
          constants.PREDICTIONS_KEY: np.array([1]),
          constants.FEATURES_KEY: {
              'sampled_key': i
          }
      })

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create(examples, reshuffle=False)
          | 'PreProcess' >> beam.ParDo(metric.preprocessors[0])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeMetric' >> beam.CombinePerKey(metric.combiner))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          fixed_sized_sample_key = metric_types.MetricKey(
              name='fixed_size_sample')
          np.testing.assert_equal(got_metrics,
                                  {fixed_sized_sample_key: np.array([4, 0])})

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result)

  def testFixedSizeSampleWeighted(self):
    metric = sample_metrics.FixedSizeSample(
        sampled_key='sampled_key', size=2,
        random_seed=0).computations(example_weighted=True)[0]

    examples = []
    for i in range(5):
      examples.append({
          constants.LABELS_KEY: np.array([0]),
          constants.PREDICTIONS_KEY: np.array([1]),
          constants.EXAMPLE_WEIGHTS_KEY: np.array([10**i]),
          constants.FEATURES_KEY: {
              'sampled_key': i
          }
      })

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create(examples, reshuffle=False)
          # | 'Process' >> beam.ParDo(metric.preprocessors[0])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeMetric' >> beam.CombinePerKey(metric.combiner))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          fixed_sized_sample_key = metric_types.MetricKey(
              name='fixed_size_sample', example_weighted=True)
          np.testing.assert_equal(got_metrics,
                                  {fixed_sized_sample_key: np.array([4, 3])})

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result)


