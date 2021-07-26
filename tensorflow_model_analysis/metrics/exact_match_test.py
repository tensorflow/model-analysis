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
"""Tests for exact match metric."""
from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.metrics import exact_match
from tensorflow_model_analysis.metrics import metric_util


class ExactMatchTest(testutil.TensorflowModelAnalysisTest):

  def testExactMatchWithoutWeights(self):
    computations = exact_match.ExactMatch().computations()
    metric = computations[0]
    example1 = {
        'labels': np.array(['Test 1 two 3']),
        'predictions': np.array(['Test 1 two 3']),
        'example_weights': np.array([1.0]),
    }
    example2 = {
        'labels': np.array(['Testing']),
        'predictions': np.array(['Dog']),
        'example_weights': np.array([1.0]),
    }
    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example1, example2])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeMetric' >> beam.CombinePerKey(metric.combiner))

      # pylint: enable=no-value-for-parameter
      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          key = metric.keys[0]
          # example1 is a perfect match (score 100)
          # example2 is a complete miss (score 0)
          # average score: 0.50
          self.assertDictElementsAlmostEqual(got_metrics, {key: 0.50}, places=5)
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testExactMatchScoreWithWeights(self):
    computations = exact_match.ExactMatch().computations()
    metric = computations[0]
    example1 = {
        'labels': np.array(['Test 1 two 3']),
        'predictions': np.array(['Test 1 two 3']),
        'example_weights': np.array([3.0]),
    }
    example2 = {
        'labels': np.array(['Testing']),
        'predictions': np.array(['Dog']),
        'example_weights': np.array([1.0]),
    }
    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example1, example2])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeMetric' >> beam.CombinePerKey(metric.combiner))

      # pylint: enable=no-value-for-parameter
      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          key = metric.keys[0]
          # example1 is a perfect match (score 100)
          # example2 is a complete miss (score 0)
          # average score: (1*3 + 0*1) / 4 = 0.75
          self.assertDictElementsAlmostEqual(got_metrics, {key: 0.75}, places=5)
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')


if __name__ == '__main__':
  tf.test.main()
