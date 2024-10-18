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

import json

from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis.metrics import exact_match
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.utils import test_util


class ExactMatchTest(
    test_util.TensorflowModelAnalysisTest, parameterized.TestCase
):

  @parameterized.named_parameters(('text', False), ('json', True))
  def testExactMatchWithoutWeights(self, test_json):
    convert_to = 'json' if test_json else None
    computations = exact_match.ExactMatch(convert_to=convert_to).computations()
    metric = computations[0]

    def _maybe_convert_feature(f):
      return json.dumps(f) if test_json else f

    example1 = {
        'labels': np.array([_maybe_convert_feature('Test 1 two 3')]),
        'predictions': np.array([_maybe_convert_feature('Test 1 two 3')]),
        'example_weights': np.array([1.0]),
    }
    example2 = {
        'labels': np.array([_maybe_convert_feature('Testing')]),
        'predictions': np.array([_maybe_convert_feature('Dog')]),
        'example_weights': np.array([1.0]),
    }
    example3 = {
        'labels': np.array([_maybe_convert_feature('Test 1 two 3') + ' ']),
        'predictions': np.array([_maybe_convert_feature('Test 1 two 3')]),
        'example_weights': np.array([1.0]),
    }
    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example1, example2, example3])
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
          # example3 is a perfect match for json but a miss for text.
          # average score: 0.6666.. for json, 0.3333... for text.
          score = 2.0 / 3 if test_json else 1.0 / 3
          self.assertDictElementsAlmostEqual(
              got_metrics, {key: score}, places=5)
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testExactMatchScoreWithWeights(self):
    computations = exact_match.ExactMatch().computations(example_weighted=True)
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


