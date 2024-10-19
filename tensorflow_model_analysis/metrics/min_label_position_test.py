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
"""Tests for mean min label position metric."""


import pytest
import math

from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.metrics import min_label_position
from tensorflow_model_analysis.utils import test_util
from tensorflow_model_analysis.utils import util as tfma_util


@pytest.mark.xfail(run=False, reason="PR 183 This class contains tests that fail and needs to be fixed. "
"If all tests pass, please remove this mark.")
class MinLabelPositionTest(
    test_util.TensorflowModelAnalysisTest, parameterized.TestCase
):

  def testRaisesErrorIfNoQueryKey(self):
    with self.assertRaises(ValueError):
      min_label_position.MinLabelPosition().computations()

  def testRaisesErrorWhenExampleWeightsDiffer(self):
    with self.assertRaises(ValueError):
      metric = min_label_position.MinLabelPosition().computations(
          query_key='query', example_weighted=True)[0]

      query1_example1 = {
          'labels': np.array([0.0]),
          'predictions': np.array([0.2]),
          'example_weights': np.array([1.0]),
          'features': {
              'query': np.array(['query1'])
          }
      }
      query1_example2 = {
          'labels': np.array([1.0]),
          'predictions': np.array([0.8]),
          'example_weights': np.array([0.5]),
          'features': {
              'query': np.array(['query1'])
          }
      }

      with beam.Pipeline() as pipeline:
        # pylint: disable=no-value-for-parameter
        _ = (
            pipeline
            | 'Create' >> beam.Create(
                [tfma_util.merge_extracts([query1_example1, query1_example2])])
            | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs, True)
            | 'AddSlice' >> beam.Map(lambda x: ((), x))
            | 'Combine' >> beam.CombinePerKey(metric.combiner))

  @parameterized.named_parameters(('default_label', None),
                                  ('custom_label', 'custom_label'))
  def testMinLabelPosition(self, label_key):
    metric = min_label_position.MinLabelPosition(
        label_key=label_key).computations(
            query_key='query', example_weighted=True)[0]

    query1_example1 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.2]),
        'example_weights': np.array([1.0]),
        'features': {
            'custom_label': np.array([0.0]),
            'query': np.array(['query1'])
        }
    }
    query1_example2 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.8]),
        'example_weights': np.array([1.0]),
        'features': {
            'custom_label': np.array([1.0]),
            'query': np.array(['query1'])
        }
    }
    query2_example1 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.9]),
        'example_weights': np.array([2.0]),
        'features': {
            'custom_label': np.array([0.0]),
            'query': np.array(['query2'])
        }
    }
    query2_example2 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.1]),
        'example_weights': np.array([2.0]),
        'features': {
            'custom_label': np.array([1.0]),
            'query': np.array(['query2'])
        }
    }
    query2_example3 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.5]),
        'example_weights': np.array([2.0]),
        'features': {
            'custom_label': np.array([0.0]),
            'query': np.array(['query2'])
        }
    }
    query3_example1 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.9]),
        'example_weights': np.array([3.0]),
        'features': {
            'custom_label': np.array([0.0]),
            'query': np.array(['query3'])
        }
    }
    examples = [
        tfma_util.merge_extracts([query1_example1, query1_example2]),
        tfma_util.merge_extracts(
            [query2_example1, query2_example2, query2_example3]),
        tfma_util.merge_extracts([query3_example1])
    ]

    if label_key:
      self.assertIsNotNone(metric.preprocessors)

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create(examples)
          | 'Process' >> beam.Map(
              metric_util.to_standard_metric_inputs, include_features=True)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'Combine' >> beam.CombinePerKey(metric.combiner))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          key = metric_types.MetricKey(
              name='min_label_position', example_weighted=True)
          self.assertIn(key, got_metrics)
          if label_key == 'custom_label':
            # (1*1.0 + 3*2.0) / (1.0 + 2.0) = 2.333333
            self.assertAllClose(got_metrics[key], 2.333333)
          else:
            # (2*1.0 + 1*2.0 + 1*3.0) / (1.0 + 2.0 + 3.0) = 1.166666
            self.assertAllClose(got_metrics[key], 1.166666)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testMinLabelPositionWithNoWeightedExamples(self):
    metric = min_label_position.MinLabelPosition().computations(
        query_key='query', example_weighted=True)[0]

    query1_example1 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.2]),
        'example_weights': np.array([0.0]),
        'features': {
            'query': np.array(['query1'])
        }
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          |
          'Create' >> beam.Create([tfma_util.merge_extracts([query1_example1])])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs, True)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'Combine' >> beam.CombinePerKey(metric.combiner))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          key = metric_types.MetricKey(
              name='min_label_position', example_weighted=True)
          self.assertIn(key, got_metrics)
          self.assertTrue(math.isnan(got_metrics[key]))

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')


