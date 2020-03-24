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
"""Tests for mean min label position metric."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import math

import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.metrics import min_label_position


class MinLabelPositionTest(testutil.TensorflowModelAnalysisTest):

  def testRaisesErrorIfNoQueryKey(self):
    with self.assertRaises(ValueError):
      min_label_position.MinLabelPosition().computations()

  def testRaisesErrorWhenExampleWeightsDiffer(self):
    with self.assertRaises(ValueError):
      metric = min_label_position.MinLabelPosition().computations(
          query_key='query')[0]

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

      def to_standard_metric_inputs_list(list_of_extracts):
        return [
            metric_util.to_standard_metric_inputs(e, True)
            for e in list_of_extracts
        ]

      with beam.Pipeline() as pipeline:
        # pylint: disable=no-value-for-parameter
        _ = (
            pipeline
            | 'Create' >> beam.Create([[query1_example1, query1_example2]])
            | 'Process' >> beam.Map(to_standard_metric_inputs_list)
            | 'AddSlice' >> beam.Map(lambda x: ((), x))
            | 'Combine' >> beam.CombinePerKey(metric.combiner))

  def testMinLabelPosition(self):
    metric = min_label_position.MinLabelPosition().computations(
        query_key='query')[0]

    query1_example1 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.2]),
        'example_weights': np.array([1.0]),
        'features': {
            'query': np.array(['query1'])
        }
    }
    query1_example2 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.8]),
        'example_weights': np.array([1.0]),
        'features': {
            'query': np.array(['query1'])
        }
    }
    query2_example1 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.5]),
        'example_weights': np.array([2.0]),
        'features': {
            'query': np.array(['query2'])
        }
    }
    query2_example2 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.9]),
        'example_weights': np.array([2.0]),
        'features': {
            'query': np.array(['query2'])
        }
    }
    query2_example3 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.1]),
        'example_weights': np.array([2.0]),
        'features': {
            'query': np.array(['query2'])
        }
    }
    query3_example1 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.9]),
        'example_weights': np.array([3.0]),
        'features': {
            'query': np.array(['query3'])
        }
    }
    examples = [[query1_example1, query1_example2],
                [query2_example1, query2_example2, query2_example3],
                [query3_example1]]

    def to_standard_metric_inputs_list(list_of_extracts):
      return [
          metric_util.to_standard_metric_inputs(e, True)
          for e in list_of_extracts
      ]

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create(examples)
          | 'Process' >> beam.Map(to_standard_metric_inputs_list)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'Combine' >> beam.CombinePerKey(metric.combiner))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          key = metric_types.MetricKey(name='min_label_position')
          self.assertDictElementsAlmostEqual(
              got_metrics, {
                  key: 0.66667,
              }, places=5)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testMinLabelPositionWithNoWeightedExamples(self):
    metric = min_label_position.MinLabelPosition().computations(
        query_key='query')[0]

    query1_example1 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.2]),
        'example_weights': np.array([0.0]),
        'features': {
            'query': np.array(['query1'])
        }
    }

    def to_standard_metric_inputs_list(list_of_extracts):
      return [
          metric_util.to_standard_metric_inputs(e, True)
          for e in list_of_extracts
      ]

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([[query1_example1]])
          | 'Process' >> beam.Map(to_standard_metric_inputs_list)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'Combine' >> beam.CombinePerKey(metric.combiner))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          key = metric_types.MetricKey(name='min_label_position')
          self.assertIn(key, got_metrics)
          self.assertTrue(math.isnan(got_metrics[key]))

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')


if __name__ == '__main__':
  tf.test.main()
