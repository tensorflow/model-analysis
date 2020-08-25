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
"""Tests for query statistics metrics."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import util as tfma_util
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.metrics import query_statistics


class QueryStatisticsTest(testutil.TensorflowModelAnalysisTest):

  def testQueryStatistics(self):
    metrics = query_statistics.QueryStatistics().computations(
        query_key='query')[0]

    query1_example1 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.2]),
        'example_weights': np.array([1.0]),
        'features': {
            'query': np.array(['query1']),
            'gain': np.array([1.0])
        }
    }
    query1_example2 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.8]),
        'example_weights': np.array([1.0]),
        'features': {
            'query': np.array(['query1']),
            'gain': np.array([0.5])
        }
    }
    query2_example1 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.5]),
        'example_weights': np.array([2.0]),
        'features': {
            'query': np.array(['query2']),
            'gain': np.array([0.5])
        }
    }
    query2_example2 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.9]),
        'example_weights': np.array([2.0]),
        'features': {
            'query': np.array(['query2']),
            'gain': np.array([1.0])
        }
    }
    query2_example3 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.1]),
        'example_weights': np.array([2.0]),
        'features': {
            'query': np.array(['query2']),
            'gain': np.array([0.1])
        }
    }
    query3_example1 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.9]),
        'example_weights': np.array([3.0]),
        'features': {
            'query': np.array(['query3']),
            'gain': np.array([1.0])
        }
    }
    examples = [
        tfma_util.merge_extracts([query1_example1, query1_example2]),
        tfma_util.merge_extracts(
            [query2_example1, query2_example2, query2_example3]),
        tfma_util.merge_extracts([query3_example1])
    ]

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create(examples)
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs, True)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'Combine' >> beam.CombinePerKey(metrics.combiner))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          total_queries_key = metric_types.MetricKey(name='total_queries')
          total_documents_key = metric_types.MetricKey(name='total_documents')
          min_documents_key = metric_types.MetricKey(name='min_documents')
          max_documents_key = metric_types.MetricKey(name='max_documents')
          self.assertDictElementsAlmostEqual(
              got_metrics, {
                  total_queries_key: 3,
                  total_documents_key: 6,
                  min_documents_key: 1,
                  max_documents_key: 3
              },
              places=5)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')


if __name__ == '__main__':
  tf.test.main()
