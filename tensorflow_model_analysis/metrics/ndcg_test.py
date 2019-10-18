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
"""Tests for NDCG metric."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.metrics import ndcg


class NDCGMetricsTest(testutil.TensorflowModelAnalysisTest):

  def testNDCG(self):
    metric = ndcg.NDCG(gain_key='gain').computations(
        sub_keys=[metric_types.SubKey(top_k=1),
                  metric_types.SubKey(top_k=2)],
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
          ndcg1_key = metric_types.MetricKey(
              name='ndcg', sub_key=metric_types.SubKey(top_k=1))
          ndcg2_key = metric_types.MetricKey(
              name='ndcg', sub_key=metric_types.SubKey(top_k=2))
          # Query1 (weight=1): (p=0.8, g=0.5) (p=0.2, g=1.0)
          # Query2 (weight=2): (p=0.9, g=1.0) (p=0.5, g=0.5) (p=0.1, g=0.1)
          # Query3 (weight=3): (p=0.9, g=1.0)
          #
          # DCG@1:  0.5, 1.0, 1.0
          # NDCG@1: 0.5, 1.0, 1.0
          # Average NDCG@1: (1 * 0.5 + 2 * 1.0 + 3 * 1.0) / (1 + 2 + 3) ~ 0.92
          #
          # DCG@2: (0.5 + 1.0/log(3), (1.0 + 0.5/log(3), (1.0)
          # NDCG@2: (0.5 + 1.0/log(3)) / (1.0 + 0.5/log(3)),
          #         (1.0 + 0.5/log(3)) / (1.0 + 0.5/log(3)),
          #         1.0
          # Average NDCG@2: (1 * 0.860 + 2 * 1.0 + 3 * 1.0) / (1 + 2 + 3) ~ 0.97
          self.assertDictElementsAlmostEqual(
              got_metrics, {
                  ndcg1_key: 0.9166667,
                  ndcg2_key: 0.9766198
              },
              places=5)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')


if __name__ == '__main__':
  tf.test.main()
