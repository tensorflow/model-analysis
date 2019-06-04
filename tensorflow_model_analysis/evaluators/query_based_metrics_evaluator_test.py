# Copyright 2018 Google LLC
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
"""Test for using the QueryBasedEvaluator API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Standard Imports

import apache_beam as beam
from apache_beam.testing import util
import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.api import tfma_unit
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.eval_saved_model.example_trainers import fixed_prediction_estimator_extra_fields
from tensorflow_model_analysis.evaluators import query_based_metrics_evaluator
from tensorflow_model_analysis.evaluators.query_metrics import min_label_position
from tensorflow_model_analysis.evaluators.query_metrics import ndcg
from tensorflow_model_analysis.evaluators.query_metrics import query_statistics
from tensorflow_model_analysis.extractors import predict_extractor
from tensorflow_model_analysis.extractors import slice_key_extractor


class QueryBasedMetricsEvaluatorTest(testutil.TensorflowModelAnalysisTest):

  def setUp(self):
    self.longMessage = True  # pylint: disable=invalid-name

  def _getEvalExportDir(self):
    return os.path.join(self._getTempDir(), 'eval_export_dir')

  def _get_examples(self):
    # fixed_string used as query_id
    # fixed_float used as gain_key for NDCG
    # fixed_int used as weight_key for NDCG
    #
    # Query1 (weight=1): (p=0.8, g=0.5) (p=0.2, g=1.0)
    # Query2 (weight=2): (p=0.9, g=1.0) (p=0.5, g=0.5) (p=0.1, g=0.1)
    # Query3 (weight=3): (p=0.9, g=1.0)
    # DCG@1:  0.5, 1.0, 1.0
    # NDCG@1: 0.5, 1.0, 1.0
    # Average NDCG@1: (1 * 0.5 + 2 * 1.0 + 3 * 1.0) / (1 + 2 + 3)
    #               = 5.5 / 6.0
    #               ~ 0.917
    # Using log2(3.0) ~ 1.585 and hence 1 / log2(3.0) ~ 0.631
    # DCG@2: (0.5 + 0.631), (1.0 + 0.315), (1.0)
    # NDCG@2: (0.5 + 0.631) / (1.0 + 0.315), (1.0 + 0.315) / (1.0 + 0.315), 1.0
    # Average NDCG@2: (1 * 0.860 + 2 * 1.0 + 3 * 1.0) / (1 + 2 + 3)
    #               ~ 0.977
    query1_example1 = self._makeExample(
        prediction=0.2,
        label=1.0,
        fixed_float=1.0,
        fixed_string='query1',
        fixed_int=1)
    query1_example2 = self._makeExample(
        prediction=0.8,
        label=0.0,
        fixed_float=0.5,
        fixed_string='query1',
        fixed_int=1)

    query2_example1 = self._makeExample(
        prediction=0.5,
        label=0.0,
        fixed_float=0.5,
        fixed_string='query2',
        fixed_int=2)
    query2_example2 = self._makeExample(
        prediction=0.9,
        label=1.0,
        fixed_float=1.0,
        fixed_string='query2',
        fixed_int=2)
    query2_example3 = self._makeExample(
        prediction=0.1,
        label=0.0,
        fixed_float=0.1,
        fixed_string='query2',
        fixed_int=2)

    query3_example1 = self._makeExample(
        prediction=0.9,
        label=1.0,
        fixed_float=1.0,
        fixed_string='query3',
        fixed_int=3)

    serialized_examples = [
        query1_example1.SerializeToString(),
        query1_example2.SerializeToString(),
        query2_example1.SerializeToString(),
        query2_example2.SerializeToString(),
        query2_example3.SerializeToString(),
        query3_example1.SerializeToString(),
    ]

    return serialized_examples

  def testEvaluateQueryBasedMetrics(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        fixed_prediction_estimator_extra_fields
        .simple_fixed_prediction_estimator_extra_fields(None,
                                                        temp_eval_export_dir))
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=eval_export_dir)
    extractors = [
        predict_extractor.PredictExtractor(eval_shared_model),
        slice_key_extractor.SliceKeyExtractor()
    ]

    with beam.Pipeline() as pipeline:
      metrics = (
          pipeline
          | 'Create' >> beam.Create(self._get_examples())
          | 'InputsToExtracts' >> model_eval_lib.InputsToExtracts()
          | 'Extract' >> tfma_unit.Extract(extractors=extractors)  # pylint: disable=no-value-for-parameter
          | 'EvaluateQueryBasedMetrics' >>
          query_based_metrics_evaluator.EvaluateQueryBasedMetrics(
              prediction_key='',
              query_id='fixed_string',
              combine_fns=[
                  query_statistics.QueryStatisticsCombineFn(),
                  ndcg.NdcgMetricCombineFn(
                      at_vals=[1, 2],
                      gain_key='fixed_float',
                      weight_key='fixed_int'),
                  min_label_position.MinLabelPositionCombineFn(
                      label_key='', weight_key='fixed_int'),
              ]))

      def check_metrics(got):
        try:
          self.assertEqual(1, len(got), 'got: %s' % got)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          self.assertDictElementsAlmostEqual(
              got_metrics, {
                  'post_export_metrics/total_queries':
                      3.0,
                  'post_export_metrics/total_documents':
                      6.0,
                  'post_export_metrics/min_documents':
                      1.0,
                  'post_export_metrics/max_documents':
                      3.0,
                  'post_export_metrics/ndcg@1':
                      0.9166667,
                  'post_export_metrics/ndcg@2':
                      0.9766198,
                  'post_export_metrics/average_min_label_position/__labels':
                      0.6666667,
              })

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(
          metrics[constants.METRICS_KEY], check_metrics, label='metrics')


if __name__ == '__main__':
  tf.test.main()
