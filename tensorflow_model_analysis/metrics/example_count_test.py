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
"""Tests for example count metric."""


import pytest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
import tensorflow_model_analysis as tfma
from tensorflow_model_analysis.metrics import example_count
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.utils import test_util

from google.protobuf import text_format


@pytest.mark.xfail(run=False, reason="PR 183 This class contains tests that fail and needs to be fixed. "
"If all tests pass, please remove this mark.")
class ExampleCountTest(
    test_util.TensorflowModelAnalysisTest, parameterized.TestCase
):

  @parameterized.named_parameters(
      ('unweighted', '', '', False), ('basic', '', '', True),
      ('multi-model', 'model', '', True), ('multi-output', '', 'output', True),
      ('multi-model-multi-output', 'model', 'output', True))
  def testExampleCount(self, model_name, output_name, example_weighted):
    metric = example_count.ExampleCount().computations(
        model_names=[model_name],
        output_names=[output_name],
        example_weighted=example_weighted)[0]

    example0 = {'labels': None, 'predictions': None, 'example_weights': [0.0]}
    example1 = {'labels': None, 'predictions': None, 'example_weights': [0.5]}
    example2 = {'labels': None, 'predictions': None, 'example_weights': [1.0]}
    example3 = {'labels': None, 'predictions': None, 'example_weights': [0.7]}

    if output_name:
      example0['example_weights'] = {output_name: example0['example_weights']}
      example1['example_weights'] = {output_name: example1['example_weights']}
      example2['example_weights'] = {output_name: example2['example_weights']}
      example3['example_weights'] = {output_name: example3['example_weights']}

    if model_name:
      example0['example_weights'] = {model_name: example0['example_weights']}
      example1['example_weights'] = {model_name: example1['example_weights']}
      example2['example_weights'] = {model_name: example2['example_weights']}
      example3['example_weights'] = {model_name: example3['example_weights']}

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example0, example1, example2, example3])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeMetric' >> beam.CombinePerKey(metric.combiner))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          example_count_key = metric_types.MetricKey(
              name='example_count',
              model_name=model_name,
              output_name=output_name,
              example_weighted=example_weighted)
          if example_weighted:
            self.assertDictElementsAlmostEqual(
                got_metrics, {example_count_key: (0.0 + 0.5 + 1.0 + 0.7)})
          else:
            self.assertDictElementsAlmostEqual(
                got_metrics, {example_count_key: (1.0 + 1.0 + 1.0 + 1.0)})

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')


@pytest.mark.xfail(run=False, reason="PR 183 This class contains tests that fail and needs to be fixed. "
"If all tests pass, please remove this mark.")
class ExampleCountEnd2EndTest(parameterized.TestCase):

  def testExampleCountsWithoutLabelPredictions(self):
    eval_config = text_format.Parse(
        """
        model_specs {
          signature_name: "serving_default"
          example_weight_key: "example_weights"
        }
        slicing_specs {
        }
        metrics_specs {
          metrics {
            class_name: "ExampleCount"
          }
        }
        """,
        tfma.EvalConfig(),
    )
    name_list = ['example_count']
    expected_results = [0.6]
    extracts = [
        {
            'features': {
                'example_weights': np.array([0.5]),
            }
        },
        {'features': {}, 'example_weights': np.array([0.1])},
    ]

    evaluators = tfma.default_evaluators(eval_config=eval_config)
    extractors = tfma.default_extractors(
        eval_shared_model=None, eval_config=eval_config
    )

    with beam.Pipeline() as p:
      result = (
          p
          | 'LoadData' >> beam.Create(extracts)
          | 'ExtractEval'
          >> tfma.ExtractAndEvaluate(
              extractors=extractors, evaluators=evaluators
          )
      )

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          self.assertLen(got_metrics, len(name_list))
          for name, expected_result in zip(name_list, expected_results):
            key = metric_types.MetricKey(name=name, example_weighted=True)
            self.assertIn(key, got_metrics)
            got_metric = got_metrics[key]
            np.testing.assert_allclose(
                expected_result,
                got_metric,
                rtol=1e-3,
                err_msg=f'This {name} metric fails.',
            )
        except AssertionError as err:
          raise util.BeamAssertException(err)

      self.assertIn('metrics', result)
      util.assert_that(result['metrics'], check_result, label='result')


