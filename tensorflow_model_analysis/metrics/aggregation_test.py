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
"""Tests for aggregation metrics."""


import pytest
import copy
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis.metrics import aggregation
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.utils import test_util


@pytest.mark.xfail(run=False, reason="PR 183 This class contains tests that fail and needs to be fixed. "
"If all tests pass, please remove this mark.")
class AggregationMetricsTest(test_util.TensorflowModelAnalysisTest):

  def testOutputAverage(self):
    metric_name = 'test'
    computations = aggregation.output_average(
        metric_name, output_weights={
            'output_1': 0.3,
            'output_2': 0.7
        })
    metric = computations[0]

    sub_metrics = {}
    output_names = ('output_1', 'output_2', 'output_3')
    output_values = (0.1, 0.2, 0.3)
    for output_name, output_value in zip(output_names, output_values):
      key = metric_types.MetricKey(name=metric_name, output_name=output_name)
      sub_metrics[key] = output_value

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([((), sub_metrics)])
          | 'ComputeMetric' >> beam.Map(lambda x: (x[0], metric.result(x[1]))))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          key = metric.keys[0]
          expected_value = (0.3 * 0.1 + 0.7 * 0.2) / (0.3 + 0.7)
          self.assertDictElementsAlmostEqual(
              got_metrics, {key: expected_value}, places=5)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testMacroAverage(self):
    metric_name = 'test'
    class_ids = [0, 1, 2]
    sub_keys = [metric_types.SubKey(class_id=i) for i in class_ids]
    sub_key_values = [0.1, 0.2, 0.3]
    computations = aggregation.macro_average(
        metric_name,
        sub_keys,
        eval_config=config_pb2.EvalConfig(),
        class_weights={
            0: 1.0,
            1: 1.0,
            2: 1.0
        })
    metric = computations[0]

    sub_metrics = {}
    for sub_key, value in zip(sub_keys, sub_key_values):
      key = metric_types.MetricKey(name=metric_name, sub_key=sub_key)
      sub_metrics[key] = value

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([((), sub_metrics)])
          | 'ComputeMetric' >> beam.Map(lambda x: (x[0], metric.result(x[1]))))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          key = metric.keys[0]
          expected_value = (0.1 + 0.2 + 0.3) / 3.0
          self.assertDictElementsAlmostEqual(
              got_metrics, {key: expected_value}, places=5)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testMacroAverageWithWeights(self):
    metric_name = 'test'
    class_ids = [0, 1, 2]
    class_weights = {0: 0.2, 1: 0.3, 2: 0.5}
    sub_keys = [metric_types.SubKey(class_id=i) for i in class_ids]
    sub_key_values = [0.1, 0.2, 0.3]
    computations = aggregation.macro_average(
        metric_name,
        sub_keys,
        eval_config=config_pb2.EvalConfig(),
        class_weights=class_weights)
    metric = computations[0]

    sub_metrics = {}
    for sub_key, value in zip(sub_keys, sub_key_values):
      key = metric_types.MetricKey(name=metric_name, sub_key=sub_key)
      sub_metrics[key] = value

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([((), sub_metrics)])
          | 'ComputeMetric' >> beam.Map(lambda x: (x[0], metric.result(x[1]))))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          key = metric.keys[0]
          expected_value = (0.1 * 0.2 + 0.2 * 0.3 + 0.3 * 0.5) / (0.2 + 0.3 +
                                                                  0.5)
          self.assertDictElementsAlmostEqual(
              got_metrics, {key: expected_value}, places=5)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testWeightedMacroAverage(self):
    example1 = {
        'labels': np.array([0.0, 1.0, 1.0]),
        'predictions': np.array([0.0, 0.3, 0.7]),
        'example_weights': np.array([1.0]),
    }
    example2 = {
        'labels': np.array([0.0, 1.0, 1.0]),
        'predictions': np.array([0.5, 0.3, 0.8]),
        'example_weights': np.array([1.0]),
    }
    example3 = {
        'labels': np.array([1.0, 0.0, 1.0]),
        'predictions': np.array([0.3, 0.7, 0.9]),
        'example_weights': np.array([1.0]),
    }

    metric_name = 'test'
    class_ids = [0, 1, 2]
    sub_keys = [metric_types.SubKey(class_id=i) for i in class_ids]
    sub_key_values = [0.1, 0.2, 0.3]
    computations = aggregation.weighted_macro_average(
        metric_name, sub_keys, eval_config=config_pb2.EvalConfig())
    class_weights = computations[0]
    metric = computations[1]

    def create_sub_metrics(sliced_metrics):
      slice_value, metrics = sliced_metrics
      metrics = copy.copy(metrics)
      for sub_key, value in zip(sub_keys, sub_key_values):
        key = metric_types.MetricKey(name=metric_name, sub_key=sub_key)
        metrics[key] = value
      return (slice_value, metrics)

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example1, example2, example3])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeClassWeights' >> beam.CombinePerKey(class_weights.combiner)
          | 'CreateSubMetric' >> beam.Map(create_sub_metrics)
          | 'ComputeMetric' >> beam.Map(lambda x: (x[0], metric.result(x[1]))))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          key = metric.keys[0]
          # Labels: 1 x class_0, 2 x class_1, 3 x class_2
          # Class weights:  0.125, .3333, .5
          expected_value = (0.1 * 1.0 / 6.0) + (0.2 * 2.0 / 6.0) + (0.3 * 3.0 /
                                                                    6.0)
          self.assertDictElementsAlmostEqual(
              got_metrics, {key: expected_value}, places=5)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')


