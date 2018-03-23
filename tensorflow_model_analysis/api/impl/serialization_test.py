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
"""Test for using the serialization library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import apache_beam as beam
import numpy as np

import tensorflow as tf
from tensorflow_model_analysis.api.impl import api_types
from tensorflow_model_analysis.api.impl import serialization
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.slicer import slicer


class SerializationTest(testutil.TensorflowModelAnalysisTest):

  def setUp(self):
    self.longMessage = True

  def assertMetricsAlmostEqual(self, expected_metrics, got_metrics):
    self.assertItemsEqual(
        expected_metrics.keys(),
        got_metrics.keys(),
        msg='keys do not match. expected_metrics: %s, got_metrics: %s' %
        (expected_metrics, got_metrics))
    for key in expected_metrics.keys():
      self.assertAlmostEqual(
          expected_metrics[key],
          got_metrics[key],
          msg='value for key %s does not match' % key)

  def assertSliceMetricsListEqual(self, expected_list, got_list):
    self.assertEqual(
        len(expected_list),
        len(got_list),
        msg='expected_list: %s, got_list: %s' % (expected_list, got_list))
    for index, (expected, got) in enumerate(zip(expected_list, got_list)):
      (expected_key, expected_metrics) = expected
      (got_key, got_metrics) = got
      self.assertEqual(
          expected_key, got_key, msg='key mismatch at index %d' % index)
      self.assertMetricsAlmostEqual(expected_metrics, got_metrics)

  def testSerializeDeserializeMetrics(self):
    slice_key1 = (('fruit', 'apple'),)
    metrics1 = {
        'alpha': np.array([1.0]),
        'bravo': np.array([1.0, 2.0, 3.0]),
        'charlie': np.float32(4.0)
    }
    expected_metrics1 = {
        'alpha': [1.0],
        'bravo': [1.0, 2.0, 3.0],
        'charlie': 4.0
    }
    slice_key2 = (('fruit', 'pear'),)
    metrics2 = {
        'alpha': np.array([10.0]),
        'bravo': np.array([10.0, 20.0, 30.0]),
        'charlie': np.float32(40.0)
    }
    expected_metrics2 = {
        'alpha': [10.0],
        'bravo': [10.0, 20.0, 30.0],
        'charlie': 40.0
    }
    slice_key3 = (('fruit', 'pear'), ('animal', 'duck'))
    metrics3 = {
        'alpha': np.array([0.1]),
        'bravo': np.array([0.2]),
        'charlie': np.float32(0.3)
    }
    expected_metrics3 = {
        'alpha': [0.1],
        'bravo': [0.2],
        'charlie': 0.3,
    }
    slice_metrics_list = [(slice_key1, metrics1), (slice_key2, metrics2),
                          (slice_key3, metrics3)]
    expected_slice_metrics_list = [(slice_key1, expected_metrics1),
                                   (slice_key2, expected_metrics2),
                                   (slice_key3, expected_metrics3)]

    serialized = serialization._serialize_metrics(
        slice_metrics_list, serialization._METRICS_METRICS_TYPE)
    deserialized = serialization._deserialize_metrics_raw(serialized)
    got_slice_metrics_list = deserialized[serialization._SLICE_METRICS_LIST_KEY]
    self.assertSliceMetricsListEqual(expected_slice_metrics_list,
                                     got_slice_metrics_list)

  def testSerializeDeserializeEvalConfig(self):
    eval_config = api_types.EvalConfig(
        model_location='/path/to/model',
        data_location='/path/to/data',
        slice_spec=[
            slicer.SingleSliceSpec(
                features=[('age', 5), ('gender', 'f')], columns=['country']),
            slicer.SingleSliceSpec(
                features=[('age', 6), ('gender', 'm')], columns=['interest'])
        ],
        example_weight_metric_key='key')
    serialized = serialization._serialize_eval_config(eval_config)
    deserialized = serialization._deserialize_eval_config_raw(serialized)
    got_eval_config = deserialized[serialization._EVAL_CONFIG_KEY]
    self.assertEqual(eval_config, got_eval_config)

  def testSerializeDeserializeToFile(self):
    metrics_slice_key1 = (('fruit', 'pear'), ('animal', 'duck'))
    metrics1 = {
        'alpha': np.array([0.1]),
        'bravo': np.array([0.2]),
        'charlie': np.float32(0.3)
    }
    expected_metrics1 = {
        'alpha': [0.1],
        'bravo': [0.2],
        'charlie': 0.3,
    }
    plots_slice_key1 = (('fruit', 'peach'), ('animal', 'cow'))
    plots1 = {
        'alpha': np.array([0.5, 0.6, 0.7]),
        'bravo': np.array([0.6, 0.7, 0.8]),
        'charlie': np.float32(0.7)
    }
    expected_plots1 = {
        'alpha': [0.5, 0.6, 0.7],
        'bravo': [0.6, 0.7, 0.8],
        'charlie': 0.7,
    }
    eval_config = api_types.EvalConfig(
        model_location='/path/to/model',
        data_location='/path/to/data',
        slice_spec=[
            slicer.SingleSliceSpec(
                features=[('age', 5), ('gender', 'f')], columns=['country']),
            slicer.SingleSliceSpec(
                features=[('age', 6), ('gender', 'm')], columns=['interest'])
        ],
        example_weight_metric_key='key')

    output_path = self._getTempDir()
    with beam.Pipeline() as pipeline:
      metrics = (
          pipeline
          | 'CreateMetrics' >> beam.Create([(metrics_slice_key1, metrics1)]))
      plots = (
          pipeline | 'CreatePlots' >> beam.Create([(plots_slice_key1, plots1)]))

      _ = ((metrics, plots)
           | 'WriteMetricsPlotsAndConfig' >>
           serialization.WriteMetricsPlotsAndConfig(
               output_path=output_path, eval_config=eval_config))

    metrics, plots = serialization.load_plots_and_metrics(output_path)
    self.assertSliceMetricsListEqual([(metrics_slice_key1, expected_metrics1)],
                                     metrics)
    self.assertSliceMetricsListEqual([(plots_slice_key1, expected_plots1)],
                                     plots)

    got_eval_config = serialization.load_eval_config(output_path)
    self.assertEqual(eval_config, got_eval_config)


if __name__ == '__main__':
  tf.test.main()
