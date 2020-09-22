# Lint as: python3
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
"""Tests for counter utility to count all the metrics computed."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import apache_beam as beam
import tensorflow as tf

from tensorflow_model_analysis import config
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.evaluators import counter_util
from tensorflow_model_analysis.post_export_metrics import post_export_metrics


class CounterUtilTest(tf.test.TestCase):

  def testMetricComputedBeamCounter(self):
    with beam.Pipeline() as pipeline:
      auc = post_export_metrics.auc()
      _ = pipeline | counter_util.IncrementMetricsCallbacksCounters([auc])

    result = pipeline.run()
    metric_filter = beam.metrics.metric.MetricsFilter().with_namespace(
        constants.METRICS_NAMESPACE).with_name('metric_computed_auc_v1')
    actual_metrics_count = result.metrics().query(
        filter=metric_filter)['counters'][0].committed

    self.assertEqual(actual_metrics_count, 1)

  def testSliceSpecBeamCounter(self):
    with beam.Pipeline() as pipeline:
      _ = (
          pipeline
          | beam.Create([[[('slice_key', 'first_slice')]]])
          | counter_util.IncrementSliceSpecCounters())

    result = pipeline.run()

    slice_spec_filter = beam.metrics.metric.MetricsFilter().with_namespace(
        constants.METRICS_NAMESPACE).with_name(
            'slice_computed_slice_key_first_slice')
    slice_count = result.metrics().query(
        filter=slice_spec_filter)['counters'][0].committed
    self.assertEqual(slice_count, 1)

  def testMetricsSpecBeamCounter(self):
    with beam.Pipeline() as pipeline:
      metrics_spec = config.MetricsSpec(
          metrics=[config.MetricConfig(class_name='FairnessIndicators')])
      _ = pipeline | counter_util.IncrementMetricsSpecsCounters([metrics_spec])

    result = pipeline.run()
    metric_filter = beam.metrics.metric.MetricsFilter().with_namespace(
        constants.METRICS_NAMESPACE).with_name(
            'metric_computed_FairnessIndicators_v2')
    actual_metrics_count = result.metrics().query(
        filter=metric_filter)['counters'][0].committed

    self.assertEqual(actual_metrics_count, 1)


if __name__ == '__main__':
  tf.test.main()
