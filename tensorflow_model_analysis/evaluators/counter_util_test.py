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

# Standard Imports
import apache_beam as beam
import tensorflow as tf

from tensorflow_model_analysis import constants
from tensorflow_model_analysis.evaluators import counter_util
from tensorflow_model_analysis.post_export_metrics import post_export_metrics


class CounterUtilTest(tf.test.TestCase):

  def testMetricComputedBeamCounter(self):
    with beam.Pipeline() as pipeline:
      auc = post_export_metrics.auc()
      _ = pipeline | counter_util.IncrementMetricsComputationCounters([auc])

    result = pipeline.run()
    metric_filter = beam.metrics.metric.MetricsFilter().with_namespace(
        constants.METRICS_NAMESPACE).with_name('metric_computed_auc')
    actual_metrics_count = result.metrics().query(
        filter=metric_filter)['counters'][0].committed

    self.assertEqual(actual_metrics_count, 1)


if __name__ == '__main__':
  tf.test.main()
