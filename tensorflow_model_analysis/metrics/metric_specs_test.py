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
"""Tests for metric specs."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import json
import tensorflow as tf
from tensorflow_model_analysis import config
from tensorflow_model_analysis.metrics import calibration
from tensorflow_model_analysis.metrics import metric_specs
from tensorflow_model_analysis.metrics import metric_types


class MetricSpecsTest(tf.test.TestCase):

  def testSpecsFromMetrics(self):
    metrics_specs = metric_specs.specs_from_metrics(
        {
            'output_name1': [
                tf.keras.metrics.MeanSquaredError('mse'),
                calibration.MeanLabel('mean_label')
            ],
            'output_name2': [
                tf.keras.metrics.RootMeanSquaredError('rmse'),
                calibration.MeanPrediction('mean_prediction')
            ]
        },
        model_names=['model_name1', 'model_name2'],
        binarize=config.BinarizationOptions(class_ids=[0, 1]),
        aggregate=config.AggregationOptions(macro_average=True))

    self.assertLen(metrics_specs, 5)
    self.assertProtoEquals(
        metrics_specs[0],
        config.MetricsSpec(metrics=[
            config.MetricConfig(
                class_name='ExampleCount',
                config=json.dumps({'name': 'example_count'})),
        ]))
    self.assertProtoEquals(
        metrics_specs[1],
        config.MetricsSpec(
            metrics=[
                config.MetricConfig(
                    class_name='WeightedExampleCount',
                    config=json.dumps({'name': 'weighted_example_count'})),
            ],
            model_names=['model_name1', 'model_name2'],
            output_names=['output_name1']))
    self.assertProtoEquals(
        metrics_specs[2],
        config.MetricsSpec(
            metrics=[
                config.MetricConfig(
                    class_name='MeanSquaredError',
                    config=json.dumps({
                        'name': 'mse',
                        'dtype': 'float32'
                    },
                                      sort_keys=True)),
                config.MetricConfig(
                    class_name='MeanLabel',
                    config=json.dumps({'name': 'mean_label'}))
            ],
            model_names=['model_name1', 'model_name2'],
            output_names=['output_name1'],
            binarize=config.BinarizationOptions(class_ids=[0, 1]),
            aggregate=config.AggregationOptions(macro_average=True)))
    self.assertProtoEquals(
        metrics_specs[3],
        config.MetricsSpec(
            metrics=[
                config.MetricConfig(
                    class_name='WeightedExampleCount',
                    config=json.dumps({'name': 'weighted_example_count'})),
            ],
            model_names=['model_name1', 'model_name2'],
            output_names=['output_name2']))
    self.assertProtoEquals(
        metrics_specs[4],
        config.MetricsSpec(
            metrics=[
                config.MetricConfig(
                    class_name='RootMeanSquaredError',
                    config=json.dumps({
                        'name': 'rmse',
                        'dtype': 'float32'
                    },
                                      sort_keys=True)),
                config.MetricConfig(
                    class_name='MeanPrediction',
                    config=json.dumps({'name': 'mean_prediction'}))
            ],
            model_names=['model_name1', 'model_name2'],
            output_names=['output_name2'],
            binarize=config.BinarizationOptions(class_ids=[0, 1]),
            aggregate=config.AggregationOptions(macro_average=True)))

  def testToComputations(self):
    computations = metric_specs.to_computations(
        metric_specs.specs_from_metrics(
            {
                'output_name': [
                    tf.keras.metrics.MeanSquaredError('mse'),
                    calibration.MeanLabel('mean_label')
                ]
            },
            model_names=['model_name'],
            binarize=config.BinarizationOptions(class_ids=[0, 1]),
            aggregate=config.AggregationOptions(macro_average=True)),
        config.EvalConfig())

    keys = []
    for m in computations:
      for k in m.keys:
        if not k.name.startswith('_'):
          keys.append(k)
    self.assertLen(keys, 8)
    self.assertIn(metric_types.MetricKey(name='example_count'), keys)
    self.assertIn(
        metric_types.MetricKey(
            name='weighted_example_count',
            model_name='model_name',
            output_name='output_name'), keys)
    self.assertIn(
        metric_types.MetricKey(
            name='mse',
            model_name='model_name',
            output_name='output_name',
            sub_key=metric_types.SubKey(class_id=0)), keys)
    self.assertIn(
        metric_types.MetricKey(
            name='mse',
            model_name='model_name',
            output_name='output_name',
            sub_key=metric_types.SubKey(class_id=1)), keys)
    self.assertIn(
        metric_types.MetricKey(
            name='mse', model_name='model_name', output_name='output_name'),
        keys)
    self.assertIn(
        metric_types.MetricKey(
            name='mean_label',
            model_name='model_name',
            output_name='output_name',
            sub_key=metric_types.SubKey(class_id=0)), keys)
    self.assertIn(
        metric_types.MetricKey(
            name='mean_label',
            model_name='model_name',
            output_name='output_name',
            sub_key=metric_types.SubKey(class_id=1)), keys)
    self.assertIn(
        metric_types.MetricKey(
            name='mean_label',
            model_name='model_name',
            output_name='output_name'), keys)


if __name__ == '__main__':
  tf.test.main()
