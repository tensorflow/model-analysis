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
                tf.keras.losses.MeanAbsoluteError(name='mae'),
                calibration.MeanLabel('mean_label')
            ],
            'output_name2': [
                tf.keras.metrics.RootMeanSquaredError('rmse'),
                tf.keras.losses.MeanAbsolutePercentageError(name='mape'),
                calibration.MeanPrediction('mean_prediction')
            ]
        },
        model_names=['model_name1', 'model_name2'],
        binarize=config.BinarizationOptions(class_ids={'values': [0, 1]}),
        aggregate=config.AggregationOptions(macro_average=True))

    self.assertLen(metrics_specs, 5)
    self.assertProtoEquals(
        metrics_specs[0],
        config.MetricsSpec(
            metrics=[
                config.MetricConfig(
                    class_name='ExampleCount',
                    config=json.dumps({'name': 'example_count'})),
            ],
            model_names=['model_name1', 'model_name2']))
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
                    class_name='MeanAbsoluteError',
                    module=metric_specs._TF_LOSSES_MODULE,
                    config=json.dumps({
                        'reduction': 'auto',
                        'name': 'mae'
                    },
                                      sort_keys=True)),
                config.MetricConfig(
                    class_name='MeanLabel',
                    config=json.dumps({'name': 'mean_label'}))
            ],
            model_names=['model_name1', 'model_name2'],
            output_names=['output_name1'],
            binarize=config.BinarizationOptions(class_ids={'values': [0, 1]}),
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
                    class_name='MeanAbsolutePercentageError',
                    module=metric_specs._TF_LOSSES_MODULE,
                    config=json.dumps({
                        'reduction': 'auto',
                        'name': 'mape'
                    },
                                      sort_keys=True)),
                config.MetricConfig(
                    class_name='MeanPrediction',
                    config=json.dumps({'name': 'mean_prediction'}))
            ],
            model_names=['model_name1', 'model_name2'],
            output_names=['output_name2'],
            binarize=config.BinarizationOptions(class_ids={'values': [0, 1]}),
            aggregate=config.AggregationOptions(macro_average=True)))

  def testMetricKeysToSkipForConfidenceIntervals(self):
    metrics_specs = [
        config.MetricsSpec(
            metrics=[
                config.MetricConfig(
                    class_name='ExampleCount',
                    config=json.dumps({'name': 'example_count'}),
                    threshold=config.MetricThreshold(
                        value_threshold=config.GenericValueThreshold())),
                config.MetricConfig(
                    class_name='MeanLabel',
                    config=json.dumps({'name': 'mean_label'}),
                    threshold=config.MetricThreshold(
                        change_threshold=config.GenericChangeThreshold())),
                config.MetricConfig(
                    class_name='MeanSquaredError',
                    config=json.dumps({'name': 'mse'}),
                    threshold=config.MetricThreshold(
                        change_threshold=config.GenericChangeThreshold()))
            ],
            model_names=['model_name1', 'model_name2'],
            output_names=['output_name1', 'output_name2']),
    ]
    metrics_specs += metric_specs.specs_from_metrics(
        [tf.keras.metrics.MeanSquaredError('mse')],
        model_names=['model_name1', 'model_name2'])
    keys = metric_specs.metric_keys_to_skip_for_confidence_intervals(
        metrics_specs)
    self.assertLen(keys, 6)
    self.assertIn(
        metric_types.MetricKey(name='example_count', model_name='model_name1'),
        keys)
    self.assertIn(
        metric_types.MetricKey(name='example_count', model_name='model_name2'),
        keys)
    self.assertIn(
        metric_types.MetricKey(
            name='example_count',
            model_name='model_name1',
            output_name='output_name1'), keys)
    self.assertIn(
        metric_types.MetricKey(
            name='example_count',
            model_name='model_name1',
            output_name='output_name2'), keys)
    self.assertIn(
        metric_types.MetricKey(
            name='example_count',
            model_name='model_name2',
            output_name='output_name1'), keys)
    self.assertIn(
        metric_types.MetricKey(
            name='example_count',
            model_name='model_name2',
            output_name='output_name2'), keys)

  def testMetricThresholdsFromMetricsSpecs(self):
    slice_specs = [
        config.SlicingSpec(feature_keys=['feature1']),
        config.SlicingSpec(feature_values={'feature2': 'value1'})
    ]

    # For cross slice tests.
    baseline_slice_spec = config.SlicingSpec(feature_keys=['feature3'])

    metrics_specs = [
        config.MetricsSpec(
            thresholds={
                'auc':
                    config.MetricThreshold(
                        value_threshold=config.GenericValueThreshold()),
                'mean/label':
                    config.MetricThreshold(
                        value_threshold=config.GenericValueThreshold(),
                        change_threshold=config.GenericChangeThreshold()),
                'mse':
                    config.MetricThreshold(
                        change_threshold=config.GenericChangeThreshold())
            },
            per_slice_thresholds={
                'auc':
                    config.PerSliceMetricThresholds(thresholds=[
                        config.PerSliceMetricThreshold(
                            slicing_specs=slice_specs,
                            threshold=config.MetricThreshold(
                                value_threshold=config.GenericValueThreshold()))
                    ]),
                'mean/label':
                    config.PerSliceMetricThresholds(thresholds=[
                        config.PerSliceMetricThreshold(
                            slicing_specs=slice_specs,
                            threshold=config.MetricThreshold(
                                value_threshold=config.GenericValueThreshold(),
                                change_threshold=config.GenericChangeThreshold(
                                )))
                    ])
            },
            cross_slice_thresholds={
                'auc':
                    config.CrossSliceMetricThresholds(thresholds=[
                        config.CrossSliceMetricThreshold(
                            cross_slicing_specs=[
                                config.CrossSlicingSpec(
                                    baseline_spec=baseline_slice_spec,
                                    slicing_specs=slice_specs)
                            ],
                            threshold=config.MetricThreshold(
                                value_threshold=config.GenericValueThreshold(),
                                change_threshold=config.GenericChangeThreshold(
                                )))
                    ]),
                'mse':
                    config.CrossSliceMetricThresholds(thresholds=[
                        config.CrossSliceMetricThreshold(
                            cross_slicing_specs=[
                                config.CrossSlicingSpec(
                                    baseline_spec=baseline_slice_spec,
                                    slicing_specs=slice_specs)
                            ],
                            threshold=config.MetricThreshold(
                                change_threshold=config.GenericChangeThreshold(
                                ))),
                        # Test for duplicate cross_slicing_spec.
                        config.CrossSliceMetricThreshold(
                            cross_slicing_specs=[
                                config.CrossSlicingSpec(
                                    baseline_spec=baseline_slice_spec,
                                    slicing_specs=slice_specs)
                            ],
                            threshold=config.MetricThreshold(
                                value_threshold=config.GenericValueThreshold()))
                    ])
            },
            model_names=['model_name'],
            output_names=['output_name']),
        config.MetricsSpec(
            metrics=[
                config.MetricConfig(
                    class_name='ExampleCount',
                    config=json.dumps({'name': 'example_count'}),
                    threshold=config.MetricThreshold(
                        value_threshold=config.GenericValueThreshold()))
            ],
            model_names=['model_name1', 'model_name2'],
            output_names=['output_name1', 'output_name2']),
        config.MetricsSpec(
            metrics=[
                config.MetricConfig(
                    class_name='WeightedExampleCount',
                    config=json.dumps({'name': 'weighted_example_count'}),
                    threshold=config.MetricThreshold(
                        value_threshold=config.GenericValueThreshold()))
            ],
            model_names=['model_name1', 'model_name2'],
            output_names=['output_name1', 'output_name2']),
        config.MetricsSpec(
            metrics=[
                config.MetricConfig(
                    class_name='MeanSquaredError',
                    config=json.dumps({'name': 'mse'}),
                    threshold=config.MetricThreshold(
                        change_threshold=config.GenericChangeThreshold())),
                config.MetricConfig(
                    class_name='MeanLabel',
                    config=json.dumps({'name': 'mean_label'}),
                    threshold=config.MetricThreshold(
                        change_threshold=config.GenericChangeThreshold()),
                    per_slice_thresholds=[
                        config.PerSliceMetricThreshold(
                            slicing_specs=slice_specs,
                            threshold=config.MetricThreshold(
                                change_threshold=config.GenericChangeThreshold(
                                ))),
                    ],
                    cross_slice_thresholds=[
                        config.CrossSliceMetricThreshold(
                            cross_slicing_specs=[
                                config.CrossSlicingSpec(
                                    baseline_spec=baseline_slice_spec,
                                    slicing_specs=slice_specs)
                            ],
                            threshold=config.MetricThreshold(
                                change_threshold=config.GenericChangeThreshold(
                                )))
                    ]),
            ],
            model_names=['model_name'],
            output_names=['output_name'],
            binarize=config.BinarizationOptions(class_ids={'values': [0, 1]}),
            aggregate=config.AggregationOptions(
                macro_average=True, class_weights={
                    0: 1.0,
                    1: 1.0
                }))
    ]

    thresholds = metric_specs.metric_thresholds_from_metrics_specs(
        metrics_specs)

    expected_keys_and_threshold_counts = {
        metric_types.MetricKey(
            name='auc',
            model_name='model_name',
            output_name='output_name',
            is_diff=False):
            4,
        metric_types.MetricKey(
            name='auc',
            model_name='model_name',
            output_name='output_name',
            is_diff=True):
            1,
        metric_types.MetricKey(
            name='mean/label',
            model_name='model_name',
            output_name='output_name',
            is_diff=True):
            3,
        metric_types.MetricKey(
            name='mean/label',
            model_name='model_name',
            output_name='output_name',
            is_diff=False):
            3,
        metric_types.MetricKey(
            name='example_count',
            model_name='model_name1',
            output_name='output_name1'):
            1,
        metric_types.MetricKey(
            name='example_count',
            model_name='model_name1',
            output_name='output_name2'):
            1,
        metric_types.MetricKey(
            name='example_count',
            model_name='model_name2',
            output_name='output_name1'):
            1,
        metric_types.MetricKey(
            name='example_count',
            model_name='model_name2',
            output_name='output_name2'):
            1,
        metric_types.MetricKey(
            name='weighted_example_count',
            model_name='model_name1',
            output_name='output_name1'):
            1,
        metric_types.MetricKey(
            name='weighted_example_count',
            model_name='model_name1',
            output_name='output_name2'):
            1,
        metric_types.MetricKey(
            name='weighted_example_count',
            model_name='model_name2',
            output_name='output_name1'):
            1,
        metric_types.MetricKey(
            name='weighted_example_count',
            model_name='model_name2',
            output_name='output_name2'):
            1,
        metric_types.MetricKey(
            name='mse',
            model_name='model_name',
            output_name='output_name',
            sub_key=metric_types.SubKey(class_id=0),
            is_diff=True):
            1,
        metric_types.MetricKey(
            name='mse',
            model_name='model_name',
            output_name='output_name',
            sub_key=metric_types.SubKey(class_id=1),
            is_diff=True):
            1,
        metric_types.MetricKey(
            name='mse',
            model_name='model_name',
            output_name='output_name',
            is_diff=True):
            2,
        metric_types.MetricKey(
            name='mse',
            model_name='model_name',
            output_name='output_name',
            is_diff=False):
            1,
        metric_types.MetricKey(
            name='mse',
            model_name='model_name',
            output_name='output_name',
            aggregation_type=metric_types.AggregationType(macro_average=True),
            is_diff=True):
            1,
        metric_types.MetricKey(
            name='mean_label',
            model_name='model_name',
            output_name='output_name',
            sub_key=metric_types.SubKey(class_id=0),
            is_diff=True):
            4,
        metric_types.MetricKey(
            name='mean_label',
            model_name='model_name',
            output_name='output_name',
            sub_key=metric_types.SubKey(class_id=1),
            is_diff=True):
            4,
        metric_types.MetricKey(
            name='mean_label',
            model_name='model_name',
            output_name='output_name',
            aggregation_type=metric_types.AggregationType(macro_average=True),
            is_diff=True):
            4
    }
    self.assertLen(thresholds, len(expected_keys_and_threshold_counts))
    for key, count in expected_keys_and_threshold_counts.items():
      self.assertIn(key, thresholds)
      self.assertLen(thresholds[key], count, 'failed for key {}'.format(key))

  def testToComputations(self):
    computations = metric_specs.to_computations(
        metric_specs.specs_from_metrics(
            {
                'output_name': [
                    tf.keras.metrics.MeanSquaredError('mse'),
                    # Add a loss exactly same as metric
                    # (https://github.com/tensorflow/tfx/issues/1550)
                    tf.keras.losses.MeanSquaredError(name='loss'),
                    calibration.MeanLabel('mean_label')
                ]
            },
            model_names=['model_name'],
            binarize=config.BinarizationOptions(class_ids={'values': [0, 1]}),
            aggregate=config.AggregationOptions(
                macro_average=True, class_weights={
                    0: 1.0,
                    1: 1.0
                })),
        config.EvalConfig())

    keys = []
    for m in computations:
      for k in m.keys:
        if not k.name.startswith('_'):
          keys.append(k)
    self.assertLen(keys, 11)
    self.assertIn(
        metric_types.MetricKey(name='example_count', model_name='model_name'),
        keys)
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
            name='loss',
            model_name='model_name',
            output_name='output_name',
            sub_key=metric_types.SubKey(class_id=0)), keys)
    self.assertIn(
        metric_types.MetricKey(
            name='loss',
            model_name='model_name',
            output_name='output_name',
            sub_key=metric_types.SubKey(class_id=1)), keys)
    self.assertIn(
        metric_types.MetricKey(
            name='loss', model_name='model_name', output_name='output_name'),
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

  # This tests b/155810786
  def testToComputationsWithMixedAggregationAndNonAggregationMetrics(self):
    computations = metric_specs.to_computations([
        config.MetricsSpec(
            metrics=[config.MetricConfig(class_name='CategoricalAccuracy')]),
        config.MetricsSpec(
            metrics=[config.MetricConfig(class_name='BinaryCrossentropy')],
            binarize=config.BinarizationOptions(class_ids={'values': [1]}),
            aggregate=config.AggregationOptions(micro_average=True))
    ], config.EvalConfig())

    # 3 separate computations should be used (one for aggregated metrics, one
    # for non-aggregated metrics, and one for metrics associated with class 1)
    self.assertLen(computations, 3)


if __name__ == '__main__':
  tf.test.main()
