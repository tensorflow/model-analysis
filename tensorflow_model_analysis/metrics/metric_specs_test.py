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


import pytest
import json

import tensorflow as tf
from tensorflow_model_analysis.metrics import calibration
from tensorflow_model_analysis.metrics import confusion_matrix_metrics
# This module should be included as the tests involves replacement of keras
# mean squared error and mean absolute error with native TFMA metrics.x
from tensorflow_model_analysis.metrics import mean_regression_error  # pylint: disable=unused-import
from tensorflow_model_analysis.metrics import metric_specs
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.utils.keras_lib import tf_keras


# TODO(b/272542795): Remove once the Keras version has caught up.
def _maybe_add_fn_name(kv, name):
  # Check new Keras version behavior per b/272542795#comment17.
  if 'fn' in tf_keras.losses.MeanAbsoluteError().get_config():
    kv['fn'] = name
  return kv


@pytest.mark.xfail(run=False, reason="PR 183 This class contains tests that fail and needs to be fixed. "
"If all tests pass, please remove this mark.")
class MetricSpecsTest(tf.test.TestCase):

  def testSpecsFromMetrics(self):
    metrics_specs = metric_specs.specs_from_metrics(
        {
            'output_name1': [
                tf_keras.metrics.Precision(name='precision'),
                tf_keras.metrics.MeanSquaredError('mse'),
                tf_keras.losses.MeanAbsoluteError(name='mae'),
            ],
            'output_name2': [
                confusion_matrix_metrics.Precision(name='precision'),
                tf_keras.losses.MeanAbsolutePercentageError(name='mape'),
                calibration.MeanPrediction('mean_prediction'),
            ],
        },
        unweighted_metrics={
            'output_name1': [calibration.MeanLabel('mean_label')],
            'output_name2': [tf_keras.metrics.RootMeanSquaredError('rmse')],
        },
        model_names=['model_name1', 'model_name2'],
        binarize=config_pb2.BinarizationOptions(class_ids={'values': [0, 1]}),
        aggregate=config_pb2.AggregationOptions(macro_average=True),
    )

    self.assertLen(metrics_specs, 7)
    self.assertProtoEquals(
        metrics_specs[0],
        config_pb2.MetricsSpec(
            metrics=[
                config_pb2.MetricConfig(
                    class_name='ExampleCount',
                    config=json.dumps({'name': 'example_count'})),
            ],
            model_names=['model_name1', 'model_name2'],
            example_weights=config_pb2.ExampleWeightOptions(unweighted=True)))
    self.assertProtoEquals(
        metrics_specs[1],
        config_pb2.MetricsSpec(
            metrics=[
                config_pb2.MetricConfig(
                    class_name='WeightedExampleCount',
                    config=json.dumps({'name': 'weighted_example_count'})),
            ],
            model_names=['model_name1', 'model_name2'],
            output_names=['output_name1'],
            example_weights=config_pb2.ExampleWeightOptions(weighted=True)))
    self.assertProtoEquals(
        metrics_specs[2],
        config_pb2.MetricsSpec(
            metrics=[
                config_pb2.MetricConfig(
                    class_name='Precision',
                    config=json.dumps(
                        {
                            'name': 'precision',
                            'class_id': None,
                            'thresholds': None,
                            'top_k': None,
                        },
                        sort_keys=True,
                    ),
                ),
                config_pb2.MetricConfig(
                    class_name='MeanSquaredError',
                    config=json.dumps(
                        {
                            'name': 'mse',
                        },
                        sort_keys=True,
                    ),
                ),
                config_pb2.MetricConfig(
                    class_name='MeanAbsoluteError',
                    config=json.dumps(
                        _maybe_add_fn_name(
                            {'name': 'mae'}, 'mean_absolute_error'
                        ),
                        sort_keys=True,
                    ),
                ),
            ],
            model_names=['model_name1', 'model_name2'],
            output_names=['output_name1'],
            binarize=config_pb2.BinarizationOptions(
                class_ids={'values': [0, 1]}
            ),
            aggregate=config_pb2.AggregationOptions(macro_average=True),
        ),
    )
    self.assertProtoEquals(
        metrics_specs[3],
        config_pb2.MetricsSpec(
            metrics=[
                config_pb2.MetricConfig(
                    class_name='MeanLabel',
                    config=json.dumps({'name': 'mean_label'}))
            ],
            model_names=['model_name1', 'model_name2'],
            output_names=['output_name1'],
            binarize=config_pb2.BinarizationOptions(
                class_ids={'values': [0, 1]}),
            aggregate=config_pb2.AggregationOptions(macro_average=True),
            example_weights=config_pb2.ExampleWeightOptions(unweighted=True)))
    self.assertProtoEquals(
        metrics_specs[4],
        config_pb2.MetricsSpec(
            metrics=[
                config_pb2.MetricConfig(
                    class_name='WeightedExampleCount',
                    config=json.dumps({'name': 'weighted_example_count'})),
            ],
            model_names=['model_name1', 'model_name2'],
            output_names=['output_name2'],
            example_weights=config_pb2.ExampleWeightOptions(weighted=True)))
    self.assertProtoEquals(
        metrics_specs[5],
        config_pb2.MetricsSpec(
            metrics=[
                config_pb2.MetricConfig(
                    class_name='Precision',
                    config=json.dumps(
                        {
                            'name': 'precision',
                        },
                        sort_keys=True,
                    ),
                ),
                config_pb2.MetricConfig(
                    class_name='MeanAbsolutePercentageError',
                    config=json.dumps(
                        _maybe_add_fn_name(
                            {'name': 'mape'}, 'mean_absolute_percentage_error'
                        ),
                        sort_keys=True,
                    ),
                ),
                config_pb2.MetricConfig(
                    class_name='MeanPrediction',
                    config=json.dumps({'name': 'mean_prediction'}),
                ),
            ],
            model_names=['model_name1', 'model_name2'],
            output_names=['output_name2'],
            binarize=config_pb2.BinarizationOptions(
                class_ids={'values': [0, 1]}
            ),
            aggregate=config_pb2.AggregationOptions(macro_average=True),
        ),
    )
    # This is for the compatibility issue when migrating from Keras 2 to Keras
    # 3. In the older versions, TFMA is using tf.keras which is pointing to
    # Keras 2. However, the newest tf.keras is pointing to Keras 3, and TFMA is
    # using tf_keras pacakge for Keras 2. In this case, there is a module name
    # discrepancy, where tf.keras modulue is pointing to keras.src.metrics but
    # tf_keras modulue is pointing to tf_keras.src.metrics.

    version_fn = getattr(tf.keras, '__version__', None)
    if not version_fn or (version_fn and version_fn.startswith('3.')):
      self.assertProtoEquals(
          metrics_specs[6],
          config_pb2.MetricsSpec(
              metrics=[
                  config_pb2.MetricConfig(
                      class_name='RootMeanSquaredError',
                      module='tf_keras.src.metrics.regression_metrics',
                      config=json.dumps(
                          {'name': 'rmse', 'dtype': 'float32'}, sort_keys=True
                      ),
                  )
              ],
              model_names=['model_name1', 'model_name2'],
              output_names=['output_name2'],
              binarize=config_pb2.BinarizationOptions(
                  class_ids={'values': [0, 1]}
              ),
              aggregate=config_pb2.AggregationOptions(macro_average=True),
              example_weights=config_pb2.ExampleWeightOptions(unweighted=True),
          ),
      )
    else:
      self.assertProtoEquals(
          metrics_specs[6],
          config_pb2.MetricsSpec(
              metrics=[
                  config_pb2.MetricConfig(
                      class_name='RootMeanSquaredError',
                      module='tf_keras.metrics.regression_metrics',
                      config=json.dumps(
                          {'name': 'rmse', 'dtype': 'float32'}, sort_keys=True
                      ),
                  )
              ],
              model_names=['model_name1', 'model_name2'],
              output_names=['output_name2'],
              binarize=config_pb2.BinarizationOptions(
                  class_ids={'values': [0, 1]}
              ),
              aggregate=config_pb2.AggregationOptions(macro_average=True),
              example_weights=config_pb2.ExampleWeightOptions(unweighted=True),
          ),
      )

  def testMetricKeysToSkipForConfidenceIntervals(self):
    metrics_specs = [
        config_pb2.MetricsSpec(
            metrics=[
                config_pb2.MetricConfig(
                    class_name='ExampleCount',
                    config=json.dumps({'name': 'example_count'}),
                    threshold=config_pb2.MetricThreshold(
                        value_threshold=config_pb2.GenericValueThreshold())),
                config_pb2.MetricConfig(
                    class_name='MeanLabel',
                    config=json.dumps({'name': 'mean_label'}),
                    threshold=config_pb2.MetricThreshold(
                        change_threshold=config_pb2.GenericChangeThreshold())),
                config_pb2.MetricConfig(
                    class_name='MeanSquaredError',
                    config=json.dumps({'name': 'mse'}),
                    threshold=config_pb2.MetricThreshold(
                        change_threshold=config_pb2.GenericChangeThreshold()))
            ],
            model_names=['model_name1', 'model_name2'],
            output_names=['output_name1', 'output_name2']),
    ]
    metrics_specs += metric_specs.specs_from_metrics(
        [tf_keras.metrics.MeanSquaredError('mse')],
        model_names=['model_name1', 'model_name2'],
    )
    keys = metric_specs.metric_keys_to_skip_for_confidence_intervals(
        metrics_specs, eval_config=config_pb2.EvalConfig())
    self.assertLen(keys, 8)
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
    self.assertIn(
        metric_types.MetricKey(name='example_count', model_name='model_name1'),
        keys)
    self.assertIn(
        metric_types.MetricKey(
            name='weighted_example_count',
            model_name='model_name1',
            example_weighted=True), keys)
    self.assertIn(
        metric_types.MetricKey(name='example_count', model_name='model_name2'),
        keys)
    self.assertIn(
        metric_types.MetricKey(
            name='weighted_example_count',
            model_name='model_name2',
            example_weighted=True), keys)

  def testMetricThresholdsFromMetricsSpecs(self):
    slice_specs = [
        config_pb2.SlicingSpec(feature_keys=['feature1']),
        config_pb2.SlicingSpec(feature_values={'feature2': 'value1'})
    ]

    # For cross slice tests.
    baseline_slice_spec = config_pb2.SlicingSpec(feature_keys=['feature3'])

    metrics_specs = [
        config_pb2.MetricsSpec(
            thresholds={
                'auc':
                    config_pb2.MetricThreshold(
                        value_threshold=config_pb2.GenericValueThreshold()),
                'mean/label':
                    config_pb2.MetricThreshold(
                        value_threshold=config_pb2.GenericValueThreshold(),
                        change_threshold=config_pb2.GenericChangeThreshold()),
                'mse':
                    config_pb2.MetricThreshold(
                        change_threshold=config_pb2.GenericChangeThreshold())
            },
            per_slice_thresholds={
                'auc':
                    config_pb2.PerSliceMetricThresholds(thresholds=[
                        config_pb2.PerSliceMetricThreshold(
                            slicing_specs=slice_specs,
                            threshold=config_pb2.MetricThreshold(
                                value_threshold=config_pb2
                                .GenericValueThreshold()))
                    ]),
                'mean/label':
                    config_pb2.PerSliceMetricThresholds(thresholds=[
                        config_pb2.PerSliceMetricThreshold(
                            slicing_specs=slice_specs,
                            threshold=config_pb2.MetricThreshold(
                                value_threshold=config_pb2
                                .GenericValueThreshold(),
                                change_threshold=config_pb2
                                .GenericChangeThreshold()))
                    ])
            },
            cross_slice_thresholds={
                'auc':
                    config_pb2.CrossSliceMetricThresholds(thresholds=[
                        config_pb2.CrossSliceMetricThreshold(
                            cross_slicing_specs=[
                                config_pb2.CrossSlicingSpec(
                                    baseline_spec=baseline_slice_spec,
                                    slicing_specs=slice_specs)
                            ],
                            threshold=config_pb2.MetricThreshold(
                                value_threshold=config_pb2
                                .GenericValueThreshold(),
                                change_threshold=config_pb2
                                .GenericChangeThreshold()))
                    ]),
                'mse':
                    config_pb2.CrossSliceMetricThresholds(thresholds=[
                        config_pb2.CrossSliceMetricThreshold(
                            cross_slicing_specs=[
                                config_pb2.CrossSlicingSpec(
                                    baseline_spec=baseline_slice_spec,
                                    slicing_specs=slice_specs)
                            ],
                            threshold=config_pb2.MetricThreshold(
                                change_threshold=config_pb2
                                .GenericChangeThreshold())),
                        # Test for duplicate cross_slicing_spec.
                        config_pb2.CrossSliceMetricThreshold(
                            cross_slicing_specs=[
                                config_pb2.CrossSlicingSpec(
                                    baseline_spec=baseline_slice_spec,
                                    slicing_specs=slice_specs)
                            ],
                            threshold=config_pb2.MetricThreshold(
                                value_threshold=config_pb2
                                .GenericValueThreshold()))
                    ])
            },
            model_names=['model_name'],
            output_names=['output_name']),
        config_pb2.MetricsSpec(
            metrics=[
                config_pb2.MetricConfig(
                    class_name='ExampleCount',
                    config=json.dumps({'name': 'example_count'}),
                    threshold=config_pb2.MetricThreshold(
                        value_threshold=config_pb2.GenericValueThreshold()))
            ],
            model_names=['model_name1', 'model_name2'],
            example_weights=config_pb2.ExampleWeightOptions(unweighted=True)),
        config_pb2.MetricsSpec(
            metrics=[
                config_pb2.MetricConfig(
                    class_name='WeightedExampleCount',
                    config=json.dumps({'name': 'weighted_example_count'}),
                    threshold=config_pb2.MetricThreshold(
                        value_threshold=config_pb2.GenericValueThreshold()))
            ],
            model_names=['model_name1', 'model_name2'],
            output_names=['output_name1', 'output_name2'],
            example_weights=config_pb2.ExampleWeightOptions(weighted=True)),
        config_pb2.MetricsSpec(
            metrics=[
                config_pb2.MetricConfig(
                    class_name='MeanSquaredError',
                    config=json.dumps({'name': 'mse'}),
                    threshold=config_pb2.MetricThreshold(
                        change_threshold=config_pb2.GenericChangeThreshold())),
                config_pb2.MetricConfig(
                    class_name='MeanLabel',
                    config=json.dumps({'name': 'mean_label'}),
                    threshold=config_pb2.MetricThreshold(
                        change_threshold=config_pb2.GenericChangeThreshold()),
                    per_slice_thresholds=[
                        config_pb2.PerSliceMetricThreshold(
                            slicing_specs=slice_specs,
                            threshold=config_pb2.MetricThreshold(
                                change_threshold=config_pb2
                                .GenericChangeThreshold())),
                    ],
                    cross_slice_thresholds=[
                        config_pb2.CrossSliceMetricThreshold(
                            cross_slicing_specs=[
                                config_pb2.CrossSlicingSpec(
                                    baseline_spec=baseline_slice_spec,
                                    slicing_specs=slice_specs)
                            ],
                            threshold=config_pb2.MetricThreshold(
                                change_threshold=config_pb2
                                .GenericChangeThreshold()))
                    ]),
            ],
            model_names=['model_name'],
            output_names=['output_name'],
            binarize=config_pb2.BinarizationOptions(
                class_ids={'values': [0, 1]}),
            aggregate=config_pb2.AggregationOptions(
                macro_average=True, class_weights={
                    0: 1.0,
                    1: 1.0
                }))
    ]

    thresholds = metric_specs.metric_thresholds_from_metrics_specs(
        metrics_specs, eval_config=config_pb2.EvalConfig())

    expected_keys_and_threshold_counts = {
        metric_types.MetricKey(
            name='auc',
            model_name='model_name',
            output_name='output_name',
            is_diff=False,
            example_weighted=None):
            4,
        metric_types.MetricKey(
            name='auc',
            model_name='model_name',
            output_name='output_name',
            is_diff=True,
            example_weighted=None):
            1,
        metric_types.MetricKey(
            name='mean/label',
            model_name='model_name',
            output_name='output_name',
            is_diff=True,
            example_weighted=None):
            3,
        metric_types.MetricKey(
            name='mean/label',
            model_name='model_name',
            output_name='output_name',
            is_diff=False,
            example_weighted=None):
            3,
        metric_types.MetricKey(name='example_count', model_name='model_name1'):
            1,
        metric_types.MetricKey(name='example_count', model_name='model_name2'):
            1,
        metric_types.MetricKey(
            name='weighted_example_count',
            model_name='model_name1',
            output_name='output_name1',
            example_weighted=True):
            1,
        metric_types.MetricKey(
            name='weighted_example_count',
            model_name='model_name1',
            output_name='output_name2',
            example_weighted=True):
            1,
        metric_types.MetricKey(
            name='weighted_example_count',
            model_name='model_name2',
            output_name='output_name1',
            example_weighted=True):
            1,
        metric_types.MetricKey(
            name='weighted_example_count',
            model_name='model_name2',
            output_name='output_name2',
            example_weighted=True):
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
            is_diff=False,
            example_weighted=None):
            1,
        metric_types.MetricKey(
            name='mse',
            model_name='model_name',
            output_name='output_name',
            is_diff=True,
            example_weighted=None):
            2,
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
            [
                tf_keras.metrics.MeanSquaredError('mse'),
                # Add a loss exactly same as metric
                # (https://github.com/tensorflow/tfx/issues/1550)
                tf_keras.losses.MeanSquaredError(name='loss'),
                calibration.MeanLabel('mean_label'),
            ],
            model_names=['model_name'],
            output_names=['output_1', 'output_2'],
            output_weights={'output_1': 1.0, 'output_2': 1.0},
            binarize=config_pb2.BinarizationOptions(
                class_ids={'values': [0, 1]}
            ),
            aggregate=config_pb2.AggregationOptions(
                macro_average=True, class_weights={0: 1.0, 1: 1.0}
            ),
        ),
        config_pb2.EvalConfig(),
    )

    keys = []
    for m in computations:
      for k in m.keys:
        if not k.name.startswith('_'):
          keys.append(k)
    self.assertLen(keys, 31)
    self.assertIn(
        metric_types.MetricKey(name='example_count', model_name='model_name'),
        keys)
    for output_name in ('output_1', 'output_2', ''):
      self.assertIn(
          metric_types.MetricKey(
              name='weighted_example_count',
              model_name='model_name',
              output_name=output_name,
              example_weighted=True), keys)
      self.assertIn(
          metric_types.MetricKey(
              name='mse',
              model_name='model_name',
              output_name=output_name,
              sub_key=metric_types.SubKey(class_id=0)), keys)
      self.assertIn(
          metric_types.MetricKey(
              name='mse',
              model_name='model_name',
              output_name=output_name,
              sub_key=metric_types.SubKey(class_id=1)), keys)
      aggregation_type = metric_types.AggregationType(
          macro_average=True) if output_name else None
      self.assertIn(
          metric_types.MetricKey(
              name='mse',
              model_name='model_name',
              output_name=output_name,
              aggregation_type=aggregation_type), keys)
      self.assertIn(
          metric_types.MetricKey(
              name='loss',
              model_name='model_name',
              output_name=output_name,
              sub_key=metric_types.SubKey(class_id=0)), keys)
      self.assertIn(
          metric_types.MetricKey(
              name='loss',
              model_name='model_name',
              output_name=output_name,
              sub_key=metric_types.SubKey(class_id=1)), keys)
      aggregation_type = metric_types.AggregationType(
          macro_average=True) if output_name else None
      self.assertIn(
          metric_types.MetricKey(
              name='loss',
              model_name='model_name',
              output_name=output_name,
              aggregation_type=aggregation_type), keys)
      self.assertIn(
          metric_types.MetricKey(
              name='mean_label',
              model_name='model_name',
              output_name=output_name,
              sub_key=metric_types.SubKey(class_id=0)), keys)
      self.assertIn(
          metric_types.MetricKey(
              name='mean_label',
              model_name='model_name',
              output_name=output_name,
              sub_key=metric_types.SubKey(class_id=1)), keys)
      aggregation_type = metric_types.AggregationType(
          macro_average=True) if output_name else None
      self.assertIn(
          metric_types.MetricKey(
              name='mean_label',
              model_name='model_name',
              output_name=output_name,
              aggregation_type=aggregation_type), keys)

  # This tests b/155810786
  def testToComputationsWithMixedAggregationAndNonAggregationMetrics(self):
    computations = metric_specs.to_computations([
        config_pb2.MetricsSpec(
            metrics=[config_pb2.MetricConfig(
                class_name='CategoricalAccuracy')]),
        config_pb2.MetricsSpec(
            metrics=[config_pb2.MetricConfig(class_name='BinaryCrossentropy')],
            binarize=config_pb2.BinarizationOptions(class_ids={'values': [1]}),
            aggregate=config_pb2.AggregationOptions(micro_average=True))
    ], config_pb2.EvalConfig())

    # 3 separate computations should be used (one for aggregated metrics, one
    # for non-aggregated metrics, and one for metrics associated with class 1)
    self.assertLen(computations, 3)


