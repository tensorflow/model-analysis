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
"""Tests for metric_types."""

import tensorflow as tf
from tensorflow_model_analysis.metrics import metric_types


class MetricTypesTest(tf.test.TestCase):

  def testMetricKeyStrForMetricKeyWithOneField(self):
    self.assertEqual(
        str(metric_types.MetricKey(name='metric_name')),
        'name: "metric_name" example_weighted: { }')

  def testMetricKeyStrForMetricKeyWithAllFields(self):
    self.assertEqual(
        str(
            metric_types.MetricKey(
                name='metric_name',
                model_name='model_name',
                output_name='output_name',
                sub_key=metric_types.SubKey(class_id=1),
                example_weighted=True,
                is_diff=True)),
        'name: "metric_name" output_name: "output_name" ' +
        'sub_key: { class_id: { value: 1 } } model_name: "model_name" ' +
        'is_diff: true example_weighted: { value: true }')

  def testMetricKeyFromProto(self):
    metric_keys = [
        metric_types.MetricKey(name='metric_name'),
        metric_types.MetricKey(
            name='metric_name',
            model_name='model_name',
            output_name='output_name',
            sub_key=metric_types.SubKey(class_id=1),
            is_diff=True),
        metric_types.MetricKey(
            name='metric_name',
            model_name='model_name',
            output_name='output_name',
            sub_key=metric_types.SubKey(top_k=2),
            example_weighted=None,
            aggregation_type=metric_types.AggregationType(micro_average=True)),
        metric_types.MetricKey(
            name='metric_name',
            model_name='model_name',
            output_name='output_name',
            example_weighted=False)
    ]
    for key in metric_keys:
      got_key = metric_types.MetricKey.from_proto(key.to_proto())
      self.assertEqual(key, got_key, '{} != {}'.format(key, got_key))

  def testPlotKeyFromProto(self):
    plot_keys = [
        metric_types.PlotKey(name='plot_name'),
        metric_types.PlotKey(
            name='plot_name',
            model_name='model_name',
            output_name='output_name',
            sub_key=metric_types.SubKey(class_id=1),
        ),
        metric_types.MetricKey(
            name='plot_name',
            model_name='model_name',
            output_name='output_name',
            sub_key=metric_types.SubKey(top_k=2),
        ),
    ]
    for key in plot_keys:
      got_key = metric_types.PlotKey.from_proto(key.to_proto())
      self.assertEqual(key, got_key, '{} != {}'.format(key, got_key))

  def testSubKeyStr(self):
    self.assertEqual(str(metric_types.SubKey(class_id=1)), 'classId:1')
    self.assertEqual(str(metric_types.SubKey(top_k=2)), 'topK:2')
    self.assertEqual(str(metric_types.SubKey(k=3)), 'k:3')
    self.assertEqual(
        str(metric_types.SubKey(class_id=1, top_k=2)), 'classId:1 topK:2'
    )

  def testSubKeySetUp(self):
    with self.assertRaises(
        NotImplementedError,
        msg=(
            'A non-existent SubKey should be represented as None, not as ',
            'SubKey(None, None, None).',
        ),
    ):
      str(metric_types.SubKey())
    with self.assertRaises(
        ValueError,
        msg=('k and top_k cannot both be set at the same time',),
    ):
      str(metric_types.SubKey(k=2, top_k=2))
    with self.assertRaises(
        ValueError,
        msg=('k and class_id cannot both be set at the same time',),
    ):
      str(metric_types.SubKey(k=2, class_id=2))

  def testAggregationTypeLessThan(self):
    self.assertLess(
        metric_types.AggregationType(macro_average=True),
        metric_types.AggregationType(micro_average=True),
    )
    self.assertLess(
        metric_types.AggregationType(weighted_macro_average=True),
        metric_types.AggregationType(macro_average=True),
    )

  def testPreprocessors(self):
    preprocessor = metric_types.StandardMetricInputsPreprocessorList([
        metric_types.FeaturePreprocessor(feature_keys=['feature1', 'feature2']),
        metric_types.TransformedFeaturePreprocessor(feature_keys=['feature1']),
        metric_types.AttributionPreprocessor(feature_keys=['feature1']),
    ])
    self.assertEqual(
        preprocessor.include_filter,
        {
            'labels': {},
            'predictions': {},
            'example_weights': {},
            'features': {
                'feature1': {},
                'feature2': {},
            },
            'transformed_features': {
                'feature1': {},
            },
            'attributions': {
                'feature1': {},
            },
        },
    )

  def testPreprocessorsWithoutDefaults(self):
    preprocessor = metric_types.StandardMetricInputsPreprocessorList([
        metric_types.FeaturePreprocessor(
            feature_keys=['feature1', 'feature2'],
            include_default_inputs=False),
        metric_types.TransformedFeaturePreprocessor(
            feature_keys=['feature1'], include_default_inputs=False),
        metric_types.AttributionPreprocessor(
            feature_keys=['feature1'], include_default_inputs=False)
    ])
    self.assertEqual(
        preprocessor.include_filter, {
            'features': {
                'feature1': {},
                'feature2': {},
            },
            'transformed_features': {
                'feature1': {},
            },
            'attributions': {
                'feature1': {},
            },
        })

  def testMultiModelMultiOutputPreprocessors(self):
    preprocessor = metric_types.StandardMetricInputsPreprocessorList([
        metric_types.FeaturePreprocessor(
            feature_keys=['feature1', 'feature2'],
            model_names=['model1', 'model2'],
            output_names=['output1', 'output2']),
        metric_types.TransformedFeaturePreprocessor(
            feature_keys=['feature1'],
            model_names=['model1', 'model2'],
            output_names=['output1', 'output2']),
        metric_types.AttributionPreprocessor(
            feature_keys=['feature1'],
            model_names=['model1'],
            output_names=['output2'])
    ])
    self.assertEqual(
        preprocessor.include_filter, {
            'labels': {
                'model1': {
                    'output1': {},
                    'output2': {},
                },
                'model2': {
                    'output1': {},
                    'output2': {},
                },
            },
            'predictions': {
                'model1': {
                    'output1': {},
                    'output2': {},
                },
                'model2': {
                    'output1': {},
                    'output2': {},
                },
            },
            'example_weights': {
                'model1': {
                    'output1': {},
                    'output2': {},
                },
                'model2': {
                    'output1': {},
                    'output2': {},
                },
            },
            'features': {
                'feature1': {},
                'feature2': {},
            },
            'transformed_features': {
                'model1': {
                    'feature1': {},
                },
                'model2': {
                    'feature1': {},
                },
            },
            'attributions': {
                'model1': {
                    'output2': {
                        'feature1': {},
                    }
                },
            },
        })


