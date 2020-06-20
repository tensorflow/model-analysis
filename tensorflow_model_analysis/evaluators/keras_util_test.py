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
"""Tests for utils for evaluations using keras_util."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import json
import os

import tensorflow as tf
from tensorflow_model_analysis import config
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.evaluators import keras_util


class KerasSavedModelUtilTest(testutil.TensorflowModelAnalysisTest):

  def _comparable_spec(self, spec):
    """Normalizes spec so it can be used in comparisons."""
    # Some keras versions store losses as metrics and some store them as loss
    # functions. This makes the module setting sometimes be used and sometimes
    # not. For test consistency we will clear the module setting.
    for metric in spec.metrics:
      metric.ClearField('module')
    return spec

  def _loss_name(self, model, metric_name, output_name):
    # The new keras models prefix losses with the output name, the old didn't.
    if not hasattr(model, 'loss_functions'):
      # TODO(b/149780822): Update after we get an API from keras.
      return output_name + '_' + metric_name
    else:
      return metric_name

  def testMetricSpecsFromKeras(self):
    export_dir = os.path.join(self._getTempDir(), 'export_dir')
    dummy_layer = tf.keras.layers.Input(shape=(1,))
    model = tf.keras.models.Model([dummy_layer], [dummy_layer])
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.MeanSquaredError(name='mse')])
    features = [[0.0], [1.0]]
    labels = [[1], [0]]
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(buffer_size=1).repeat().batch(2)
    model.fit(dataset, steps_per_epoch=1)
    model.save(export_dir, save_format='tf')

    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir)

    metrics_specs = (
        keras_util.metrics_specs_from_keras('', eval_shared_model.model_loader))

    # TODO(b/149995449): Keras does not support re-loading metrics with the new
    #   API. Re-enable after this is fixed.
    model = eval_shared_model.model_loader.construct_fn()
    if not hasattr(model, 'loss_functions'):
      return

    self.assertLen(metrics_specs, 1)
    self.assertProtoEquals(
        self._comparable_spec(metrics_specs[0]),
        config.MetricsSpec(
            metrics=[
                config.MetricConfig(
                    class_name='BinaryCrossentropy',
                    config=json.dumps(
                        {
                            'from_logits': False,
                            'label_smoothing': 0,
                            'reduction': 'auto',
                            'name': 'binary_crossentropy'
                        },
                        sort_keys=True)),
                config.MetricConfig(
                    class_name='MeanSquaredError',
                    config=json.dumps({
                        'name': 'mse',
                        'dtype': 'float32'
                    },
                                      sort_keys=True))
            ],
            model_names=['']))

  def testMetricSpecsFromKerasSequential(self):
    export_dir = os.path.join(self._getTempDir(), 'export_dir')
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(1,), name='test'),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.MeanSquaredError(name='mse')])
    features = [[0.0], [1.0]]
    labels = [[1], [0]]
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(buffer_size=1).repeat().batch(2)
    model.fit(dataset, steps_per_epoch=1)
    model.save(export_dir, save_format='tf')

    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir, tags=[tf.saved_model.SERVING])

    metrics_specs = (
        keras_util.metrics_specs_from_keras('', eval_shared_model.model_loader))

    # TODO(b/149995449): Keras does not support re-loading metrics with the new
    #   API. Re-enable after this is fixed.
    model = eval_shared_model.model_loader.construct_fn()
    if not hasattr(model, 'loss_functions'):
      return

    self.assertLen(metrics_specs, 1)
    self.assertProtoEquals(
        self._comparable_spec(metrics_specs[0]),
        config.MetricsSpec(
            metrics=[
                config.MetricConfig(
                    class_name='BinaryCrossentropy',
                    config=json.dumps(
                        {
                            'from_logits': False,
                            'label_smoothing': 0,
                            'reduction': 'auto',
                            'name': 'binary_crossentropy'
                        },
                        sort_keys=True)),
                config.MetricConfig(
                    class_name='MeanSquaredError',
                    config=json.dumps({
                        'name': 'mse',
                        'dtype': 'float32'
                    },
                                      sort_keys=True))
            ],
            model_names=['']))

  def testMetricSpecsFromKerasWithMultipleOutputs(self):
    export_dir = os.path.join(self._getTempDir(), 'export_dir')
    input_layer = tf.keras.layers.Input(shape=(1,))
    output_layer1 = tf.keras.layers.Dense(1, name='output_1')(input_layer)
    output_layer2 = tf.keras.layers.Dense(1, name='output_2')(input_layer)
    model = tf.keras.models.Model([input_layer], [output_layer1, output_layer2])
    model.compile(
        loss={
            'output_1':
                (tf.keras.losses.BinaryCrossentropy(name='binary_crossentropy')
                ),
            'output_2':
                (tf.keras.losses.BinaryCrossentropy(name='binary_crossentropy'))
        },
        metrics=[tf.keras.metrics.MeanSquaredError(name='mse')])
    features = [[0.0], [1.0]]
    labels = [[1], [0]]
    dataset = tf.data.Dataset.from_tensor_slices((features, {
        'output_1': labels,
        'output_2': labels
    }))
    dataset = dataset.shuffle(buffer_size=1).repeat().batch(2)
    model.fit(dataset, steps_per_epoch=1)
    model.save(export_dir, save_format='tf')

    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir)

    metrics_specs = (
        keras_util.metrics_specs_from_keras('', eval_shared_model.model_loader))

    # TODO(b/149995449): Keras does not support re-loading metrics with the new
    #   API. Re-enable after this is fixed.
    model = eval_shared_model.model_loader.construct_fn()
    if not hasattr(model, 'loss_functions'):
      return

    self.assertLen(metrics_specs, 2)
    self.assertProtoEquals(
        self._comparable_spec(metrics_specs[0]),
        config.MetricsSpec(
            metrics=[
                config.MetricConfig(
                    class_name='BinaryCrossentropy',
                    config=json.dumps(
                        {
                            'from_logits':
                                False,
                            'label_smoothing':
                                0,
                            'reduction':
                                'auto',
                            'name':
                                self._loss_name(model, 'binary_crossentropy',
                                                'output_1')
                        },
                        sort_keys=True)),
                config.MetricConfig(
                    class_name='MeanSquaredError',
                    config=json.dumps(
                        {
                            'name': 'output_1_mse',
                            'dtype': 'float32'
                        },
                        sort_keys=True))
            ],
            model_names=[''],
            output_names=['output_1']))
    self.assertProtoEquals(
        self._comparable_spec(metrics_specs[1]),
        config.MetricsSpec(
            metrics=[
                config.MetricConfig(
                    class_name='BinaryCrossentropy',
                    config=json.dumps(
                        {
                            'from_logits':
                                False,
                            'label_smoothing':
                                0,
                            'reduction':
                                'auto',
                            'name':
                                self._loss_name(model, 'binary_crossentropy',
                                                'output_2')
                        },
                        sort_keys=True)),
                config.MetricConfig(
                    class_name='MeanSquaredError',
                    config=json.dumps(
                        {
                            'name': 'output_2_mse',
                            'dtype': 'float32'
                        },
                        sort_keys=True))
            ],
            model_names=[''],
            output_names=['output_2']))


if __name__ == '__main__':
  tf.test.main()
