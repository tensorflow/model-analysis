# Lint as: python3
# Copyright 2020 Google LLC
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
"""Tests for tfjs predict extractor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import config
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.extractors import legacy_tfjs_predict_extractor as tfjs_predict_extractor
from tensorflowjs.converters import converter


class TFJSPredictExtractorTest(testutil.TensorflowModelAnalysisTest,
                               parameterized.TestCase):

  @parameterized.named_parameters(
      ('single_model_single_output_batched_examples_batched_inputs', False,
       False, True, True),
      ('single_model_single_output_batched_examples_not_batched_inputs', False,
       False, True, False),
      ('single_model_single_output_single_examples_batched_inputs', False,
       False, False, True),
      ('single_model_single_output_single_examples_not_batched_inputs', False,
       False, False, False),
      ('single_model_multi_output_batched_examples_batched_inputs', False, True,
       True, True),
      ('single_model_multi_output_batched_examples_not_batched_inputs', False,
       True, True, False),
      ('single_model_multi_output_single_examples_batched_inputs', False, True,
       False, True),
      ('single_model_multi_output_single_examples_not_batched_inputs', False,
       True, False, False),
      ('multi_model_single_output_batched_examples_batched_inputs', True, False,
       True, True),
      ('multi_model_single_output_batched_examples_not_batched_inputs', True,
       False, True, False),
      ('multi_model_single_output_single_examples_batched_inputs', True, False,
       False, True),
      ('multi_model_single_output_single_examples_not_batched_inputs', True,
       False, False, False),
      ('multi_model_multi_output_batched_examples_batched_inputs', True, True,
       True, True),
      ('multi_model_multi_output_batched_examples_not_batched_inputs', True,
       True, True, False),
      ('multi_model_multi_output_single_examples_batched_inputs', True, True,
       False, True),
      ('mult_model_multi_output_single_examples_not_batched_inputs', True, True,
       False, False))
  def testTFJSPredictExtractorWithSingleOutputModel(self, multi_model,
                                                    multi_output,
                                                    batch_examples,
                                                    batch_inputs):
    input1 = tf.keras.layers.Input(shape=(1,), name='input1')
    input2 = tf.keras.layers.Input(shape=(1,), name='input2')
    inputs = [input1, input2]
    input_layer = tf.keras.layers.concatenate(inputs)
    output_layers = {}
    output_layers['output1'] = (
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid,
                              name='output1')(input_layer))
    if multi_output:
      output_layers['output2'] = (
          tf.keras.layers.Dense(1, activation=tf.nn.sigmoid,
                                name='output2')(input_layer))

    model = tf.keras.models.Model(inputs, output_layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=.001),
        loss=tf.keras.losses.binary_crossentropy,
        metrics=['accuracy'])

    train_features = {'input1': [[0.0], [1.0]], 'input2': [[1.0], [0.0]]}
    labels = {'output1': [[1], [0]]}
    if multi_output:
      labels['output2'] = [[1], [0]]

    example_weights = {'output1': [1.0, 0.5]}
    if multi_output:
      example_weights['output2'] = [1.0, 0.5]
    dataset = tf.data.Dataset.from_tensor_slices(
        (train_features, labels, example_weights))
    dataset = dataset.shuffle(buffer_size=1).repeat().batch(2)
    model.fit(dataset, steps_per_epoch=1)

    src_model_path = tempfile.mkdtemp()
    model.save(src_model_path)

    dst_model_path = tempfile.mkdtemp()
    converter.convert([
        '--input_format=tf_saved_model',
        '--saved_model_tags=serve',
        '--signature_name=serving_default',
        src_model_path,
        dst_model_path,
    ])

    model_specs = [config.ModelSpec(name='model1', model_type='tf_js')]
    if multi_model:
      model_specs.append(config.ModelSpec(name='model2', model_type='tf_js'))

    eval_config = config.EvalConfig(model_specs=model_specs)
    eval_shared_models = [
        self.createTestEvalSharedModel(
            model_name='model1',
            eval_saved_model_path=dst_model_path,
            model_type='tf_js')
    ]
    if multi_model:
      eval_shared_models.append(
          self.createTestEvalSharedModel(
              model_name='model2',
              eval_saved_model_path=dst_model_path,
              model_type='tf_js'))

    desired_batch_size = 2 if batch_examples else None
    predictor = tfjs_predict_extractor.TFJSPredictExtractor(
        eval_config=eval_config,
        eval_shared_model=eval_shared_models,
        desired_batch_size=desired_batch_size)

    predict_features = [
        {
            'input1': np.array([0.0], dtype=np.float32),
            'input2': np.array([1.0], dtype=np.float32),
            'non_model_feature': np.array([0]),  # should be ignored by model
        },
        {
            'input1': np.array([1.0], dtype=np.float32),
            'input2': np.array([0.0], dtype=np.float32),
            'non_model_feature': np.array([1]),  # should be ignored by model
        }
    ]

    if batch_inputs:
      predict_features = [{k: np.expand_dims(v, 0)
                           for k, v in p.items()}
                          for p in predict_features]

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create(predict_features)
          | 'FeaturesToExtracts' >>
          beam.Map(lambda x: {constants.FEATURES_KEY: x})
          | predictor.stage_name >> predictor.ptransform)

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 2)
          # We can't verify the actual predictions, but we can verify the keys.
          for item in got:
            self.assertIn(constants.PREDICTIONS_KEY, item)

            if multi_model:
              self.assertIn('model1', item[constants.PREDICTIONS_KEY])
              self.assertIn('model2', item[constants.PREDICTIONS_KEY])
              if multi_output:
                self.assertIn('Identity',
                              item[constants.PREDICTIONS_KEY]['model1'])
                self.assertIn('Identity_1',
                              item[constants.PREDICTIONS_KEY]['model1'])

            elif multi_output:
              self.assertIn('Identity', item[constants.PREDICTIONS_KEY])
              self.assertIn('Identity_1', item[constants.PREDICTIONS_KEY])

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
