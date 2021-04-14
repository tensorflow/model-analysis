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
import tensorflow as tf
from tensorflow_model_analysis import config
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.extractors import features_extractor
from tensorflow_model_analysis.extractors import tfjs_predict_extractor
from tensorflowjs.converters import converter
from tfx_bsl.tfxio import test_util

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2


class TFJSPredictExtractorTest(testutil.TensorflowModelAnalysisTest,
                               parameterized.TestCase):

  @parameterized.named_parameters(
      ('single_model_single_output', False, False),
      ('single_model_multi_output', False, True),
      ('multi_model_single_output', True, False),
      ('multi_model_multi_output_batched_examples_batched_inputs', True, True))
  def testTFJSPredictExtractorWithKerasModel(self, multi_model, multi_output):
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

    schema = text_format.Parse(
        """
        feature {
          name: "input1"
          type: FLOAT
        }
        feature {
          name: "input2"
          type: FLOAT
        }
        feature {
          name: "non_model_feature"
          type: INT
        }
        """, schema_pb2.Schema())
    tfx_io = test_util.InMemoryTFExampleRecord(
        schema=schema, raw_record_column_name=constants.ARROW_INPUT_COLUMN)
    feature_extractor = features_extractor.FeaturesExtractor(eval_config)
    predictor = tfjs_predict_extractor.TFJSPredictExtractor(
        eval_config=eval_config, eval_shared_model=eval_shared_models)

    examples = [
        self._makeExample(input1=0.0, input2=1.0, non_model_feature=0),
        self._makeExample(input1=1.0, input2=0.0, non_model_feature=1),
    ]

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([e.SerializeToString() for e in examples],
                                    reshuffle=False)
          | 'BatchExamples' >> tfx_io.BeamSource(batch_size=2)
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | feature_extractor.stage_name >> feature_extractor.ptransform
          | predictor.stage_name >> predictor.ptransform)

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got = got[0]
          self.assertIn(constants.PREDICTIONS_KEY, got)
          self.assertLen(got[constants.PREDICTIONS_KEY], 2)

          for item in got[constants.PREDICTIONS_KEY]:
            if multi_model:
              self.assertIn('model1', item)
              self.assertIn('model2', item)
              if multi_output:
                self.assertIn('Identity', item['model1'])
                self.assertIn('Identity_1', item['model1'])

            elif multi_output:
              self.assertIn('Identity', item)
              self.assertIn('Identity_1', item)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
