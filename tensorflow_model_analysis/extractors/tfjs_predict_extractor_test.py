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


import pytest
import tempfile

from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.extractors import features_extractor
from tensorflow_model_analysis.extractors import tfjs_predict_extractor
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.utils import test_util as testutil
from tensorflow_model_analysis.utils.keras_lib import tf_keras
from tfx_bsl.tfxio import test_util

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2

try:
  from tensorflowjs.converters import converter  # pylint: disable=g-import-not-at-top

  _TFJS_IMPORTED = True
except ModuleNotFoundError:
  _TFJS_IMPORTED = False


@pytest.mark.xfail(run=False, reason="PR 183 This class contains tests that fail and needs to be fixed. "
"If all tests pass, please remove this mark.")
class TFJSPredictExtractorTest(
    testutil.TensorflowModelAnalysisTest, parameterized.TestCase
):

  @parameterized.named_parameters(
      ('single_model_single_output', False, False),
      ('single_model_multi_output', False, True),
      ('multi_model_single_output', True, False),
      ('multi_model_multi_output_batched_examples_batched_inputs', True, True),
  )
  def testTFJSPredictExtractorWithKerasModel(self, multi_model, multi_output):
    if not _TFJS_IMPORTED:
      self.skipTest('This test requires TensorFlow JS.')

    input1 = tf_keras.layers.Input(shape=(1,), name='input1')
    input2 = tf_keras.layers.Input(shape=(1,), name='input2', dtype=tf.int64)
    input3 = tf_keras.layers.Input(shape=(1,), name='input3', dtype=tf.string)
    inputs = [input1, input2, input3]
    input_layer = tf_keras.layers.concatenate([
        inputs[0],
        tf.cast(inputs[1], tf.float32),
        tf.cast(inputs[2] == 'a', tf.float32),
    ])
    output_layers = {}
    output_layers['output1'] = tf_keras.layers.Dense(
        1, activation=tf.nn.sigmoid, name='output1'
    )(input_layer)
    if multi_output:
      output_layers['output2'] = tf_keras.layers.Dense(
          1, activation=tf.nn.sigmoid, name='output2'
      )(input_layer)

    model = tf_keras.models.Model(inputs, output_layers)
    model.compile(
        optimizer=tf_keras.optimizers.Adam(lr=0.001),
        loss=tf_keras.losses.binary_crossentropy,
        metrics=['accuracy'],
    )

    train_features = {
        'input1': [[0.0], [1.0]],
        'input2': [[1], [0]],
        'input3': [[b'a'], [b'b']],
    }
    labels = {'output1': [[1], [0]]}
    if multi_output:
      labels['output2'] = [[1], [0]]

    example_weights = {'output1': [1.0, 0.5]}
    if multi_output:
      example_weights['output2'] = [1.0, 0.5]
    dataset = tf.data.Dataset.from_tensor_slices(
        (train_features, labels, example_weights)
    )
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

    model_specs = [config_pb2.ModelSpec(name='model1', model_type='tf_js')]
    if multi_model:
      model_specs.append(
          config_pb2.ModelSpec(name='model2', model_type='tf_js')
      )

    eval_config = config_pb2.EvalConfig(model_specs=model_specs)
    eval_shared_models = [
        self.createTestEvalSharedModel(
            model_name='model1',
            model_path=dst_model_path,
            model_type='tf_js',
        )
    ]
    if multi_model:
      eval_shared_models.append(
          self.createTestEvalSharedModel(
              model_name='model2',
              model_path=dst_model_path,
              model_type='tf_js',
          )
      )

    schema = text_format.Parse(
        """
        feature {
          name: "input1"
          type: FLOAT
        }
        feature {
          name: "input2"
          type: INT
        }
        feature {
          name: "input3"
          type: BYTES
        }
        feature {
          name: "non_model_feature"
          type: INT
        }
        """,
        schema_pb2.Schema(),
    )
    tfx_io = test_util.InMemoryTFExampleRecord(
        schema=schema, raw_record_column_name=constants.ARROW_INPUT_COLUMN
    )
    feature_extractor = features_extractor.FeaturesExtractor(eval_config)
    predictor = tfjs_predict_extractor.TFJSPredictExtractor(
        eval_config=eval_config, eval_shared_model=eval_shared_models
    )

    examples = [
        self._makeExample(
            input1=0.0, input2=1, input3=b'a', non_model_feature=0
        ),
        self._makeExample(
            input1=1.0, input2=0, input3=b'b', non_model_feature=1
        ),
    ]

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create'
          >> beam.Create(
              [e.SerializeToString() for e in examples], reshuffle=False
          )
          | 'BatchExamples' >> tfx_io.BeamSource(batch_size=2)
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | feature_extractor.stage_name >> feature_extractor.ptransform
          | predictor.stage_name >> predictor.ptransform
      )

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got = got[0]
          self.assertIn(constants.PREDICTIONS_KEY, got)
          for model in ('model1', 'model2') if multi_model else (''):
            per_model_result = got[constants.PREDICTIONS_KEY]
            if model:
              self.assertIn(model, per_model_result)
              per_model_result = per_model_result[model]
            for output in ('Identity', 'Identity_1') if multi_output else (''):
              per_output_result = per_model_result
              if output:
                self.assertIn(output, per_output_result)
                per_output_result = per_output_result[output]
              self.assertLen(per_output_result, 2)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')


