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
"""Tests for tflite predict extractor."""

import itertools
import os
import tempfile

from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.extractors import features_extractor
from tensorflow_model_analysis.extractors import tflite_predict_extractor
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.utils.keras_lib import tf_keras
from tfx_bsl.tfxio import test_util

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2


_TF_MAJOR_VERSION = int(tf.version.VERSION.split('.')[0])

_MULTI_MODEL_CASES = [False, True]
_MULTI_OUTPUT_CASES = [False, True]
# Equality op not supported in TF1. See b/242088810
_BYTES_FEATURE_CASES = [False] if _TF_MAJOR_VERSION < 2 else [False, True]
_QUANTIZATION_CASES = [False, True]


def random_genenerator():
  generator: tf.random.Generator = tf.random.Generator.from_seed(42)
  for unused_i in range(10):
    r = {
        'input1': generator.uniform(shape=(2, 1), minval=0.0, maxval=1.0),
        'input2': generator.uniform(shape=(2, 1), minval=0.0, maxval=1.0),
        'input3': tf.constant([[b'a'], [b'b']], shape=(2, 1), dtype=tf.string),
    }
    yield r


class TFLitePredictExtractorTest(
    testutil.TensorflowModelAnalysisTest, parameterized.TestCase
):

  @parameterized.parameters(
      itertools.product(
          _MULTI_MODEL_CASES,
          _MULTI_OUTPUT_CASES,
          _BYTES_FEATURE_CASES,
          _QUANTIZATION_CASES,
      )
  )
  def testTFlitePredictExtractorWithKerasModel(
      self, multi_model, multi_output, use_bytes_feature, use_quantization
  ):
    input1 = tf_keras.layers.Input(shape=(1,), name='input1')
    input2 = tf_keras.layers.Input(shape=(1,), name='input2')
    input3 = tf_keras.layers.Input(shape=(1,), name='input3', dtype=tf.string)
    inputs = [input1, input2, input3]
    if use_bytes_feature:
      input_layer = tf_keras.layers.concatenate(
          [inputs[0], inputs[1], tf.cast(inputs[2] == 'a', tf.float32)]
      )
    else:
      input_layer = tf_keras.layers.concatenate([inputs[0], inputs[1]])
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
        'input2': [[1.0], [0.0]],
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

    converter = tf.compat.v2.lite.TFLiteConverter.from_keras_model(model)
    if use_quantization:
      converter.optimizations = [tf.lite.Optimize.DEFAULT]
      converter.target_spec.supported_ops = [
          tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
          tf.lite.OpsSet.SELECT_TF_OPS,
      ]
      converter.inference_input_type = tf.uint8
      converter.inference_output_type = tf.uint8
      converter.representative_dataset = random_genenerator
    tflite_model = converter.convert()

    tflite_model_dir = tempfile.mkdtemp()
    with tf.io.gfile.GFile(os.path.join(tflite_model_dir, 'tflite'), 'wb') as f:
      f.write(tflite_model)

    model_specs = [config_pb2.ModelSpec(name='model1', model_type='tf_lite')]
    if multi_model:
      model_specs.append(
          config_pb2.ModelSpec(name='model2', model_type='tf_lite')
      )

    eval_config = config_pb2.EvalConfig(model_specs=model_specs)
    eval_shared_models = [
        self.createTestEvalSharedModel(
            model_name='model1',
            eval_saved_model_path=tflite_model_dir,
            model_type='tf_lite',
        )
    ]
    if multi_model:
      eval_shared_models.append(
          self.createTestEvalSharedModel(
              model_name='model2',
              eval_saved_model_path=tflite_model_dir,
              model_type='tf_lite',
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
          type: FLOAT
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
    predictor = tflite_predict_extractor.TFLitePredictExtractor(
        eval_config=eval_config, eval_shared_model=eval_shared_models
    )

    examples = [
        self._makeExample(
            input1=0.0, input2=1.0, input3=b'a', non_model_feature=0
        ),
        self._makeExample(
            input1=1.0, input2=0.0, input3=b'b', non_model_feature=1
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


