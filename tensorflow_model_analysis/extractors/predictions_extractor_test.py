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
"""Test for batched predict extractor."""


import pytest
import os

from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.extractors import features_extractor
from tensorflow_model_analysis.extractors import predictions_extractor
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.utils import test_util
from tensorflow_model_analysis.utils.keras_lib import tf_keras
from tfx_bsl.tfxio import tensor_adapter
from tfx_bsl.tfxio import test_util as tfx_bsl_test_util

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2



@pytest.mark.xfail(run=False, reason="PR 183 This class contains tests that fail and needs to be fixed. "
"If all tests pass, please remove this mark.")
class PredictionsExtractorTest(
    test_util.TensorflowModelAnalysisTest, parameterized.TestCase
):

  def _getExportDir(self):
    return os.path.join(self._getTempDir(), 'export_dir')

  def _create_tfxio_and_feature_extractor(
      self, eval_config: config_pb2.EvalConfig, schema: schema_pb2.Schema
  ):
    tfx_io = tfx_bsl_test_util.InMemoryTFExampleRecord(
        schema=schema, raw_record_column_name=constants.ARROW_INPUT_COLUMN
    )
    tensor_adapter_config = tensor_adapter.TensorAdapterConfig(
        arrow_schema=tfx_io.ArrowSchema(),
        tensor_representations=tfx_io.TensorRepresentations(),
    )
    feature_extractor = features_extractor.FeaturesExtractor(
        eval_config=eval_config,
        tensor_representations=tensor_adapter_config.tensor_representations,
    )
    return tfx_io, feature_extractor

  # Note: The funtionality covered in this unit test is not supported by
  # PredictionExtractorOSS. This Keras model accepts multiple input tensors,
  # and does not include a signature that # accepts serialized input
  # (i.e. string). This is a requirement for using the bulk inference APIs which
  # only support serialized input right now.
  @parameterized.named_parameters(
      ('ModelSignaturesDoFnInferenceCallableModel', ''),
      ('ModelSignaturesDoFnInferenceServingDefault', 'serving_default'),
  )
  def testPredictionsExtractorWithKerasModel(self, signature_name):
    input1 = tf_keras.layers.Input(shape=(2,), name='input1')
    input2 = tf_keras.layers.Input(shape=(2,), name='input2')
    inputs = [input1, input2]
    input_layer = tf_keras.layers.concatenate(inputs)
    output_layer = tf_keras.layers.Dense(
        1, activation=tf.nn.sigmoid, name='output'
    )(input_layer)
    model = tf_keras.models.Model(inputs, output_layer)
    model.compile(
        optimizer=tf_keras.optimizers.Adam(lr=0.001),
        loss=tf_keras.losses.binary_crossentropy,
        metrics=['accuracy'],
    )

    train_features = {
        'input1': [[0.0, 0.0], [1.0, 1.0]],
        'input2': [[1.0, 1.0], [0.0, 0.0]],
    }
    labels = [[1], [0]]
    example_weights = [1.0, 0.5]
    dataset = tf.data.Dataset.from_tensor_slices(
        (train_features, labels, example_weights)
    )
    dataset = dataset.shuffle(buffer_size=1).repeat().batch(2)
    model.fit(dataset, steps_per_epoch=1)

    export_dir = self._getExportDir()
    model.save(export_dir, save_format='tf')

    eval_config = config_pb2.EvalConfig(
        model_specs=[config_pb2.ModelSpec(signature_name=signature_name)]
    )
    eval_shared_model = self.createKerasTestEvalSharedModel(
        eval_saved_model_path=export_dir, eval_config=eval_config
    )
    tfx_io, feature_extractor = self._create_tfxio_and_feature_extractor(
        eval_config,
        text_format.Parse(
            """
        tensor_representation_group {
          key: ""
          value {
            tensor_representation {
              key: "input1"
              value {
                dense_tensor {
                  column_name: "input1"
                  shape { dim { size: 2 } }
                }
              }
            }
            tensor_representation {
              key: "input2"
              value {
                dense_tensor {
                  column_name: "input2"
                  shape { dim { size: 2 } }
                }
              }
            }
          }
        }
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
        """,
            schema_pb2.Schema(),
        ),
    )
    prediction_extractor = predictions_extractor.PredictionsExtractor(
        eval_config=eval_config, eval_shared_model=eval_shared_model
    )

    examples = [
        self._makeExample(
            input1=[0.0, 0.0], input2=[1.0, 1.0], non_model_feature=0
        ),  # should be ignored by model
        self._makeExample(
            input1=[1.0, 1.0], input2=[0.0, 0.0], non_model_feature=1
        ),  # should be ignored by model
    ]
    num_examples = len(examples)

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create'
          >> beam.Create(
              [e.SerializeToString() for e in examples], reshuffle=False
          )
          | 'BatchExamples' >> tfx_io.BeamSource(batch_size=num_examples)
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | feature_extractor.stage_name >> feature_extractor.ptransform
          | prediction_extractor.stage_name >> prediction_extractor.ptransform
      )
      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          # We can't verify the actual predictions, but we can verify the keys.
          self.assertIn(constants.PREDICTIONS_KEY, got[0])

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result)

  # Note: The funtionality covered in this unit test is not supported by
  # PredictionExtractorOSS. This Keras model does not include a signature that
  # accepts serialized input (i.e. string). This is a requirement for using the
  # bulk inference APIs which only support serialized input right now.
  @parameterized.named_parameters(
      ('ModelSignaturesDoFnInferenceCallableModel', ''),
      ('ModelSignaturesDoFnInferenceServingDefault', 'serving_default'),
  )
  def testPredictionsExtractorWithSequentialKerasModel(self, signature_name):
    # Note that the input will be called 'test_input'
    model = tf_keras.models.Sequential([
        tf_keras.layers.Dense(
            1, activation=tf.nn.sigmoid, input_shape=(2,), name='test'
        )
    ])
    model.compile(
        optimizer=tf_keras.optimizers.Adam(lr=0.001),
        loss=tf_keras.losses.binary_crossentropy,
        metrics=['accuracy'],
    )

    train_features = {'test_input': [[0.0, 0.0], [1.0, 1.0]]}
    labels = [[1], [0]]
    example_weights = [1.0, 0.5]
    dataset = tf.data.Dataset.from_tensor_slices(
        (train_features, labels, example_weights)
    )
    dataset = dataset.shuffle(buffer_size=1).repeat().batch(2)
    model.fit(dataset, steps_per_epoch=1)

    export_dir = self._getExportDir()
    model.save(export_dir, save_format='tf')

    eval_config = config_pb2.EvalConfig(
        model_specs=[config_pb2.ModelSpec(signature_name=signature_name)]
    )
    eval_shared_model = self.createKerasTestEvalSharedModel(
        eval_saved_model_path=export_dir, eval_config=eval_config
    )
    tfx_io, feature_extractor = self._create_tfxio_and_feature_extractor(
        eval_config,
        text_format.Parse(
            """
        tensor_representation_group {
          key: ""
          value {
            tensor_representation {
              key: "test"
              value {
                dense_tensor {
                  column_name: "test"
                  shape { dim { size: 2 } }
                }
              }
            }
          }
        }
        feature {
          name: "test"
          type: FLOAT
        }
        feature {
          name: "non_model_feature"
          type: INT
        }
        """,
            schema_pb2.Schema(),
        ),
    )
    prediction_extractor = predictions_extractor.PredictionsExtractor(
        eval_config=eval_config, eval_shared_model=eval_shared_model
    )

    # Notice that the features are 'test' but the model expects 'test_input'.
    # This tests that the PredictExtractor properly handles this case.
    examples = [
        self._makeExample(
            test=[0.0, 0.0], non_model_feature=0
        ),  # should be ignored by model
        self._makeExample(
            test=[1.0, 1.0], non_model_feature=1
        ),  # should be ignored by model
    ]
    num_examples = len(examples)

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create'
          >> beam.Create(
              [e.SerializeToString() for e in examples], reshuffle=False
          )
          | 'BatchExamples' >> tfx_io.BeamSource(batch_size=num_examples)
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | feature_extractor.stage_name >> feature_extractor.ptransform
          | prediction_extractor.stage_name >> prediction_extractor.ptransform
      )
      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          # We can't verify the actual predictions, but we can verify the keys.
          self.assertIn(constants.PREDICTIONS_KEY, got[0])

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result)

  # Note: The funtionality covered in this unit test is not supported by
  # PredictionExtractorOSS. This Keras model accepts multiple input tensors,
  # and does not include a signature that # accepts serialized input
  # (i.e. string). This is a requirement for using the bulk inference APIs which
  # only support serialized input right now.
  def testBatchSizeLimitWithKerasModel(self):
    input1 = tf_keras.layers.Input(shape=(1,), batch_size=1, name='input1')
    input2 = tf_keras.layers.Input(shape=(1,), batch_size=1, name='input2')

    inputs = [input1, input2]
    input_layer = tf_keras.layers.concatenate(inputs)

    def add_1(tensor):
      return tf.add_n([tensor, tf.constant(1.0, shape=(1, 2))])

    assert_layer = tf_keras.layers.Lambda(add_1)(input_layer)

    model = tf_keras.models.Model(inputs, assert_layer)
    model.compile(
        optimizer=tf_keras.optimizers.Adam(lr=0.001),
        loss=tf_keras.losses.binary_crossentropy,
        metrics=['accuracy'],
    )

    export_dir = self._getExportDir()
    model.save(export_dir, save_format='tf')

    eval_config = config_pb2.EvalConfig(model_specs=[config_pb2.ModelSpec()])
    eval_shared_model = self.createKerasTestEvalSharedModel(
        eval_saved_model_path=export_dir, eval_config=eval_config
    )
    tfx_io, feature_extractor = self._create_tfxio_and_feature_extractor(
        eval_config,
        text_format.Parse(
            """
        tensor_representation_group {
          key: ""
          value {
            tensor_representation {
              key: "input1"
              value {
                dense_tensor {
                  column_name: "input1"
                  shape { dim { size: 1 } }
                }
              }
            }
            tensor_representation {
              key: "input2"
              value {
                dense_tensor {
                  column_name: "input2"
                  shape { dim { size: 1 } }
                }
              }
            }
          }
        }
        feature {
          name: "input1"
          type: FLOAT
        }
        feature {
          name: "input2"
          type: FLOAT
        }
        """,
            schema_pb2.Schema(),
        ),
    )
    prediction_extractor = predictions_extractor.PredictionsExtractor(
        eval_config=eval_config, eval_shared_model=eval_shared_model
    )

    examples = []
    for _ in range(4):
      examples.append(self._makeExample(input1=0.0, input2=1.0))

    with beam.Pipeline() as pipeline:
      predict_extracts = (
          pipeline
          | 'Create'
          >> beam.Create(
              [e.SerializeToString() for e in examples], reshuffle=False
          )
          | 'BatchExamples' >> tfx_io.BeamSource(batch_size=1)
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | feature_extractor.stage_name >> feature_extractor.ptransform
          | prediction_extractor.stage_name >> prediction_extractor.ptransform
      )

      def check_result(got):
        try:
          self.assertLen(got, 4)
          # We can't verify the actual predictions, but we can verify the keys.
          for item in got:
            self.assertIn(constants.PREDICTIONS_KEY, item)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(predict_extracts, check_result)

  # TODO(b/239975835): Remove this test for version 1.0.
  def testRekeyPredictionsInFeaturesForPrematerializedPredictions(self):
    model_spec1 = config_pb2.ModelSpec(
        name='model1', prediction_key='prediction'
    )
    model_spec2 = config_pb2.ModelSpec(
        name='model2',
        prediction_keys={'output1': 'prediction1', 'output2': 'prediction2'},
    )
    eval_config = config_pb2.EvalConfig(model_specs=[model_spec1, model_spec2])
    schema = text_format.Parse(
        """
        tensor_representation_group {
          key: ""
          value {
            tensor_representation {
              key: "fixed_int"
              value {
                dense_tensor {
                  column_name: "fixed_int"
                }
              }
            }
          }
        }
        feature {
          name: "prediction"
          type: FLOAT
          shape: { }
          presence: { min_fraction: 1 }
        }
        feature {
          name: "prediction1"
          type: FLOAT
          shape: { }
          presence: { min_fraction: 1 }
        }
        feature {
          name: "prediction2"
          type: FLOAT
          shape: { }
          presence: { min_fraction: 1 }
        }
        feature {
          name: "fixed_int"
          type: INT
        }
        """,
        schema_pb2.Schema(),
    )
    # TODO(b/73109633): Remove when field is removed or its default changes to
    # False.
    if hasattr(schema, 'generate_legacy_feature_spec'):
      schema.generate_legacy_feature_spec = False
    tfx_io, feature_extractor = self._create_tfxio_and_feature_extractor(
        eval_config, schema
    )

    examples = [
        self._makeExample(
            prediction=1.0, prediction1=1.0, prediction2=0.0, fixed_int=1
        ),
        self._makeExample(
            prediction=1.0, prediction1=1.0, prediction2=1.0, fixed_int=1
        ),
    ]
    num_examples = len(examples)

    prediction_extractor = predictions_extractor.PredictionsExtractor(
        eval_config=eval_config, eval_shared_model=None
    )

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create'
          >> beam.Create(
              [e.SerializeToString() for e in examples], reshuffle=False
          )
          | 'BatchExamples' >> tfx_io.BeamSource(batch_size=num_examples)
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | feature_extractor.stage_name >> feature_extractor.ptransform
          | prediction_extractor.stage_name >> prediction_extractor.ptransform
      )
      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          for model_name in ('model1', 'model2'):
            self.assertIn(model_name, got[0][constants.PREDICTIONS_KEY])
          self.assertAllClose(
              np.array([1.0, 1.0]), got[0][constants.PREDICTIONS_KEY]['model1']
          )

          self.assertAllClose(
              {
                  'output1': np.array([1.0, 1.0]),
                  'output2': np.array([0.0, 1.0]),
              },
              got[0][constants.PREDICTIONS_KEY]['model2'],
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result)


