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
"""Test for transformed features extractor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import unittest

from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import tensorflow as tf
from tensorflow_model_analysis import config
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.extractors import features_extractor
from tensorflow_model_analysis.extractors import transformed_features_extractor
from tfx_bsl.tfxio import tensor_adapter
from tfx_bsl.tfxio import test_util

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2

_TF_MAJOR_VERSION = int(tf.version.VERSION.split('.')[0])


class TransformedFeaturesExtractorTest(testutil.TensorflowModelAnalysisTest,
                                       parameterized.TestCase):

  def createDenseInputsSchema(self):
    return text_format.Parse(
        """
        tensor_representation_group {
          key: ""
          value {
            tensor_representation {
              key: "input_1"
              value {
                dense_tensor {
                  column_name: "input_1"
                  shape { dim { size: 1 } }
                }
              }
            }
            tensor_representation {
              key: "input_2"
              value {
                dense_tensor {
                  column_name: "input_2"
                  shape { dim { size: 1 } }
                }
              }
            }
          }
        }
        feature {
          name: "input_1"
          type: FLOAT
        }
        feature {
          name: "input_2"
          type: FLOAT
        }
        feature {
          name: "non_model_feature"
          type: INT
        }
        """, schema_pb2.Schema())

  def createModelWithMultipleDenseInputs(self, save_as_keras):
    input1 = tf.keras.layers.Input(shape=(1,), name='input_1')
    input2 = tf.keras.layers.Input(shape=(1,), name='input_2')
    inputs = [input1, input2]
    input_layer = tf.keras.layers.concatenate(inputs)
    output_layer = tf.keras.layers.Dense(
        1, activation=tf.nn.sigmoid, name='output')(
            input_layer)
    model = tf.keras.models.Model(inputs, output_layer)

    # Add tft_layer to model to test callables stored as attributes
    model.tft_layer = tf.keras.models.Model(inputs, {
        'tft_feature': output_layer,
        'tft_label': output_layer
    })

    @tf.function
    def serving_default(serialized_tf_examples):
      parsed_features = tf.io.parse_example(
          serialized_tf_examples, {
              'input_1': tf.io.FixedLenFeature([1], dtype=tf.float32),
              'input_2': tf.io.FixedLenFeature([1], dtype=tf.float32)
          })
      return model(parsed_features)

    @tf.function
    def transformed_features(features):
      return {
          'transformed_feature': features['input_1'],
      }

    @tf.function
    def transformed_labels(features):
      return {'transformed_label': features['input_2']}

    @tf.function
    def custom_preprocessing(features):
      return {
          'custom_feature': features['input_1'],
          'custom_label': features['input_2']
      }

    single_input_spec = tf.TensorSpec(
        shape=(None,), dtype=tf.string, name='examples')
    multi_input_spec = {
        'input_1':
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32, name='input_1'),
        'input_2':
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32, name='input_2')
    }
    signatures = {
        'serving_default':
            serving_default.get_concrete_function(single_input_spec),
        'transformed_labels':
            transformed_labels.get_concrete_function(multi_input_spec),
        'transformed_features':
            transformed_features.get_concrete_function(multi_input_spec),
        'custom_preprocessing':
            custom_preprocessing.get_concrete_function(multi_input_spec)
    }

    export_path = tempfile.mkdtemp()
    if save_as_keras:
      model.save(export_path, save_format='tf', signatures=signatures)
    else:
      tf.saved_model.save(model, export_path, signatures=signatures)
    return export_path

  @parameterized.named_parameters(
      (
          'keras_defaults',
          True,
          [],
          {
              'features': [
                  'input_1',  # raw feature
                  'input_2',  # raw feature
                  'non_model_feature',  # from schema
              ],
              'transformed_features': [
                  # TODO(b/173029091): Re-add tft_layer
                  # 'tft_feature',  # added by tft_layer
                  # 'tft_label',  # added by tft_layer
                  'transformed_feature',  # added by transformed_features
                  'transformed_label',  # added by transformed_labels
              ]
          }),
      (
          'tf_defaults',
          False,
          [],
          {
              'features': [
                  'input_1',  # raw feature
                  'input_2',  # raw feature
                  'non_model_feature',  # from schema
              ],
              'transformed_features': [
                  # TODO(b/173029091): Re-add tft_layer
                  # 'tft_feature',  # added by tft_layer
                  # 'tft_label',  # added by tft_layer
                  'transformed_feature',  # added by transformed_features
                  'transformed_label',  # added by transformed_labels
              ]
          }),
      (
          'keras_custom',
          True,
          ['custom_preprocessing'],
          {
              'features': [
                  'input_1',  # raw feature
                  'input_2',  # raw feature
                  'non_model_feature',  # from schema
              ],
              'transformed_features': [
                  'custom_feature',  # added by custom_preprocessing
                  'custom_label',  # added by custom_preprocessing
              ]
          }),
      (
          'tf_custom',
          False,
          ['custom_preprocessing'],
          {
              'features': [
                  'input_1',  # raw feature
                  'input_2',  # raw feature
                  'non_model_feature',  # from schema
              ],
              'transformed_features': [
                  'custom_feature',  # added by custom_preprocessing
                  'custom_label',  # added by custom_preprocessing
              ]
          }),
  )
  @unittest.skipIf(_TF_MAJOR_VERSION < 2,
                   'not all signatures supported for TF1')
  def testPreprocessedFeaturesExtractor(self, save_as_keras,
                                        preprocessing_function_names,
                                        expected_extract_keys):
    export_path = self.createModelWithMultipleDenseInputs(save_as_keras)

    eval_config = config.EvalConfig(model_specs=[
        config.ModelSpec(
            preprocessing_function_names=preprocessing_function_names)
    ])
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_path, tags=[tf.saved_model.SERVING])
    schema = self.createDenseInputsSchema()
    tfx_io = test_util.InMemoryTFExampleRecord(
        schema=schema, raw_record_column_name=constants.ARROW_INPUT_COLUMN)
    tensor_adapter_config = tensor_adapter.TensorAdapterConfig(
        arrow_schema=tfx_io.ArrowSchema(),
        tensor_representations=tfx_io.TensorRepresentations())
    feature_extractor = features_extractor.FeaturesExtractor(eval_config)
    transformation_extractor = (
        transformed_features_extractor.TransformedFeaturesExtractor(
            eval_config=eval_config,
            eval_shared_model=eval_shared_model,
            tensor_adapter_config=tensor_adapter_config))

    examples = [
        self._makeExample(input_1=1.0, input_2=2.0),
        self._makeExample(input_1=3.0, input_2=4.0),
        self._makeExample(input_1=5.0, input_2=6.0),
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
          | transformation_extractor.stage_name >>
          transformation_extractor.ptransform)

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 2)
          for item in got:
            for extracts_key, feature_keys in expected_extract_keys.items():
              self.assertIn(extracts_key, item)
              for value in item[extracts_key]:
                self.assertEqual(
                    set(feature_keys),
                    set(value.keys()),
                    msg='got={}'.format(item))

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
