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
"""Test for batched predict extractor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import config
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.eval_saved_model.example_trainers import batch_size_limited_classifier
from tensorflow_model_analysis.eval_saved_model.example_trainers import dnn_classifier
from tensorflow_model_analysis.eval_saved_model.example_trainers import fixed_prediction_estimator_extra_fields
from tensorflow_model_analysis.eval_saved_model.example_trainers import multi_head
from tensorflow_model_analysis.extractors import features_extractor
from tensorflow_model_analysis.extractors import predictions_extractor
from tfx_bsl.tfxio import tensor_adapter
from tfx_bsl.tfxio import test_util

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2


class PredictionsExtractorTest(testutil.TensorflowModelAnalysisTest):

  def _getExportDir(self):
    return os.path.join(self._getTempDir(), 'export_dir')

  def testPredictionsExtractorWithRegressionModel(self):
    temp_export_dir = self._getExportDir()
    export_dir, _ = (
        fixed_prediction_estimator_extra_fields
        .simple_fixed_prediction_estimator_extra_fields(temp_export_dir, None))

    eval_config = config.EvalConfig(model_specs=[config.ModelSpec()])
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir, tags=[tf.saved_model.SERVING])
    schema = text_format.Parse(
        """
        feature {
          name: "prediction"
          type: FLOAT
        }
        feature {
          name: "label"
          type: FLOAT
        }
        feature {
          name: "fixed_int"
          type: INT
        }
        feature {
          name: "fixed_float"
          type: FLOAT
        }
        feature {
          name: "fixed_string"
          type: BYTES
        }
        """, schema_pb2.Schema())
    tfx_io = test_util.InMemoryTFExampleRecord(
        schema=schema, raw_record_column_name=constants.ARROW_INPUT_COLUMN)
    tensor_adapter_config = tensor_adapter.TensorAdapterConfig(
        arrow_schema=tfx_io.ArrowSchema(),
        tensor_representations=tfx_io.TensorRepresentations())
    feature_extractor = features_extractor.FeaturesExtractor(eval_config)
    prediction_extractor = predictions_extractor.PredictionsExtractor(
        eval_config=eval_config,
        eval_shared_model=eval_shared_model,
        tensor_adapter_config=tensor_adapter_config)

    examples = [
        self._makeExample(
            prediction=0.2,
            label=1.0,
            fixed_int=1,
            fixed_float=1.0,
            fixed_string='fixed_string1'),
        self._makeExample(
            prediction=0.8,
            label=0.0,
            fixed_int=1,
            fixed_float=1.0,
            fixed_string='fixed_string2'),
        self._makeExample(
            prediction=0.5,
            label=0.0,
            fixed_int=2,
            fixed_float=1.0,
            fixed_string='fixed_string3')
    ]

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([e.SerializeToString() for e in examples],
                                    reshuffle=False)
          | 'BatchExamples' >> tfx_io.BeamSource(batch_size=3)
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | feature_extractor.stage_name >> feature_extractor.ptransform
          | prediction_extractor.stage_name >> prediction_extractor.ptransform)

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          self.assertIn(constants.PREDICTIONS_KEY, got[0])
          expected_preds = [0.2, 0.8, 0.5]
          self.assertAlmostEqual(got[0][constants.PREDICTIONS_KEY],
                                 expected_preds)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testPredictionsExtractorWithBinaryClassificationModel(self):
    temp_export_dir = self._getExportDir()
    export_dir, _ = dnn_classifier.simple_dnn_classifier(
        temp_export_dir, None, n_classes=2)

    eval_config = config.EvalConfig(model_specs=[config.ModelSpec()])
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir, tags=[tf.saved_model.SERVING])
    schema = text_format.Parse(
        """
        feature {
          name: "age"
          type: FLOAT
        }
        feature {
          name: "langauge"
          type: BYTES
        }
        feature {
          name: "label"
          type: INT
        }
        """, schema_pb2.Schema())
    tfx_io = test_util.InMemoryTFExampleRecord(
        schema=schema, raw_record_column_name=constants.ARROW_INPUT_COLUMN)
    tensor_adapter_config = tensor_adapter.TensorAdapterConfig(
        arrow_schema=tfx_io.ArrowSchema(),
        tensor_representations=tfx_io.TensorRepresentations())
    feature_extractor = features_extractor.FeaturesExtractor(eval_config)
    prediction_extractor = predictions_extractor.PredictionsExtractor(
        eval_config=eval_config,
        eval_shared_model=eval_shared_model,
        tensor_adapter_config=tensor_adapter_config)

    examples = [
        self._makeExample(age=1.0, language='english', label=0),
        self._makeExample(age=2.0, language='chinese', label=1),
        self._makeExample(age=3.0, language='chinese', label=0),
    ]

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([e.SerializeToString() for e in examples],
                                    reshuffle=False)
          | 'BatchExamples' >> tfx_io.BeamSource(batch_size=3)
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | feature_extractor.stage_name >> feature_extractor.ptransform
          | prediction_extractor.stage_name >> prediction_extractor.ptransform)

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          # We can't verify the actual predictions, but we can verify the keys.
          for item in got:
            self.assertIn(constants.PREDICTIONS_KEY, item)
            for pred in item[constants.PREDICTIONS_KEY]:
              for pred_key in ('logistic', 'probabilities', 'all_classes'):
                self.assertIn(pred_key, pred)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testPredictionsExtractorWithMultiClassModel(self):
    temp_export_dir = self._getExportDir()
    export_dir, _ = dnn_classifier.simple_dnn_classifier(
        temp_export_dir, None, n_classes=3)

    eval_config = config.EvalConfig(model_specs=[config.ModelSpec()])
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir, tags=[tf.saved_model.SERVING])
    schema = text_format.Parse(
        """
        feature {
          name: "age"
          type: FLOAT
        }
        feature {
          name: "langauge"
          type: BYTES
        }
        feature {
          name: "label"
          type: INT
        }
        """, schema_pb2.Schema())
    tfx_io = test_util.InMemoryTFExampleRecord(
        schema=schema, raw_record_column_name=constants.ARROW_INPUT_COLUMN)
    tensor_adapter_config = tensor_adapter.TensorAdapterConfig(
        arrow_schema=tfx_io.ArrowSchema(),
        tensor_representations=tfx_io.TensorRepresentations())
    feature_extractor = features_extractor.FeaturesExtractor(eval_config)
    prediction_extractor = predictions_extractor.PredictionsExtractor(
        eval_config=eval_config,
        eval_shared_model=eval_shared_model,
        tensor_adapter_config=tensor_adapter_config)

    examples = [
        self._makeExample(age=1.0, language='english', label=0),
        self._makeExample(age=2.0, language='chinese', label=1),
        self._makeExample(age=3.0, language='english', label=2),
        self._makeExample(age=4.0, language='chinese', label=1),
    ]

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([e.SerializeToString() for e in examples],
                                    reshuffle=False)
          | 'BatchExamples' >> tfx_io.BeamSource(batch_size=4)
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | feature_extractor.stage_name >> feature_extractor.ptransform
          | prediction_extractor.stage_name >> prediction_extractor.ptransform)

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          # We can't verify the actual predictions, but we can verify the keys.
          for item in got:
            self.assertIn(constants.PREDICTIONS_KEY, item)
            for pred in item[constants.PREDICTIONS_KEY]:
              for pred_key in ('probabilities', 'all_classes'):
                self.assertIn(pred_key, pred)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testPredictionsExtractorWithMultiOutputModel(self):
    temp_export_dir = self._getExportDir()
    export_dir, _ = multi_head.simple_multi_head(temp_export_dir, None)

    eval_config = config.EvalConfig(model_specs=[config.ModelSpec()])
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir, tags=[tf.saved_model.SERVING])
    schema = text_format.Parse(
        """
        feature {
          name: "age"
          type: FLOAT
        }
        feature {
          name: "langauge"
          type: BYTES
        }
        feature {
          name: "english_label"
          type: FLOAT
        }
        feature {
          name: "chinese_label"
          type: FLOAT
        }
        feature {
          name: "other_label"
          type: FLOAT
        }
        """, schema_pb2.Schema())
    tfx_io = test_util.InMemoryTFExampleRecord(
        schema=schema, raw_record_column_name=constants.ARROW_INPUT_COLUMN)
    tensor_adapter_config = tensor_adapter.TensorAdapterConfig(
        arrow_schema=tfx_io.ArrowSchema(),
        tensor_representations=tfx_io.TensorRepresentations())
    feature_extractor = features_extractor.FeaturesExtractor(eval_config)
    prediction_extractor = predictions_extractor.PredictionsExtractor(
        eval_config=eval_config,
        eval_shared_model=eval_shared_model,
        tensor_adapter_config=tensor_adapter_config)

    examples = [
        self._makeExample(
            age=1.0,
            language='english',
            english_label=1.0,
            chinese_label=0.0,
            other_label=0.0),
        self._makeExample(
            age=1.0,
            language='chinese',
            english_label=0.0,
            chinese_label=1.0,
            other_label=0.0),
        self._makeExample(
            age=2.0,
            language='english',
            english_label=1.0,
            chinese_label=0.0,
            other_label=0.0),
        self._makeExample(
            age=2.0,
            language='other',
            english_label=0.0,
            chinese_label=1.0,
            other_label=1.0)
    ]

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([e.SerializeToString() for e in examples],
                                    reshuffle=False)
          | 'BatchExamples' >> tfx_io.BeamSource(batch_size=4)
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | feature_extractor.stage_name >> feature_extractor.ptransform
          | prediction_extractor.stage_name >> prediction_extractor.ptransform)

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          # We can't verify the actual predictions, but we can verify the keys.
          for item in got:
            self.assertIn(constants.PREDICTIONS_KEY, item)
            for pred in item[constants.PREDICTIONS_KEY]:
              for output_name in ('chinese_head', 'english_head', 'other_head'):
                for pred_key in ('logistic', 'probabilities', 'all_classes'):
                  self.assertIn(output_name + '/' + pred_key, pred)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testPredictionsExtractorWithMultiModels(self):
    temp_export_dir = self._getExportDir()
    export_dir1, _ = multi_head.simple_multi_head(temp_export_dir, None)
    export_dir2, _ = multi_head.simple_multi_head(temp_export_dir, None)

    eval_config = config.EvalConfig(model_specs=[
        config.ModelSpec(name='model1'),
        config.ModelSpec(name='model2')
    ])
    eval_shared_model1 = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir1, tags=[tf.saved_model.SERVING])
    eval_shared_model2 = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir2, tags=[tf.saved_model.SERVING])
    schema = text_format.Parse(
        """
        feature {
          name: "age"
          type: FLOAT
        }
        feature {
          name: "langauge"
          type: BYTES
        }
        feature {
          name: "english_label"
          type: FLOAT
        }
        feature {
          name: "chinese_label"
          type: FLOAT
        }
        feature {
          name: "other_label"
          type: FLOAT
        }
        """, schema_pb2.Schema())
    tfx_io = test_util.InMemoryTFExampleRecord(
        schema=schema, raw_record_column_name=constants.ARROW_INPUT_COLUMN)
    tensor_adapter_config = tensor_adapter.TensorAdapterConfig(
        arrow_schema=tfx_io.ArrowSchema(),
        tensor_representations=tfx_io.TensorRepresentations())
    feature_extractor = features_extractor.FeaturesExtractor(eval_config)
    prediction_extractor = predictions_extractor.PredictionsExtractor(
        eval_config=eval_config,
        eval_shared_model={
            'model1': eval_shared_model1,
            'model2': eval_shared_model2
        },
        tensor_adapter_config=tensor_adapter_config)

    examples = [
        self._makeExample(
            age=1.0,
            language='english',
            english_label=1.0,
            chinese_label=0.0,
            other_label=0.0),
        self._makeExample(
            age=1.0,
            language='chinese',
            english_label=0.0,
            chinese_label=1.0,
            other_label=0.0),
        self._makeExample(
            age=2.0,
            language='english',
            english_label=1.0,
            chinese_label=0.0,
            other_label=0.0),
        self._makeExample(
            age=2.0,
            language='other',
            english_label=0.0,
            chinese_label=1.0,
            other_label=1.0)
    ]

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([e.SerializeToString() for e in examples],
                                    reshuffle=False)
          | 'BatchExamples' >> tfx_io.BeamSource(batch_size=4)
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | feature_extractor.stage_name >> feature_extractor.ptransform
          | prediction_extractor.stage_name >> prediction_extractor.ptransform)

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          for item in got:
            # We can't verify the actual predictions, but we can verify the keys
            self.assertIn(constants.PREDICTIONS_KEY, item)
            for pred in item[constants.PREDICTIONS_KEY]:
              for model_name in ('model1', 'model2'):
                self.assertIn(model_name, pred)
                for output_name in ('chinese_head', 'english_head',
                                    'other_head'):
                  for pred_key in ('logistic', 'probabilities', 'all_classes'):
                    self.assertIn(output_name + '/' + pred_key,
                                  pred[model_name])

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testPredictionsExtractorWithKerasModel(self):
    input1 = tf.keras.layers.Input(shape=(2,), name='input1')
    input2 = tf.keras.layers.Input(shape=(2,), name='input2')
    inputs = [input1, input2]
    input_layer = tf.keras.layers.concatenate(inputs)
    output_layer = tf.keras.layers.Dense(
        1, activation=tf.nn.sigmoid, name='output')(
            input_layer)
    model = tf.keras.models.Model(inputs, output_layer)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=.001),
        loss=tf.keras.losses.binary_crossentropy,
        metrics=['accuracy'])

    train_features = {
        'input1': [[0.0, 0.0], [1.0, 1.0]],
        'input2': [[1.0, 1.0], [0.0, 0.0]]
    }
    labels = [[1], [0]]
    example_weights = [1.0, 0.5]
    dataset = tf.data.Dataset.from_tensor_slices(
        (train_features, labels, example_weights))
    dataset = dataset.shuffle(buffer_size=1).repeat().batch(2)
    model.fit(dataset, steps_per_epoch=1)

    export_dir = self._getExportDir()
    model.save(export_dir, save_format='tf')

    eval_config = config.EvalConfig(model_specs=[config.ModelSpec()])
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir, tags=[tf.saved_model.SERVING])
    schema = text_format.Parse(
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
        """, schema_pb2.Schema())
    tfx_io = test_util.InMemoryTFExampleRecord(
        schema=schema, raw_record_column_name=constants.ARROW_INPUT_COLUMN)
    tensor_adapter_config = tensor_adapter.TensorAdapterConfig(
        arrow_schema=tfx_io.ArrowSchema(),
        tensor_representations=tfx_io.TensorRepresentations())
    feature_extractor = features_extractor.FeaturesExtractor(eval_config)
    prediction_extractor = predictions_extractor.PredictionsExtractor(
        eval_config=eval_config,
        eval_shared_model=eval_shared_model,
        tensor_adapter_config=tensor_adapter_config)

    examples = [
        self._makeExample(
            input1=[0.0, 0.0], input2=[1.0, 1.0],
            non_model_feature=0),  # should be ignored by model
        self._makeExample(
            input1=[1.0, 1.0], input2=[0.0, 0.0],
            non_model_feature=1),  # should be ignored by model
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
          | prediction_extractor.stage_name >> prediction_extractor.ptransform)

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          # We can't verify the actual predictions, but we can verify the keys.
          for item in got:
            self.assertIn(constants.PREDICTIONS_KEY, item)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testPredictionsExtractorWithSequentialKerasModel(self):
    # Note that the input will be called 'test_input'
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(
            1, activation=tf.nn.sigmoid, input_shape=(2,), name='test')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=.001),
        loss=tf.keras.losses.binary_crossentropy,
        metrics=['accuracy'])

    train_features = {'test_input': [[0.0, 0.0], [1.0, 1.0]]}
    labels = [[1], [0]]
    example_weights = [1.0, 0.5]
    dataset = tf.data.Dataset.from_tensor_slices(
        (train_features, labels, example_weights))
    dataset = dataset.shuffle(buffer_size=1).repeat().batch(2)
    model.fit(dataset, steps_per_epoch=1)

    export_dir = self._getExportDir()
    model.save(export_dir, save_format='tf')

    eval_config = config.EvalConfig(model_specs=[config.ModelSpec()])
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir, tags=[tf.saved_model.SERVING])
    schema = text_format.Parse(
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
        """, schema_pb2.Schema())
    tfx_io = test_util.InMemoryTFExampleRecord(
        schema=schema, raw_record_column_name=constants.ARROW_INPUT_COLUMN)
    tensor_adapter_config = tensor_adapter.TensorAdapterConfig(
        arrow_schema=tfx_io.ArrowSchema(),
        tensor_representations=tfx_io.TensorRepresentations())
    feature_extractor = features_extractor.FeaturesExtractor(eval_config)
    prediction_extractor = predictions_extractor.PredictionsExtractor(
        eval_config=eval_config,
        eval_shared_model=eval_shared_model,
        tensor_adapter_config=tensor_adapter_config)

    # Notice that the features are 'test' but the model expects 'test_input'.
    # This tests that the PredictExtractor properly handles this case.
    examples = [
        self._makeExample(test=[0.0, 0.0],
                          non_model_feature=0),  # should be ignored by model
        self._makeExample(test=[1.0, 1.0],
                          non_model_feature=1),  # should be ignored by model
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
          | prediction_extractor.stage_name >> prediction_extractor.ptransform)

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          # We can't verify the actual predictions, but we can verify the keys.
          for item in got:
            self.assertIn(constants.PREDICTIONS_KEY, item)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testBatchSizeLimitWithKerasModel(self):
    input1 = tf.keras.layers.Input(shape=(1,), batch_size=1, name='input1')
    input2 = tf.keras.layers.Input(shape=(1,), batch_size=1, name='input2')

    inputs = [input1, input2]
    input_layer = tf.keras.layers.concatenate(inputs)

    def add_1(tensor):
      return tf.add_n([tensor, tf.constant(1.0, shape=(1, 2))])

    assert_layer = tf.keras.layers.Lambda(add_1)(input_layer)

    model = tf.keras.models.Model(inputs, assert_layer)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=.001),
        loss=tf.keras.losses.binary_crossentropy,
        metrics=['accuracy'])

    export_dir = self._getExportDir()
    model.save(export_dir, save_format='tf')

    eval_config = config.EvalConfig(model_specs=[config.ModelSpec()])
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir, tags=[tf.saved_model.SERVING])
    schema = text_format.Parse(
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
        """, schema_pb2.Schema())
    tfx_io = test_util.InMemoryTFExampleRecord(
        schema=schema, raw_record_column_name=constants.ARROW_INPUT_COLUMN)
    tensor_adapter_config = tensor_adapter.TensorAdapterConfig(
        arrow_schema=tfx_io.ArrowSchema(),
        tensor_representations=tfx_io.TensorRepresentations())
    feature_extractor = features_extractor.FeaturesExtractor(eval_config)
    prediction_extractor = predictions_extractor.PredictionsExtractor(
        eval_config=eval_config,
        eval_shared_model=eval_shared_model,
        tensor_adapter_config=tensor_adapter_config)

    examples = []
    for _ in range(4):
      examples.append(self._makeExample(input1=0.0, input2=1.0))

    with beam.Pipeline() as pipeline:
      predict_extracts = (
          pipeline
          | 'Create' >> beam.Create([e.SerializeToString() for e in examples],
                                    reshuffle=False)
          | 'BatchExamples' >> tfx_io.BeamSource(batch_size=1)
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | feature_extractor.stage_name >> feature_extractor.ptransform
          | prediction_extractor.stage_name >> prediction_extractor.ptransform)

      # pylint: enable=no-value-for-parameter
      def check_result(got):
        try:
          self.assertLen(got, 4)
          # We can't verify the actual predictions, but we can verify the keys.
          for item in got:
            self.assertIn(constants.PREDICTIONS_KEY, item)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(predict_extracts, check_result, label='result')

  def testBatchSizeLimit(self):
    temp_export_dir = self._getExportDir()
    _, export_dir = batch_size_limited_classifier.simple_batch_size_limited_classifier(
        None, temp_export_dir)
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir, tags=[tf.saved_model.SERVING])
    eval_config = config.EvalConfig(model_specs=[config.ModelSpec()])
    schema = text_format.Parse(
        """
        feature {
          name: "classes"
          type: BYTES
        }
        feature {
          name: "scores"
          type: FLOAT
        }
        feature {
          name: "labels"
          type: BYTES
        }
        """, schema_pb2.Schema())
    tfx_io = test_util.InMemoryTFExampleRecord(
        schema=schema, raw_record_column_name=constants.ARROW_INPUT_COLUMN)
    tensor_adapter_config = tensor_adapter.TensorAdapterConfig(
        arrow_schema=tfx_io.ArrowSchema(),
        tensor_representations=tfx_io.TensorRepresentations())
    feature_extractor = features_extractor.FeaturesExtractor(eval_config)
    prediction_extractor = predictions_extractor.PredictionsExtractor(
        eval_config=eval_config,
        eval_shared_model=eval_shared_model,
        tensor_adapter_config=tensor_adapter_config)

    examples = []
    for _ in range(4):
      examples.append(
          self._makeExample(classes='first', scores=0.0, labels='third'))

    with beam.Pipeline() as pipeline:
      predict_extracts = (
          pipeline
          | 'Create' >> beam.Create([e.SerializeToString() for e in examples],
                                    reshuffle=False)
          | 'BatchExamples' >> tfx_io.BeamSource(batch_size=1)
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | feature_extractor.stage_name >> feature_extractor.ptransform
          | prediction_extractor.stage_name >> prediction_extractor.ptransform)

      def check_result(got):
        try:
          self.assertLen(got, 4)
          # We can't verify the actual predictions, but we can verify the keys.
          for item in got:
            self.assertIn(constants.PREDICTIONS_KEY, item)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(predict_extracts, check_result, label='result')

  def testPredictionsExtractorWithoutEvalSharedModel(self):
    model_spec1 = config.ModelSpec(name='model1', prediction_key='prediction')
    model_spec2 = config.ModelSpec(
        name='model2',
        prediction_keys={
            'output1': 'prediction1',
            'output2': 'prediction2'
        })
    eval_config = config.EvalConfig(model_specs=[model_spec1, model_spec2])
    feature_extractor = features_extractor.FeaturesExtractor(eval_config)
    prediction_extractor = predictions_extractor.PredictionsExtractor(
        eval_config)

    schema = text_format.Parse(
        """
        feature {
          name: "prediction"
          type: FLOAT
        }
        feature {
          name: "prediction1"
          type: FLOAT
        }
        feature {
          name: "prediction2"
          type: FLOAT
        }
        feature {
          name: "fixed_int"
          type: INT
        }
        """, schema_pb2.Schema())
    tfx_io = test_util.InMemoryTFExampleRecord(
        schema=schema, raw_record_column_name=constants.ARROW_INPUT_COLUMN)

    examples = [
        self._makeExample(
            prediction=1.0, prediction1=1.0, prediction2=0.0, fixed_int=1),
        self._makeExample(
            prediction=1.0, prediction1=1.0, prediction2=1.0, fixed_int=1)
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
          | prediction_extractor.stage_name >> prediction_extractor.ptransform)

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          for model_name in ('model1', 'model2'):
            self.assertIn(model_name, got[0][constants.PREDICTIONS_KEY][0])
          self.assertAlmostEqual(got[0][constants.PREDICTIONS_KEY][0]['model1'],
                                 np.array([1.0]))
          self.assertDictElementsAlmostEqual(
              got[0][constants.PREDICTIONS_KEY][0]['model2'], {
                  'output1': np.array([1.0]),
                  'output2': np.array([0.0])
              })

          for model_name in ('model1', 'model2'):
            self.assertIn(model_name, got[0][constants.PREDICTIONS_KEY][1])
          self.assertAlmostEqual(got[0][constants.PREDICTIONS_KEY][1]['model1'],
                                 np.array([1.0]))
          self.assertDictElementsAlmostEqual(
              got[0][constants.PREDICTIONS_KEY][1]['model2'], {
                  'output1': np.array([1.0]),
                  'output2': np.array([1.0])
              })

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
