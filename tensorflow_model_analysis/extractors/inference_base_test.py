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
"""Tests for inference_base.

For more test coverage, see servo_beam_predictions_extractor_test.py and
tfx_bsl_predictions_extractor_test.py.
"""


import pytest
import os

import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.eval_saved_model.example_trainers import fixed_prediction_estimator_extra_fields
from tensorflow_model_analysis.extractors import features_extractor
from tensorflow_model_analysis.extractors import inference_base
from tensorflow_model_analysis.extractors import tfx_bsl_predictions_extractor
from tensorflow_model_analysis.proto import config_pb2
from tfx_bsl.tfxio import tensor_adapter
from tfx_bsl.tfxio import test_util

from google.protobuf import text_format
from tensorflow.core.protobuf import saved_model_pb2  # pylint: disable=g-direct-tensorflow-import
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_serving.apis import logging_pb2
from tensorflow_serving.apis import prediction_log_pb2


@pytest.mark.xfail(run=False, reason="PR 183 This class contains tests that fail and needs to be fixed. "
"If all tests pass, please remove this mark.")
class TfxBslPredictionsExtractorTest(testutil.TensorflowModelAnalysisTest):

  def setUp(self):
    super().setUp()
    log_metadata1 = logging_pb2.LogMetadata(timestamp_secs=1)
    predict_log1 = prediction_log_pb2.PredictLog()
    self.prediction_log1 = prediction_log_pb2.PredictionLog(
        predict_log=predict_log1, log_metadata=log_metadata1
    )

    log_metadata2 = logging_pb2.LogMetadata(timestamp_secs=2)
    predict_log2 = prediction_log_pb2.PredictLog()
    self.prediction_log2 = prediction_log_pb2.PredictionLog(
        predict_log=predict_log2, log_metadata=log_metadata2
    )

  def _getExportDir(self):
    return os.path.join(self._getTempDir(), 'export_dir')

  def _create_tfxio_and_feature_extractor(
      self, eval_config: config_pb2.EvalConfig, schema: schema_pb2.Schema
  ):
    tfx_io = test_util.InMemoryTFExampleRecord(
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

  def testRegressionModel(self):
    temp_export_dir = self._getExportDir()
    export_dir, _ = (
        fixed_prediction_estimator_extra_fields.simple_fixed_prediction_estimator_extra_fields(
            temp_export_dir, None
        )
    )

    eval_config = config_pb2.EvalConfig(
        model_specs=[
            config_pb2.ModelSpec(
                name='model_1', signature_name='serving_default'
            )
        ]
    )
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir, tags=[tf.saved_model.SERVING]
    )
    tfx_io, feature_extractor = self._create_tfxio_and_feature_extractor(
        eval_config,
        text_format.Parse(
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
        """,
            schema_pb2.Schema(),
        ),
    )

    examples = [
        self._makeExample(
            prediction=0.2,
            label=1.0,
            fixed_int=1,
            fixed_float=1.0,
            fixed_string='fixed_string1',
        ),
        self._makeExample(
            prediction=0.8,
            label=0.0,
            fixed_int=1,
            fixed_float=1.0,
            fixed_string='fixed_string2',
        ),
        self._makeExample(
            prediction=0.5,
            label=0.0,
            fixed_int=2,
            fixed_float=1.0,
            fixed_string='fixed_string3',
        ),
    ]
    num_examples = len(examples)

    tfx_bsl_inference_ptransform = inference_base.RunInference(
        tfx_bsl_predictions_extractor.TfxBslInferenceWrapper(
            eval_config.model_specs, {'': eval_shared_model}
        ),
        output_batch_size=num_examples,
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
          | 'RunInferenceBase' >> tfx_bsl_inference_ptransform
      )

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          self.assertIn(constants.PREDICTIONS_KEY, got[0])
          self.assertAllClose(
              np.array([[0.2], [0.8], [0.5]]), got[0][constants.PREDICTIONS_KEY]
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result)

  def testIsValidConfigForBulkInferencePass(self):
    saved_model_proto = text_format.Parse(
        """
      saved_model_schema_version: 1
      meta_graphs {
        meta_info_def {
          tags: "serve"
        }
        signature_def: {
          key: "serving_default"
          value: {
            inputs: {
              key: "inputs"
              value {
                dtype: DT_STRING
                name: "input_node:0"
              }
            }
            method_name: "predict"
            outputs: {
              key: "outputs"
              value {
                dtype: DT_FLOAT
                tensor_shape {
                  dim { size: -1 }
                  dim { size: 100 }
                }
              }
            }
          }
        }
      }
      """,
        saved_model_pb2.SavedModel(),
    )
    temp_dir = self.create_tempdir()
    temp_dir.create_file(
        'saved_model.pb', content=saved_model_proto.SerializeToString()
    )
    eval_config = config_pb2.EvalConfig(
        model_specs=[
            config_pb2.ModelSpec(
                name='model_1', signature_name='serving_default'
            )
        ]
    )
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=temp_dir.full_path,
        model_name='model_1',
        tags=[tf.saved_model.SERVING],
        model_type=constants.TF_GENERIC,
    )

    self.assertTrue(
        inference_base.is_valid_config_for_bulk_inference(
            eval_config, eval_shared_model
        )
    )

  def testIsValidConfigForBulkInferencePassDefaultSignatureLookUp(self):
    saved_model_proto = text_format.Parse(
        """
      saved_model_schema_version: 1
      meta_graphs {
        meta_info_def {
          tags: "serve"
        }
        signature_def: {
          key: "serving_default"
          value: {
            inputs: {
              key: "inputs"
              value {
                dtype: DT_STRING
                name: "input_node:0"
              }
            }
            method_name: "predict"
            outputs: {
              key: "outputs"
              value {
                dtype: DT_FLOAT
                tensor_shape {
                  dim { size: -1 }
                  dim { size: 100 }
                }
              }
            }
          }
        }
      }
      """,
        saved_model_pb2.SavedModel(),
    )
    temp_dir = self.create_tempdir()
    temp_dir.create_file(
        'saved_model.pb', content=saved_model_proto.SerializeToString()
    )
    eval_config = config_pb2.EvalConfig(
        model_specs=[config_pb2.ModelSpec(name='model_1')]
    )
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=temp_dir.full_path,
        model_name='model_1',
        tags=[tf.saved_model.SERVING],
        model_type=constants.TF_GENERIC,
    )

    self.assertTrue(
        inference_base.is_valid_config_for_bulk_inference(
            eval_config, eval_shared_model
        )
    )

  def testIsValidConfigForBulkInferenceFailNoSignatureFound(self):
    saved_model_proto = text_format.Parse(
        """
      saved_model_schema_version: 1
      meta_graphs {
        meta_info_def {
          tags: "serve"
        }
        signature_def: {
          key: "serving_default"
          value: {
            inputs: {
              key: "inputs"
              value {
                dtype: DT_STRING
                name: "input_node:0"
              }
            }
            method_name: "predict"
            outputs: {
              key: "outputs"
              value {
                dtype: DT_FLOAT
                tensor_shape {
                  dim { size: -1 }
                  dim { size: 100 }
                }
              }
            }
          }
        }
      }
      """,
        saved_model_pb2.SavedModel(),
    )
    temp_dir = self.create_tempdir()
    temp_dir.create_file(
        'saved_model.pb', content=saved_model_proto.SerializeToString()
    )
    eval_config = config_pb2.EvalConfig(
        model_specs=[
            config_pb2.ModelSpec(name='model_1', signature_name='not_found')
        ]
    )
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=temp_dir.full_path,
        model_name='model_1',
        model_type=constants.TF_GENERIC,
    )
    self.assertFalse(
        inference_base.is_valid_config_for_bulk_inference(
            eval_config, eval_shared_model
        )
    )

  def testIsValidConfigForBulkInferenceFailKerasModel(self):
    saved_model_proto = text_format.Parse(
        """
      saved_model_schema_version: 1
      meta_graphs {
        meta_info_def {
          tags: "serve"
        }
        signature_def: {
          key: "serving_default"
          value: {
            inputs: {
              key: "inputs"
              value {
                dtype: DT_STRING
                name: "input_node:0"
              }
            }
            method_name: "predict"
            outputs: {
              key: "outputs"
              value {
                dtype: DT_FLOAT
                tensor_shape {
                  dim { size: -1 }
                  dim { size: 100 }
                }
              }
            }
          }
        }
      }
      """,
        saved_model_pb2.SavedModel(),
    )
    temp_dir = self.create_tempdir()
    temp_dir.create_file(
        'saved_model.pb', content=saved_model_proto.SerializeToString()
    )
    eval_config = config_pb2.EvalConfig(
        model_specs=[
            config_pb2.ModelSpec(
                name='model_1', signature_name='serving_default'
            )
        ]
    )
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=temp_dir.full_path,
        model_name='model_1',
        model_type=constants.TF_KERAS,
    )
    self.assertFalse(
        inference_base.is_valid_config_for_bulk_inference(
            eval_config, eval_shared_model
        )
    )

  def testIsValidConfigForBulkInferenceFailMoreThanOneInput(self):
    saved_model_proto = text_format.Parse(
        """
      saved_model_schema_version: 1
      meta_graphs {
        meta_info_def {
          tags: "serve"
        }
        signature_def: {
          key: "serving_default"
          value: {
            inputs: {
              key: "inputs"
              value {
                dtype: DT_STRING
                name: "input_node:0"
              }
            }
            method_name: "predict"
            outputs: {
              key: "outputs"
              value {
                dtype: DT_FLOAT
                tensor_shape {
                  dim { size: -1 }
                  dim { size: 100 }
                }
              }
            }
          }
        }
      }
      """,
        saved_model_pb2.SavedModel(),
    )
    temp_dir = self.create_tempdir()
    temp_dir.create_file(
        'saved_model.pb', content=saved_model_proto.SerializeToString()
    )
    eval_config = config_pb2.EvalConfig(
        model_specs=[
            config_pb2.ModelSpec(
                name='model_1', signature_name='serving_default'
            )
        ]
    )
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=temp_dir.full_path,
        model_name='model_1',
        model_type=constants.TF_GENERIC,
    )
    self.assertFalse(
        inference_base.is_valid_config_for_bulk_inference(
            eval_config, eval_shared_model
        )
    )

  def testIsValidConfigForBulkInferenceFailWrongInputType(self):
    saved_model_proto = text_format.Parse(
        """
      saved_model_schema_version: 1
      meta_graphs {
        meta_info_def {
          tags: "serve"
        }
        signature_def: {
          key: "serving_default"
          value: {
            inputs: {
              key: "inputs"
              value {
                dtype: DT_FLOAT
                name: "input_node:0"
              }
            }
            method_name: "predict"
            outputs: {
              key: "outputs"
              value {
                dtype: DT_FLOAT
                tensor_shape {
                  dim { size: -1 }
                  dim { size: 100 }
                }
              }
            }
          }
        }
      }
      """,
        saved_model_pb2.SavedModel(),
    )
    temp_dir = self.create_tempdir()
    temp_dir.create_file(
        'saved_model.pb', content=saved_model_proto.SerializeToString()
    )
    eval_config = config_pb2.EvalConfig(
        model_specs=[
            config_pb2.ModelSpec(
                name='model_1', signature_name='serving_default'
            )
        ]
    )
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=temp_dir.full_path,
        model_name='model_1',
        model_type=constants.TF_GENERIC,
    )
    self.assertFalse(
        inference_base.is_valid_config_for_bulk_inference(
            eval_config, eval_shared_model
        )
    )

  def testInsertSinglePredictionLogIntoExtract(self):
    model_names_to_prediction_logs = {'prediction_log1': self.prediction_log1}
    inference_tuple = ({}, model_names_to_prediction_logs)
    output_extracts = inference_base.insert_predictions_into_extracts(
        inference_tuple=inference_tuple,
        prediction_log_keypath=[constants.PREDICTION_LOG_KEY],
    )

    ref_extracts = {constants.PREDICTION_LOG_KEY: self.prediction_log1}

    self.assertEqual(
        output_extracts[constants.PREDICTION_LOG_KEY],
        ref_extracts[constants.PREDICTION_LOG_KEY],
    )

  def testInsertTwoPredictionLogsIntoExtracts(self):
    model_names_to_prediction_logs = {
        'prediction_log1': self.prediction_log1,
        'prediction_log2': self.prediction_log2,
    }
    inference_tuple = ({}, model_names_to_prediction_logs)
    extracts = inference_base.insert_predictions_into_extracts(
        inference_tuple,
        prediction_log_keypath=[constants.PREDICTION_LOG_KEY],
    )

    ref_extracts = {
        constants.PREDICTION_LOG_KEY: model_names_to_prediction_logs
    }

    self.assertEqual(
        extracts[constants.PREDICTION_LOG_KEY],
        ref_extracts[constants.PREDICTION_LOG_KEY],
    )

  def testInsertPredictionLogsWithCustomPathIntoExtracts(self):
    model_names_to_prediction_logs = {
        'prediction_log1': self.prediction_log1,
        'prediction_log2': self.prediction_log2,
    }
    inference_tuple = ({}, model_names_to_prediction_logs)
    extracts = inference_base.insert_predictions_into_extracts(
        inference_tuple,
        prediction_log_keypath=['foo', 'bar'],
    )

    ref_extracts = {'foo': {'bar': model_names_to_prediction_logs}}
    self.assertEqual(extracts['foo']['bar'], ref_extracts['foo']['bar'])


