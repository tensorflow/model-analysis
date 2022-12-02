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

import os

import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
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
from tensorflow_metadata.proto.v0 import schema_pb2


class TfxBslPredictionsExtractorTest(testutil.TensorflowModelAnalysisTest):

  def _getExportDir(self):
    return os.path.join(self._getTempDir(), 'export_dir')

  def _create_tfxio_and_feature_extractor(self,
                                          eval_config: config_pb2.EvalConfig,
                                          schema: schema_pb2.Schema):
    tfx_io = test_util.InMemoryTFExampleRecord(
        schema=schema, raw_record_column_name=constants.ARROW_INPUT_COLUMN)
    tensor_adapter_config = tensor_adapter.TensorAdapterConfig(
        arrow_schema=tfx_io.ArrowSchema(),
        tensor_representations=tfx_io.TensorRepresentations())
    feature_extractor = features_extractor.FeaturesExtractor(
        eval_config=eval_config,
        tensor_representations=tensor_adapter_config.tensor_representations)
    return tfx_io, feature_extractor

  def testRegressionModel(self):
    temp_export_dir = self._getExportDir()
    export_dir, _ = (
        fixed_prediction_estimator_extra_fields
        .simple_fixed_prediction_estimator_extra_fields(temp_export_dir, None))

    eval_config = config_pb2.EvalConfig(model_specs=[config_pb2.ModelSpec()])
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir, tags=[tf.saved_model.SERVING])
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
        """, schema_pb2.Schema()))

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
    num_examples = len(examples)

    tfx_bsl_inference_ptransform = inference_base.RunInference(
        tfx_bsl_predictions_extractor.TfxBslInferenceWrapper(
            eval_config.model_specs, {'': eval_shared_model}),
        batch_size=num_examples)

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([e.SerializeToString() for e in examples],
                                    reshuffle=False)
          | 'BatchExamples' >> tfx_io.BeamSource(batch_size=num_examples)
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | feature_extractor.stage_name >> feature_extractor.ptransform
          | 'RunInferenceBase' >> tfx_bsl_inference_ptransform)

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          self.assertIn(constants.PREDICTIONS_KEY, got[0])
          self.assertAllClose(
              np.array([[0.2], [0.8], [0.5]]),
              got[0][constants.PREDICTIONS_KEY])

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result)

  def testGetEvalSharedModelTwoModelCase(self):
    model_name = 'model_1'
    name_to_eval_shared_model = {
        model_name: types.EvalSharedModel(model_name=model_name),
        'model_2': types.EvalSharedModel(model_name='model_2')
    }
    returned_model = inference_base.get_eval_shared_model(
        model_name, name_to_eval_shared_model)
    self.assertEqual(model_name, returned_model.model_name)

  def testGetEvalSharedModelOneModelCase(self):
    model_name = 'model_1'
    name_to_eval_shared_model = {
        '': types.EvalSharedModel(model_name=model_name)
    }
    returned_model = inference_base.get_eval_shared_model(
        model_name, name_to_eval_shared_model)
    self.assertEqual(model_name, returned_model.model_name)

  def testGetEvalSharedModelRaisesKeyError(self):
    model_name = 'model_1'
    name_to_eval_shared_model = {
        'not_model_1': types.EvalSharedModel(model_name=model_name)
    }
    with self.assertRaises(ValueError):
      inference_base.get_eval_shared_model(model_name,
                                           name_to_eval_shared_model)
