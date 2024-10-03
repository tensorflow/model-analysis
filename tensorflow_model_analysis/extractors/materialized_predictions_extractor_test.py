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
"""Test for batched materialized predictions extractor."""


import pytest
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.extractors import features_extractor
from tensorflow_model_analysis.extractors import materialized_predictions_extractor
from tensorflow_model_analysis.proto import config_pb2
from tfx_bsl.tfxio import tensor_adapter
from tfx_bsl.tfxio import test_util

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2



@pytest.mark.xfail(run=False, reason="PR 183 This class contains tests that fail and needs to be fixed. "
"If all tests pass, please remove this mark.")
class MaterializedPredictionsExtractorTest(
    testutil.TensorflowModelAnalysisTest
):

  def test_rekey_predictions_in_features(self):
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
    prediction_extractor = (
        materialized_predictions_extractor.MaterializedPredictionsExtractor(
            eval_config
        )
    )

    examples = [
        self._makeExample(
            prediction=1.0, prediction1=1.0, prediction2=0.0, fixed_int=1
        ),
        self._makeExample(
            prediction=1.0, prediction1=1.0, prediction2=1.0, fixed_int=1
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
          | prediction_extractor.stage_name >> prediction_extractor.ptransform
      )

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          for model_name in ('model1', 'model2'):
            self.assertIn(model_name, got[0][constants.PREDICTIONS_KEY])
          self.assertAllClose(
              got[0][constants.PREDICTIONS_KEY]['model1'], np.array([1.0, 1.0])
          )
          self.assertAllClose(
              got[0][constants.PREDICTIONS_KEY]['model2'],
              {
                  'output1': np.array([1.0, 1.0]),
                  'output2': np.array([0.0, 1.0]),
              },
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')


