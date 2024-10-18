# Copyright 2022 Google LLC
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
"""Tests for counterfactual_predictions_extactor."""


import pytest
import os
import tempfile

from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.api import types
from tensorflow_model_analysis.extractors import counterfactual_predictions_extractor
from tensorflow_model_analysis.extractors import features_extractor
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.utils import test_util
from tensorflow_model_analysis.utils.keras_lib import tf_keras
from tfx_bsl.tfxio import tensor_adapter
from tfx_bsl.tfxio import test_util as tfx_bsl_test_util

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2


class IdentityParsingLayer(tf_keras.layers.Layer):
  """A Kears layer which performs parsing and returns a single tensor."""

  def __init__(self, feature_key):
    self._feature_key = feature_key
    super(IdentityParsingLayer, self).__init__(trainable=False)

  def call(self, serialized_example):
    parsed = tf.io.parse_example(
        serialized_example,
        {self._feature_key: tf.io.FixedLenFeature(shape=[], dtype=tf.int64)},
    )
    return parsed[self._feature_key]


@pytest.mark.xfail(run=False, reason="PR 183 This class contains tests that fail and needs to be fixed. "
"If all tests pass, please remove this mark.")
class CounterfactualPredictionsExtactorTest(
    test_util.TensorflowModelAnalysisTest, parameterized.TestCase
):

  def _makeIdentityParsingModel(self):
    """Builds a Keras model that parses and returns a single input tensor."""
    inputs = tf_keras.Input(shape=(), dtype=tf.string)
    outputs = IdentityParsingLayer(feature_key='x')(inputs)
    model = tf_keras.Model(inputs=inputs, outputs=outputs)
    path = os.path.join(tempfile.mkdtemp(), 'export_dir')
    tf.saved_model.save(model, path)
    return path

  @parameterized.named_parameters(
      {
          'testcase_name': 'empty_eval_shared_models',
          'eval_shared_models': [],
          'cf_configs': {},
          'expected_exception_regex': r'requires at least one EvalSharedModel',
      },
      {
          'testcase_name': 'empty_cf_configs',
          'eval_shared_models': [
              types.EvalSharedModel(
                  model_path='', model_type=constants.TF_KERAS
              )
          ],
          'cf_configs': {},
          'expected_exception_regex': r'requires at least one cf_configs',
      },
      {
          'testcase_name': 'unsupported_type',
          'eval_shared_models': [
              types.EvalSharedModel(
                  model_path='', model_type=constants.TF_ESTIMATOR
              )
          ],
          'cf_configs': {},
          'expected_exception_regex': r'found model types.*tf_estimator',
      },
      {
          'testcase_name': 'not_exacty_one_config',
          'eval_shared_models': [
              types.EvalSharedModel(
                  model_path='', model_type=constants.TF_KERAS
              )
          ],
          'cf_configs': {'orig': {}, 'cf': {}},
          'expected_exception_regex': r'one config is expected, but got 2',
      },
      {
          'testcase_name': 'unmatched_configs',
          'eval_shared_models': [
              types.EvalSharedModel(
                  model_name='orig',
                  model_path='',
                  model_type=constants.TF_KERAS,
              ),
              types.EvalSharedModel(
                  model_name='cf', model_path='', model_type=constants.TF_KERAS
              ),
          ],
          'cf_configs': {'orig': {}, 'cf': {}, 'cf1': {}},
          'expected_exception_regex': r'Unmatched configured model names:.*cf1',
      },
  )
  def test_validate_and_update_models_and_configs(
      self, eval_shared_models, cf_configs, expected_exception_regex
  ):
    with self.assertRaisesRegex(ValueError, expected_exception_regex):
      (
          counterfactual_predictions_extractor._validate_and_update_models_and_configs(
              eval_shared_models, cf_configs
          )
      )

  @parameterized.named_parameters(
      {
          'testcase_name': 'single_non_cf_single_cf',
          'eval_shared_model_names': ['orig', 'cf'],
          'model_specs': [
              config_pb2.ModelSpec(
                  name='orig', signature_name='serving_default'
              ),
              config_pb2.ModelSpec(name='cf', signature_name='serving_default'),
          ],
          'cf_configs': {'cf': {'x': 'x_cf1'}},
          'expected_predictions': {
              'orig': np.array([1, 2]),
              'cf': np.array([1, 1]),
          },
      },
      {
          'testcase_name': 'single_cf',
          'eval_shared_model_names': [''],
          'model_specs': [
              config_pb2.ModelSpec(signature_name='serving_default')
          ],
          'cf_configs': {'cf': {'x': 'x_cf1'}},
          'expected_predictions': {'': np.array([1, 1])},
      },
      {
          'testcase_name': 'single_non_cf_multiple_cf',
          'eval_shared_model_names': ['orig', 'cf1', 'cf2'],
          'model_specs': [
              config_pb2.ModelSpec(
                  name='orig', signature_name='serving_default'
              ),
              config_pb2.ModelSpec(
                  name='cf1', signature_name='serving_default'
              ),
              config_pb2.ModelSpec(
                  name='cf2', signature_name='serving_default'
              ),
          ],
          'cf_configs': {'cf1': {'x': 'x_cf1'}, 'cf2': {'x': 'x_cf2'}},
          'expected_predictions': {
              'orig': np.array([1, 2]),
              'cf1': np.array([1, 1]),
              'cf2': np.array([2, 2]),
          },
      },
  )
  def test_cf_predictions_extractor(
      self,
      eval_shared_model_names,
      model_specs,
      cf_configs,
      expected_predictions,
  ):
    model_path = self._makeIdentityParsingModel()
    eval_config = config_pb2.EvalConfig(model_specs=model_specs)
    eval_shared_models = []
    for model_name in eval_shared_model_names:
      eval_shared_models.append(
          model_eval_lib.default_eval_shared_model(
              eval_saved_model_path=model_path,
              tags=[tf.saved_model.SERVING],
              model_name=model_name,
              eval_config=eval_config,
          )
      )
    schema = text_format.Parse(
        """
        feature {
          name: "x"
          type: INT
        }
        feature {
          name: "x_cf1"
          type: INT
        }
        feature {
          name: "x_cf2"
          type: INT
        }
        """,
        schema_pb2.Schema(),
    )

    tfx_io = tfx_bsl_test_util.InMemoryTFExampleRecord(
        schema=schema, raw_record_column_name=constants.ARROW_INPUT_COLUMN
    )
    examples = [
        self._makeExample(x=1, x_cf1=1, x_cf2=2),
        self._makeExample(x=2, x_cf1=1, x_cf2=2),
    ]
    num_examples = len(examples)
    tensor_adapter_config = tensor_adapter.TensorAdapterConfig(
        arrow_schema=tfx_io.ArrowSchema(),
        tensor_representations=tfx_io.TensorRepresentations(),
    )
    feature_extractor = features_extractor.FeaturesExtractor(
        eval_config=eval_config,
        tensor_representations=tensor_adapter_config.tensor_representations,
    )

    cf_predictions_extractor = (
        counterfactual_predictions_extractor.CounterfactualPredictionsExtractor(
            eval_shared_models=eval_shared_models,
            eval_config=eval_config,
            cf_configs=cf_configs,
        )
    )

    with beam.Pipeline() as pipeline:
      result = (
          pipeline
          | 'Create'
          >> beam.Create(
              [e.SerializeToString() for e in examples], reshuffle=False
          )
          | 'BatchExamples' >> tfx_io.BeamSource(batch_size=num_examples)
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | 'ExtractFeatures' >> feature_extractor.ptransform
          | cf_predictions_extractor.stage_name
          >> cf_predictions_extractor.ptransform
      )

      def check_result(got):
        try:
          self.assertLen(got, 1)
          # exact outputs are non-deterministic because model is trained
          # so we can't assert full extracts. Instead, we just assert keys.
          self.assertIn(constants.PREDICTIONS_KEY, got[0])
          np.testing.assert_equal(
              got[0][constants.PREDICTIONS_KEY],
              expected_predictions,
              err_msg=(
                  f'actual:{got[0][constants.PREDICTIONS_KEY]}\n'
                  f'expected:{expected_predictions}'
              ),
          )
          self.assertNotIn(
              counterfactual_predictions_extractor._TEMP_ORIG_INPUT_KEY, got[0]
          )
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')


