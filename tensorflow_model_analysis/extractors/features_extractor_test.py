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
"""Test for features extractor."""

from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.extractors import features_extractor
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.utils import test_util
from tfx_bsl.tfxio import tf_example_record

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2


class FeaturesExtractorTest(
    test_util.TensorflowModelAnalysisTest, parameterized.TestCase
):

  def test_features_extractor_no_features(self):
    model_spec = config_pb2.ModelSpec()
    eval_config = config_pb2.EvalConfig(model_specs=[model_spec])
    feature_extractor = features_extractor.FeaturesExtractor(eval_config)
    tfx_io = tf_example_record.TFExampleBeamRecord(
        raw_record_column_name=constants.ARROW_INPUT_COLUMN,
        physical_format='inmem',
        telemetry_descriptors=['testing'],
    )

    with beam.Pipeline() as pipeline:
      result = (
          pipeline
          | 'Create' >> beam.Create([b''] * 3)
          | 'DecodeToRecordBatch' >> tfx_io.BeamSource(batch_size=3)
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | feature_extractor.stage_name >> feature_extractor.ptransform
      )

      def check_result(got):
        self.assertLen(got, 1)
        self.assertIn(constants.FEATURES_KEY, got[0])
        self.assertEmpty(got[0][constants.FEATURES_KEY])
        self.assertIn(constants.INPUT_KEY, got[0])
        self.assertLen(got[0][constants.INPUT_KEY], 3)

      util.assert_that(result, check_result, label='CheckResult')

  def test_features_extractor(self):
    model_spec = config_pb2.ModelSpec()
    eval_config = config_pb2.EvalConfig(model_specs=[model_spec])
    feature_extractor = features_extractor.FeaturesExtractor(eval_config)

    schema = text_format.Parse(
        """
        feature {
          name: "example_weight"
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
    )
    tfx_io = tf_example_record.TFExampleBeamRecord(
        schema=schema,
        raw_record_column_name=constants.ARROW_INPUT_COLUMN,
        physical_format='inmem',
        telemetry_descriptors=['testing'],
    )

    example_kwargs = [
        {'fixed_int': 1, 'fixed_float': 1.0, 'fixed_string': 'fixed_string1'},
        {'fixed_int': 1, 'fixed_float': 1.0, 'fixed_string': 'fixed_string2'},
        {'fixed_int': 2, 'fixed_float': 0.0, 'fixed_string': 'fixed_string3'},
    ]

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create'
          >> beam.Create(
              [
                  self._makeExample(**kwargs).SerializeToString()
                  for kwargs in example_kwargs
              ],
              reshuffle=False,
          )
          | 'DecodeToRecordBatch' >> tfx_io.BeamSource(batch_size=3)
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | feature_extractor.stage_name >> feature_extractor.ptransform
      )

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          self.assertIn(constants.FEATURES_KEY, got[0])
          self.assertLen(got[0][constants.FEATURES_KEY], 4)  # 4 features
          self.assertIn('example_weight', got[0][constants.FEATURES_KEY])
          # Arrays of type np.object won't compare with assertAllClose
          self.assertEqual(
              got[0][constants.FEATURES_KEY]['example_weight'].tolist(),
              [None, None, None],
          )
          self.assertIn('fixed_int', got[0][constants.FEATURES_KEY])
          self.assertAllClose(
              got[0][constants.FEATURES_KEY]['fixed_int'],
              np.array([[1], [1], [2]]),
          )
          self.assertIn('fixed_float', got[0][constants.FEATURES_KEY])
          self.assertAllClose(
              got[0][constants.FEATURES_KEY]['fixed_float'],
              np.array([[1.0], [1.0], [0.0]]),
          )
          self.assertIn('fixed_string', got[0][constants.FEATURES_KEY])
          # Arrays of type np.object won't compare with assertAllClose
          self.assertEqual(
              got[0][constants.FEATURES_KEY]['fixed_string'].tolist(),
              [[b'fixed_string1'], [b'fixed_string2'], [b'fixed_string3']],
          )
          self.assertIn(constants.INPUT_KEY, got[0])
          self.assertLen(got[0][constants.INPUT_KEY], 3)  # 3 examples

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')


