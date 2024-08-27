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
"""Test for labels extractor."""


import pytest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.extractors import features_extractor
from tensorflow_model_analysis.extractors import labels_extractor
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.utils import test_util
from tfx_bsl.tfxio import test_util as tfx_bsl_test_util

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2


@pytest.mark.xfail(run=False, reason="PR 183 This class contains tests that fail and needs to be fixed. "
"If all tests pass, please remove this mark.")
class LabelsExtractorTest(
    test_util.TensorflowModelAnalysisTest, parameterized.TestCase
):

  @parameterized.named_parameters(
      ('with_label', 'label'), ('without_label', None)
  )
  def testLabelsExtractor(self, label):
    model_spec = config_pb2.ModelSpec(label_key=label)
    eval_config = config_pb2.EvalConfig(model_specs=[model_spec])
    feature_extractor = features_extractor.FeaturesExtractor(eval_config)
    label_extractor = labels_extractor.LabelsExtractor(eval_config)

    label_feature = ''
    if label is not None:
      label_feature = """
          feature {
            name: "%s"
            type: FLOAT
          }
          """ % label
    schema = text_format.Parse(
        label_feature + """
        feature {
          name: "fixed_int"
          type: INT
        }
        """,
        schema_pb2.Schema(),
    )
    tfx_io = tfx_bsl_test_util.InMemoryTFExampleRecord(
        schema=schema, raw_record_column_name=constants.ARROW_INPUT_COLUMN
    )

    def maybe_add_key(d, key, value):
      if key is not None:
        d[key] = value
      return d

    example_kwargs = [
        maybe_add_key(
            {
                'fixed_int': 1,
            },
            label,
            1.0,
        ),
        maybe_add_key(
            {
                'fixed_int': 1,
            },
            label,
            0.0,
        ),
        maybe_add_key(
            {
                'fixed_int': 2,
            },
            label,
            0.0,
        ),
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
          | 'BatchExamples' >> tfx_io.BeamSource(batch_size=3)
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | feature_extractor.stage_name >> feature_extractor.ptransform
          | label_extractor.stage_name >> label_extractor.ptransform
      )

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          if label is None:
            self.assertIsNone(got[0][constants.LABELS_KEY])
          else:
            self.assertAllClose(
                got[0][constants.LABELS_KEY], np.array([[1.0], [0.0], [0.0]])
            )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testLabelsExtractorMultiOutput(self):
    model_spec = config_pb2.ModelSpec(
        label_keys={
            'output1': 'label1',
            'output2': 'label2',
            'output3': 'label3',
        }
    )
    eval_config = config_pb2.EvalConfig(model_specs=[model_spec])
    feature_extractor = features_extractor.FeaturesExtractor(eval_config)
    label_extractor = labels_extractor.LabelsExtractor(eval_config)

    schema = text_format.Parse(
        """
        feature {
          name: "label1"
          type: FLOAT
        }
        feature {
          name: "label2"
          type: FLOAT
        }
        feature {
          name: "fixed_int"
          type: INT
        }
        """,
        schema_pb2.Schema(),
    )
    tfx_io = tfx_bsl_test_util.InMemoryTFExampleRecord(
        schema=schema, raw_record_column_name=constants.ARROW_INPUT_COLUMN
    )

    examples = [
        self._makeExample(label1=1.0, label2=0.0, fixed_int=1),
        self._makeExample(label1=1.0, label2=1.0, fixed_int=1),
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
          | label_extractor.stage_name >> label_extractor.ptransform
      )

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          # None cannot be compared with assertAllClose
          self.assertIn('output3', got[0][constants.LABELS_KEY])
          self.assertIsNone(got[0][constants.LABELS_KEY]['output3'])
          del got[0][constants.LABELS_KEY]['output3']
          self.assertAllClose(
              got[0][constants.LABELS_KEY],
              {
                  'output1': np.array([[1.0], [1.0]]),
                  'output2': np.array([[0.0], [1.0]]),
              },
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testLabelsExtractorMultiModel(self):
    model_spec1 = config_pb2.ModelSpec(name='model1', label_key='label')
    model_spec2 = config_pb2.ModelSpec(
        name='model2', label_keys={'output1': 'label1', 'output2': 'label2'}
    )
    eval_config = config_pb2.EvalConfig(model_specs=[model_spec1, model_spec2])
    feature_extractor = features_extractor.FeaturesExtractor(eval_config)
    label_extractor = labels_extractor.LabelsExtractor(eval_config)

    schema = text_format.Parse(
        """
        feature {
          name: "label"
          type: FLOAT
        }
        feature {
          name: "label1"
          type: FLOAT
        }
        feature {
          name: "label2"
          type: FLOAT
        }
        feature {
          name: "fixed_int"
          type: INT
        }
        """,
        schema_pb2.Schema(),
    )
    tfx_io = tfx_bsl_test_util.InMemoryTFExampleRecord(
        schema=schema, raw_record_column_name=constants.ARROW_INPUT_COLUMN
    )

    examples = [
        self._makeExample(label=1.0, label1=1.0, label2=0.0, fixed_int=1),
        self._makeExample(label=1.0, label1=1.0, label2=1.0, fixed_int=1),
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
          | label_extractor.stage_name >> label_extractor.ptransform
      )

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          for model_name in ('model1', 'model2'):
            self.assertIn(model_name, got[0][constants.LABELS_KEY])
          self.assertAllClose(
              got[0][constants.LABELS_KEY]['model1'], np.array([[1.0], [1.0]])
          )
          self.assertAllClose(
              got[0][constants.LABELS_KEY]['model2'],
              {
                  'output1': np.array([[1.0], [1.0]]),
                  'output2': np.array([[0.0], [1.0]]),
              },
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')


