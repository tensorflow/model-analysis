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
"""Test for unbatch extractor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import config
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.extractors import example_weights_extractor
from tensorflow_model_analysis.extractors import features_extractor
from tensorflow_model_analysis.extractors import labels_extractor
from tensorflow_model_analysis.extractors import predictions_extractor
from tensorflow_model_analysis.extractors import unbatch_extractor
from tfx_bsl.tfxio import test_util

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2


class UnbatchExtractorTest(testutil.TensorflowModelAnalysisTest):

  def testUnbatchExtractor(self):
    model_spec = config.ModelSpec(
        label_key='label', example_weight_key='example_weight')
    eval_config = config.EvalConfig(model_specs=[model_spec])
    feature_extractor = features_extractor.FeaturesExtractor(eval_config)
    label_extractor = labels_extractor.LabelsExtractor(eval_config)
    example_weight_extractor = (
        example_weights_extractor.ExampleWeightsExtractor(eval_config))
    predict_extractor = predictions_extractor.PredictionsExtractor(eval_config)
    unbatch_inputs_extractor = unbatch_extractor.UnbatchExtractor()

    schema = text_format.Parse(
        """
        feature {
          name: "label"
          type: FLOAT
        }
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
        """, schema_pb2.Schema())
    tfx_io = test_util.InMemoryTFExampleRecord(
        schema=schema, raw_record_column_name=constants.ARROW_INPUT_COLUMN)
    examples = [
        self._makeExample(
            label=1.0,
            example_weight=0.5,
            fixed_int=1,
            fixed_float=1.0,
            fixed_string='fixed_string1'),
        self._makeExample(
            label=0.0,
            example_weight=0.0,
            fixed_int=1,
            fixed_float=1.0,
            fixed_string='fixed_string2'),
        self._makeExample(
            label=0.0,
            example_weight=1.0,
            fixed_int=2,
            fixed_float=0.0,
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
          | label_extractor.stage_name >> label_extractor.ptransform
          | example_weight_extractor.stage_name >>
          example_weight_extractor.ptransform
          | predict_extractor.stage_name >> predict_extractor.ptransform
          | unbatch_inputs_extractor.stage_name >>
          unbatch_inputs_extractor.ptransform)

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 3)
          self.assertDictElementsAlmostEqual(got[0][constants.FEATURES_KEY], {
              'fixed_int': np.array([1]),
              'fixed_float': np.array([1.0]),
          })
          self.assertEqual(got[0][constants.FEATURES_KEY]['fixed_string'],
                           np.array([b'fixed_string1']))
          self.assertAlmostEqual(got[0][constants.LABELS_KEY], np.array([1.0]))
          self.assertAlmostEqual(got[0][constants.EXAMPLE_WEIGHTS_KEY],
                                 np.array([0.5]))
          self.assertDictElementsAlmostEqual(got[1][constants.FEATURES_KEY], {
              'fixed_int': np.array([1]),
              'fixed_float': np.array([1.0]),
          })
          self.assertEqual(got[1][constants.FEATURES_KEY]['fixed_string'],
                           np.array([b'fixed_string2']))
          self.assertAlmostEqual(got[1][constants.LABELS_KEY], np.array([0.0]))
          self.assertAlmostEqual(got[1][constants.EXAMPLE_WEIGHTS_KEY],
                                 np.array([0.0]))
          self.assertDictElementsAlmostEqual(got[2][constants.FEATURES_KEY], {
              'fixed_int': np.array([2]),
              'fixed_float': np.array([0.0]),
          })
          self.assertEqual(got[2][constants.FEATURES_KEY]['fixed_string'],
                           np.array([b'fixed_string3']))
          self.assertAlmostEqual(got[2][constants.LABELS_KEY], np.array([0.0]))
          self.assertAlmostEqual(got[2][constants.EXAMPLE_WEIGHTS_KEY],
                                 np.array([1.0]))

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testUnbatchExtractorMultiOutput(self):
    model_spec = config.ModelSpec(
        label_keys={
            'output1': 'label1',
            'output2': 'label2'
        },
        example_weight_keys={
            'output1': 'example_weight1',
            'output2': 'example_weight2'
        })
    eval_config = config.EvalConfig(model_specs=[model_spec])
    feature_extractor = features_extractor.FeaturesExtractor(eval_config)
    label_extractor = labels_extractor.LabelsExtractor(eval_config)
    example_weight_extractor = (
        example_weights_extractor.ExampleWeightsExtractor(eval_config))
    predict_extractor = predictions_extractor.PredictionsExtractor(eval_config)
    unbatch_inputs_extractor = unbatch_extractor.UnbatchExtractor()

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
          name: "example_weight1"
          type: FLOAT
        }
        feature {
          name: "example_weight2"
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

    examples = [
        self._makeExample(
            label1=1.0,
            label2=0.0,
            example_weight1=0.5,
            example_weight2=0.5,
            fixed_int=1,
            fixed_float=1.0,
            fixed_string='fixed_string1'),
        self._makeExample(
            label1=1.0,
            label2=1.0,
            example_weight1=0.0,
            example_weight2=1.0,
            fixed_int=1,
            fixed_float=1.0,
            fixed_string='fixed_string2'),
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
          | label_extractor.stage_name >> label_extractor.ptransform
          | example_weight_extractor.stage_name >>
          example_weight_extractor.ptransform
          | predict_extractor.stage_name >> predict_extractor.ptransform
          | unbatch_inputs_extractor.stage_name >>
          unbatch_inputs_extractor.ptransform)

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 2)
          self.assertDictElementsAlmostEqual(got[0][constants.FEATURES_KEY], {
              'fixed_int': np.array([1]),
              'fixed_float': np.array([1.0]),
          })
          self.assertEqual(got[0][constants.FEATURES_KEY]['fixed_string'],
                           np.array([b'fixed_string1']))
          self.assertDictElementsAlmostEqual(got[0][constants.LABELS_KEY], {
              'output1': np.array([1.0]),
              'output2': np.array([0.0])
          })
          self.assertDictElementsAlmostEqual(
              got[0][constants.EXAMPLE_WEIGHTS_KEY], {
                  'output1': np.array([0.5]),
                  'output2': np.array([0.5])
              })
          self.assertDictElementsAlmostEqual(got[1][constants.FEATURES_KEY], {
              'fixed_int': np.array([1]),
              'fixed_float': np.array([1.0]),
          })
          self.assertEqual(got[1][constants.FEATURES_KEY]['fixed_string'],
                           np.array([b'fixed_string2']))
          self.assertDictElementsAlmostEqual(got[1][constants.LABELS_KEY], {
              'output1': np.array([1.0]),
              'output2': np.array([1.0])
          })
          self.assertDictElementsAlmostEqual(
              got[1][constants.EXAMPLE_WEIGHTS_KEY], {
                  'output1': np.array([0.0]),
                  'output2': np.array([1.0])
              })

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testUnbatchExtractorMultiModel(self):
    model_spec1 = config.ModelSpec(
        name='model1',
        label_key='label',
        example_weight_key='example_weight',
        prediction_key='fixed_float')
    model_spec2 = config.ModelSpec(
        name='model2',
        label_keys={
            'output1': 'label1',
            'output2': 'label2'
        },
        example_weight_keys={
            'output1': 'example_weight1',
            'output2': 'example_weight2'
        },
        prediction_keys={
            'output1': 'fixed_float',
            'output2': 'fixed_float'
        })
    eval_config = config.EvalConfig(model_specs=[model_spec1, model_spec2])
    feature_extractor = features_extractor.FeaturesExtractor(eval_config)
    label_extractor = labels_extractor.LabelsExtractor(eval_config)
    example_weight_extractor = (
        example_weights_extractor.ExampleWeightsExtractor(eval_config))
    predict_extractor = predictions_extractor.PredictionsExtractor(eval_config)
    unbatch_inputs_extractor = unbatch_extractor.UnbatchExtractor()

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
          name: "example_weight"
          type: FLOAT
        }
        feature {
          name: "example_weight1"
          type: FLOAT
        }
        feature {
          name: "example_weight2"
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

    examples = [
        self._makeExample(
            label=1.0,
            label1=1.0,
            label2=0.0,
            example_weight=0.5,
            example_weight1=0.5,
            example_weight2=0.5,
            fixed_int=1,
            fixed_float=1.0,
            fixed_string='fixed_string1'),
        self._makeExample(
            label=1.0,
            label1=1.0,
            label2=1.0,
            example_weight=0.0,
            example_weight1=0.0,
            example_weight2=1.0,
            fixed_int=1,
            fixed_float=2.0,
            fixed_string='fixed_string2'),
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
          | label_extractor.stage_name >> label_extractor.ptransform
          | example_weight_extractor.stage_name >>
          example_weight_extractor.ptransform
          | predict_extractor.stage_name >> predict_extractor.ptransform
          | unbatch_inputs_extractor.stage_name >>
          unbatch_inputs_extractor.ptransform)

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 2)
          self.assertDictElementsAlmostEqual(got[0][constants.FEATURES_KEY], {
              'fixed_int': np.array([1]),
          })
          self.assertEqual(got[0][constants.FEATURES_KEY]['fixed_string'],
                           np.array([b'fixed_string1']))
          for model_name in ('model1', 'model2'):
            self.assertIn(model_name, got[0][constants.LABELS_KEY])
            self.assertIn(model_name, got[0][constants.EXAMPLE_WEIGHTS_KEY])
            self.assertIn(model_name, got[0][constants.PREDICTIONS_KEY])
          self.assertAlmostEqual(got[0][constants.LABELS_KEY]['model1'],
                                 np.array([1.0]))
          self.assertDictElementsAlmostEqual(
              got[0][constants.LABELS_KEY]['model2'], {
                  'output1': np.array([1.0]),
                  'output2': np.array([0.0])
              })
          self.assertAlmostEqual(
              got[0][constants.EXAMPLE_WEIGHTS_KEY]['model1'], np.array([0.5]))
          self.assertDictElementsAlmostEqual(
              got[0][constants.EXAMPLE_WEIGHTS_KEY]['model2'], {
                  'output1': np.array([0.5]),
                  'output2': np.array([0.5])
              })
          self.assertAlmostEqual(got[0][constants.PREDICTIONS_KEY]['model1'],
                                 np.array([1.0]))
          self.assertDictElementsAlmostEqual(
              got[0][constants.PREDICTIONS_KEY]['model2'], {
                  'output1': np.array([1.0]),
                  'output2': np.array([1.0])
              })

          self.assertDictElementsAlmostEqual(got[1][constants.FEATURES_KEY], {
              'fixed_int': np.array([1]),
          })
          self.assertEqual(got[1][constants.FEATURES_KEY]['fixed_string'],
                           np.array([b'fixed_string2']))
          for model_name in ('model1', 'model2'):
            self.assertIn(model_name, got[1][constants.LABELS_KEY])
            self.assertIn(model_name, got[1][constants.EXAMPLE_WEIGHTS_KEY])
            self.assertIn(model_name, got[1][constants.PREDICTIONS_KEY])
          self.assertAlmostEqual(got[1][constants.LABELS_KEY]['model1'],
                                 np.array([1.0]))
          self.assertDictElementsAlmostEqual(
              got[1][constants.LABELS_KEY]['model2'], {
                  'output1': np.array([1.0]),
                  'output2': np.array([1.0])
              })
          self.assertAlmostEqual(
              got[1][constants.EXAMPLE_WEIGHTS_KEY]['model1'], np.array([0.0]))
          self.assertDictElementsAlmostEqual(
              got[1][constants.EXAMPLE_WEIGHTS_KEY]['model2'], {
                  'output1': np.array([0.0]),
                  'output2': np.array([1.0])
              })
          self.assertAlmostEqual(got[1][constants.PREDICTIONS_KEY]['model1'],
                                 np.array([2.0]))
          self.assertDictElementsAlmostEqual(
              got[1][constants.PREDICTIONS_KEY]['model2'], {
                  'output1': np.array([2.0]),
                  'output2': np.array([2.0])
              })
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')


if __name__ == '__main__':
  tf.test.main()
