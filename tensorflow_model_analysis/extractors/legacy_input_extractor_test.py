# Lint as: python3
# Copyright 2019 Google LLC
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
"""Tests for input extractor."""
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
from tensorflow_model_analysis.extractors import legacy_input_extractor as input_extractor


class InputExtractorTest(testutil.TensorflowModelAnalysisTest):

  def testInputExtractor(self):
    model_spec = config.ModelSpec(
        label_key='label', example_weight_key='example_weight')
    extractor = input_extractor.InputExtractor(
        eval_config=config.EvalConfig(model_specs=[model_spec]))

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
          | 'InputsToExtracts' >> model_eval_lib.InputsToExtracts()
          | extractor.stage_name >> extractor.ptransform)

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 3)
          self.assertDictElementsAlmostEqual(
              got[0][constants.FEATURES_KEY], {
                  'fixed_int': np.array([1]),
                  'fixed_float': np.array([1.0]),
                  'label': np.array([1.0]),
                  'example_weight': np.array([0.5]),
              })
          self.assertEqual(got[0][constants.FEATURES_KEY]['fixed_string'],
                           np.array([b'fixed_string1']))
          self.assertAlmostEqual(got[0][constants.LABELS_KEY], np.array([1.0]))
          self.assertAlmostEqual(got[0][constants.EXAMPLE_WEIGHTS_KEY],
                                 np.array([0.5]))
          self.assertDictElementsAlmostEqual(
              got[1][constants.FEATURES_KEY], {
                  'fixed_int': np.array([1]),
                  'fixed_float': np.array([1.0]),
                  'label': np.array([0.0]),
                  'example_weight': np.array([0.0]),
              })
          self.assertEqual(got[1][constants.FEATURES_KEY]['fixed_string'],
                           np.array([b'fixed_string2']))
          self.assertAlmostEqual(got[1][constants.LABELS_KEY], np.array([0.0]))
          self.assertAlmostEqual(got[1][constants.EXAMPLE_WEIGHTS_KEY],
                                 np.array([0.0]))
          self.assertDictElementsAlmostEqual(
              got[2][constants.FEATURES_KEY], {
                  'fixed_int': np.array([2]),
                  'fixed_float': np.array([0.0]),
                  'label': np.array([0.0]),
                  'example_weight': np.array([1.0]),
              })
          self.assertEqual(got[2][constants.FEATURES_KEY]['fixed_string'],
                           np.array([b'fixed_string3']))
          self.assertAlmostEqual(got[2][constants.LABELS_KEY], np.array([0.0]))
          self.assertAlmostEqual(got[2][constants.EXAMPLE_WEIGHTS_KEY],
                                 np.array([1.0]))

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testInputExtractorMultiOutput(self):
    model_spec = config.ModelSpec(
        label_keys={
            'output1': 'label1',
            'output2': 'label2'
        },
        example_weight_keys={
            'output1': 'example_weight1',
            'output2': 'example_weight2'
        })
    extractor = input_extractor.InputExtractor(
        eval_config=config.EvalConfig(model_specs=[model_spec]))

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
          | 'InputsToExtracts' >> model_eval_lib.InputsToExtracts()
          | extractor.stage_name >> extractor.ptransform)

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 2)
          self.assertDictElementsAlmostEqual(
              got[0][constants.FEATURES_KEY], {
                  'fixed_int': np.array([1]),
                  'fixed_float': np.array([1.0]),
                  'label1': np.array([1.0]),
                  'label2': np.array([0.0]),
                  'example_weight1': np.array([0.5]),
                  'example_weight2': np.array([0.5]),
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
          self.assertDictElementsAlmostEqual(
              got[1][constants.FEATURES_KEY], {
                  'fixed_int': np.array([1]),
                  'fixed_float': np.array([1.0]),
                  'label1': np.array([1.0]),
                  'label2': np.array([1.0]),
                  'example_weight1': np.array([0.0]),
                  'example_weight2': np.array([1.0]),
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

  def testInputExtractorMultiModel(self):
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
    extractor = input_extractor.InputExtractor(
        eval_config=config.EvalConfig(model_specs=[model_spec1, model_spec2]))

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
          | 'InputsToExtracts' >> model_eval_lib.InputsToExtracts()
          | extractor.stage_name >> extractor.ptransform)

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 2)
          self.assertDictElementsAlmostEqual(
              got[0][constants.FEATURES_KEY], {
                  'fixed_int': np.array([1]),
                  'label': np.array([1.0]),
                  'label1': np.array([1.0]),
                  'label2': np.array([0.0]),
                  'example_weight': np.array([0.5]),
                  'example_weight1': np.array([0.5]),
                  'example_weight2': np.array([0.5]),
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

          self.assertDictElementsAlmostEqual(
              got[1][constants.FEATURES_KEY], {
                  'fixed_int': np.array([1]),
                  'label': np.array([1.0]),
                  'label1': np.array([1.0]),
                  'label2': np.array([1.0]),
                  'example_weight': np.array([0.0]),
                  'example_weight1': np.array([0.0]),
                  'example_weight2': np.array([1.0]),
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
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
