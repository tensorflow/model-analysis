# Copyright 2018 Google LLC
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
"""Test for using the Evaluate API.

Note that we actually train and export models within these tests.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.api.impl import api_types
from tensorflow_model_analysis.api.impl import slice as slice_api
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.extractors import slice_key_extractor
from tensorflow_model_analysis.slicer import slicer


def make_features_dict(features_dict):
  result = {}
  for key, value in features_dict.items():
    result[key] = {'node': np.array(value)}
  return result


def create_fpls():
  fpl1 = api_types.FeaturesPredictionsLabels(
      example_ref=0,
      features=make_features_dict({
          'gender': ['f'],
          'age': [13],
          'interest': ['cars']
      }),
      predictions=make_features_dict({
          'kb': [1],
      }),
      labels=make_features_dict({
          'ad_risk_score': [0]
      }))
  fpl2 = api_types.FeaturesPredictionsLabels(
      example_ref=0,
      features=make_features_dict({
          'gender': ['m'],
          'age': [10],
          'interest': ['cars']
      }),
      predictions=make_features_dict({
          'kb': [1],
      }),
      labels=make_features_dict({
          'ad_risk_score': [0]
      }))
  return [fpl1, fpl2]


def wrap_fpl(fpl):
  return types.ExampleAndExtracts(
      example=fpl, extracts={constants.FEATURES_PREDICTIONS_LABELS_KEY: fpl})


class SliceTest(testutil.TensorflowModelAnalysisTest):

  def testSliceDefaultSlice(self):
    with beam.Pipeline() as pipeline:
      fpls = create_fpls()

      metrics = (
          pipeline
          | 'CreateTestInput' >> beam.Create(fpls)
          | 'WrapFpls' >> beam.Map(wrap_fpl)
          | 'ExtractSlices' >> slice_key_extractor.ExtractSliceKeys(
              [slicer.SingleSliceSpec()])
          | 'FanoutSlices' >> slice_api.FanoutSlices())

      def check_result(got):
        try:
          self.assertEqual(2, len(got), 'got: %s' % got)
          expected_result = [
              ((), fpls[0]),
              ((), fpls[1]),
          ]
          self.assertEqual(len(got), len(expected_result))
          self.assertTrue(
              got[0] == expected_result[0] and got[1] == expected_result[1] or
              got[1] == expected_result[0] and got[0] == expected_result[1])
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(metrics, check_result)

  def testSliceOneSlice(self):
    with beam.Pipeline() as pipeline:
      fpls = create_fpls()
      metrics = (
          pipeline
          | 'CreateTestInput' >> beam.Create(fpls)
          | 'WrapFpls' >> beam.Map(wrap_fpl)
          | 'ExtractSlices' >> slice_key_extractor.ExtractSliceKeys([
              slicer.SingleSliceSpec(),
              slicer.SingleSliceSpec(columns=['gender'])
          ])
          | 'FanoutSlices' >> slice_api.FanoutSlices())

      def check_result(got):
        try:
          self.assertEqual(4, len(got), 'got: %s' % got)
          expected_result = [
              ((), fpls[0]),
              ((), fpls[1]),
              ((('gender', 'f'),), fpls[0]),
              ((('gender', 'm'),), fpls[1]),
          ]
          self.assertEqual(
              sorted(got, key=lambda x: x[0]),
              sorted(expected_result, key=lambda x: x[0]))
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(metrics, check_result)


if __name__ == '__main__':
  tf.test.main()
