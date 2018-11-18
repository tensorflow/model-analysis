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
"""Test for slice_key_extractor."""

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

  def testSliceKeys(self):
    with beam.Pipeline() as pipeline:
      fpls = create_fpls()
      slice_keys_extracts = (
          pipeline
          | 'CreateTestInput' >> beam.Create(fpls)
          | 'WrapFpls' >> beam.Map(wrap_fpl)
          | 'ExtractSlices' >> slice_key_extractor.ExtractSliceKeys([
              slicer.SingleSliceSpec(),
              slicer.SingleSliceSpec(columns=['gender'])
          ]))

      def check_result(got):
        try:
          self.assertEqual(2, len(got), 'got: %s' % got)
          expected_results = sorted([[(), (('gender', 'f'),)],
                                     [(), (('gender', 'm'),)]])
          got_results = []
          for item in got:
            extracts_dict = item.extracts
            self.assertTrue('slice_keys' in extracts_dict)
            got_results.append(sorted(extracts_dict['slice_keys']))
          self.assertEqual(sorted(got_results), sorted(expected_results))
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(slice_keys_extracts, check_result)

  def testMaterializedSliceKeys(self):
    with beam.Pipeline() as pipeline:
      fpls = create_fpls()
      slice_keys_extracts = (
          pipeline
          | 'CreateTestInput' >> beam.Create(fpls)
          | 'WrapFpls' >> beam.Map(wrap_fpl)
          | 'ExtractSlices' >> slice_key_extractor.ExtractSliceKeys(
              [
                  slicer.SingleSliceSpec(),
                  slicer.SingleSliceSpec(columns=['gender'])
              ],
              materialize=True))

      def check_result(got):
        try:
          self.assertEqual(2, len(got), 'got: %s' % got)
          expected_results = sorted([
              types.MaterializedColumn(
                  name='materialized_slice_keys',
                  value=[b'Overall', b'gender:f']),
              types.MaterializedColumn(
                  name='materialized_slice_keys',
                  value=[b'Overall', b'gender:m'])
          ])
          got_results = []
          for item in got:
            extracts_dict = item.extracts
            self.assertTrue('materialized_slice_keys' in extracts_dict)
            got_results.append(extracts_dict['materialized_slice_keys'])
          self.assertEqual(sorted(got_results), sorted(expected_results))
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(slice_keys_extracts, check_result)


if __name__ == '__main__':
  tf.test.main()
