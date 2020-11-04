# Lint as: python3
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
"""Test for using the MetaFeatureExtractor as part of TFMA."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.extractors import legacy_meta_feature_extractor as meta_feature_extractor
from tensorflow_model_analysis.extractors import slice_key_extractor
from tensorflow_model_analysis.slicer import slicer_lib as slicer


def make_features_dict(features_dict):
  result = {}
  for key, value in features_dict.items():
    result[key] = {'node': np.array(value)}
  return result


def create_fpls():
  """Create test FPL dicts that can be used for verification."""
  fpl1 = types.FeaturesPredictionsLabels(
      input_ref=0,
      features=make_features_dict({
          'gender': ['f'],
          'age': [13],
          'interest': ['cars']
      }),
      predictions=make_features_dict({
          'kb': [1],
      }),
      labels=make_features_dict({'ad_risk_score': [0]}))
  fpl2 = types.FeaturesPredictionsLabels(
      input_ref=1,
      features=make_features_dict({
          'gender': ['m'],
          'age': [10],
          'interest': ['cars', 'movies']
      }),
      predictions=make_features_dict({
          'kb': [1],
      }),
      labels=make_features_dict({'ad_risk_score': [0]}))
  return [fpl1, fpl2]


def wrap_fpl(fpl):
  return {
      constants.INPUT_KEY: 'xyz',
      constants.FEATURES_PREDICTIONS_LABELS_KEY: fpl
  }


def get_num_interests(fpl):
  interests = meta_feature_extractor.get_feature_value(fpl, 'interest')
  new_features = {'num_interests': len(interests)}
  return new_features


class MetaFeatureExtractorTest(testutil.TensorflowModelAnalysisTest):

  def testMetaFeatures(self):
    with beam.Pipeline() as pipeline:
      fpls = create_fpls()

      metrics = (
          pipeline
          | 'CreateTestInput' >> beam.Create(fpls)
          | 'WrapFpls' >> beam.Map(wrap_fpl)
          | 'ExtractInterestsNum' >>
          meta_feature_extractor.ExtractMetaFeature(get_num_interests))

      def check_result(got):
        try:
          self.assertEqual(2, len(got), 'got: %s' % got)
          for res in got:
            self.assertIn(
                'num_interests',
                res[constants.FEATURES_PREDICTIONS_LABELS_KEY].features)
            self.assertEqual(
                len(
                    meta_feature_extractor.get_feature_value(
                        res[constants.FEATURES_PREDICTIONS_LABELS_KEY],
                        'interest')),
                meta_feature_extractor.get_feature_value(
                    res[constants.FEATURES_PREDICTIONS_LABELS_KEY],
                    'num_interests'))
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(metrics, check_result)

  def testNoModificationOfExistingKeys(self):

    def bad_meta_feature_fn(_):
      return {'interest': ['bad', 'key']}

    with self.assertRaises(ValueError):
      with beam.Pipeline() as pipeline:
        fpls = create_fpls()

        _ = (
            pipeline
            | 'CreateTestInput' >> beam.Create(fpls)
            | 'WrapFpls' >> beam.Map(wrap_fpl)
            | 'ExtractInterestsNum' >>
            meta_feature_extractor.ExtractMetaFeature(bad_meta_feature_fn))

  def testSliceOnMetaFeature(self):
    # We want to make sure that slicing on the newly added feature works, so
    # pulling in slice here.
    with beam.Pipeline() as pipeline:
      fpls = create_fpls()
      metrics = (
          pipeline
          | 'CreateTestInput' >> beam.Create(fpls)
          | 'WrapFpls' >> beam.Map(wrap_fpl)
          | 'ExtractInterestsNum' >>
          meta_feature_extractor.ExtractMetaFeature(get_num_interests)
          | 'ExtractSlices' >> slice_key_extractor.ExtractSliceKeys([
              slicer.SingleSliceSpec(),
              slicer.SingleSliceSpec(columns=['num_interests'])
          ])
          | 'FanoutSlices' >> slicer.FanoutSlices())

      def check_result(got):
        try:
          self.assertEqual(4, len(got), 'got: %s' % got)
          expected_slice_keys = [
              (),
              (),
              (('num_interests', 1),),
              (('num_interests', 2),),
          ]
          self.assertCountEqual(
              sorted(slice_key for slice_key, _ in got),
              sorted(expected_slice_keys))
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(metrics, check_result)

  def testGetSparseTensorValue(self):
    sparse_tensor_value = tf.compat.v1.SparseTensorValue(
        indices=[[0, 0, 0], [0, 1, 0], [0, 1, 1]],
        values=['', 'one', 'two'],
        dense_shape=[1, 2, 2])
    fpl_with_sparse_tensor = types.FeaturesPredictionsLabels(
        input_ref=0, features={}, predictions={}, labels={})

    meta_feature_extractor._set_feature_value(fpl_with_sparse_tensor.features,
                                              'sparse', sparse_tensor_value)
    self.assertEqual(['', 'one', 'two'],
                     meta_feature_extractor.get_feature_value(
                         fpl_with_sparse_tensor, 'sparse'))


if __name__ == '__main__':
  tf.test.main()
