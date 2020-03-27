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
"""Test for auto_slice_key_extractor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.extractors import auto_slice_key_extractor
from tensorflow_model_analysis.slicer import slicer_lib as slicer

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import statistics_pb2


class AutoSliceKeyExtractorTest(testutil.TensorflowModelAnalysisTest):

  def test_slice_spec_from_stats_and_schema(self):
    stats = text_format.Parse(
        """
        datasets {
          features: {
            path { step: 'feature1' }
            type: STRING
            string_stats: {
              unique: 10
            }
          }
          features: {
            path { step: 'feature2' }
            type: STRING
            string_stats: {
              unique: 200
            }
          }
          features: {
            path { step: 'feature3' }
            type: INT
            string_stats: {
              unique: 10
            }
          }
          features: {
            path { step: 'feature4' }
            type: INT
            string_stats: {
              unique: 200
            }
          }
          features: {
            path { step: 'feature5' }
            type: INT
            num_stats: {
              min: 1
              max: 10
            }
          }
          features: {
            path { step: 'feature6' }
            type: FLOAT
            num_stats: {
              min: 1
              max: 10
            }
          }
          features: {
            path { step: 'feature7' }
            type: FLOAT
            num_stats: {
              min: 1
              max: 1
            }
          }
        }
        """, statistics_pb2.DatasetFeatureStatisticsList())
    transformed_feature5 = (
        auto_slice_key_extractor.TRANSFORMED_FEATURE_PREFIX + 'feature5')
    transformed_feature6 = (
        auto_slice_key_extractor.TRANSFORMED_FEATURE_PREFIX + 'feature6')
    expected_slice_spec = [
        slicer.SingleSliceSpec(columns=['feature1']),
        slicer.SingleSliceSpec(columns=['feature3']),
        slicer.SingleSliceSpec(columns=[transformed_feature5]),
        slicer.SingleSliceSpec(columns=[transformed_feature6]),
        slicer.SingleSliceSpec(columns=['feature1', 'feature3']),
        slicer.SingleSliceSpec(columns=['feature1', transformed_feature5]),
        slicer.SingleSliceSpec(columns=['feature1', transformed_feature6]),
        slicer.SingleSliceSpec(columns=['feature3', transformed_feature5]),
        slicer.SingleSliceSpec(columns=['feature3', transformed_feature6]),
        slicer.SingleSliceSpec(
            columns=[transformed_feature5, transformed_feature6]),
        slicer.SingleSliceSpec()
    ]
    actual_slice_spec = auto_slice_key_extractor.slice_spec_from_stats(stats)
    self.assertEqual(actual_slice_spec, expected_slice_spec)

  def test_auto_extract_slice_keys(self):
    features = [
        {
            'gender': np.array(['f']),
            'age': np.array([20])
        },
        {
            'gender': np.array(['m']),
            'age': np.array([45])
        },
        {
            'gender': np.array(['f']),
            'age': np.array([15])
        },
        {
            'gender': np.array(['m']),
            'age': np.array([90])
        },
    ]
    stats = text_format.Parse(
        """
        datasets {
          features: {
            path { step: 'gender' }
            type: STRING
            string_stats: {
              unique: 10
            }
          }
          features: {
            path { step: 'age' }
            type: INT
            num_stats: {
              min: 1
              max: 10
              histograms {
                buckets {
                  low_value:  18
                  high_value: 35
                }
                buckets {
                  low_value:  35
                  high_value: 80
                }
                type: QUANTILES
              }
              histograms {
                buckets {
                  low_value:  18
                  high_value: 80
                }
                type: STANDARD
              }
            }
          }
        }
        """, statistics_pb2.DatasetFeatureStatisticsList())
    transformed_age_feat_name = (
        auto_slice_key_extractor.TRANSFORMED_FEATURE_PREFIX + 'age')
    with beam.Pipeline() as pipeline:
      slice_keys_extracts = (
          pipeline
          | 'CreateTestInput' >> beam.Create(features)
          | 'FeaturesToExtracts' >>
          beam.Map(lambda x: {constants.FEATURES_KEY: x})
          |
          'AutoExtractSlices' >> auto_slice_key_extractor._AutoExtractSliceKeys(
              slice_spec=[
                  slicer.SingleSliceSpec(),
                  slicer.SingleSliceSpec(columns=[transformed_age_feat_name]),
                  slicer.SingleSliceSpec(columns=['gender']),
                  slicer.SingleSliceSpec(
                      columns=['gender', transformed_age_feat_name])
              ],
              statistics=stats))

      def check_result(got):
        try:
          self.assertEqual(4, len(got), 'got: %s' % got)
          expected_results = sorted([
              [(), (('gender', 'f'),),
               (
                   ('gender', 'f'),
                   (transformed_age_feat_name, 1),
               ), ((transformed_age_feat_name, 1),)],
              [(), (('gender', 'm'),),
               (
                   ('gender', 'm'),
                   (transformed_age_feat_name, 2),
               ), ((transformed_age_feat_name, 2),)],
              [(), (('gender', 'f'),),
               (
                   ('gender', 'f'),
                   (transformed_age_feat_name, 0),
               ), ((transformed_age_feat_name, 0),)],
              [(), (('gender', 'm'),),
               (
                   ('gender', 'm'),
                   (transformed_age_feat_name, 3),
               ), ((transformed_age_feat_name, 3),)],
          ])
          got_results = []
          for item in got:
            self.assertIn(constants.SLICE_KEY_TYPES_KEY, item)
            got_results.append(sorted(item[constants.SLICE_KEY_TYPES_KEY]))
          self.assertCountEqual(sorted(got_results), sorted(expected_results))
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(slice_keys_extracts, check_result)


if __name__ == '__main__':
  tf.test.main()
