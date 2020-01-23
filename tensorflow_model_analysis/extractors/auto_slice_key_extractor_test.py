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

# Standard Imports

import tensorflow as tf
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
            }
          }
          features: {
            path { step: 'feature6' }
            type: FLOAT
            num_stats: {
            }
          }
        }
        """, statistics_pb2.DatasetFeatureStatisticsList())
    expected_slice_spec = [
        slicer.SingleSliceSpec(columns=['feature1']),
        slicer.SingleSliceSpec(columns=['feature3']),
        slicer.SingleSliceSpec(columns=['feature1', 'feature3']),
        slicer.SingleSliceSpec()
    ]
    actual_slice_spec = auto_slice_key_extractor.slice_spec_from_stats(stats)
    self.assertEqual(actual_slice_spec, expected_slice_spec)


if __name__ == '__main__':
  tf.test.main()
