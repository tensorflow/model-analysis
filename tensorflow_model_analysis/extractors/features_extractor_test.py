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
"""Test for features extractor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import config
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.extractors import features_extractor
from tfx_bsl.tfxio import test_util

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2


class FeaturesExtractorTest(testutil.TensorflowModelAnalysisTest,
                            parameterized.TestCase):

  def testFeaturesExtractor(self):
    model_spec = config.ModelSpec()
    eval_config = config.EvalConfig(model_specs=[model_spec])
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
        """, schema_pb2.Schema())
    tfx_io = test_util.InMemoryTFExampleRecord(
        schema=schema, raw_record_column_name=constants.ARROW_INPUT_COLUMN)

    example_kwargs = [
        {
            'fixed_int': 1,
            'fixed_float': 1.0,
            'fixed_string': 'fixed_string1'
        },
        {
            'fixed_int': 1,
            'fixed_float': 1.0,
            'fixed_string': 'fixed_string2'
        },
        {
            'fixed_int': 2,
            'fixed_float': 0.0,
            'fixed_string': 'fixed_string3'
        },
    ]

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([
              self._makeExample(**kwargs).SerializeToString()
              for kwargs in example_kwargs
          ],
                                    reshuffle=False)
          | 'BatchExamples' >> tfx_io.BeamSource(batch_size=3)
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | feature_extractor.stage_name >> feature_extractor.ptransform)

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          self.assertDictElementsAlmostEqual(got[0][constants.FEATURES_KEY][0],
                                             {
                                                 'fixed_int': np.array([1]),
                                                 'fixed_float': np.array([1.0]),
                                             })
          self.assertEqual(got[0][constants.FEATURES_KEY][0]['fixed_string'],
                           np.array([b'fixed_string1']))
          self.assertDictElementsAlmostEqual(got[0][constants.FEATURES_KEY][1],
                                             {
                                                 'fixed_int': np.array([1]),
                                                 'fixed_float': np.array([1.0]),
                                             })
          self.assertEqual(got[0][constants.FEATURES_KEY][1]['fixed_string'],
                           np.array([b'fixed_string2']))
          self.assertDictElementsAlmostEqual(got[0][constants.FEATURES_KEY][2],
                                             {
                                                 'fixed_int': np.array([2]),
                                                 'fixed_float': np.array([0.0]),
                                             })
          self.assertEqual(got[0][constants.FEATURES_KEY][2]['fixed_string'],
                           np.array([b'fixed_string3']))

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')


if __name__ == '__main__':
  tf.test.main()
