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


import pytest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import types
from tensorflow_model_analysis.extractors import slice_key_extractor
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.slicer import slicer_lib as slicer
from tensorflow_model_analysis.utils import test_util


def make_features_dict(features_dict):
  result = {}
  for key, value in features_dict.items():
    result[key] = {'node': np.array(value)}
  return result


def create_fpls():
  fpl1 = types.FeaturesPredictionsLabels(
      input_ref=0,
      features=make_features_dict(
          {'gender': ['f'], 'age': [13], 'interest': ['cars']}
      ),
      predictions=make_features_dict({
          'kb': [1],
      }),
      labels=make_features_dict({'ad_risk_score': [0]}),
  )
  fpl2 = types.FeaturesPredictionsLabels(
      input_ref=0,
      features=make_features_dict(
          {'gender': ['m'], 'age': [10], 'interest': ['cars']}
      ),
      predictions=make_features_dict({
          'kb': [1],
      }),
      labels=make_features_dict({'ad_risk_score': [0]}),
  )
  return [fpl1, fpl2]


def wrap_fpl(fpl):
  return {
      constants.INPUT_KEY: fpl,
      constants.FEATURES_PREDICTIONS_LABELS_KEY: fpl,
  }


@pytest.mark.xfail(run=False, reason="PR 183 This class contains tests that fail and needs to be fixed. "
"If all tests pass, please remove this mark.")
class SliceTest(test_util.TensorflowModelAnalysisTest, parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'features_only',
          [''],
          [
              {
                  constants.FEATURES_KEY: make_features_dict(
                      {'gender': ['m'], 'age': [10], 'interest': ['cars']}
                  )
              },
              {
                  constants.FEATURES_KEY: make_features_dict(
                      {'gender': ['f'], 'age': [12], 'interest': ['cars']}
                  )
              },
          ],
          [slicer.SingleSliceSpec(columns=['gender'])],
          [[(('gender', 'm'),)], [(('gender', 'f'),)]],
      ),
      (
          'duplicate_feature_keys',
          [''],
          [
              {
                  constants.FEATURES_KEY: make_features_dict(
                      {'gender': ['m'], 'age': [10], 'interest': ['cars']}
                  )
              },
              {
                  constants.FEATURES_KEY: make_features_dict(
                      {'gender': ['f'], 'age': [12], 'interest': ['cars']}
                  )
              },
          ],
          [
              slicer.SingleSliceSpec(columns=['gender']),
              slicer.SingleSliceSpec(columns=['gender']),
          ],
          [[(('gender', 'm'),)], [(('gender', 'f'),)]],
      ),
      (
          'transformed_features',
          [''],
          [
              {
                  constants.FEATURES_KEY: make_features_dict(
                      {'gender': ['m'], 'age': [10], 'interest': ['cars']}
                  ),
                  constants.TRANSFORMED_FEATURES_KEY: make_features_dict(
                      {'gender': ['m'], 'age': [10], 'interest': ['boats']}
                  ),
              },
              {
                  constants.FEATURES_KEY: make_features_dict(
                      {'gender': ['f'], 'age': [12], 'interest': ['cars']}
                  ),
                  constants.TRANSFORMED_FEATURES_KEY: make_features_dict(
                      {'gender': ['m'], 'age': [10], 'interest': ['planes']}
                  ),
              },
          ],
          [slicer.SingleSliceSpec(columns=['interest'])],
          [[(('interest', 'boats'),)], [(('interest', 'planes'),)]],
      ),
      (
          'missing_features',
          [''],
          [
              {
                  constants.TRANSFORMED_FEATURES_KEY: make_features_dict(
                      {'gender': ['m'], 'age': [10], 'interest': ['boats']}
                  )
              },
              {
                  constants.TRANSFORMED_FEATURES_KEY: make_features_dict(
                      {'gender': ['m'], 'age': [10], 'interest': ['planes']}
                  )
              },
          ],
          [slicer.SingleSliceSpec(columns=['interest'])],
          [[(('interest', 'boats'),)], [(('interest', 'planes'),)]],
      ),
      (
          'transformed_features_with_multiple_models',
          ['model1', 'model2'],
          [
              {
                  constants.FEATURES_KEY: make_features_dict(
                      {'gender': ['m'], 'age': [10], 'interest': ['cars']}
                  ),
                  constants.TRANSFORMED_FEATURES_KEY: {
                      'model1': make_features_dict({'interest': ['boats']}),
                      'model2': make_features_dict({'interest': ['planes']}),
                  },
              },
              {
                  constants.FEATURES_KEY: make_features_dict(
                      {'gender': ['f'], 'age': [12], 'interest': ['planes']}
                  ),
                  constants.TRANSFORMED_FEATURES_KEY: {
                      'model1': make_features_dict({'interest': ['trains']}),
                      'model2': make_features_dict({'interest': ['planes']}),
                  },
              },
          ],
          [slicer.SingleSliceSpec(columns=['interest'])],
          [
              [(('interest', 'boats'),), (('interest', 'planes'),)],
              [(('interest', 'planes'),), (('interest', 'trains'),)],
          ],
      ),
      (
          'features_with_batched_slices_keys',
          [''],
          [
              {
                  constants.FEATURES_KEY: make_features_dict(
                      {'gender': ['m'], 'age': [10], 'interest': ['cars']}
                  ),
                  constants.SLICE_KEY_TYPES_KEY: [(
                      ('age', '10'),
                      ('interest', 'cars'),
                  )],
              },
              {
                  constants.FEATURES_KEY: make_features_dict(
                      {'gender': ['f'], 'age': [12], 'interest': ['cars']}
                  ),
                  constants.SLICE_KEY_TYPES_KEY: [(
                      ('age', '12'),
                      ('interest', 'cars'),
                  )],
              },
          ],
          [slicer.SingleSliceSpec(columns=['gender'])],
          [
              [
                  (
                      ('age', '10'),
                      ('interest', 'cars'),
                  ),
                  (('gender', 'm'),),
              ],
              [
                  (
                      ('age', '12'),
                      ('interest', 'cars'),
                  ),
                  (('gender', 'f'),),
              ],
          ],
      ),
  )
  def testSliceKeys(self, model_names, extracts, slice_specs, expected_slices):
    eval_config = config_pb2.EvalConfig(
        model_specs=[config_pb2.ModelSpec(name=name) for name in model_names]
    )
    with beam.Pipeline() as pipeline:
      slice_keys_extracts = (
          pipeline
          | 'CreateTestInput' >> beam.Create(extracts)
          | 'ExtractSlices'
          >> slice_key_extractor.ExtractSliceKeys(
              slice_spec=slice_specs, eval_config=eval_config
          )
      )

      def check_result(got):
        try:
          self.assertLen(got, 2)
          got_results = []
          for item in got:
            self.assertIn(constants.SLICE_KEY_TYPES_KEY, item)
            got_results.append(sorted(item[constants.SLICE_KEY_TYPES_KEY]))
          self.assertCountEqual(got_results, expected_slices)
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(slice_keys_extracts, check_result)

  def testLegacySliceKeys(self):
    with beam.Pipeline() as pipeline:
      fpls = create_fpls()
      slice_keys_extracts = (
          pipeline
          | 'CreateTestInput' >> beam.Create(fpls)
          | 'WrapFpls' >> beam.Map(wrap_fpl)
          | 'ExtractSlices'
          >> slice_key_extractor.ExtractSliceKeys([
              slicer.SingleSliceSpec(),
              slicer.SingleSliceSpec(columns=['gender']),
          ])
      )

      def check_result(got):
        try:
          self.assertLen(got, 2)
          expected_results = sorted(
              [[(), (('gender', 'f'),)], [(), (('gender', 'm'),)]]
          )
          got_results = []
          for item in got:
            self.assertIn(constants.SLICE_KEY_TYPES_KEY, item)
            got_results.append(sorted(item[constants.SLICE_KEY_TYPES_KEY]))
          self.assertCountEqual(got_results, expected_results)
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(slice_keys_extracts, check_result)

  def testMaterializedLegacySliceKeys(self):
    with beam.Pipeline() as pipeline:
      fpls = create_fpls()
      slice_keys_extracts = (
          pipeline
          | 'CreateTestInput' >> beam.Create(fpls)
          | 'WrapFpls' >> beam.Map(wrap_fpl)
          | 'ExtractSlices'
          >> slice_key_extractor.ExtractSliceKeys(
              [
                  slicer.SingleSliceSpec(),
                  slicer.SingleSliceSpec(columns=['gender']),
              ],
              materialize=True,
          )
      )

      def check_result(got):
        try:
          self.assertLen(got, 2)
          expected_results = [
              types.MaterializedColumn(
                  name=constants.SLICE_KEYS_KEY, value=[b'Overall', b'gender:f']
              ),
              types.MaterializedColumn(
                  name=constants.SLICE_KEYS_KEY, value=[b'Overall', b'gender:m']
              ),
          ]
          got_results = []
          for item in got:
            self.assertIn(constants.SLICE_KEYS_KEY, item)
            got_result = item[constants.SLICE_KEYS_KEY]
            got_results.append(
                got_result._replace(value=sorted(got_result.value))
            )
          self.assertCountEqual(got_results, expected_results)
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(slice_keys_extracts, check_result)


