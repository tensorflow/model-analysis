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
"""Slicer test."""

from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import six
import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import types
from tensorflow_model_analysis.extractors import slice_key_extractor
from tensorflow_model_analysis.post_export_metrics import metric_keys
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.proto import metrics_for_slice_pb2
from tensorflow_model_analysis.slicer import slicer_lib as slicer
from tensorflow_model_analysis.utils import test_util
from tensorflow_model_analysis.utils import util as tfma_util

from google.protobuf import text_format


def make_features_dict(features_dict):
  result = {}
  for key, value in features_dict.items():
    result[key] = {'node': np.array(value)}
  return result


def create_fpls():
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
      input_ref=0,
      features=make_features_dict({
          'gender': ['m'],
          'age': [10],
          'interest': ['cars']
      }),
      predictions=make_features_dict({
          'kb': [1],
      }),
      labels=make_features_dict({'ad_risk_score': [0]}))
  return [fpl1, fpl2]


def wrap_fpl(fpl):
  return {
      constants.INPUT_KEY: fpl,
      constants.FEATURES_PREDICTIONS_LABELS_KEY: fpl
  }


class SlicerTest(test_util.TensorflowModelAnalysisTest, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.longMessage = True  # pylint: disable=invalid-name
    beam.typehints.disable_type_annotations()

  def _makeFeaturesDict(self, features_dict):
    result = {}
    for key, value in features_dict.items():
      result[key] = {'node': np.array(value)}
    return result

  def assertSliceResult(self, name, features_dict, columns, features, expected):
    spec = slicer.SingleSliceSpec(columns=columns, features=features)
    msg = 'Test case %s: slice on columns %s, features %s' % (name, columns,
                                                              features)
    six.assertCountEqual(
        self, expected,
        slicer.get_slices_for_features_dicts([features_dict], None, [spec]),
        msg)

  def testDeserializeSliceKey(self):
    slice_metrics = text_format.Parse(
        """
          single_slice_keys {
            column: 'age'
            int64_value: 5
          }
          single_slice_keys {
            column: 'language'
            bytes_value: 'english'
          }
          single_slice_keys {
            column: 'price'
            float_value: 1.0
          }
        """, metrics_for_slice_pb2.SliceKey())

    got_slice_key = slicer.deserialize_slice_key(slice_metrics)
    self.assertCountEqual([('age', 5), ('language', 'english'), ('price', 1.0)],
                          got_slice_key)

  def testDeserializeCrossSliceKey(self):
    slice_metrics = text_format.Parse(
        """
          baseline_slice_key {
            single_slice_keys {
              column: 'age'
              int64_value: 5
            }
            single_slice_keys {
              column: 'language'
              bytes_value: 'english'
            }
            single_slice_keys {
              column: 'price'
              float_value: 1.0
            }
          }
          comparison_slice_key {
            single_slice_keys {
              column: 'age'
              int64_value: 8
            }
            single_slice_keys {
              column: 'language'
              bytes_value: 'hindi'
            }
          }
        """, metrics_for_slice_pb2.CrossSliceKey())

    got_slice_key = slicer.deserialize_cross_slice_key(slice_metrics)
    self.assertCountEqual(
        ((('age', 5), ('language', 'english'), ('price', 1.0)),
         (('age', 8), ('language', 'hindi'))), got_slice_key)

  def testSliceEquality(self):
    overall = slicer.SingleSliceSpec()
    age_column = slicer.SingleSliceSpec(columns=['age'])
    age_feature = slicer.SingleSliceSpec(features=[('age', 5)])
    age_and_gender = slicer.SingleSliceSpec(
        columns=['age'], features=[('gender', 'f')])

    # Note that we construct new instances of the slices to ensure that we
    # aren't just checking object identity.
    def check_equality_and_hash_equality(left, right):
      self.assertEqual(left, right)
      self.assertEqual(hash(left), hash(right))

    check_equality_and_hash_equality(overall, slicer.SingleSliceSpec())
    check_equality_and_hash_equality(age_column,
                                     slicer.SingleSliceSpec(columns=['age']))
    check_equality_and_hash_equality(
        age_feature, slicer.SingleSliceSpec(features=[('age', 5)]))
    check_equality_and_hash_equality(
        age_and_gender,
        slicer.SingleSliceSpec(columns=['age'], features=[('gender', 'f')]))

    self.assertNotEqual(overall, age_column)
    self.assertNotEqual(age_column, age_feature)
    self.assertNotEqual(age_column, age_and_gender)
    self.assertNotEqual(age_feature, age_and_gender)

    self.assertCountEqual([slicer.SingleSliceSpec()], [overall])
    self.assertCountEqual([
        slicer.SingleSliceSpec(columns=['age']),
        slicer.SingleSliceSpec(),
        slicer.SingleSliceSpec(features=[('age', 5)]),
        slicer.SingleSliceSpec(columns=['age'], features=[('gender', 'f')])
    ], [age_and_gender, age_feature, overall, age_column])

  def testNoOverlappingColumns(self):
    self.assertRaises(ValueError, slicer.SingleSliceSpec, ['age'], [('age', 5)])

  def testNonUTF8ValueRaisesValueError(self):
    column_name = 'column_name'
    invalid_value = b'\x8a'
    spec = slicer.SingleSliceSpec(columns=[column_name])
    features_dict = self._makeFeaturesDict({
        column_name: [invalid_value],
    })
    with self.assertRaisesRegex(ValueError, column_name):
      list(slicer.get_slices_for_features_dicts([features_dict], None, [spec]))

  def testGetSlicesForFeaturesDictUnivalent(self):
    test_cases = [
        ('Overall', [], [], [()]),
        ('Feature does not match', [], [('age', 99)], []),
        ('No such column', ['no_such_column'], [], []),
        ('Single column', ['age'], [], [(('age', 5),)]),
        ('Single feature', [], [('age', 5)], [(('age', 5),)]),
        ('Single feature type mismatch', [], [('age', '5')], [(('age', 5),)]),
        ('One column, one feature',
         ['gender'], [('age', 5)], [(('age', 5), ('gender', 'f'))]),
        ('Two features', ['interest', 'gender'], [('age', 5)],
         [(('age', 5), ('gender', 'f'), ('interest', 'cars'))]),
    ]  # pyformat: disable
    features_dict = self._makeFeaturesDict({
        'gender': ['f'],
        'age': [5],
        'interest': ['cars']
    })
    for (name, columns, features, expected) in test_cases:
      self.assertSliceResult(name, features_dict, columns, features, expected)

  def testGetSlicesForFeaturesDictMultivalent(self):
    test_cases = [
        (
            'One column',
            ['fruits'],
            [],
            [
                (('fruits', 'apples'),),
                (('fruits', 'pears'),)
            ],
        ),
        (
            'Two columns',
            ['fruits', 'interests'],
            [],
            [
                (('fruits', 'apples'), ('interests', 'cars')),
                (('fruits', 'apples'), ('interests', 'dogs')),
                (('fruits', 'pears'), ('interests', 'cars')),
                (('fruits', 'pears'), ('interests', 'dogs'))
            ],
        ),
        (
            'One feature',
            [],
            [('interests', 'cars')],
            [
                (('interests', 'cars'),)
            ],
        ),
        (
            'Two features',
            [],
            [('gender', 'f'), ('interests', 'cars')],
            [
                (('gender', 'f'), ('interests', 'cars'))
            ],
        ),
        (
            'One column, one feature',
            ['fruits'],
            [('interests', 'cars')],
            [
                (('fruits', 'apples'), ('interests', 'cars')),
                (('fruits', 'pears'), ('interests', 'cars'))
            ],
        ),
        (
            'One column, two features',
            ['fruits'],
            [('gender', 'f'), ('interests', 'cars')],
            [
                (('fruits', 'apples'), ('gender', 'f'), ('interests', 'cars')),
                (('fruits', 'pears'), ('gender', 'f'), ('interests', 'cars')),
            ],
        ),
        (
            'Two columns, one feature',
            ['interests', 'fruits'], [('gender', 'f')],
            [
                (('fruits', 'apples'), ('gender', 'f'), ('interests', 'cars')),
                (('fruits', 'pears'), ('gender', 'f'), ('interests', 'cars')),
                (('fruits', 'apples'), ('gender', 'f'), ('interests', 'dogs')),
                (('fruits', 'pears'), ('gender', 'f'), ('interests', 'dogs'))
            ],
        ),
        (
            'Two columns, two features',
            ['interests', 'fruits'],
            [('gender', 'f'), ('age', 5)],
            [
                (('age', 5), ('fruits', 'apples'), ('gender', 'f'),
                 ('interests', 'cars')),
                (('age', 5), ('fruits', 'pears'), ('gender', 'f'),
                 ('interests', 'cars')),
                (('age', 5), ('fruits', 'apples'), ('gender', 'f'),
                 ('interests', 'dogs')),
                (('age', 5), ('fruits', 'pears'), ('gender', 'f'),
                 ('interests', 'dogs'))
            ],
        )
    ]  # pyformat: disable

    features_dict = self._makeFeaturesDict({
        'gender': ['f'],
        'age': [5],
        'interests': ['cars', 'dogs'],
        'fruits': ['apples', 'pears']
    })

    for (name, columns, features, expected) in test_cases:
      self.assertSliceResult(name, features_dict, columns, features, expected)

  def testGetSlicesForFeaturesDictMultipleSingleSliceSpecs(self):
    features_dict = self._makeFeaturesDict({
        'gender': ['f'],
        'age': [5],
        'interest': ['cars']
    })

    spec_overall = slicer.SingleSliceSpec()
    spec_age = slicer.SingleSliceSpec(columns=['age'])
    spec_age4 = slicer.SingleSliceSpec(features=[('age', 4)])
    spec_age5_gender = slicer.SingleSliceSpec(
        columns=['gender'], features=[('age', 5)])

    slice_spec = [spec_overall, spec_age, spec_age4, spec_age5_gender]
    expected = [(), (('age', 5),), (('age', 5), ('gender', 'f'))]
    self.assertCountEqual(
        expected,
        slicer.get_slices_for_features_dicts([features_dict], None, slice_spec))

  def testStringifySliceKey(self):
    test_cases = [
        ('overall', (), 'Overall'),
        ('one bytes feature', (('age_str', '5'),), 'age_str:5'),
        ('one int64 feature', (('age', 1),), 'age:1'),
        ('mixed', (('age', 1), ('gender', 'f')), 'age_X_gender:1_X_f'),
        ('more', (('age', 1), ('gender', 'f'), ('interest', 'cars')),
         'age_X_gender_X_interest:1_X_f_X_cars'),
        ('unicode', (('text', b'\xe4\xb8\xad\xe6\x96\x87'),), u'text:\u4e2d\u6587'),
    ]  # pyformat: disable
    for (name, slice_key, stringified_key) in test_cases:
      self.assertEqual(
          stringified_key, slicer.stringify_slice_key(slice_key), msg=name)

  @parameterized.named_parameters(('empty_slice_keys', [], np.array([])),
                                  ('specific_and_overall_slice_key', [
                                      ('f', 1), ()
                                  ], np.array([('f', 1), ()], dtype=object)))
  def testSliceKeysToNumpy(self, slice_keys_tuples, expected_slice_keys_array):
    np.testing.assert_array_equal(
        slicer.slice_keys_to_numpy_array(slice_keys_tuples),
        expected_slice_keys_array)

  def testSliceKeysToNumpyOverall(self):
    actual = slicer.slice_keys_to_numpy_array([()])
    self.assertIsInstance(actual, np.ndarray)
    self.assertEqual(actual.dtype, object)
    self.assertEqual(actual.shape, (1,))
    self.assertEqual(actual[0], ())

  def testIsCrossSliceApplicable(self):
    test_cases = [
        (True, 'overall pass', ((), (('b', 2),)), config_pb2.CrossSlicingSpec(
            baseline_spec=config_pb2.SlicingSpec(),
            slicing_specs=[config_pb2.SlicingSpec(feature_values={'b': '2'})])),
        (True, 'value pass', ((('a', 1),), (('b', 2),)),
         config_pb2.CrossSlicingSpec(
             baseline_spec=config_pb2.SlicingSpec(feature_values={'a': '1'}),
             slicing_specs=[config_pb2.SlicingSpec(feature_values={'b': '2'})])),
        (True, 'baseline key pass', ((('a', 1),), (('b', 2),)),
         config_pb2.CrossSlicingSpec(
             baseline_spec=config_pb2.SlicingSpec(feature_keys=['a']),
             slicing_specs=[config_pb2.SlicingSpec(feature_values={'b': '2'})])),
        (True, 'comparison key pass', ((('a', 1),), (('b', 2),)),
         config_pb2.CrossSlicingSpec(
             baseline_spec=config_pb2.SlicingSpec(feature_values={'a': '1'}),
             slicing_specs=[config_pb2.SlicingSpec(feature_keys=['b'])])),
        (True, 'comparison multiple key pass', ((('a', 1),), (('c', 3),)),
         config_pb2.CrossSlicingSpec(
             baseline_spec=config_pb2.SlicingSpec(feature_values={'a': '1'}),
             slicing_specs=[config_pb2.SlicingSpec(feature_keys=['b']),
                            config_pb2.SlicingSpec(feature_keys=['c'])])),
        (False, 'overall fail', ((('a', 1),), (('b', 2),)),
         config_pb2.CrossSlicingSpec(
             baseline_spec=config_pb2.SlicingSpec(),
             slicing_specs=[config_pb2.SlicingSpec(feature_values={'b': '2'})])),
        (False, 'value fail', ((('a', 1),), (('b', 3),)),
         config_pb2.CrossSlicingSpec(
             baseline_spec=config_pb2.SlicingSpec(feature_values={'a': '1'}),
             slicing_specs=[config_pb2.SlicingSpec(feature_values={'b': '2'})])),
        (False, 'baseline key fail', ((('c', 1),), (('b', 2),)),
         config_pb2.CrossSlicingSpec(
             baseline_spec=config_pb2.SlicingSpec(feature_keys=['a']),
             slicing_specs=[config_pb2.SlicingSpec(feature_values={'b': '2'})])),
        (False, 'comparison key fail', ((('a', 1),), (('c', 3),)),
         config_pb2.CrossSlicingSpec(
             baseline_spec=config_pb2.SlicingSpec(feature_values={'a': '1'}),
             slicing_specs=[config_pb2.SlicingSpec(feature_keys=['b'])])),
        (False, 'comparison multiple key fail', ((('a', 1),), (('d', 3),)),
         config_pb2.CrossSlicingSpec(
             baseline_spec=config_pb2.SlicingSpec(feature_values={'a': '1'}),
             slicing_specs=[config_pb2.SlicingSpec(feature_keys=['b']),
                            config_pb2.SlicingSpec(feature_keys=['c'])])),
    ]  # pyformat: disable
    for (expected_result, name, sliced_key, slicing_spec) in test_cases:
      self.assertEqual(
          expected_result,
          slicer.is_cross_slice_applicable(
              cross_slice_key=sliced_key, cross_slicing_spec=slicing_spec),
          msg=name)

  def testGetSliceKeyType(self):
    test_cases = [
        (slicer.SliceKeyType, 'overall', ()),
        (slicer.SliceKeyType, 'one bytes feature', (('a', '5'),)),
        (slicer.SliceKeyType, 'one int64 feature', (('a', 1),)),
        (slicer.SliceKeyType, 'mixed', (('a', 1), ('b', 'f'))),
        (slicer.SliceKeyType, 'more', (('a', 1), ('b', 'f'), ('c', 'cars'))),
        (slicer.SliceKeyType, 'unicode',
         (('a', b'\xe4\xb8\xad\xe6\x96\x87'),)),
        (slicer.CrossSliceKeyType, 'CrossSlice overall', ((), ())),
        (slicer.CrossSliceKeyType, 'CrossSlice one slice key baseline',
         ((('a', '5'),), ())),
        (slicer.CrossSliceKeyType, 'CrossSlice one slice key comparison',
         ((), (('a', 1),))),
        (slicer.CrossSliceKeyType, 'CrossSlice two simple slice key',
         ((('a', 1),), (('b', 'f'),))),
        (slicer.CrossSliceKeyType, 'CrossSlice two multiple slice key',
         ((('a', 1), ('b', 'f'), ('c', '11')),
          (('a2', 1), ('b', 'm'), ('c', '11')))),
    ]  # pyformat: disable
    for (expected_result, name, slice_key) in test_cases:
      self.assertEqual(
          expected_result, slicer.get_slice_key_type(slice_key), msg=name)

    unrecognized_test_cases = [
        ('Unrecognized 1: ', ('a')),
        ('Unrecognized 2: ', ('a',)),
        ('Unrecognized 3: ', ('a', 1)),
        ('Unrecognized 4: ', (('a'))),
        ('Unrecognized 5: ', (('a',))),
        ('Unrecognized 6: ', ((), (), ())),
        ('Unrecognized 7: ', ((('a', 1),), (('b', 1),), (('c', 1),))),
        ('Unrecognized 8: ', ((('a', 1),), ('b', 1))),
        ('Unrecognized 9: ', (('a', 1), (('b', 1),))),
    ]  # pyformat: disable
    for (name, slice_key) in unrecognized_test_cases:
      with self.assertRaises(TypeError, msg=name + str(slice_key)):
        slicer.get_slice_key_type(slice_key)

  @parameterized.named_parameters(
      {
          'testcase_name': '_single_slice_spec',
          'slice_type': slicer.SingleSliceSpec,
          'slicing_spec': config_pb2.SlicingSpec(feature_values={'a': '1'}),
      }, {
          'testcase_name':
              '_cross_slice_spec',
          'slice_type':
              slicer.CrossSliceSpec,
          'slicing_spec':
              config_pb2.CrossSlicingSpec(
                  baseline_spec=config_pb2.SlicingSpec(),
                  slicing_specs=[
                      config_pb2.SlicingSpec(feature_values={'b': '2'})
                  ]),
      })
  def testDeserializeSliceSpec(self, slice_type, slicing_spec):
    slice_spec = slicer.deserialize_slice_spec(slicing_spec)
    self.assertIsInstance(slice_spec, slice_type)

  def testDeserializeSliceSpec_hashable(self):
    single_slice_spec = slicer.deserialize_slice_spec(
        config_pb2.SlicingSpec(feature_values={'a': '1'}))
    cross_slice_spec = slicer.deserialize_slice_spec(
        slicer.config_pb2.CrossSlicingSpec(
            baseline_spec=config_pb2.SlicingSpec(),
            slicing_specs=[config_pb2.SlicingSpec(feature_values={'b': '2'})]))
    # Check either of them can be hashed and used as keys.
    slice_map = {single_slice_spec: 1, cross_slice_spec: 2}
    self.assertEqual(slice_map[single_slice_spec], 1)
    self.assertEqual(slice_map[cross_slice_spec], 2)

  def testIsSliceApplicable(self):
    test_cases = [
        ('applicable', ['column1'],
         [('column3', 'value3'), ('column4', 'value4')],
         (('column1', 'value1'), ('column3', 'value3'), ('column4', 'value4')),
         True),
        ('wrongcolumns', ['column1', 'column2'],
         [('column3', 'value3'), ('column4', 'value4')],
         (('column1', 'value1'), ('column3', 'value3'), ('column4', 'value4')),
         False),
        ('wrongfeatures', ['column1'], [('column3', 'value3')],
         (('column1', 'value1'), ('column3', 'value3'), ('column4', 'value4')),
         False),
        ('nocolumns', [], [('column3', 'value3')],
         (('column1', 'value1'), ('column3', 'value3'), ('column4', 'value4')),
         False),
        ('nofeatures', ['column1'], [], (('column1', 'value1'),), True),
        ('empty slice key', ['column1'], [('column2', 'value1')], (), False),
        ('overall', [], [], (), True)
    ]  # pyformat: disable

    for (name, columns, features, slice_key, result) in test_cases:
      slice_spec = slicer.SingleSliceSpec(columns=columns, features=features)
      self.assertEqual(
          slice_spec.is_slice_applicable(slice_key), result, msg=name)

  def testSliceDefaultSlice(self):
    with beam.Pipeline() as pipeline:
      fpls = create_fpls()

      metrics = (
          pipeline
          | 'CreateTestInput' >> beam.Create(fpls)
          | 'WrapFpls' >> beam.Map(wrap_fpl)
          | 'ExtractSlices' >> slice_key_extractor.ExtractSliceKeys(
              [slicer.SingleSliceSpec()])
          | 'FanoutSlices' >> slicer.FanoutSlices())

      def check_result(got):
        try:
          self.assertLen(got, 2)
          expected_result = [
              ((), wrap_fpl(fpls[0])),
              ((), wrap_fpl(fpls[1])),
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
          | 'CreateTestInput' >> beam.Create(fpls, reshuffle=False)
          | 'WrapFpls' >> beam.Map(wrap_fpl)
          | 'ExtractSlices' >> slice_key_extractor.ExtractSliceKeys([
              slicer.SingleSliceSpec(),
              slicer.SingleSliceSpec(columns=['gender'])
          ])
          | 'FanoutSlices' >> slicer.FanoutSlices())

      def check_result(got):
        try:
          self.assertLen(got, 4)
          expected_result = [
              ((), wrap_fpl(fpls[0])),
              ((), wrap_fpl(fpls[1])),
              ((('gender', 'f'),), wrap_fpl(fpls[0])),
              ((('gender', 'm'),), wrap_fpl(fpls[1])),
          ]
          self.assertCountEqual(got, expected_result)
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(metrics, check_result)

  def testMultidimSlices(self):
    data = [{
        'features': {
            'gender': [['f'], ['f']],
            'age': [[13], [13]],
            'interest': [['cars'], ['cars']]
        },
        'predictions': [[1], [1]],
        'labels': [[0], [0]],
        constants.SLICE_KEY_TYPES_KEY:
            np.array([
                slicer.slice_keys_to_numpy_array([(), (('gender', 'f'),)]),
                slicer.slice_keys_to_numpy_array([(), (('gender', 'f'),)])
            ])
    }, {
        'features': {
            'gender': [['f'], ['m']],
            'age': [[13], [10]],
            'interest': [['cars'], ['cars']]
        },
        'predictions': [[1], [1]],
        'labels': [[0], [0]],
        constants.SLICE_KEY_TYPES_KEY:
            np.array([
                slicer.slice_keys_to_numpy_array([(), (('gender', 'f'),)]),
                slicer.slice_keys_to_numpy_array([(), (('gender', 'm'),)])
            ])
    }]

    with beam.Pipeline() as pipeline:
      result = (
          pipeline
          | 'CreateTestInput' >> beam.Create(data, reshuffle=False)
          | 'FanoutSlices' >> slicer.FanoutSlices())

      def check_result(got):
        try:
          self.assertLen(got, 5)
          del data[0][constants.SLICE_KEY_TYPES_KEY]
          del data[1][constants.SLICE_KEY_TYPES_KEY]
          expected_result = [
              ((), data[0]),
              ((), data[1]),
              ((('gender', 'f'),), data[0]),
              ((('gender', 'f'),), data[1]),
              ((('gender', 'm'),), data[1]),
          ]
          self.assertCountEqual(got, expected_result)
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result)

  def testMultidimOverallSlices(self):
    data = [
        {
            constants.SLICE_KEY_TYPES_KEY: (  # variable length batch case
                types.VarLenTensorValue.from_dense_rows([
                    slicer.slice_keys_to_numpy_array([(('gender', 'f'),), ()]),
                    slicer.slice_keys_to_numpy_array([()]),
                ])
            )
        },
        {
            constants.SLICE_KEY_TYPES_KEY: np.array([  # fixed length batch case
                slicer.slice_keys_to_numpy_array([()]),
                slicer.slice_keys_to_numpy_array([()]),
            ])
        },
    ]
    data = [tfma_util.StandardExtracts(d) for d in data]
    with beam.Pipeline() as pipeline:
      # Fix the typehint infer error
      beam.typehints.disable_type_annotations()
      result = (
          pipeline
          | 'CreateTestInput'
          >> beam.Create(data, reshuffle=False).with_output_types(
              types.Extracts
          )
          | 'FanoutSlices' >> slicer.FanoutSlices()
      )

      def check_result(got):
        try:
          del data[0][constants.SLICE_KEY_TYPES_KEY]
          del data[1][constants.SLICE_KEY_TYPES_KEY]
          expected_result = [
              ((('gender', 'f'),), data[0]),
              ((), data[0]),
              ((), data[1]),
          ]
          self.assertCountEqual(got, expected_result)
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result)

  def testFilterOutSlices(self):
    slice_key_1 = (('slice_key', 'slice1'),)
    slice_key_2 = (('slice_key', 'slice2'),)
    slice_key_3 = (('slice_key', 'slice3'),)

    values_list = [(slice_key_1, {
        'val11': 'val12'
    }), (slice_key_2, {
        'val21': 'val22'
    })]
    slice_counts_list = [(slice_key_1, 2), (slice_key_2, 1), (slice_key_3, 0)]

    def check_output(got):
      try:
        self.assertLen(got, 2)
        slices = {}
        for (k, v) in got:
          slices[k] = v

        self.assertEqual(slices[slice_key_1], {'val11': 'val12'})
        self.assertIn(metric_keys.ERROR_METRIC, slices[slice_key_2])
      except AssertionError as err:
        raise util.BeamAssertException(err)

    with beam.Pipeline() as pipeline:
      slice_counts_pcoll = (
          pipeline | 'CreateSliceCountsPColl' >> beam.Create(slice_counts_list))
      output_dict = (
          pipeline
          | 'CreateValuesPColl' >> beam.Create(values_list)
          | 'FilterOutSlices' >> slicer.FilterOutSlices(
              slice_counts_pcoll,
              min_slice_size=2,
              error_metric_key=metric_keys.ERROR_METRIC))
      util.assert_that(output_dict, check_output)

  @parameterized.named_parameters(
      {
          'testcase_name': 'matching_single_spec',
          'slice_key': (('f1', 1),),
          'slice_specs': [slicer.SingleSliceSpec(features=[('f1', 1)])],
          'expected_result': True
      },
      {
          'testcase_name': 'matching_single_spec_with_float',
          'slice_key': (('f1', '1.0'),),
          'slice_specs': [slicer.SingleSliceSpec(features=[('f1', '1.0')])],
          'expected_result': True
      },
      {
          'testcase_name': 'non_matching_single_spec',
          'slice_key': (('f1', 1),),
          'slice_specs': [slicer.SingleSliceSpec(columns=['f2'])],
          'expected_result': False
      },
      {
          'testcase_name': 'matching_multiple_specs',
          'slice_key': (('f1', 1),),
          'slice_specs': [
              slicer.SingleSliceSpec(columns=['f1']),
              slicer.SingleSliceSpec(columns=['f2'])
          ],
          'expected_result': True
      },
      {
          'testcase_name': 'empty_specs',
          'slice_key': (('f1', 1),),
          'slice_specs': [],
          'expected_result': False
      },
  )
  def testSliceKeyMatchesSliceSpecs(self, slice_key, slice_specs,
                                    expected_result):
    self.assertEqual(
        expected_result,
        slicer.slice_key_matches_slice_specs(slice_key, slice_specs))


