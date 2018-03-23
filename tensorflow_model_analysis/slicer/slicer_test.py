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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import six

import tensorflow as tf
from tensorflow_model_analysis.slicer import slicer


class SlicerTest(tf.test.TestCase):

  def setUp(self):
    self.longMessage = True

  def _makeFeaturesDict(self, features_dict):
    result = {}
    for key, value in features_dict.items():
      result[key] = {'node': np.array(value)}
    return result

  def assertSliceResult(self, name, features_dict, columns, features, expected):
    spec = slicer.SingleSliceSpec(columns=columns, features=features)
    msg = 'Test case %s: slice on columns %s, features %s' % (name, columns,
                                                              features)
    six.assertCountEqual(self, expected,
                         slicer.get_slices_for_features_dict(
                             features_dict, [spec]), msg)

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
    check_equality_and_hash_equality(
        age_column, slicer.SingleSliceSpec(columns=['age']))
    check_equality_and_hash_equality(
        age_feature, slicer.SingleSliceSpec(features=[('age', 5)]))
    check_equality_and_hash_equality(age_and_gender,
                                     slicer.SingleSliceSpec(
                                         columns=['age'],
                                         features=[('gender', 'f')]))

    self.assertNotEqual(overall, age_column)
    self.assertNotEqual(age_column, age_feature)
    self.assertNotEqual(age_column, age_and_gender)
    self.assertNotEqual(age_feature, age_and_gender)

    self.assertItemsEqual([slicer.SingleSliceSpec()], [overall])
    self.assertItemsEqual(
        [
            slicer.SingleSliceSpec(columns=['age']),
            slicer.SingleSliceSpec(),
            slicer.SingleSliceSpec(features=[('age', 5)]),
            slicer.SingleSliceSpec(columns=['age'], features=[('gender', 'f')])
        ], [age_and_gender, age_feature, overall, age_column])

  def testNoOverlappingColumns(self):
    self.assertRaises(ValueError, slicer.SingleSliceSpec, ['age'], [('age', 5)])

  def testGetSlicesForFeaturesDictUnivalent(self):
    test_cases = [
        ('Overall', [], [], [()]),
        ('Feature does not match', [], [('age', 99)], []),
        ('No such column', ['no_such_column'], [], []),
        ('Single column', ['age'], [], [(('age', 5),)]),
        ('Single feature', [], [('age', 5)], [(('age', 5),)]),
        ('Single feature type mismatch', [], [('age', '5')], []),
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
    self.assertItemsEqual(expected,
                          slicer.get_slices_for_features_dict(
                              features_dict, slice_spec))

  def testStringifySliceKey(self):
    test_cases = [
        ('overall', (), 'Overall'),
        ('one bytes feature', (('age_str', '5'),), 'age_str:5'),
        ('one int64 feature', (('age', 1),), 'age:1'),
        ('mixed', (('age', 1), ('gender', 'f')), 'age_X_gender:1_X_f'),
        ('more', (('age', 1), ('gender', 'f'), ('interest', 'cars')),
         'age_X_gender_X_interest:1_X_f_X_cars')
    ]  # pyformat: disable
    for (name, slice_key, stringified_key) in test_cases:
      self.assertEqual(
          stringified_key, slicer.stringify_slice_key(slice_key), msg=name)

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


if __name__ == '__main__':
  tf.test.main()
