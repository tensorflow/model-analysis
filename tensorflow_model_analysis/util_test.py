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
"""Simple tests for util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_model_analysis import util


class UtilTest(tf.test.TestCase):

  def testUniqueKey(self):
    self.assertEqual('key', util.unique_key('key', ['key1', 'key2']))
    self.assertEqual('key1_1', util.unique_key('key1', ['key1', 'key2']))
    self.assertEqual('key1_2', util.unique_key('key1', ['key1', 'key1_1']))

  def testUniqueKeyWithUpdateKeys(self):
    keys = ['key1', 'key2']
    util.unique_key('key1', keys, update_keys=True)
    self.assertEqual(['key1', 'key2', 'key1_1'], keys)

  def testCompoundKey(self):
    self.assertEqual('a_b', util.compound_key(['a_b']))
    self.assertEqual('a__b', util.compound_key(['a', 'b']))
    self.assertEqual('a__b____c__d', util.compound_key(['a', 'b__c', 'd']))

  def testGetByKeys(self):
    self.assertEqual([1], util.get_by_keys({'labels': [1]}, ['labels']))

  def testGetByKeysMissingAndDefault(self):
    self.assertEqual('a', util.get_by_keys({}, ['labels'], default_value='a'))
    self.assertEqual(
        'a', util.get_by_keys({'labels': {}}, ['labels'], default_value='a'))

  def testGetByKeysMissingAndOptional(self):
    self.assertEqual(None, util.get_by_keys({}, ['labels'], optional=True))
    self.assertEqual(
        None, util.get_by_keys({'labels': {}}, ['labels'], optional=True))

  def testGetByKeysMissingAndNonOptional(self):
    with self.assertRaisesRegexp(ValueError, 'not found'):
      util.get_by_keys({}, ['labels'])
    with self.assertRaisesRegexp(ValueError, 'not found'):
      util.get_by_keys({'labels': {}}, ['labels'])

  def testGetByKeysWitMultiLevel(self):
    self.assertEqual([1],
                     util.get_by_keys({'predictions': {
                         'output': [1]
                     }}, ['predictions', 'output']))

    self.assertEqual([1],
                     util.get_by_keys(
                         {'predictions': {
                             'model': {
                                 'output': [1],
                             },
                         }}, ['predictions', 'model', 'output']))

  def testGetByKeysWithPrefix(self):
    self.assertEqual({
        'all_classes': ['a', 'b'],
        'probabilities': [1]
    },
                     util.get_by_keys(
                         {
                             'predictions': {
                                 'output/all_classes': ['a', 'b'],
                                 'output/probabilities': [1],
                             },
                         }, ['predictions', 'output']))
    self.assertEqual({
        'all_classes': ['a', 'b'],
        'probabilities': [1]
    },
                     util.get_by_keys(
                         {
                             'predictions': {
                                 'model': {
                                     'output/all_classes': ['a', 'b'],
                                     'output/probabilities': [1],
                                 },
                             },
                         }, ['predictions', 'model', 'output']))

  def testGetByKeysMissingSecondaryKey(self):
    with self.assertRaisesRegexp(ValueError, 'not found'):
      util.get_by_keys({'predictions': {
          'missing': [1]
      }}, ['predictions', 'output'])

  def testKwargsOnly(self):

    @util.kwargs_only
    def fn(a, b, c, d=None, e=5):
      if d is None:
        d = 100
      if e is None:
        e = 1000
      return a + b + c + d + e

    self.assertEqual(1 + 2 + 3 + 100 + 5, fn(a=1, b=2, c=3))
    self.assertEqual(1 + 2 + 3 + 100 + 1000, fn(a=1, b=2, c=3, e=None))
    with self.assertRaisesRegexp(TypeError, 'keyword-arguments only'):
      fn(1, 2, 3)
    with self.assertRaisesRegexp(TypeError, 'with c specified'):
      fn(a=1, b=2, e=5)  # pylint: disable=no-value-for-parameter
    with self.assertRaisesRegexp(TypeError, 'with extraneous kwargs'):
      fn(a=1, b=2, c=3, f=11)  # pylint: disable=unexpected-keyword-arg


if __name__ == '__main__':
  tf.test.main()
