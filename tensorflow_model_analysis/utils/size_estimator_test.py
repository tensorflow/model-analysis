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
"""Tests for size estimator."""

import tensorflow as tf
from tensorflow_model_analysis.utils import size_estimator


class SizeEstimatorTest(tf.test.TestCase):

  def testRefCountAmortization(self):
    estimator = size_estimator.SizeEstimator(size_threshold=10, size_fn=len)
    self.assertEqual(estimator.get_estimate(), 0)
    a = b'fasjg'
    b, c = a, a
    # The test string should not use sys reference count, which may lead to
    # unexpected string reference increase/decrease.
    expected_size_estimate = 4
    estimator.update(a)
    estimator.update(b)
    estimator.update(c)
    estimator.update(a)
    self.assertEqual(estimator.get_estimate(), expected_size_estimate)

    self.assertFalse(estimator.should_flush())

  def testFlush(self):
    estimator = size_estimator.SizeEstimator(size_threshold=10, size_fn=len)
    self.assertEqual(estimator.get_estimate(), 0)
    estimator.update(b'plmjh')
    estimator.update(b'plmjhghytfghsggssss')
    self.assertTrue(estimator.should_flush())
    estimator.clear()
    self.assertEqual(estimator.get_estimate(), 0)

  def testMergeEstimators(self):
    estimator1 = size_estimator.SizeEstimator(size_threshold=10, size_fn=len)
    self.assertEqual(estimator1.get_estimate(), 0)
    estimator2 = size_estimator.SizeEstimator(size_threshold=10, size_fn=len)
    self.assertEqual(estimator2.get_estimate(), 0)
    a = b'pkmiz'
    b, c = a, a
    # The test string should not use sys reference count, which may lead to
    # unexpected string reference increase/decrease.

    expected_size_estimate = 4
    estimator1.update(a)
    estimator1.update(b)
    estimator2.update(c)
    estimator2.update(a)
    estimator1 += estimator2
    self.assertEqual(estimator1.get_estimate(), expected_size_estimate)


