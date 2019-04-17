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
"""Tests for math_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
from tensorflow_model_analysis import math_util
from tensorflow_model_analysis import types


class MathUtilTest(tf.test.TestCase):

  def testCalculateConfidenceInterval(self):
    self.assertEqual(
        math_util.calculate_confidence_interval(
            types.ValueWithTDistribution(10, 2, 9, 10)),
        (10, 8.5692861880948552, 11.430713811905145))
    mean, lb, ub = math_util.calculate_confidence_interval(
        types.ValueWithTDistribution(-1, -1, -1, -1))
    self.assertEqual(mean, -1)
    self.assertTrue(math.isnan(lb))
    self.assertTrue(math.isnan(ub))


if __name__ == '__main__':
  tf.test.main()
