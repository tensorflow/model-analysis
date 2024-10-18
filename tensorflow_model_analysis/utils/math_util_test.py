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

import math
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis.api import types
from tensorflow_model_analysis.metrics import binary_confusion_matrices
from tensorflow_model_analysis.utils import math_util


class MathUtilTest(tf.test.TestCase):

  def testCalculateConfidenceInterval(self):
    np.testing.assert_almost_equal(
        math_util.calculate_confidence_interval(
            types.ValueWithTDistribution(10, 2, 9, 10)),
        (10, 5.4756856744035902196, 14.524314325596410669))
    mid, lb, ub = math_util.calculate_confidence_interval(
        types.ValueWithTDistribution(-1, -1, -1, -1))
    self.assertEqual(mid, -1)
    self.assertTrue(math.isnan(lb))
    self.assertTrue(math.isnan(ub))

  def testCalculateConfidenceIntervalConfusionMatrices(self):
    mid, lb, ub = math_util.calculate_confidence_interval(
        types.ValueWithTDistribution(
            sample_mean=binary_confusion_matrices.Matrices(
                thresholds=[0.5], tp=[0.0], tn=[2.0], fp=[1.0], fn=[1.0]),
            sample_standard_deviation=binary_confusion_matrices.Matrices(
                thresholds=[0.5],
                tp=[0.0],
                tn=[2.051956704170308],
                fp=[1.025978352085154],
                fn=[1.2139539573337679]),
            sample_degrees_of_freedom=19,
            unsampled_value=binary_confusion_matrices.Matrices(
                thresholds=[0.5], tp=[0.0], tn=[2.0], fp=[1.0], fn=[1.0])))

    expected_mid = binary_confusion_matrices.Matrices(
        thresholds=[0.5], tp=[0.0], tn=[2.0], fp=[1.0], fn=[1.0])
    self.assertEqual(expected_mid, mid)

    expected_lb = binary_confusion_matrices.Matrices(
        thresholds=[0.5],
        tp=[0.0],
        tn=[-2.2947947404327547],
        fp=[-1.1473973702163773],
        fn=[-1.5408348336436783])
    self.assertEqual(expected_lb.thresholds, lb.thresholds)
    np.testing.assert_almost_equal(lb.tp, expected_lb.tp)
    np.testing.assert_almost_equal(lb.fp, expected_lb.fp)
    np.testing.assert_almost_equal(lb.tn, expected_lb.tn)
    np.testing.assert_almost_equal(lb.fn, expected_lb.fn)

    expected_ub = binary_confusion_matrices.Matrices(
        thresholds=[0.5],
        tp=[0.0],
        tn=[6.294794740432755],
        fp=[3.1473973702163773],
        fn=[3.5408348336436783])
    self.assertEqual(expected_ub.thresholds, ub.thresholds)
    np.testing.assert_almost_equal(ub.tp, expected_ub.tp)
    np.testing.assert_almost_equal(ub.fp, expected_ub.fp)
    np.testing.assert_almost_equal(ub.tn, expected_ub.tn)
    np.testing.assert_almost_equal(ub.fn, expected_ub.fn)


