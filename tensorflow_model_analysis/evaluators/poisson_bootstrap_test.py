# Lint as: python3
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
"""Test for using the poisson bootstrap API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import types
from tensorflow_model_analysis.evaluators import poisson_bootstrap


class PoissonBootstrapTest(tf.test.TestCase):

  def testCalculateConfidenceInterval(self):
    sampling_data_list = [
        np.array([
            [0, 0, 2, 7, 0.77777779, 1],
            [1, 0, 2, 6, 0.75, 0.85714287],
            [4, 0, 2, 3, 0.60000002, 0.42857143],
            [4, 2, 0, 3, 1, 0.42857143],
            [7, 2, 0, 0, float('nan'), 0],
        ]),
        np.array([
            [7, 2, 0, 0, float('nan'), 0],
            [0, 0, 2, 7, 0.77777779, 1],
            [1, 0, 2, 6, 0.75, 0.85714287],
            [4, 0, 2, 3, 0.60000002, 0.42857143],
            [4, 2, 0, 3, 1, 0.42857143],
        ]),
    ]
    unsampled_data = np.array([
        [4, 2, 0, 3, 1, 0.42857143],
        [7, 2, 0, 0, float('nan'), 0],
        [0, 0, 2, 7, 0.77777779, 1],
        [1, 0, 2, 6, 0.75, 0.85714287],
        [4, 0, 2, 3, 0.60000002, 0.42857143],
    ])
    result = poisson_bootstrap._calculate_t_distribution(
        sampling_data_list, unsampled_data)
    self.assertIsInstance(result, np.ndarray)
    self.assertEqual(result.shape, (5, 6))
    self.assertAlmostEqual(result[0][0].sample_mean, 3.5, delta=0.1)
    self.assertAlmostEqual(
        result[0][0].sample_standard_deviation, 4.94, delta=0.1)
    self.assertEqual(result[0][0].sample_degrees_of_freedom, 1)
    self.assertEqual(result[0][0].unsampled_value, 4.0)
    self.assertAlmostEqual(result[0][4].sample_mean, 0.77, delta=0.1)
    self.assertTrue(np.isnan(result[0][4].sample_standard_deviation))
    self.assertEqual(result[0][4].sample_degrees_of_freedom, 0)
    self.assertEqual(result[0][4].unsampled_value, 1.0)

    sampling_data_list = [
        np.array([1, 2]),
        np.array([1, 2]),
        np.array([1, float('nan')])
    ]
    unsampled_data = np.array([1, 2])
    result = poisson_bootstrap._calculate_t_distribution(
        sampling_data_list, unsampled_data)
    self.assertIsInstance(result, np.ndarray)
    self.assertEqual(result.tolist(), [
        types.ValueWithTDistribution(
            sample_mean=1.0,
            sample_standard_deviation=0.0,
            sample_degrees_of_freedom=2,
            unsampled_value=1),
        types.ValueWithTDistribution(
            sample_mean=2.0,
            sample_standard_deviation=0.0,
            sample_degrees_of_freedom=1,
            unsampled_value=2)
    ])


if __name__ == '__main__':
  tf.test.main()
