# Copyright 2021 Google LLC
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
"""Tests for TF metric accumulators."""

import numpy as np
import tensorflow as tf
from tensorflow_model_analysis.metrics import tf_metric_accumulators
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.utils import test_util



class TfMetricAccumulatorsTest(test_util.TensorflowModelAnalysisTest):

  def testTFMetricsAccumulator(self):
    # This test uses strings as inputs, but it works similarly to how an
    # accumulator based on tf.Examples would work.
    acc = tf_metric_accumulators.TFMetricsAccumulator(
        input_counts=[1, 1], metric_counts=[1, 2], size_estimator_fn=len)

    self.assertEqual(0, acc.len_inputs())

    acc.add_input(0, 'output_1_input_1')
    acc.add_input(0, 'output_1_input_2')
    acc.add_input(1, 'output_2_input_1')
    acc.add_input(1, 'output_2_input_2')
    acc.add_input(1, 'output_2_input_3')
    self.assertEqual(
        acc.get_inputs(0), (['output_1_input_1', 'output_1_input_2'],))
    self.assertEqual(
        acc.get_inputs(1),
        (['output_2_input_1', 'output_2_input_2', 'output_2_input_3'],))

    acc.clear_inputs()
    self.assertEqual(0, acc.len_inputs())

    acc.add_weights(0, 0, np.array([1.0, 2.0]))
    acc.add_weights(0, 0, np.array([3.0, 4.0]))
    acc.add_weights(1, 0, np.array([5.0, 6.0]))
    acc.add_weights(1, 1, np.array([7.0, 8.0]))
    acc.add_weights(1, 1, np.array([9.0, 10.0]))
    self.assertAllClose(acc.get_weights(0, 0), np.array([4.0, 6.0]))
    self.assertAllClose(acc.get_weights(1, 0), np.array([5.0, 6.0]))
    self.assertAllClose(acc.get_weights(1, 1), np.array([16.0, 18.0]))

  def testTFCompilableMetricsAccumulator(self):
    acc = tf_metric_accumulators.TFCompilableMetricsAccumulator(
        metric_counts=[1, 2], padding_options=None)

    self.assertEqual(0, acc.len_inputs())

    acc.add_input(0, np.array([1.0, 0.0]), np.array([0.5, 0.5]),
                  np.array([1.0]))
    acc.add_input(0, np.array([1.0, 1.0]), np.array([0.3, 0.7]),
                  np.array([1.0]))
    acc.add_input(1, np.array([0.0, 0.0]), np.array([0.2, 0.8]),
                  np.array([0.5]))
    acc.add_input(1, np.array([0.0, 1.0]), np.array([0.1, 0.9]),
                  np.array([0.5]))
    acc.add_input(1, np.array([1.0, 1.0]), np.array([0.6, 0.4]),
                  np.array([0.7]))
    self.assertAllClose(
        acc.get_inputs(0), (np.array([
            np.array([1.0, 0.0]), np.array([1.0, 1.0])
        ]), np.array([
            np.array([0.5, 0.5]), np.array([0.3, 0.7])
        ]), np.array([np.array([1.0]), np.array([1.0])])))
    self.assertAllClose(
        acc.get_inputs(1),
        (np.array(
            [np.array([0.0, 0.0]),
             np.array([0.0, 1.0]),
             np.array([1.0, 1.0])]),
         np.array([
             np.array([0.2, 0.8]),
             np.array([0.1, 0.9]),
             np.array([0.6, 0.4])
         ]), np.array([np.array([0.5]),
                       np.array([0.5]),
                       np.array([0.7])])))

    acc.clear_inputs()
    self.assertEqual(0, acc.len_inputs())

    acc.add_weights(0, 0, np.array([1.0, 2.0]))
    acc.add_weights(1, 0, np.array([3.0, 4.0]))
    acc.add_weights(1, 1, np.array([5.0, 6.0]))
    acc.add_weights(1, 1, np.array([7.0, 8.0]))
    self.assertAllClose(acc.get_weights(0, 0), np.array([1.0, 2.0]))
    self.assertAllClose(acc.get_weights(1, 0), np.array([3.0, 4.0]))
    self.assertAllClose(acc.get_weights(1, 1), np.array([12.0, 14.0]))

  def testTFCompilableMetricsAccumulatorWithFirstEmptyInput(self):
    acc = tf_metric_accumulators.TFCompilableMetricsAccumulator(
        metric_counts=[1, 2, 3],
        padding_options=config_pb2.PaddingOptions(
            label_float_padding=-1.0,
            prediction_float_padding=-1.0,
        ),
    )

    self.assertEqual(0, acc.len_inputs())

    acc.add_input(0, None, None, None)

    acc.add_input(
        1, np.array([1.0, 1.0, 1.0]), np.array([0.3, 0.7]), np.array([1.0])
    )
    acc.add_input(
        2, np.array([0.0, 0.0]), np.array([0.2, 0.8]), np.array([0.5])
    )
    self.assertAllClose(
        acc.get_inputs(1),
        (
            np.array([[1.0, 1.0, 1.0]]),
            np.array([[0.3, 0.7, -1.0]]),
            np.array([[1.0]]),
        ),
    )
    self.assertAllClose(
        acc.get_inputs(2),
        (
            np.array([[0.0, 0.0, -1.0]]),
            np.array([[0.2, 0.8, -1.0]]),
            np.array([[0.5]]),
        ),
    )


