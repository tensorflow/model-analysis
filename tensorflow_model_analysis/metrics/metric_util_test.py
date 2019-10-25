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
"""Tests for metric utils."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util


class UtilTest(tf.test.TestCase):

  def testToScalar(self):
    self.assertEqual(1, metric_util.to_scalar(np.array([1])))
    self.assertEqual(1.0, metric_util.to_scalar(np.array(1.0)))
    self.assertEqual('string', metric_util.to_scalar(np.array([['string']])))
    sparse_tensor = tf.compat.v1.SparseTensorValue(
        indices=np.array([0]), values=np.array([1]), dense_shape=(1,))
    self.assertEqual(1, metric_util.to_scalar(sparse_tensor))

  def testStandardMetricInputsToNumpy(self):
    example = metric_types.StandardMetricInputs(
        {'output_name': np.array([2])},
        {'output_name': np.array([0, 0.5, 0.3, 0.9])},
        {'output_name': np.array([1.0])})
    got_label, got_pred, got_example_weight = (
        metric_util.to_label_prediction_example_weight(
            example, output_name='output_name'))

    self.assertAllClose(got_label, np.array([2]))
    self.assertAllClose(got_pred, np.array([0, 0.5, 0.3, 0.9]))
    self.assertAllClose(got_example_weight, np.array([1.0]))

  def testStandardMetricInputsWithZeroWeightsToNumpy(self):
    example = metric_types.StandardMetricInputs(
        np.array([2]), np.array([0, 0.5, 0.3, 0.9]), np.array([0.0]))
    got_label, got_pred, got_example_weight = (
        metric_util.to_label_prediction_example_weight(example))

    self.assertAllClose(got_label, np.array([2]))
    self.assertAllClose(got_pred, np.array([0, 0.5, 0.3, 0.9]))
    self.assertAllClose(got_example_weight, np.array([0.0]))

  def testStandardMetricInputsWithClassIDToNumpy(self):
    example = metric_types.StandardMetricInputs(
        {'output_name': np.array([2])},
        {'output_name': np.array([0, 0.5, 0.3, 0.9])},
        {'output_name': np.array([1.0])})
    got_label, got_pred, got_example_weight = (
        metric_util.to_label_prediction_example_weight(
            example,
            output_name='output_name',
            sub_key=metric_types.SubKey(class_id=2)))

    self.assertAllClose(got_label, np.array([1.0]))
    self.assertAllClose(got_pred, np.array([0.3]))
    self.assertAllClose(got_example_weight, np.array([1.0]))

  def testStandardMetricInputsWithKToNumpy(self):
    example = metric_types.StandardMetricInputs(
        {'output_name': np.array([2])},
        {'output_name': np.array([0, 0.5, 0.3, 0.9])},
        {'output_name': np.array([1.0])})
    got_label, got_pred, got_example_weight = (
        metric_util.to_label_prediction_example_weight(
            example,
            output_name='output_name',
            sub_key=metric_types.SubKey(k=2)))

    self.assertAllClose(got_label, np.array([0.0]))
    self.assertAllClose(got_pred, np.array([0.5]))
    self.assertAllClose(got_example_weight, np.array([1.0]))

  def testStandardMetricInputsWithTopKToNumpy(self):
    example = metric_types.StandardMetricInputs(
        {'output_name': np.array([1])},
        {'output_name': np.array([0, 0.5, 0.3, 0.9])},
        {'output_name': np.array([1.0])})
    got_label, got_pred, got_example_weight = (
        metric_util.to_label_prediction_example_weight(
            example,
            output_name='output_name',
            sub_key=metric_types.SubKey(top_k=2)))

    self.assertAllClose(got_label, np.array([0.0, 1.0]))
    self.assertAllClose(got_pred, np.array([0.9, 0.5]))
    self.assertAllClose(got_example_weight, np.array([1.0]))

  def testPrepareLabelsAndPredictions(self):
    labels = [0]
    preds = {
        'logistic': np.array([0.8]),
    }
    got_labels, got_preds = metric_util.prepare_labels_and_predictions(
        labels, preds)

    self.assertAllClose(got_labels, np.array([0]))
    self.assertAllClose(got_preds, np.array([0.8]))

  def testPrepareLabelsAndPredictionsBatched(self):
    labels = [['b']]
    preds = {
        'logistic': np.array([[0.8]]),
        'all_classes': np.array([['a', 'b', 'c']])
    }
    got_labels, got_preds = metric_util.prepare_labels_and_predictions(
        labels, preds)

    self.assertAllClose(got_labels, np.array([[1]]))
    self.assertAllClose(got_preds, np.array([[0.8]]))

  def testPrepareLabelsAndPredictionsMixedBatching(self):
    labels = np.array([1])
    preds = {
        'predictions': np.array([[0.8]]),
    }
    got_labels, got_preds = metric_util.prepare_labels_and_predictions(
        labels, preds)

    self.assertAllClose(got_labels, np.array([1]))
    self.assertAllClose(got_preds, np.array([[0.8]]))

  def testPrepareMultipleLabelsAndPredictions(self):
    labels = np.array(['b', 'c', 'a'])
    preds = {
        'scores': np.array([0.2, 0.7, 0.1]),
        'classes': np.array(['a', 'b', 'c'])
    }
    got_labels, got_preds = metric_util.prepare_labels_and_predictions(
        labels, preds)

    self.assertAllClose(got_labels, np.array([1, 2, 0]))
    self.assertAllClose(got_preds, np.array([0.2, 0.7, 0.1]))

  def testPrepareMultipleLabelsAndPredictionsPythonList(self):
    labels = ['b', 'c', 'a']
    preds = {'probabilities': [0.2, 0.7, 0.1], 'all_classes': ['a', 'b', 'c']}
    got_labels, got_preds = metric_util.prepare_labels_and_predictions(
        labels, preds)

    self.assertAllClose(got_labels, np.array([1, 2, 0]))
    self.assertAllClose(got_preds, np.array([0.2, 0.7, 0.1]))

  def testPrepareMultipleLabelsAndPredictionsMultiDimension(self):
    labels = [[0], [1]]
    preds = {'probabilities': [[0.2, 0.8], [0.3, 0.7]]}
    got_labels, got_preds = metric_util.prepare_labels_and_predictions(
        labels, preds)

    self.assertAllClose(got_labels, np.array([[0], [1]]))
    self.assertAllClose(got_preds, np.array([[0.2, 0.8], [0.3, 0.7]]))

  def testPrepareLabelsAndPredictionsEmpty(self):
    labels = []
    preds = {'logistic': [], 'all_classes': ['a', 'b', 'c']}
    got_labels, got_preds = metric_util.prepare_labels_and_predictions(
        labels, preds)

    self.assertAllClose(got_labels, np.array([]))
    self.assertAllClose(got_preds, np.array([]))

  def testPrepareLabelsAndPredictionsWithVocab(self):
    labels = np.array(['e', 'f'])
    preds = {'probabilities': [0.2, 0.8], 'all_classes': ['a', 'b', 'c']}
    got_labels, got_preds = metric_util.prepare_labels_and_predictions(
        labels, preds, label_vocabulary=['e', 'f'])

    self.assertAllClose(got_labels, np.array([0, 1]))
    self.assertAllClose(got_preds, np.array([0.2, 0.8]))

  def testSelectClassID(self):
    labels = np.array([2])
    preds = np.array([0.2, 0.7, 0.1])
    got_labels, got_preds = metric_util.select_class_id(1, labels, preds)

    self.assertAllClose(got_labels, np.array([0]))
    self.assertAllClose(got_preds, np.array([0.7]))

  def testSelectClassIDWithMultipleValues(self):
    labels = np.array([0, 2, 1])
    preds = np.array([[0.2, 0.7, 0.1], [0.3, 0.6, 0.1], [0.1, 0.2, 0.7]])
    got_labels, got_preds = metric_util.select_class_id(1, labels, preds)

    self.assertAllClose(got_labels, np.array([0, 0, 1]))
    self.assertAllClose(got_preds, np.array([0.7, 0.6, 0.2]))

  def testSelectClassIDWithOnehot(self):
    labels = np.array([[0, 0, 1]])
    preds = np.array([0.2, 0.7, 0.1])
    got_labels, got_preds = metric_util.select_class_id(1, labels, preds)

    self.assertAllClose(got_labels, np.array([0]))
    self.assertAllClose(got_preds, np.array([0.7]))

  def testSelectClassIDWithOnehotAndMultipleValues(self):
    labels = np.array([[0, 0, 1], [0, 0, 1], [0, 1, 0]])
    preds = np.array([[0.2, 0.7, 0.1], [0.3, 0.6, 0.1], [0.1, 0.2, 0.7]])
    got_labels, got_preds = metric_util.select_class_id(1, labels, preds)

    self.assertAllClose(got_labels, np.array([0, 0, 1]))
    self.assertAllClose(got_preds, np.array([0.7, 0.6, 0.2]))

  def testSelectClassIDWithBatchedLabels(self):
    labels = np.array([[0], [1], [2]])
    preds = np.array([[0.2, 0.7, 0.1], [0.3, 0.6, 0.1], [0.1, 0.2, 0.7]])
    got_labels, got_preds = metric_util.select_class_id(2, labels, preds)

    self.assertAllClose(got_labels, np.array([[0], [0], [1]]))
    self.assertAllClose(got_preds, np.array([[0.1], [0.1], [0.7]]))

  def testSelectClassIDEmpty(self):
    labels = np.array(np.array([]))
    preds = np.array(np.array([]))
    got_labels, got_preds = metric_util.select_class_id(1, labels, preds)

    self.assertAllClose(got_labels, np.array([]))
    self.assertAllClose(got_preds, np.array([]))

  def testSelectTopK(self):
    labels = np.array([3])
    preds = np.array([0.4, 0.1, 0.2, 0.3])
    got_labels, got_preds = metric_util.select_top_k(2, labels, preds)

    self.assertAllClose(got_labels, np.array([0, 1]))
    self.assertAllClose(got_preds, np.array([0.4, 0.3]))

  def testSelectTopKBatched(self):
    labels = np.array([[2], [3]])
    preds = np.array([[0.4, 0.1, 0.2, 0.3], [0.1, 0.2, 0.1, 0.6]])
    got_labels, got_preds = metric_util.select_top_k(2, labels, preds)

    self.assertAllClose(got_labels, np.array([[0, 0], [1, 0]]))
    self.assertAllClose(got_preds, np.array([[0.4, 0.3], [0.6, 0.2]]))

  def testSelectTopKUsingSeparateScores(self):
    labels = np.array(['', '', '', 'c'])
    preds = np.array(['b', 'c', 'a', 'd'])
    scores = np.array([0.4, 0.1, 0.2, 0.3])
    got_labels, got_preds = metric_util.select_top_k(2, labels, preds, scores)

    self.assertSequenceEqual(list(got_labels), ['', 'c'])
    self.assertSequenceEqual(list(got_preds), ['b', 'd'])


if __name__ == '__main__':
  tf.test.main()
