# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test Model Agnostic graph handler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Standard Imports
import numpy as np

import tensorflow as tf

from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.model_agnostic_eval import model_agnostic_predict


class ModelAgnosticPredictTest(testutil.TensorflowModelAnalysisTest):

  def testValidation(self):
    # Test no feature spec.
    with self.assertRaisesRegexp(
        ValueError, 'ModelAgnosticConfig must have feature_spec set.'):
      model_agnostic_predict.ModelAgnosticConfig(
          label_keys=['label'],
          prediction_keys=['probabilities'],
          feature_spec=None)

    # Test no prediction keys.
    feature_map = {
        'age':
            tf.io.FixedLenFeature([], tf.int64),
        'language':
            tf.io.VarLenFeature(tf.string),
        'probabilities':
            tf.io.FixedLenFeature([2], tf.int64, default_value=[9, 9]),
        'label':
            tf.io.FixedLenFeature([], tf.int64)
    }

    with self.assertRaisesRegexp(
        ValueError, 'ModelAgnosticConfig must have prediction keys set.'):
      model_agnostic_predict.ModelAgnosticConfig(
          label_keys=['label'], prediction_keys=[], feature_spec=feature_map)

    # Test no label keys.
    with self.assertRaisesRegexp(
        ValueError, 'ModelAgnosticConfig must have label keys set.'):
      model_agnostic_predict.ModelAgnosticConfig(
          label_keys=[],
          prediction_keys=['predictions'],
          feature_spec=feature_map)

    # Test prediction key not in feature spec.
    with self.assertRaisesRegexp(
        ValueError, 'Prediction key not_prob not defined in feature_spec.'):
      model_agnostic_predict.ModelAgnosticConfig(
          label_keys=['label'],
          prediction_keys=['not_prob'],
          feature_spec=feature_map)

    # Test label key not in feature spec.
    with self.assertRaisesRegexp(
        ValueError, 'Label key not_label not defined in feature_spec.'):
      model_agnostic_predict.ModelAgnosticConfig(
          label_keys=['not_label'],
          prediction_keys=['probabilities'],
          feature_spec=feature_map)

  def testExtractFplExampleGraph(self):
    # Set up some examples with some Sparseness.
    examples = [
        self._makeExample(
            age=0, language='english', probabilities=[0.2, 0.8], label=1),
        self._makeExample(age=1, language='chinese', label=0),
        self._makeExample(age=2, probabilities=[0.1, 0.9], label=1),
        self._makeExample(
            language='chinese', probabilities=[0.8, 0.2], label=0),
    ]

    # Set up the expected results on two of the fields. Note that directly
    # entire FPLs will fail in numpy comparison.
    expected_age = [np.array([0]), np.array([1]), np.array([2]), np.array([3])]
    expected_language = [
        tf.compat.v1.SparseTensorValue(
            indices=np.array([[0, 0]]),
            values=np.array([b'english'], dtype=np.object),
            dense_shape=np.array([1, 1])),
        tf.compat.v1.SparseTensorValue(
            indices=np.array([[0, 0]]),
            values=np.array([b'chinese'], dtype=np.object),
            dense_shape=np.array([1, 1])),
        tf.compat.v1.SparseTensorValue(
            indices=np.array([], dtype=np.int64).reshape([0, 2]),
            values=np.array([], dtype=np.object),
            dense_shape=np.array([1, 0])),
        tf.compat.v1.SparseTensorValue(
            indices=np.array([[0, 0]]),
            values=np.array([b'chinese'], dtype=np.object),
            dense_shape=np.array([1, 1]))
    ]
    expected_probabilities = [
        np.array([[0.2, 0.8]]),
        np.array([[0.5, 0.5]]),
        np.array([[0.1, 0.9]]),
        np.array([[0.8, 0.2]])
    ]
    expected_labels = [
        np.array([1]),
        np.array([0]),
        np.array([1]),
        np.array([0])
    ]

    # Serialize and feed into our graph.
    serialized_examples = [e.SerializeToString() for e in examples]

    # Set up a config to bucket our example keys.
    feature_map = {
        'age':
            tf.io.FixedLenFeature([1], tf.int64, default_value=[3]),
        'language':
            tf.io.VarLenFeature(tf.string),
        'probabilities':
            tf.io.FixedLenFeature([2], tf.float32, default_value=[0.5, 0.5]),
        'label':
            tf.io.FixedLenFeature([], tf.int64)
    }
    model_agnostic_config = model_agnostic_predict.ModelAgnosticConfig(
        label_keys=['label'],
        prediction_keys=['probabilities'],
        feature_spec=feature_map)

    # Create our model and extract our FPLs.
    agnostic_predict = model_agnostic_predict.ModelAgnosticPredict(
        model_agnostic_config)
    fpls = agnostic_predict.get_fpls_from_examples(serialized_examples)

    # Verify the result is the correct size, has all the keys, and
    # our expected values match.
    self.assertEqual(4, len(fpls))
    for i, fpl in enumerate(fpls):
      self.assertIn('language', fpl.features)
      self.assertIn('label', fpl.labels)
      self.assertIn('label', fpl.features)  # Labels should also be in features.
      self.assertIn('probabilities', fpl.predictions)
      self.assertIn('age', fpl.features)
      self.assertEquals(expected_age[i], fpl.features['age']['node'])
      self.assertSparseTensorValueEqual(expected_language[i],
                                        fpl.features['language']['node'])
      self.assertAllClose(expected_probabilities[i],
                          fpl.predictions['probabilities']['node'])
      self.assertEquals(expected_labels[i], fpl.labels['label']['node'])


if __name__ == '__main__':
  tf.test.main()
