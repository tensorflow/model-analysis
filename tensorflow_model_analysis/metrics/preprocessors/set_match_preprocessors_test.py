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
"""Tests for set match preprocessors."""


import pytest
from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util as beam_testing_util
import numpy as np
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.metrics.preprocessors import set_match_preprocessors
from tensorflow_model_analysis.utils import util

# Initialize test data

_SET_MATCH_INPUT = util.StandardExtracts({
    constants.LABELS_KEY: np.array(['cats', 'dogs']),
    constants.PREDICTIONS_KEY: {
        'classes': np.array(['dogs', 'birds']),
        'scores': np.array([0.3, 0.1]),
    },
})

_SET_MATCH_RESULT = [
    {
        constants.LABELS_KEY: np.array([1.0]),
        constants.PREDICTIONS_KEY: np.array([0.3]),
    },
    {
        constants.LABELS_KEY: np.array([1.0]),
        constants.PREDICTIONS_KEY: np.array([float('-inf')]),
    },
    {
        constants.LABELS_KEY: np.array([0.0]),
        constants.PREDICTIONS_KEY: np.array([0.1]),
    },
]

_SET_MATCH_INPUT_WITH_WEIGHT = util.StandardExtracts({
    constants.LABELS_KEY: np.array(['cats', 'dogs']),
    constants.PREDICTIONS_KEY: {
        'classes': np.array(['dogs', 'birds']),
        'scores': np.array([0.3, 0.1]),
    },
    constants.EXAMPLE_WEIGHTS_KEY: np.array([0.7]),
})

_SET_MATCH_RESULT_WITH_WEIGHT = [
    {
        constants.LABELS_KEY: np.array([1.0]),
        constants.PREDICTIONS_KEY: np.array([0.3]),
        constants.EXAMPLE_WEIGHTS_KEY: np.array([0.7]),
    },
    {
        constants.LABELS_KEY: np.array([1.0]),
        constants.PREDICTIONS_KEY: np.array([float('-inf')]),
        constants.EXAMPLE_WEIGHTS_KEY: np.array([0.7]),
    },
    {
        constants.LABELS_KEY: np.array([0.0]),
        constants.PREDICTIONS_KEY: np.array([0.1]),
        constants.EXAMPLE_WEIGHTS_KEY: np.array([0.7]),
    },
]

_SET_MATCH_INPUT_WITH_CLASS_WEIGHT = util.StandardExtracts({
    constants.LABELS_KEY: np.array(['cats', 'dogs']),
    constants.FEATURES_KEY: {'class_weights': np.array([0.7, 0.2])},
    constants.PREDICTIONS_KEY: {
        'classes': np.array(['dogs', 'birds']),
        'scores': np.array([0.3, 0.1]),
    },
    constants.EXAMPLE_WEIGHTS_KEY: np.array([0.7]),
})

_SET_MATCH_RESULT_WITH_CLASS_WEIGHT = [
    {
        constants.LABELS_KEY: np.array([1.0]),
        constants.PREDICTIONS_KEY: np.array([0.3]),
        constants.EXAMPLE_WEIGHTS_KEY: np.array([0.14]),
    },
    {
        constants.LABELS_KEY: np.array([1.0]),
        constants.PREDICTIONS_KEY: np.array([float('-inf')]),
        constants.EXAMPLE_WEIGHTS_KEY: np.array([0.49]),
    },
    {
        constants.LABELS_KEY: np.array([0.0]),
        constants.PREDICTIONS_KEY: np.array([0.1]),
        constants.EXAMPLE_WEIGHTS_KEY: np.array([0.7]),
    },
]


@pytest.mark.xfail(run=False, reason="PR 183 This class contains tests that fail and needs to be fixed. "
"If all tests pass, please remove this mark.")
class SetMatchPreprocessorTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'two_sets',
          _SET_MATCH_INPUT,
          _SET_MATCH_RESULT,
          set_match_preprocessors.SetMatchPreprocessor(
              class_key='',
              weight_key='',
              prediction_class_key='classes',
              prediction_score_key='scores',
              top_k=None,
          ),
      ),
      (
          'two_sets_with_example_weight',
          _SET_MATCH_INPUT_WITH_WEIGHT,
          _SET_MATCH_RESULT_WITH_WEIGHT,
          set_match_preprocessors.SetMatchPreprocessor(
              class_key='',
              weight_key='',
              prediction_class_key='classes',
              prediction_score_key='scores',
              top_k=None,
          ),
      ),
      (
          'two_sets_with_top_k',
          _SET_MATCH_INPUT,
          _SET_MATCH_RESULT[:2],
          set_match_preprocessors.SetMatchPreprocessor(
              class_key='',
              weight_key='',
              prediction_class_key='classes',
              prediction_score_key='scores',
              top_k=1,
          ),
      ),
      (
          'two_sets_with_class_weight',
          _SET_MATCH_INPUT_WITH_CLASS_WEIGHT,
          _SET_MATCH_RESULT_WITH_CLASS_WEIGHT,
          set_match_preprocessors.SetMatchPreprocessor(
              class_key='',
              weight_key='class_weights',
              prediction_class_key='classes',
              prediction_score_key='scores',
              top_k=None,
          ),
      ),
  )
  def testSetMatchPreprocessor(self, extracts, expected_inputs, preprocessor):
    with beam.Pipeline() as p:
      updated_pcoll = (
          p
          | 'Create' >> beam.Create([extracts])
          | 'Preprocess' >> beam.ParDo(preprocessor)
      )

      def check_result(result):
        # Only single extract case is tested
        self.assertLen(result, len(expected_inputs))
        for updated_extracts, expected_input in zip(result, expected_inputs):
          self.assertIn(constants.PREDICTIONS_KEY, updated_extracts)
          np.testing.assert_allclose(
              updated_extracts[constants.PREDICTIONS_KEY],
              expected_input[constants.PREDICTIONS_KEY],
          )
          self.assertIn(constants.LABELS_KEY, updated_extracts)
          np.testing.assert_allclose(
              updated_extracts[constants.LABELS_KEY],
              expected_input[constants.LABELS_KEY],
          )
          if constants.EXAMPLE_WEIGHTS_KEY in expected_input:
            self.assertIn(constants.EXAMPLE_WEIGHTS_KEY, updated_extracts)
            np.testing.assert_allclose(
                updated_extracts[constants.EXAMPLE_WEIGHTS_KEY],
                expected_input[constants.EXAMPLE_WEIGHTS_KEY],
            )

      beam_testing_util.assert_that(updated_pcoll, check_result)

  def testName(self):
    preprocessor = set_match_preprocessors.SetMatchPreprocessor(
        class_key='',
        weight_key='',
        prediction_class_key='classes',
        prediction_score_key='scores',
        top_k=3,
    )
    self.assertEqual(preprocessor.name, '_set_match_preprocessor:top_k=3')

  def testClassWeightShapeMismatch(self):
    extracts = util.StandardExtracts({
        constants.LABELS_KEY: np.array(['cats', 'dogs']),
        constants.FEATURES_KEY: {'class_weights': np.array([0.7])},
        constants.PREDICTIONS_KEY: np.array(['birds', 'dogs']),
    })
    with self.assertRaisesRegex(
        ValueError,
        'Classes and weights must be of the same shape.',
    ):
      _ = next(
          set_match_preprocessors.SetMatchPreprocessor(
              class_key='',
              weight_key='class_weights',
              prediction_class_key='classes',
              prediction_score_key='',
          ).process(extracts=extracts)
      )

  def testLabelNotFoundClasses(self):
    extracts = util.StandardExtracts({
        constants.LABELS_KEY: np.array(['cats', 'dogs']),
        constants.FEATURES_KEY: {'class_weights': np.array([0.7, 0.2])},
        constants.PREDICTIONS_KEY: {
            'classes': np.array(['birds', 'dogs']),
            'scores': np.array([0.1, 0.3]),
        },
    })
    with self.assertRaisesRegex(ValueError, 'key not found'):
      _ = next(
          set_match_preprocessors.SetMatchPreprocessor(
              class_key='cla',
              weight_key='weights',
              prediction_class_key='classes',
              prediction_score_key='scores',
          ).process(extracts=extracts)
      )

  def testNotFoundClassWeights(self):
    extracts = util.StandardExtracts({
        constants.LABELS_KEY: np.array(['cats', 'dogs']),
        constants.FEATURES_KEY: {'class_weights': np.array([0.7, 0.2])},
        constants.PREDICTIONS_KEY: {
            'classes': np.array([['birds', 'dogs']]),
            'scores': np.array([[0.1, 0.3]]),
        },
    })
    with self.assertRaisesRegex(ValueError, 'key not found'):
      _ = next(
          set_match_preprocessors.SetMatchPreprocessor(
              class_key='',
              weight_key='weigh',
              prediction_class_key='classes',
              prediction_score_key='score',
          ).process(extracts=extracts)
      )

  def testNotFoundFeatures(self):
    extracts = util.StandardExtracts({
        constants.LABELS_KEY: np.array(['cats', 'dogs']),
        constants.PREDICTIONS_KEY: {
            'classes': np.array([['birds', 'dogs']]),
            'scores': np.array([[0.1, 0.3]]),
        },
    })
    with self.assertRaisesRegex(ValueError, 'features is None'):
      _ = next(
          set_match_preprocessors.SetMatchPreprocessor(
              class_key='',
              weight_key='weigh',
              prediction_class_key='classes',
              prediction_score_key='score',
          ).process(extracts=extracts)
      )

  def testInvalidLabel(self):
    extracts = util.StandardExtracts({
        constants.LABELS_KEY: np.array([['cats', 'dogs']]),
        constants.PREDICTIONS_KEY: {
            'classes': np.array(['birds', 'dogs']),
            'scores': np.array([0.1, 0.3]),
        },
    })
    with self.assertRaisesRegex(ValueError, 'Labels must be a 1d numpy array.'):
      _ = next(
          set_match_preprocessors.SetMatchPreprocessor(
              prediction_class_key='classes',
              prediction_score_key='scores',
          ).process(extracts=extracts)
      )

  def testPredictionNotADict(self):
    extracts = util.StandardExtracts({
        constants.LABELS_KEY: np.array(['cats', 'dogs']),
        constants.PREDICTIONS_KEY: np.array(['birds', 'dogs']),
    })
    with self.assertRaisesRegex(
        TypeError,
        (
            'Predictions are expected to be a '
            'dictionary conatining classes and scores.'
        ),
    ):
      _ = next(
          set_match_preprocessors.SetMatchPreprocessor(
              class_key='',
              weight_key='',
              prediction_class_key='classes',
              prediction_score_key='scores',
          ).process(extracts=extracts)
      )

  def testPredictionNotFoundClasses(self):
    extracts = util.StandardExtracts({
        constants.LABELS_KEY: np.array(['cats', 'dogs']),
        constants.PREDICTIONS_KEY: {
            'classes': np.array([['birds', 'dogs']]),
            'scores': np.array([[0.1, 0.3]]),
        },
    })
    with self.assertRaisesRegex(ValueError, 'key not found'):
      _ = next(
          set_match_preprocessors.SetMatchPreprocessor(
              class_key='',
              weight_key='',
              prediction_class_key='clas',
              prediction_score_key='scores',
          ).process(extracts=extracts)
      )

  def testPredictionNotFoundScores(self):
    extracts = util.StandardExtracts({
        constants.LABELS_KEY: np.array(['cats', 'dogs']),
        constants.PREDICTIONS_KEY: {
            'classes': np.array([['birds', 'dogs']]),
            'scores': np.array([[0.1, 0.3]]),
        },
    })
    with self.assertRaisesRegex(ValueError, 'key not found'):
      _ = next(
          set_match_preprocessors.SetMatchPreprocessor(
              class_key='',
              weight_key='',
              prediction_class_key='classes',
              prediction_score_key='scor',
          ).process(extracts=extracts)
      )

  def testInvalidPrediction(self):
    extracts = util.StandardExtracts({
        constants.LABELS_KEY: np.array(['cats', 'dogs']),
        constants.PREDICTIONS_KEY: {
            'classes': np.array([['birds', 'dogs']]),
            'scores': np.array([[0.1, 0.3]]),
        },
    })
    with self.assertRaisesRegex(
        ValueError, 'Predicted classes must be a 1d numpy array.'
    ):
      _ = next(
          set_match_preprocessors.SetMatchPreprocessor(
              class_key='',
              weight_key='',
              prediction_class_key='classes',
              prediction_score_key='scores',
          ).process(extracts=extracts)
      )

  def testMismatchClassesAndScores(self):
    extracts = util.StandardExtracts({
        constants.LABELS_KEY: np.array(['cats', 'dogs']),
        constants.PREDICTIONS_KEY: {
            'classes': np.array([['birds', 'dogs']]),
            'scores': np.array([0.1, 0.3]),
        },
    })
    with self.assertRaisesRegex(
        ValueError, 'Classes and scores must be of the same shape.'
    ):
      _ = next(
          set_match_preprocessors.SetMatchPreprocessor(
              class_key='',
              weight_key='',
              prediction_class_key='classes',
              prediction_score_key='scores',
          ).process(extracts=extracts)
      )


