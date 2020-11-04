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
"""Test for using the Evaluate API.

Note that we actually train and export models within these tests.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import parameterized
import apache_beam as beam

from apache_beam.testing import util

import tensorflow as tf

from tensorflow_model_analysis import config
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.eval_saved_model.example_trainers import batch_size_limited_classifier
from tensorflow_model_analysis.eval_saved_model.example_trainers import fake_multi_examples_per_input_estimator
from tensorflow_model_analysis.eval_saved_model.example_trainers import linear_classifier
from tensorflow_model_analysis.extractors import legacy_predict_extractor as predict_extractor


class PredictExtractorTest(testutil.TensorflowModelAnalysisTest,
                           parameterized.TestCase):

  def _getEvalExportDir(self):
    return os.path.join(self._getTempDir(), 'eval_export_dir')

  @parameterized.parameters(
      {'features_blacklist': None},
      {'features_blacklist': ['age']},
  )
  def testPredict(self, features_blacklist):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = linear_classifier.simple_linear_classifier(
        None, temp_eval_export_dir)
    eval_shared_model = model_eval_lib.default_eval_shared_model(
        eval_saved_model_path=eval_export_dir,
        blacklist_feature_fetches=features_blacklist)
    with beam.Pipeline() as pipeline:
      examples = [
          self._makeExample(age=3.0, language='english', label=1.0),
          self._makeExample(age=3.0, language='chinese', label=0.0),
          self._makeExample(age=4.0, language='english', label=1.0),
          self._makeExample(age=5.0, language='chinese', label=0.0),
      ]
      serialized_examples = [e.SerializeToString() for e in examples]

      predict_extracts = (
          pipeline
          | beam.Create(serialized_examples, reshuffle=False)
          # Our diagnostic outputs, pass types.Extracts throughout, however our
          # aggregating functions do not use this interface.
          | beam.Map(lambda x: {constants.INPUT_KEY: x})
          | 'Predict' >> predict_extractor._TFMAPredict(
              eval_shared_models={'': eval_shared_model}, desired_batch_size=3))

      def check_result(got):
        try:
          self.assertLen(got, 4)
          for item in got:
            self.assertIn(constants.FEATURES_PREDICTIONS_LABELS_KEY, item)
            fpl = item[constants.FEATURES_PREDICTIONS_LABELS_KEY]
            # Verify fpl contains features, probabilities, and correct labels.
            blacklisted_features = set(features_blacklist or [])
            expected_features = (
                set(['language', 'age', 'label']) - blacklisted_features)
            for feature in expected_features:
              self.assertIn(feature, fpl.features)
            for feature in blacklisted_features:
              self.assertNotIn(feature, fpl.features)
            self.assertAlmostEqual(fpl.features['label'],
                                   fpl.labels['__labels'])

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(predict_extracts, check_result)

  def testMultiModelPredict(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, model1_dir = linear_classifier.simple_linear_classifier(
        None, temp_eval_export_dir)
    model1 = model_eval_lib.default_eval_shared_model(
        eval_saved_model_path=model1_dir)
    _, model2_dir = linear_classifier.simple_linear_classifier(
        None, temp_eval_export_dir)
    model2 = model_eval_lib.default_eval_shared_model(
        eval_saved_model_path=model2_dir)
    eval_config = config.EvalConfig(model_specs=[
        config.ModelSpec(name='model1', example_weight_key='age'),
        config.ModelSpec(name='model2', example_weight_key='age')
    ])

    with beam.Pipeline() as pipeline:
      examples = [
          self._makeExample(age=3.0, language='english', label=1.0),
          self._makeExample(age=3.0, language='chinese', label=0.0),
          self._makeExample(age=4.0, language='english', label=1.0),
          self._makeExample(age=5.0, language='chinese', label=0.0),
      ]
      serialized_examples = [e.SerializeToString() for e in examples]

      predict_extracts = (
          pipeline
          | beam.Create(serialized_examples, reshuffle=False)
          # Our diagnostic outputs, pass types.Extracts throughout, however our
          # aggregating functions do not use this interface.
          | beam.Map(lambda x: {constants.INPUT_KEY: x})
          | 'Predict' >> predict_extractor._TFMAPredict(
              eval_shared_models={
                  'model1': model1,
                  'model2': model2
              },
              desired_batch_size=3,
              eval_config=eval_config))

      def check_result(got):
        try:
          self.assertLen(got, 4)
          for item in got:
            self.assertIn(constants.FEATURES_KEY, item)
            for feature in ('language', 'age'):
              self.assertIn(feature, item[constants.FEATURES_KEY])
            self.assertIn(constants.LABELS_KEY, item)
            self.assertIn(constants.PREDICTIONS_KEY, item)
            for model in ('model1', 'model2'):
              self.assertIn(model, item[constants.PREDICTIONS_KEY])
            self.assertIn(constants.EXAMPLE_WEIGHTS_KEY, item)
            self.assertAlmostEqual(item[constants.FEATURES_KEY]['age'],
                                   item[constants.EXAMPLE_WEIGHTS_KEY])

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(predict_extracts, check_result)

  def testBatchSizeLimit(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = batch_size_limited_classifier.simple_batch_size_limited_classifier(
        None, temp_eval_export_dir)
    eval_shared_model = model_eval_lib.default_eval_shared_model(
        eval_saved_model_path=eval_export_dir)
    with beam.Pipeline() as pipeline:
      examples = [
          self._makeExample(classes='first', scores=0.0, labels='third'),
          self._makeExample(classes='first', scores=0.0, labels='third'),
          self._makeExample(classes='first', scores=0.0, labels='third'),
          self._makeExample(classes='first', scores=0.0, labels='third'),
      ]
      serialized_examples = [e.SerializeToString() for e in examples]

      predict_extracts = (
          pipeline
          | beam.Create(serialized_examples, reshuffle=False)
          # Our diagnostic outputs, pass types.Extracts throughout, however our
          # aggregating functions do not use this interface.
          | beam.Map(lambda x: {constants.INPUT_KEY: x})
          | 'Predict' >> predict_extractor._TFMAPredict(
              eval_shared_models={'': eval_shared_model}))

      def check_result(got):
        self.assertLen(got, 4)
        for item in got:
          self.assertIn(constants.PREDICTIONS_KEY, item)

      util.assert_that(predict_extracts, check_result)

  # Verify that PredictExtractor can handle models that maps one
  # raw_example_bytes to multiple examples.
  def testPredictMultipleExampleRefPerRawExampleBytes(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        fake_multi_examples_per_input_estimator
        .fake_multi_examples_per_input_estimator(None, temp_eval_export_dir))
    eval_shared_model = model_eval_lib.default_eval_shared_model(
        eval_saved_model_path=eval_export_dir)

    # The trailing zeros make an "empty" output batch.
    raw_example_bytes = ['0', '3', '1', '0', '2', '0', '0', '0', '0']

    def check_result(got):
      try:
        self.assertLen(got, 6)
        self.assertEqual(['3', '3', '3', '1', '2', '2'],
                         [extracts[constants.INPUT_KEY] for extracts in got])

        for item in got:
          self.assertIn(constants.FEATURES_PREDICTIONS_LABELS_KEY, item)
          fpl = item[constants.FEATURES_PREDICTIONS_LABELS_KEY]
          self.assertIn('input_index', fpl.features)
          self.assertIn('example_count', fpl.features)
          self.assertIn('intra_input_index', fpl.features)

      except AssertionError as err:
        raise util.BeamAssertException(err)

    with beam.Pipeline() as pipeline:
      predict_extracts = (
          pipeline
          | beam.Create(raw_example_bytes, reshuffle=False)
          # Our diagnostic outputs, pass types.Extracts throughout, however our
          # aggregating functions do not use this interface.
          | beam.Map(lambda x: {constants.INPUT_KEY: x})
          | 'Predict' >> predict_extractor._TFMAPredict(
              eval_shared_models={'': eval_shared_model}, desired_batch_size=3))

      util.assert_that(predict_extracts, check_result)


if __name__ == '__main__':
  tf.test.main()
