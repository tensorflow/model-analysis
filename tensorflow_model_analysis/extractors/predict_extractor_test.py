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


import apache_beam as beam

from apache_beam.testing import util

import tensorflow as tf

from tensorflow_model_analysis import types
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.eval_saved_model.example_trainers import fake_multi_examples_per_input_estimator
from tensorflow_model_analysis.eval_saved_model.example_trainers import linear_classifier
from tensorflow_model_analysis.extractors import predict_extractor


class PredictExtractorTest(testutil.TensorflowModelAnalysisTest):

  def _getEvalExportDir(self):
    return os.path.join(self._getTempDir(), 'eval_export_dir')

  def testPredict(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = linear_classifier.simple_linear_classifier(
        None, temp_eval_export_dir)
    eval_shared_model = types.EvalSharedModel(model_path=eval_export_dir)

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
          | beam.Create(serialized_examples)
          # Our diagnostic outputs, pass types.ExampleAndExtracts throughout,
          # however our aggregating functions do not use this interface.
          | beam.Map(lambda x: types.ExampleAndExtracts(example=x, extracts={}))
          | 'Predict' >> predict_extractor.TFMAPredict(
              eval_shared_model=eval_shared_model, desired_batch_size=3))

      def check_result(got):
        try:
          self.assertEqual(4, len(got), 'got: %s' % got)
          for item in got:
            extracts_dict = item.extracts
            self.assertTrue('fpl' in extracts_dict)
            fpl = extracts_dict['fpl']
            # Verify fpl contains features, probabilities, and correct labels.
            self.assertIn('language', fpl.features)
            self.assertIn('age', fpl.features)
            self.assertIn('label', fpl.features)
            self.assertIn('probabilities', fpl.predictions)
            self.assertAlmostEqual(fpl.features['label'],
                                   fpl.labels['__labels'])

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(predict_extracts, check_result)

  # Verify that PredictExtractor can handle models that maps one
  # raw_example_bytes to multiple examples.
  def testPredictMultipleExampleRefPerRawExampleBytes(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        fake_multi_examples_per_input_estimator
        .fake_multi_examples_per_input_estimator(None, temp_eval_export_dir))
    eval_shared_model = types.EvalSharedModel(model_path=eval_export_dir)

    # The trailing zeros make an "empty" output batch.
    raw_example_bytes = ['0', '3', '1', '0', '2', '0', '0', '0', '0']

    def check_result(got):
      try:
        self.assertEqual(6, len(got), 'got: %s' % got)
        self.assertEqual(
            ['3', '3', '3', '1', '2', '2'],
            [example_and_extracts.example for example_and_extracts in got])

        for item in got:
          extracts_dict = item.extracts
          self.assertTrue('fpl' in extracts_dict)
          fpl = extracts_dict['fpl']
          self.assertIn('input_index', fpl.features)
          self.assertIn('example_count', fpl.features)
          self.assertIn('intra_input_index', fpl.features)

      except AssertionError as err:
        raise util.BeamAssertException(err)

    with beam.Pipeline() as pipeline:
      predict_extracts = (
          pipeline
          | beam.Create(raw_example_bytes)
          # Our diagnostic outputs, pass types.ExampleAndExtracts throughout,
          # however our aggregating functions do not use this interface.
          | beam.Map(lambda x: types.ExampleAndExtracts(example=x, extracts={}))
          | 'Predict' >> predict_extractor.TFMAPredict(
              eval_shared_model=eval_shared_model, desired_batch_size=3))

      util.assert_that(predict_extracts, check_result)


if __name__ == '__main__':
  tf.test.main()
