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
"""Test for using the ModelAgnosticExtractor API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Standard Imports

import apache_beam as beam

from apache_beam.testing import util

import tensorflow as tf

from tensorflow_model_analysis import constants
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.model_agnostic_eval import model_agnostic_extractor
from tensorflow_model_analysis.model_agnostic_eval import model_agnostic_predict as agnostic_predict


class ModelAgnosticExtractorTest(testutil.TensorflowModelAnalysisTest):

  def testExtract(self):
    with beam.Pipeline() as pipeline:
      examples = [
          self._makeExample(
              age=3.0, language='english', probabilities=[1.0, 2.0], label=1.0),
          self._makeExample(
              age=3.0, language='chinese', probabilities=[2.0, 3.0], label=0.0),
          self._makeExample(
              age=4.0, language='english', probabilities=[3.0, 4.0], label=1.0),
          self._makeExample(
              age=5.0, language='chinese', probabilities=[4.0, 5.0], label=0.0),
      ]
      serialized_examples = [e.SerializeToString() for e in examples]

      # Set up a config to bucket our example keys.
      feature_map = {
          'age': tf.io.FixedLenFeature([], tf.float32),
          'language': tf.io.VarLenFeature(tf.string),
          'probabilities': tf.io.FixedLenFeature([2], tf.float32),
          'label': tf.io.FixedLenFeature([], tf.float32)
      }
      model_agnostic_config = agnostic_predict.ModelAgnosticConfig(
          label_keys=['label'],
          prediction_keys=['probabilities'],
          feature_spec=feature_map)

      fpl_extracts = (
          pipeline
          | beam.Create(serialized_examples)
          # Our diagnostic outputs, pass types.Extracts throughout, however our
          # aggregating functions do not use this interface.
          | beam.Map(lambda x: {constants.INPUT_KEY: x})
          | 'Extract' >> model_agnostic_extractor.ModelAgnosticExtract(
              model_agnostic_config=model_agnostic_config, desired_batch_size=3)
      )

      def check_result(got):
        try:
          self.assertEqual(4, len(got), 'got: %s' % got)
          for item in got:
            self.assertIn(constants.FEATURES_PREDICTIONS_LABELS_KEY, item)
            fpl = item[constants.FEATURES_PREDICTIONS_LABELS_KEY]
            # Verify fpl contains features, probabilities, and correct labels.
            self.assertIn('language', fpl.features)
            self.assertIn('age', fpl.features)
            self.assertIn('label', fpl.labels)
            self.assertIn('probabilities', fpl.predictions)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(fpl_extracts, check_result)


if __name__ == '__main__':
  tf.test.main()
