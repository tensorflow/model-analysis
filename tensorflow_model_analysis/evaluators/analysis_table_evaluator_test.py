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
"""Tests for analysis_table_evaluator."""

from __future__ import division
from __future__ import print_function

import apache_beam as beam
from apache_beam.testing import util
import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.evaluators import analysis_table_evaluator


class AnalysisTableEvaulatorTest(testutil.TensorflowModelAnalysisTest):

  def testIncludeFilter(self):
    with beam.Pipeline() as pipeline:
      got = (
          pipeline
          | 'Create' >> beam.Create([{
              'a': 1,
              'b': 2
          }])
          | 'EvaluateExtracts' >>
          analysis_table_evaluator.EvaluateExtracts(include=['a']))

      def check_result(got):
        try:
          self.assertEqual(got, [{'a': 1}])
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(got[constants.ANALYSIS_KEY], check_result)

  def testExcludeFilter(self):
    with beam.Pipeline() as pipeline:
      got = (
          pipeline
          | 'Create' >> beam.Create([{
              'a': 1,
              'b': 2
          }])
          | 'EvaluateExtracts' >>
          analysis_table_evaluator.EvaluateExtracts(exclude=['a']))

      def check_result(got):
        try:
          self.assertEqual(got, [{'b': 2}])
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(got[constants.ANALYSIS_KEY], check_result)

  def testNoIncludeOrExcludeFilters(self):
    with beam.Pipeline() as pipeline:
      got = (
          pipeline
          | 'Create' >> beam.Create([{
              constants.INPUT_KEY: 'input',
              'other': 2
          }])
          | 'EvaluateExtracts' >> analysis_table_evaluator.EvaluateExtracts())

      def check_result(got):
        try:
          self.assertEqual(got, [{'other': 2}])
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(got[constants.ANALYSIS_KEY], check_result)

  def testEmptyExcludeFilters(self):
    with beam.Pipeline() as pipeline:
      got = (
          pipeline
          | 'Create' >> beam.Create([{
              constants.INPUT_KEY: 'input',
              'other': 2
          }])
          | 'EvaluateExtracts' >>
          analysis_table_evaluator.EvaluateExtracts(exclude=[]))

      def check_result(got):
        try:
          self.assertEqual(got, [{constants.INPUT_KEY: 'input', 'other': 2}])
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(got[constants.ANALYSIS_KEY], check_result)


if __name__ == '__main__':
  tf.test.main()
