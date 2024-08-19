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
"""Test for extractor."""

import apache_beam as beam
from apache_beam.testing import util
import tensorflow as tf
from tensorflow_model_analysis.extractors import extractor
from tensorflow_model_analysis.utils import test_util


class ExtractorTest(test_util.TensorflowModelAnalysisTest):

  def testFilterRaisesValueError(self):
    with self.assertRaises(ValueError):
      with beam.Pipeline() as pipeline:
        _ = (
            pipeline
            | 'Create' >> beam.Create([])
            | 'Filter' >> extractor.Filter(include=['a'], exclude=['b'])
        )

  def testIncludeFilter(self):
    with beam.Pipeline() as pipeline:
      got = (
          pipeline
          | 'Create' >> beam.Create([{'a': 1, 'b': 2, 'c': 3, 'd': 4}])
          | 'Filter' >> extractor.Filter(include=['a', 'c'])
      )

      def check_result(got):
        try:
          self.assertEqual(got, [{'a': 1, 'c': 3}])
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(got, check_result)

  def testIncludeFilterWithDict(self):
    with beam.Pipeline() as pipeline:
      got = (
          pipeline
          | 'Create'
          >> beam.Create([{
              'a': 1,
              'b': {'b2': 2},
              'c': {'c2': {'c21': 3, 'c22': 4}},
              'd': {'d2': 4},
          }])
          | 'Filter'
          >> extractor.Filter(include={'b': {}, 'c': {'c2': {'c21': {}}}})
      )

      def check_result(got):
        try:
          self.assertEqual(got, [{'b': {'b2': 2}, 'c': {'c2': {'c21': 3}}}])
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(got, check_result)

  def testExludeFilter(self):
    with beam.Pipeline() as pipeline:
      got = (
          pipeline
          | 'Create' >> beam.Create([{'a': 1, 'b': 2, 'c': 3, 'd': 4}])
          | 'Filter' >> extractor.Filter(exclude=['b', 'd'])
      )

      def check_result(got):
        try:
          self.assertEqual(got, [{'a': 1, 'c': 3}])
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(got, check_result)

  def testExcludeFilterWithDict(self):
    with beam.Pipeline() as pipeline:
      got = (
          pipeline
          | 'Create'
          >> beam.Create([{
              'a': 1,
              'b': {'b2': 2},
              'c': {'c2': {'c21': 3, 'c22': 4}},
              'd': {'d2': 4},
          }])
          | 'Filter'
          >> extractor.Filter(exclude={'b': {}, 'c': {'c2': {'c21': {}}}})
      )

      def check_result(got):
        try:
          self.assertEqual(
              got, [{'a': 1, 'c': {'c2': {'c22': 4}}, 'd': {'d2': 4}}]
          )
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(got, check_result)


