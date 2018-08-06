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
"""Test for using the contrib model_eval_lib API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile


import apache_beam as beam
from apache_beam.testing import util

import numpy as np
import tensorflow as tf

from tensorflow_model_analysis import types
from tensorflow_model_analysis.contrib import model_eval_lib as contrib
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.eval_saved_model.example_trainers import linear_classifier


class BuildDiagnosticsTableTest(testutil.TensorflowModelAnalysisTest):

  def _getTempDir(self):
    return tempfile.mkdtemp()

  def _exportEvalSavedModel(self, classifier):
    temp_eval_export_dir = os.path.join(self._getTempDir(), 'eval_export_dir')
    _, eval_export_dir = classifier(None, temp_eval_export_dir)
    return eval_export_dir

  def _assertMaterializedColumnsExist(self, got_values_dict,
                                      expected_column_names):
    for key in expected_column_names:
      self.assertIn(key, got_values_dict)
      got_column = got_values_dict[key]
      self.assertTrue(isinstance(got_column, types.MaterializedColumn))

  def _assertMaterializedColumns(self, got_values_dict, expected_values_dict,
                                 places=3):
    for key, expected_column in expected_values_dict.items():
      self.assertIn(key, got_values_dict)

      got_column = got_values_dict[key]
      self.assertTrue(isinstance(got_column, types.MaterializedColumn))

      if (isinstance(expected_column.value, np.ndarray) or
          isinstance(expected_column.value, list)):
        # verify the arrays are identical
        for got_v, expected_v in zip(got_column.value, expected_column.value):
          self.assertAlmostEqual(got_v, expected_v, places,
                                 msg='key %s' % key)
      else:
        self.assertAlmostEqual(got_column.value, expected_column.value, places,
                               msg='key %s' % key)

  def testBuildDiagnosticsTable(self):
    model_location = self._exportEvalSavedModel(
        linear_classifier.simple_linear_classifier)

    example1 = self._makeExample(
        age=3.0, language='english', label=1.0, slice_key='first_slice')

    with beam.Pipeline() as pipeline:
      result = (pipeline
                | beam.Create([example1.SerializeToString()])
                | contrib.BuildDiagnosticTable(model_location))

      def check_result(got):
        self.assertEqual(1, len(got), 'got: %s' % got)
        (_, extracts) = got[0]

        # Values of type MaterializedColumn are emitted to signal to
        # downstream sink components to output the data to file.
        materialized_dict = dict((k, v) for k, v in extracts.iteritems()
                                 if isinstance(v, types.MaterializedColumn))
        self._assertMaterializedColumns(
            materialized_dict,
            {
                # Slice key
                'slice_key': types.MaterializedColumn(
                    name=u'slice_key', value=['first_slice']),

                # Features
                'language': types.MaterializedColumn(
                    name=u'language', value=['english']),
                'age': types.MaterializedColumn(
                    name=u'age', value=np.array([3.], dtype=np.float32)),

                # Label
                'label': types.MaterializedColumn(
                    name=u'label', value=np.array([1.], dtype=np.float32)),
                '__labels': types.MaterializedColumn(
                    name=u'__labels', value=np.array([1.], dtype=np.float32)),
            })
        self._assertMaterializedColumnsExist(
            materialized_dict,
            ['logits', 'probabilities', 'classes', 'logistic', 'class_ids'])

      util.assert_that(result, check_result)


if __name__ == '__main__':
  tf.test.main()
