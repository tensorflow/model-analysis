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
"""Tests for feature_extractor."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import numpy as np
import tensorflow as tf

from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.eval_saved_model import encoding
from tensorflow_model_analysis.eval_saved_model import load
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.extractors import feature_extractor


class BuildDiagnosticsTableTest(testutil.TensorflowModelAnalysisTest):

  def _getTempDir(self):
    return tempfile.mkdtemp()

  def _exportEvalSavedModel(self, classifier):
    temp_model_location = os.path.join(self._getTempDir(), 'eval_export_dir')
    _, model_location = classifier(None, temp_model_location)
    return model_location

  def testMaterializeFeaturesNoFpl(self):
    example1 = self._makeExample(
        age=3.0, language='english', label=1.0, slice_key='first_slice')

    example_and_extracts = types.ExampleAndExtracts(
        example=example1.SerializeToString(),
        extracts={})
    self.assertRaises(RuntimeError, feature_extractor._MaterializeFeatures,
                      example_and_extracts)

  def testMaterializeFeaturesBadFPL(self):
    example1 = self._makeExample(
        age=3.0, language='english', label=1.0, slice_key='first_slice')

    example_and_extracts = types.ExampleAndExtracts(
        example=example1.SerializeToString(),
        extracts={'fpl': 123})
    self.assertRaises(TypeError, feature_extractor._MaterializeFeatures,
                      example_and_extracts)

  def testMaterializeFeaturesNoMaterializedColumns(self):
    example1 = self._makeExample(
        age=3.0, language='english', label=1.0, slice_key='first_slice')

    features = {
        'f': {encoding.NODE_SUFFIX: np.array([1])},
        's': {encoding.NODE_SUFFIX: tf.SparseTensorValue(
            indices=[[0, 5], [1, 2], [3, 6]],
            values=[100., 200., 300.],
            dense_shape=[4, 10])}
    }
    predictions = {'p': {encoding.NODE_SUFFIX: np.array([2])}}
    labels = {'l': {encoding.NODE_SUFFIX: np.array([3])}}

    example_and_extracts = types.ExampleAndExtracts(
        example=example1.SerializeToString(),
        extracts={'fpl': load.FeaturesPredictionsLabels(features,
                                                        predictions,
                                                        labels)})
    fpl = example_and_extracts.extracts[
        constants.FEATURES_PREDICTIONS_LABELS_KEY]
    result = feature_extractor._MaterializeFeatures(example_and_extracts)
    self.assertTrue(isinstance(result, types.ExampleAndExtracts))
    self.assertEqual(result.extracts['fpl'], fpl)  # should still be there.
    self.assertEqual(result.extracts['f'],
                     types.MaterializedColumn(name='f', value=[1]))
    self.assertEqual(result.extracts['p'],
                     types.MaterializedColumn(name='p', value=[2]))
    self.assertEqual(result.extracts['l'],
                     types.MaterializedColumn(name='l', value=[3]))
    self.assertEqual(result.extracts['s'],
                     types.MaterializedColumn(
                         name='s', value=[100., 200., 300.]))

if __name__ == '__main__':
  tf.test.main()
