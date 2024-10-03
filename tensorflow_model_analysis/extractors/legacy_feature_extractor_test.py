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

import os
import tempfile
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import types
from tensorflow_model_analysis.eval_saved_model import encoding
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.extractors import legacy_feature_extractor as feature_extractor


class BuildDiagnosticsTableTest(testutil.TensorflowModelAnalysisTest):

  def _getTempDir(self):
    return tempfile.mkdtemp()

  def _exportEvalSavedModel(self, classifier):
    temp_model_location = os.path.join(self._getTempDir(), 'eval_export_dir')
    _, model_location = classifier(None, temp_model_location)
    return model_location

  def testMaterializeFeaturesNoFpl(self):
    example1 = self._makeExample(
        age=3.0, language='english', label=1.0, slice_key='first_slice'
    )

    extracts = {constants.INPUT_KEY: example1.SerializeToString()}
    self.assertRaises(
        RuntimeError, feature_extractor._MaterializeFeatures, extracts
    )

  def testMaterializeFeaturesBadFPL(self):
    example1 = self._makeExample(
        age=3.0, language='english', label=1.0, slice_key='first_slice'
    )

    extracts = {
        constants.INPUT_KEY: example1.SerializeToString(),
        constants.FEATURES_PREDICTIONS_LABELS_KEY: 123,
    }
    self.assertRaises(
        TypeError, feature_extractor._MaterializeFeatures, extracts
    )

  def testMaterializeFeaturesNoMaterializedColumns(self):
    example1 = self._makeExample(
        age=3.0, language='english', label=1.0, slice_key='first_slice'
    )

    features = {
        'f': {encoding.NODE_SUFFIX: np.array([1])},
        's': {
            encoding.NODE_SUFFIX: tf.compat.v1.SparseTensorValue(
                indices=[[0, 5], [1, 2], [3, 6]],
                values=[100.0, 200.0, 300.0],
                dense_shape=[4, 10],
            )
        },
    }
    predictions = {'p': {encoding.NODE_SUFFIX: np.array([2])}}
    labels = {'l': {encoding.NODE_SUFFIX: np.array([3])}}

    extracts = {
        constants.INPUT_KEY: example1.SerializeToString(),
        constants.FEATURES_PREDICTIONS_LABELS_KEY: (
            types.FeaturesPredictionsLabels(
                input_ref=0,
                features=features,
                predictions=predictions,
                labels=labels,
            )
        ),
    }
    fpl = extracts[constants.FEATURES_PREDICTIONS_LABELS_KEY]
    result = feature_extractor._MaterializeFeatures(extracts)
    self.assertIsInstance(result, dict)
    self.assertEqual(
        result[constants.FEATURES_PREDICTIONS_LABELS_KEY], fpl
    )  # should still be there.
    self.assertEqual(
        result['features__f'],
        types.MaterializedColumn(name='features__f', value=[1]),
    )
    self.assertEqual(
        result['predictions__p'],
        types.MaterializedColumn(name='predictions__p', value=[2]),
    )
    self.assertEqual(
        result['labels__l'],
        types.MaterializedColumn(name='labels__l', value=[3]),
    )
    self.assertEqual(
        result['features__s'],
        types.MaterializedColumn(
            name='features__s', value=[100.0, 200.0, 300.0]
        ),
    )

  def testAugmentFPLFromTfExample(self):
    example1 = self._makeExample(
        age=3.0, language='english', label=1.0, slice_key='first_slice', f=0.0
    )

    features = {
        'f': {encoding.NODE_SUFFIX: np.array([1])},
        's': {
            encoding.NODE_SUFFIX: tf.compat.v1.SparseTensorValue(
                indices=[[0, 5], [1, 2], [3, 6]],
                values=[100.0, 200.0, 300.0],
                dense_shape=[4, 10],
            )
        },
    }
    predictions = {'p': {encoding.NODE_SUFFIX: np.array([2])}}
    labels = {'l': {encoding.NODE_SUFFIX: np.array([3])}}

    extracts = {
        constants.INPUT_KEY: example1.SerializeToString(),
        constants.FEATURES_PREDICTIONS_LABELS_KEY: (
            types.FeaturesPredictionsLabels(
                input_ref=0,
                features=features,
                predictions=predictions,
                labels=labels,
            )
        ),
    }
    result = feature_extractor._MaterializeFeatures(
        extracts,
        source=constants.INPUT_KEY,
        dest=constants.FEATURES_PREDICTIONS_LABELS_KEY,
    )
    self.assertIsInstance(result, dict)
    # Assert that materialized columns are not added.
    self.assertNotIn('features__f', result)
    self.assertNotIn('features__age', result)
    # But that tf.Example features not present in FPL are.
    result_fpl = result[constants.FEATURES_PREDICTIONS_LABELS_KEY]
    self.assertEqual(
        result_fpl.features['age'], {encoding.NODE_SUFFIX: np.array([3.0])}
    )
    self.assertEqual(
        result_fpl.features['language'],
        {'node': np.array([['english']], dtype='|S7')},
    )
    self.assertEqual(
        result_fpl.features['slice_key'],
        {'node': np.array([['first_slice']], dtype='|S11')},
    )
    # And that features present in both are not overwritten by tf.Example value.
    self.assertEqual(
        result_fpl.features['f'], {encoding.NODE_SUFFIX: np.array([1])}
    )

  def testMaterializeFeaturesFromTfExample(self):
    example1 = self._makeExample(age=3.0, language='english', label=1.0)

    extracts = {constants.INPUT_KEY: example1.SerializeToString()}
    input_example = extracts[constants.INPUT_KEY]
    result = feature_extractor._MaterializeFeatures(
        extracts, source=constants.INPUT_KEY
    )
    self.assertIsInstance(result, dict)
    self.assertEqual(
        result[constants.INPUT_KEY], input_example
    )  # should still be there.
    self.assertEqual(
        result['features__age'],
        types.MaterializedColumn(name='features__age', value=[3.0]),
    )
    self.assertEqual(
        result['features__language'],
        types.MaterializedColumn(name='features__language', value=[b'english']),
    )
    self.assertEqual(
        result['features__label'],
        types.MaterializedColumn(name='features__label', value=[1.0]),
    )

  def testMaterializeFeaturesWithBadSource(self):
    example1 = self._makeExample(age=3.0, language='english', label=1.0)

    extracts = {constants.INPUT_KEY: example1.SerializeToString()}

    self.assertRaises(
        RuntimeError,
        feature_extractor._MaterializeFeatures,
        extracts,
        None,
        '10',
    )

  def testMaterializeFeaturesWithExcludes(self):
    example1 = self._makeExample(
        age=3.0, language='english', label=1.0, slice_key='first_slice'
    )

    features = {
        'f': {encoding.NODE_SUFFIX: np.array([1])},
        's': {
            encoding.NODE_SUFFIX: tf.compat.v1.SparseTensorValue(
                indices=[[0, 5], [1, 2], [3, 6]],
                values=[100.0, 200.0, 300.0],
                dense_shape=[4, 10],
            )
        },
    }
    predictions = {'p': {encoding.NODE_SUFFIX: np.array([2])}}
    labels = {'l': {encoding.NODE_SUFFIX: np.array([3])}}

    extracts = {
        constants.INPUT_KEY: example1.SerializeToString(),
        constants.FEATURES_PREDICTIONS_LABELS_KEY: (
            types.FeaturesPredictionsLabels(
                input_ref=0,
                features=features,
                predictions=predictions,
                labels=labels,
            )
        ),
    }
    result = feature_extractor._MaterializeFeatures(extracts, excludes=['s'])
    self.assertNotIn('features__s', result)


