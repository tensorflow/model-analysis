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
"""Test for using the contrib export API."""

import os
import tempfile

import tensorflow as tf
from tensorflow_model_analysis.contrib import export
from tensorflow_model_analysis.eval_saved_model import constants
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.eval_saved_model.example_trainers import dnn_classifier


class ExportTest(testutil.TensorflowModelAnalysisTest):

  def _getTempDir(self):
    return tempfile.mkdtemp()

  def testExportEvalSavedmodelWithFeatureMetadata(self):
    temp_eval_export_dir = os.path.join(self._getTempDir(), 'eval_export_dir')
    estimator_metadata = dnn_classifier.get_simple_dnn_classifier_and_metadata()
    estimator_metadata['estimator'].train(
        input_fn=estimator_metadata['train_input_fn'], steps=1000
    )

    graph = tf.Graph()
    with graph.as_default():
      eval_export_dir = export.export_eval_savedmodel_with_feature_metadata(
          estimator=estimator_metadata['estimator'],
          export_dir_base=temp_eval_export_dir,
          eval_input_receiver_fn=estimator_metadata['eval_input_receiver_fn'],
          serving_input_receiver_fn=(
              estimator_metadata['serving_input_receiver_fn']
          ),
      )

      sess = tf.compat.v1.Session(graph=graph)
      tf.compat.v1.saved_model.loader.load(
          sess, [constants.EVAL_TAG], eval_export_dir
      )

      feature_metadata = export.load_and_resolve_feature_metadata(
          eval_export_dir, graph
      )

      features = feature_metadata['features']
      feature_columns = feature_metadata['feature_columns']
      associated_tensors = feature_metadata['associated_tensors']
      self.assertSetEqual(set(['language', 'age']), set(features.keys()))
      self.assertIsInstance(features['language'], tf.SparseTensor)
      self.assertIsInstance(features['age'], tf.Tensor)

      self.assertEqual(2, len(feature_columns))
      self.assertEqual(2, len(associated_tensors))

      # Since we don't have references to the expected Tensors in the graph,
      # we simply check that the keys are present, and the resolved Tensors
      # contain the feature names.
      cols_to_tensors = {}
      cols_to_tensors[feature_columns[0]['key']] = associated_tensors[0]
      cols_to_tensors[feature_columns[1]['key']] = associated_tensors[1]

      self.assertSetEqual(
          set(['language_embedding', 'age']), set(cols_to_tensors.keys())
      )
      self.assertIn('age', cols_to_tensors['age'].name)
      self.assertIn(
          'language_embedding', cols_to_tensors['language_embedding'].name
      )


