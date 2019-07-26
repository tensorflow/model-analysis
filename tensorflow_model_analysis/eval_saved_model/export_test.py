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
"""Simple tests for export."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_model_analysis.eval_saved_model import encoding
from tensorflow_model_analysis.eval_saved_model import export
from tensorflow_model_analysis.eval_saved_model import testutil


class ExportTest(testutil.TensorflowModelAnalysisTest):

  def testEvalInputReceiverReceiverTensorKeyCheck(self):
    with self.assertRaisesRegexp(ValueError, 'exactly one key named examples'):
      export.EvalInputReceiver(
          features={},
          labels={},
          receiver_tensors={'bad_key': tf.constant(0.0)})

  def testMultipleCallsToEvalInputReceiver(self):
    graph = tf.Graph()
    with graph.as_default():
      features1 = {'apple': tf.constant(1.0), 'banana': tf.constant(2.0)}
      labels1 = tf.constant(3.0)
      receiver_tensors1 = {'examples': tf.compat.v1.placeholder(tf.string)}

      features2 = {'cherry': tf.constant(3.0)}
      labels2 = {'alpha': tf.constant(4.0), 'bravo': tf.constant(5.0)}
      receiver_tensors2 = {'examples': tf.compat.v1.placeholder(tf.string)}

      export.EvalInputReceiver(
          features=features1,
          labels=labels1,
          receiver_tensors=receiver_tensors1)

      feature_keys_collection_name = encoding.with_suffix(
          encoding.FEATURES_COLLECTION, encoding.KEY_SUFFIX)
      feature_nodes_collection_name = encoding.with_suffix(
          encoding.FEATURES_COLLECTION, encoding.NODE_SUFFIX)
      label_keys_collection_name = encoding.with_suffix(
          encoding.LABELS_COLLECTION, encoding.KEY_SUFFIX)
      label_nodes_collection_name = encoding.with_suffix(
          encoding.LABELS_COLLECTION, encoding.NODE_SUFFIX)

      self.assertEqual(
          2, len(tf.compat.v1.get_collection(feature_keys_collection_name)))
      self.assertEqual(
          2, len(tf.compat.v1.get_collection(feature_nodes_collection_name)))
      self.assertEqual(
          1, len(tf.compat.v1.get_collection(label_keys_collection_name)))
      self.assertEqual(
          1, len(tf.compat.v1.get_collection(label_nodes_collection_name)))
      self.assertEqual(
          1, len(tf.compat.v1.get_collection(encoding.EXAMPLE_REF_COLLECTION)))
      self.assertEqual(
          1, len(tf.compat.v1.get_collection(encoding.TFMA_VERSION_COLLECTION)))

      # Call again with a different set of features, labels and receiver
      # tensors, check that the latest call overrides the earlier one.
      #
      # Note that we only check the lengths of some collections: more detailed
      # checks would require the test to include more knowledge about the
      # details of how exporting is done.
      export.EvalInputReceiver(
          features=features2,
          labels=labels2,
          receiver_tensors=receiver_tensors2)
      self.assertEqual(
          1, len(tf.compat.v1.get_collection(feature_keys_collection_name)))
      self.assertEqual(
          1, len(tf.compat.v1.get_collection(feature_nodes_collection_name)))
      self.assertEqual(
          2, len(tf.compat.v1.get_collection(label_keys_collection_name)))
      self.assertEqual(
          2, len(tf.compat.v1.get_collection(label_nodes_collection_name)))
      self.assertEqual(
          1, len(tf.compat.v1.get_collection(encoding.EXAMPLE_REF_COLLECTION)))
      self.assertEqual(
          1, len(tf.compat.v1.get_collection(encoding.TFMA_VERSION_COLLECTION)))


if __name__ == '__main__':
  tf.test.main()
