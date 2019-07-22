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
"""Test for exporters.

Note that we actually train and export models within these tests.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis.eval_saved_model import exporter
from tensorflow_model_analysis.eval_saved_model import load
from tensorflow_model_analysis.eval_saved_model import testutil

from tensorflow_model_analysis.eval_saved_model.example_trainers import fixed_prediction_estimator


class ExporterTest(testutil.TensorflowModelAnalysisTest):

  def _getEvalExportDir(self):
    return os.path.join(self._getTempDir(), 'eval_export_dir')

  def runTestForExporter(self, exporter_class):
    estimator_metadata = (
        fixed_prediction_estimator
        .get_simple_fixed_prediction_estimator_and_metadata())

    exporter_name = 'TFMA'
    temp_eval_export_dir = self._getEvalExportDir()
    exporter_instance = exporter_class(
        name=exporter_name,
        eval_input_receiver_fn=estimator_metadata['eval_input_receiver_fn'],
        serving_input_receiver_fn=estimator_metadata[
            'serving_input_receiver_fn'])

    self.assertEqual(exporter_name, exporter_instance.name)

    estimator_metadata['estimator'].train(
        input_fn=estimator_metadata['train_input_fn'], steps=100)

    eval_export_dir = exporter_instance.export(
        estimator=estimator_metadata['estimator'],
        export_path=temp_eval_export_dir,
        checkpoint_path=None,
        eval_result=None,
        is_the_final_export=True)

    # Check the eval graph.
    eval_saved_model = load.EvalSavedModel(eval_export_dir)
    example1 = self._makeExample(prediction=0.9, label=0.0).SerializeToString()
    eval_saved_model.metrics_reset_update_get(example1)

    metric_values = eval_saved_model.get_metric_values()
    self.assertDictElementsAlmostEqual(metric_values, {'average_loss': 0.81})

    # Check the serving graph.
    # TODO(b/124466113): Remove tf.compat.v2 once TF 2.0 is the default.
    if hasattr(tf, 'compat.v2'):
      imported = tf.compat.v2.saved_model.load(
          eval_export_dir, tags=tf.saved_model.SERVING)
      predictions = imported.signatures[
          tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY](
              inputs=tf.constant([example1.SerializeToString()]))
      self.assertAllClose(predictions['outputs'], np.array([[0.9]]))

  def testFinalExporter(self):
    self.runTestForExporter(exporter.FinalExporter)

  def testLatestExporter(self):
    self.runTestForExporter(exporter.LatestExporter)

  def testAdaptToRemoveMetrics(self):
    estimator_metadata = (
        fixed_prediction_estimator
        .get_simple_fixed_prediction_estimator_and_metadata())

    exporter_name = 'TFMA'
    temp_eval_export_dir = self._getEvalExportDir()
    exporter_instance = exporter.FinalExporter(
        name=exporter_name,
        eval_input_receiver_fn=estimator_metadata['eval_input_receiver_fn'],
        serving_input_receiver_fn=estimator_metadata[
            'serving_input_receiver_fn'])
    exporter_instance = exporter.adapt_to_remove_metrics(
        exporter_instance, ['average_loss'])

    self.assertEqual(exporter_name, exporter_instance.name)

    estimator_metadata['estimator'].train(
        input_fn=estimator_metadata['train_input_fn'], steps=100)
    eval_export_dir = exporter_instance.export(
        estimator=estimator_metadata['estimator'],
        export_path=temp_eval_export_dir,
        checkpoint_path=None,
        eval_result=None,
        is_the_final_export=True)

    # Check the eval graph.
    eval_saved_model = load.EvalSavedModel(eval_export_dir)
    example1 = self._makeExample(prediction=0.9, label=0.0).SerializeToString()
    eval_saved_model.metrics_reset_update_get(example1)

    metric_values = eval_saved_model.get_metric_values()
    self.assertNotIn('average_loss', metric_values)

    # Check the serving graph.
    # TODO(b/124466113): Remove tf.compat.v2 once TF 2.0 is the default.
    if hasattr(tf, 'compat.v2'):
      imported = tf.compat.v2.saved_model.load(
          eval_export_dir, tags=tf.saved_model.SERVING)
      predictions = imported.signatures[
          tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY](
              inputs=tf.constant([example1.SerializeToString()]))
      self.assertAllClose(predictions['outputs'], np.array([[0.9]]))


if __name__ == '__main__':
  tf.test.main()
