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

    temp_eval_export_dir = self._getEvalExportDir()
    exporter_instance = exporter_class(
        name='TFMA',
        eval_input_receiver_fn=estimator_metadata['eval_input_receiver_fn'],
        serving_input_receiver_fn=estimator_metadata[
            'serving_input_receiver_fn'])

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
    example1 = self._makeExample(prediction=0.9, label=0.0)
    features_predictions_labels = self.predict_injective_single_example(
        eval_saved_model, example1.SerializeToString())
    eval_saved_model.perform_metrics_update(features_predictions_labels)

    metric_values = eval_saved_model.get_metric_values()
    self.assertDictElementsAlmostEqual(metric_values, {'average_loss': 0.81})

    # Check the serving graph.
    estimator = tf.contrib.estimator.SavedModelEstimator(eval_export_dir)

    def predict_input_fn():
      return {'inputs': tf.constant([example1.SerializeToString()])}

    predictions = next(estimator.predict(predict_input_fn))
    self.assertAllClose(predictions['outputs'], np.array([0.9]))

  def testFinalExporter(self):
    self.runTestForExporter(exporter.FinalExporter)

  def testLatestExporter(self):
    self.runTestForExporter(exporter.LatestExporter)


if __name__ == '__main__':
  tf.test.main()
