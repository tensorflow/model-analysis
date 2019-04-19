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
"""Test for using the tfma_unit library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Standard Imports

import apache_beam as beam
import tensorflow as tf
from tensorflow_model_analysis.api import tfma_unit
from tensorflow_model_analysis.eval_saved_model.example_trainers import fixed_prediction_estimator
from tensorflow_model_analysis.eval_saved_model.example_trainers import fixed_prediction_estimator_extra_fields
from tensorflow_model_analysis.post_export_metrics import metric_keys
from tensorflow_model_analysis.post_export_metrics import post_export_metrics
from tensorflow_model_analysis.slicer import slicer


class TFMAUnitTest(tfma_unit.TestCase):

  def _getEvalExportDir(self):
    return os.path.join(self._getTempDir(), 'eval_export_dir')

  def testAssertMetricsComputedWithoutBeamAre(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        fixed_prediction_estimator.simple_fixed_prediction_estimator(
            None, temp_eval_export_dir))
    examples = [
        self.makeExample(prediction=0.0, label=1.0),
        self.makeExample(prediction=0.7, label=0.0),
        self.makeExample(prediction=0.8, label=1.0),
        self.makeExample(prediction=1.0, label=1.0)
    ]
    self.assertMetricsComputedWithoutBeamAre(
        eval_saved_model_path=eval_export_dir,
        serialized_examples=examples,
        expected_metrics={'average_loss': (1.0 + 0.49 + 0.04 + 0.00) / 4.0})

  def testBoundedValueChecks(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        fixed_prediction_estimator.simple_fixed_prediction_estimator(
            None, temp_eval_export_dir))
    examples = [
        self.makeExample(prediction=0.8, label=1.0),
    ]

    self.assertMetricsComputedWithBeamAre(
        eval_saved_model_path=eval_export_dir,
        serialized_examples=examples,
        expected_metrics={'average_loss': 0.04})

    self.assertMetricsComputedWithoutBeamAre(
        eval_saved_model_path=eval_export_dir,
        serialized_examples=examples,
        expected_metrics={
            'average_loss':
                tfma_unit.BoundedValue(lower_bound=0.03, upper_bound=0.05)
        })

    with self.assertRaisesRegexp(
        AssertionError, 'expecting key average_loss to have value between'):
      self.assertMetricsComputedWithoutBeamAre(
          eval_saved_model_path=eval_export_dir,
          serialized_examples=examples,
          expected_metrics={
              'average_loss': tfma_unit.BoundedValue(upper_bound=0.01)
          })

    with self.assertRaisesRegexp(
        AssertionError, 'expecting key average_loss to have value between'):
      self.assertMetricsComputedWithoutBeamAre(
          eval_saved_model_path=eval_export_dir,
          serialized_examples=examples,
          expected_metrics={
              'average_loss': tfma_unit.BoundedValue(lower_bound=0.10)
          })

  def testAssertMetricsComputedWithBeamAre(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        fixed_prediction_estimator.simple_fixed_prediction_estimator(
            None, temp_eval_export_dir))
    examples = [
        self.makeExample(prediction=0.0, label=1.0),
        self.makeExample(prediction=0.7, label=0.0),
        self.makeExample(prediction=0.8, label=1.0),
        self.makeExample(prediction=1.0, label=1.0)
    ]
    self.assertMetricsComputedWithBeamAre(
        eval_saved_model_path=eval_export_dir,
        serialized_examples=examples,
        expected_metrics={'average_loss': (1.0 + 0.49 + 0.04 + 0.00) / 4.0})

  def testAssertGeneralMetricsComputedWithBeamAre(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        fixed_prediction_estimator_extra_fields
        .simple_fixed_prediction_estimator_extra_fields(None,
                                                        temp_eval_export_dir))
    examples = [
        self.makeExample(
            prediction=0.0,
            label=0.0,
            fixed_string='negative_slice',
            fixed_float=0.0,
            fixed_int=0),
        self.makeExample(
            prediction=0.2,
            label=0.0,
            fixed_string='negative_slice',
            fixed_float=0.0,
            fixed_int=0),
        self.makeExample(
            prediction=0.4,
            label=0.0,
            fixed_string='negative_slice',
            fixed_float=0.0,
            fixed_int=0),
        self.makeExample(
            prediction=0.8,
            label=1.0,
            fixed_string='positive_slice',
            fixed_float=0.0,
            fixed_int=0),
        self.makeExample(
            prediction=0.9,
            label=1.0,
            fixed_string='positive_slice',
            fixed_float=0.0,
            fixed_int=0),
        self.makeExample(
            prediction=1.0,
            label=1.0,
            fixed_string='positive_slice',
            fixed_float=0.0,
            fixed_int=0),
    ]
    expected_slice_metrics = {}
    expected_slice_metrics[()] = {
        'average_loss': (0.00 + 0.04 + 0.16 + 0.04 + 0.01 + 0.00) / 6.0,
        'mae':
            0.15,
        # Note that we don't check the exact value because of numerical errors.
        metric_keys.AUC:
            tfma_unit.BoundedValue(0.98, 1.00),
    }
    # We don't check AUC for the positive / negative only slices because
    # it's not clear what the value should be.
    expected_slice_metrics[(('fixed_string', b'negative_slice'),)] = {
        'average_loss': (0.00 + 0.04 + 0.16) / 3.0,
        'mae': 0.2,
    }
    expected_slice_metrics[(('fixed_string', b'positive_slice'),)] = {
        'average_loss': (0.04 + 0.01 + 0.00) / 3.0,
        'mae': 0.1,
    }

    def add_metrics(features, predictions, labels):
      del features
      metric_ops = {
          'mae': tf.compat.v1.metrics.mean_absolute_error(labels, predictions),
      }
      return metric_ops

    with beam.Pipeline() as pipeline:
      examples_pcollection = pipeline | 'Create' >> beam.Create(examples)
      self.assertGeneralMetricsComputedWithBeamAre(
          eval_saved_model_path=eval_export_dir,
          examples_pcollection=examples_pcollection,
          slice_spec=[
              slicer.SingleSliceSpec(),
              slicer.SingleSliceSpec(columns=['fixed_string'])
          ],
          add_metrics_callbacks=[add_metrics,
                                 post_export_metrics.auc()],
          expected_slice_metrics=expected_slice_metrics)


if __name__ == '__main__':
  tf.test.main()
