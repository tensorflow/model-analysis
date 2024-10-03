# Copyright 2019 Google LLC
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
"""Tests for utils for evaluations using the EvalMetricsGraph."""


import pytest
import os

import apache_beam as beam
from apache_beam.testing import util
import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.eval_saved_model.example_trainers import linear_classifier
from tensorflow_model_analysis.eval_saved_model.example_trainers import multi_head
from tensorflow_model_analysis.evaluators import eval_saved_model_util
from tensorflow_model_analysis.metrics import metric_types


@pytest.mark.xfail(run=False, reason="PR 183 This class contains tests that fail and needs to be fixed. "
"If all tests pass, please remove this mark.")
class EvalSavedModelUtilTest(testutil.TensorflowModelAnalysisTest):

  def _getExportDir(self):
    return os.path.join(self._getTempDir(), 'export_dir')

  def testNativeEvalSavedModelMetricComputations(self):
    temp_export_dir = self._getExportDir()
    _, export_dir = linear_classifier.simple_linear_classifier(
        None, temp_export_dir
    )

    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir
    )

    computation = (
        eval_saved_model_util.metric_computations_using_eval_saved_model(
            '', eval_shared_model.model_loader
        )[0]
    )

    examples = [
        self._makeExample(age=3.0, language='english', label=1.0),
        self._makeExample(age=3.0, language='chinese', label=0.0),
        self._makeExample(age=4.0, language='english', label=1.0),
        self._makeExample(age=5.0, language='chinese', label=0.0),
    ]

    extracts = []
    for e in examples:
      extracts.append({constants.INPUT_KEY: e.SerializeToString()})

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create(extracts)
          | 'Process' >> beam.ParDo(computation.preprocessors[0])
          | 'ToStandardMetricInputs'
          >> beam.Map(metric_types.StandardMetricInputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeMetric' >> beam.CombinePerKey(computation.combiner)
      )

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          self.assertDictElementsAlmostEqual(
              got_metrics,
              {
                  metric_types.MetricKey(
                      name='accuracy', example_weighted=None
                  ): 1.0,
                  metric_types.MetricKey(
                      name='label/mean', example_weighted=None
                  ): 0.5,
                  metric_types.MetricKey(
                      name='my_mean_age', example_weighted=None
                  ): 3.75,
                  metric_types.MetricKey(
                      name='my_mean_age_times_label', example_weighted=None
                  ): 1.75,
              },
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testNativeEvalSavedModelMetricComputationsWithMultiHead(self):
    temp_export_dir = self._getExportDir()
    _, export_dir = multi_head.simple_multi_head(None, temp_export_dir)

    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir
    )

    computation = (
        eval_saved_model_util.metric_computations_using_eval_saved_model(
            '', eval_shared_model.model_loader
        )[0]
    )

    examples = [
        self._makeExample(
            age=1.0,
            language='english',
            english_label=1.0,
            chinese_label=0.0,
            other_label=0.0,
        ),
        self._makeExample(
            age=1.0,
            language='chinese',
            english_label=0.0,
            chinese_label=1.0,
            other_label=0.0,
        ),
        self._makeExample(
            age=2.0,
            language='english',
            english_label=1.0,
            chinese_label=0.0,
            other_label=0.0,
        ),
        self._makeExample(
            age=2.0,
            language='other',
            english_label=0.0,
            chinese_label=1.0,
            other_label=1.0,
        ),
    ]

    extracts = []
    for e in examples:
      extracts.append({constants.INPUT_KEY: e.SerializeToString()})

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create(extracts)
          | 'Process' >> beam.ParDo(computation.preprocessors[0])
          | 'ToStandardMetricInputs'
          >> beam.Map(metric_types.StandardMetricInputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeMetric' >> beam.CombinePerKey(computation.combiner)
      )

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          chinese_accuracy_key = metric_types.MetricKey(
              name='accuracy', output_name='chinese_head', example_weighted=None
          )
          chinese_mean_label_key = metric_types.MetricKey(
              name='label/mean',
              output_name='chinese_head',
              example_weighted=None,
          )
          english_accuracy_key = metric_types.MetricKey(
              name='accuracy', output_name='english_head', example_weighted=None
          )
          english_mean_label_key = metric_types.MetricKey(
              name='label/mean',
              output_name='english_head',
              example_weighted=None,
          )
          other_accuracy_key = metric_types.MetricKey(
              name='accuracy', output_name='other_head', example_weighted=None
          )
          other_mean_label_key = metric_types.MetricKey(
              name='label/mean', output_name='other_head', example_weighted=None
          )
          self.assertDictElementsAlmostEqual(
              got_metrics,
              {
                  chinese_accuracy_key: 0.75,
                  chinese_mean_label_key: 0.5,
                  english_accuracy_key: 1.0,
                  english_mean_label_key: 0.5,
                  other_accuracy_key: 1.0,
                  other_mean_label_key: 0.25,
              },
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')


