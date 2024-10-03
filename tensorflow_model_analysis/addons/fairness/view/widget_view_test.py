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
"""Tests for tensorflow_model_analysis.addons.fairness.view.widget_view."""

import tensorflow as tf
from tensorflow_model_analysis.addons.fairness.view import widget_view
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.slicer import slicer_lib as slicer
from tensorflow_model_analysis.view import view_types


class WidgetViewTest(testutil.TensorflowModelAnalysisTest):

  def _makeEvalResult(
      self,
      slices=((), (('slice', '1'),)),
      metrics_names=('metrics1', 'metrics2'),
  ):
    metrics = {'': {'': {}}}
    for metrics_name in metrics_names:
      metrics[''][''][metrics_name] = {'double_value': {'value': 0.5}}

    slicing_metrics = [(s, metrics) for s in slices]
    return view_types.EvalResult(
        slicing_metrics=slicing_metrics,
        plots=None,
        attributions=None,
        config=None,
        data_location=None,
        file_format=None,
        model_location=None,
    )

  def testConvertEvalResultToUIInputWithAllDefaultParams(self):
    eval_result = self._makeEvalResult()
    result = widget_view.convert_slicing_metrics_to_ui_input(
        eval_result.slicing_metrics
    )
    self.assertEqual(
        result,
        [
            {
                'slice': 'Overall',
                'sliceValue': 'Overall',
                'metrics': {
                    'metrics2': {'double_value': {'value': 0.5}},
                    'metrics1': {'double_value': {'value': 0.5}},
                },
            },
            {
                'slice': 'slice:1',
                'sliceValue': '1',
                'metrics': {
                    'metrics2': {'double_value': {'value': 0.5}},
                    'metrics1': {'double_value': {'value': 0.5}},
                },
            },
        ],
    )

  def testConvertEvalResultToUIInputForCrossSliceKeyType(self):
    eval_result = self._makeEvalResult(
        slices=(
            ((), (('slice_1', 1), ('slice_2', 2))),
            (
                (('slice_1', 5), ('slice_2', 6)),
                (('slice_1', 3), ('slice_2', 4)),
            ),
        )
    )
    result = widget_view.convert_slicing_metrics_to_ui_input(
        eval_result.slicing_metrics
    )
    self.assertEqual(
        result,
        [
            {
                'slice': 'Overall__XX__slice_1_X_slice_2:1_X_2',
                'sliceValue': 'Overall__XX__1_X_2',
                'metrics': {
                    'metrics2': {'double_value': {'value': 0.5}},
                    'metrics1': {'double_value': {'value': 0.5}},
                },
            },
            {
                'slice': 'slice_1_X_slice_2:5_X_6__XX__slice_1_X_slice_2:3_X_4',
                'sliceValue': '5_X_6__XX__3_X_4',
                'metrics': {
                    'metrics2': {'double_value': {'value': 0.5}},
                    'metrics1': {'double_value': {'value': 0.5}},
                },
            },
        ],
    )

  def testConvertEvalResultToUIInputWithSlicingColumn(self):
    eval_result = self._makeEvalResult()
    result = widget_view.convert_slicing_metrics_to_ui_input(
        eval_result.slicing_metrics, slicing_column='slice'
    )
    self.assertEqual(
        result,
        [
            {
                'slice': 'Overall',
                'sliceValue': 'Overall',
                'metrics': {
                    'metrics2': {'double_value': {'value': 0.5}},
                    'metrics1': {'double_value': {'value': 0.5}},
                },
            },
            {
                'slice': 'slice:1',
                'sliceValue': '1',
                'metrics': {
                    'metrics2': {'double_value': {'value': 0.5}},
                    'metrics1': {'double_value': {'value': 0.5}},
                },
            },
        ],
    )

  def testConvertEvalResultToUIInputWithSlicingSpec(self):
    eval_result = self._makeEvalResult()
    result = widget_view.convert_slicing_metrics_to_ui_input(
        eval_result.slicing_metrics,
        slicing_spec=slicer.SingleSliceSpec(columns=['slice']),
    )
    self.assertEqual(
        result,
        [
            {
                'slice': 'Overall',
                'sliceValue': 'Overall',
                'metrics': {
                    'metrics2': {'double_value': {'value': 0.5}},
                    'metrics1': {'double_value': {'value': 0.5}},
                },
            },
            {
                'slice': 'slice:1',
                'sliceValue': '1',
                'metrics': {
                    'metrics2': {'double_value': {'value': 0.5}},
                    'metrics1': {'double_value': {'value': 0.5}},
                },
            },
        ],
    )

  def testConvertEvalResultToUIInputWithNoDataFound(self):
    eval_result = self._makeEvalResult(slices=((('slice', '1'),),))
    with self.assertRaises(ValueError):
      widget_view.convert_slicing_metrics_to_ui_input(
          eval_result.slicing_metrics,
          slicing_spec=slicer.SingleSliceSpec(columns=['unknown']),
      )


