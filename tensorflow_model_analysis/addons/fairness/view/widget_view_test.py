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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_model_analysis.addons.fairness.view import widget_view
from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.slicer import slicer_lib as slicer


class WidgetViewTest(testutil.TensorflowModelAnalysisTest):

  def _makeEvalResult(self,
                      slices=((), (('slice', '1'),)),
                      metrics_names=('metrics1', 'metrics2')):
    metrics = {'': {'': {}}}
    for metrics_name in metrics_names:
      metrics[''][''][metrics_name] = {'double_value': {'value': 0.5}}

    slicing_metrics = [(s, metrics) for s in slices]
    return model_eval_lib.EvalResult(
        slicing_metrics=slicing_metrics, plots=None, config=None)

  def testConvertEvalResultToUIInputWithAllDefaultParams(self):
    eval_result = self._makeEvalResult()
    result = widget_view.convert_eval_result_to_ui_input(eval_result)
    self.assertEqual(result, [{
        'slice': 'Overall',
        'sliceValue': 'Overall',
        'metrics': {
            'metrics2': {
                'double_value': {
                    'value': 0.5
                }
            },
            'metrics1': {
                'double_value': {
                    'value': 0.5
                }
            }
        }
    }, {
        'slice': u'slice:1',
        'sliceValue': u'1',
        'metrics': {
            'metrics2': {
                'double_value': {
                    'value': 0.5
                }
            },
            'metrics1': {
                'double_value': {
                    'value': 0.5
                }
            }
        }
    }])

  def testConvertEvalResultToUIInputWithSlicingColumn(self):
    eval_result = self._makeEvalResult()
    result = widget_view.convert_eval_result_to_ui_input(
        eval_result, slicing_column='slice')
    self.assertEqual(result, [{
        'slice': 'Overall',
        'sliceValue': 'Overall',
        'metrics': {
            'metrics2': {
                'double_value': {
                    'value': 0.5
                }
            },
            'metrics1': {
                'double_value': {
                    'value': 0.5
                }
            }
        }
    }, {
        'slice': u'slice:1',
        'sliceValue': u'1',
        'metrics': {
            'metrics2': {
                'double_value': {
                    'value': 0.5
                }
            },
            'metrics1': {
                'double_value': {
                    'value': 0.5
                }
            }
        }
    }])

  def testConvertEvalResultToUIInputWithSlicingSpec(self):
    eval_result = self._makeEvalResult()
    result = widget_view.convert_eval_result_to_ui_input(
        eval_result,
        slicing_spec=slicer.SingleSliceSpec(columns=['slice']),
    )
    self.assertEqual(result, [{
        'slice': 'Overall',
        'sliceValue': 'Overall',
        'metrics': {
            'metrics2': {
                'double_value': {
                    'value': 0.5
                }
            },
            'metrics1': {
                'double_value': {
                    'value': 0.5
                }
            }
        }
    }, {
        'slice': u'slice:1',
        'sliceValue': u'1',
        'metrics': {
            'metrics2': {
                'double_value': {
                    'value': 0.5
                }
            },
            'metrics1': {
                'double_value': {
                    'value': 0.5
                }
            }
        }
    }])

  def testConvertEvalResultToUIInputWithNoDataFound(self):
    eval_result = self._makeEvalResult(slices=((('slice', '1'),),))
    with self.assertRaises(ValueError):
      widget_view.convert_eval_result_to_ui_input(
          eval_result,
          slicing_spec=slicer.SingleSliceSpec(columns=['unknown']),
      )


if __name__ == '__main__':
  tf.test.main()
