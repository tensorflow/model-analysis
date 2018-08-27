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
"""Tests for tensorflow_model_analysis.view.util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api.impl import api_types
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.slicer.slicer import OVERALL_SLICE_NAME
from tensorflow_model_analysis.slicer.slicer import SingleSliceSpec
from tensorflow_model_analysis.view import util


class UtilTest(testutil.TensorflowModelAnalysisTest):
  column_1 = 'col1'
  column_2 = 'col2'

  metrics_a = {'a': 1, 'b': 2, 'example_weight': 3}
  slice_a = 'a'
  column_a = column_1 + ':' + slice_a
  result_a = ([(column_1, slice_a)], metrics_a)

  slice_b = 'b'
  metrics_b = {'a': 4, 'b': 5, 'example_weight': 6}
  column_b = column_1 + ':' + slice_b
  result_b = ([(column_1, slice_b)], metrics_b)

  slice_c = 'c'
  metrics_c = {'a': 1, 'b': 3, 'example_weight': 5}
  column_c = column_2 + ':' + slice_c
  result_c = ([(column_2, slice_c)], metrics_c)

  slice_d = 'd'
  metrics_d = {'a': 2, 'b': 4, 'example_weight': 6}
  column_d = column_1 + '_X_' + column_2 + ':' + slice_a + '_X_' + slice_d
  result_d = ([(column_1, slice_a), (column_2, slice_d)], metrics_d)

  metrics_aggregate = {'a': 10, 'b': 20, 'example_weight': 30}
  result_aggregate = ([], metrics_aggregate)

  metrics_c2 = {'a': 11, 'b': 33, 'example_weight': 55}
  result_c2 = ([(column_2, slice_c)], metrics_c2)

  data_location_1 = 'a.data'
  base_data_location_2 = 'b.data'
  full_data_location_2 = os.path.join('full', 'path', 'to', 'data',
                                      base_data_location_2)
  model_location_1 = 'a.model'
  base_model_location_2 = 'b.model'
  full_model_location_2 = os.path.join('full', 'path', 'to', 'model',
                                       base_model_location_2)

  plots_data_a = {
      'calibrationHistogramBuckets': {
          'buckets': [{
              'v': 0
          }, {
              'v': 1
          }],
      }
  }
  plots_a = ([(column_1, slice_a)], plots_data_a)

  plots_data_b = {
      'calibrationHistogramBuckets': {
          'buckets': [{
              'v': 0.25
          }, {
              'v': 0.5
          }, {
              'v': 0.75
          }],
      }
  }
  plots_b = ([(column_1, slice_b)], plots_data_b)

  plots_data_b2 = {'calibrationHistogramBuckets': {'buckets': [{'v': 0.5}],}}
  plots_b2 = ([(column_1, slice_b)], plots_data_b)

  plots_data_c = {
      'calibrationHistogramBuckets': {
          'buckets': [{
              'v': 0.5
          }, {
              'v': 'NaN'
          }],
      }
  }
  plots_c = ([(column_1, slice_c)], plots_data_c)

  def _makeTestData(self):
    return [
        self.result_a, self.result_b, self.result_c, self.result_d,
        self.result_aggregate
    ]

  def _makeTestPlotsData(self):
    return [
        self.plots_a,
        self.plots_b,
        self.plots_b2,
        self.plots_c,
    ]

  def _makeEvalResults(self):
    result_a = api_types.EvalResult(
        slicing_metrics=self._makeTestData(),
        plots=None,
        config=api_types.EvalConfig(
            example_weight_metric_key=None,
            slice_spec=None,
            data_location=self.data_location_1,
            model_location=self.model_location_1))

    result_b = api_types.EvalResult(
        slicing_metrics=[self.result_c2],
        plots=None,
        config=api_types.EvalConfig(
            example_weight_metric_key=None,
            slice_spec=None,
            data_location=self.full_data_location_2,
            model_location=self.full_model_location_2))
    return api_types.EvalResults([result_a, result_b],
                                 constants.MODEL_CENTRIC_MODE)

  def testGetSlicingMetrics(self):
    self.assertEqual(
        util.get_slicing_metrics(self._makeTestData(), self.column_1), [{
            'slice': self.column_a,
            'metrics': self.metrics_a
        }, {
            'slice': self.column_b,
            'metrics': self.metrics_b
        }])

  def testGetAggregateMetrics(self):
    self.assertEqual(
        util.get_slicing_metrics(self._makeTestData()), [{
            'slice': OVERALL_SLICE_NAME,
            'metrics': self.metrics_aggregate
        }])

  def testFilterColumnResultBySpec(self):
    self.assertEqual(
        util.get_slicing_metrics(
            self._makeTestData(),
            slicing_spec=SingleSliceSpec(columns=[self.column_2])), [{
                'slice': self.column_c,
                'metrics': self.metrics_c
            }])

  def testFilterFeatrueResultBySpec(self):
    self.assertEqual(
        util.get_slicing_metrics(
            self._makeTestData(),
            slicing_spec=SingleSliceSpec(features=[(self.column_2,
                                                    self.slice_c)])),
        [{
            'slice': self.column_c,
            'metrics': self.metrics_c
        }])

  def testFilterColumnCrossFeatrueResultBySpec(self):
    self.assertEqual(
        util.get_slicing_metrics(
            self._makeTestData(),
            slicing_spec=SingleSliceSpec(
                columns=[self.column_1],
                features=[(self.column_2, self.slice_d)])), [{
                    'slice': self.column_d,
                    'metrics': self.metrics_d
                }])

  def testFilterFeatrueCrossResultBySpec(self):
    self.assertEqual(
        util.get_slicing_metrics(
            self._makeTestData(),
            slicing_spec=SingleSliceSpec(features=[(
                self.column_1, self.slice_a), (self.column_2, self.slice_d)])),
        [{
            'slice': self.column_d,
            'metrics': self.metrics_d
        }])

  def testRaisesErrorWhenColumnNotAvailable(self):
    with self.assertRaises(ValueError):
      util.get_slicing_metrics(self._makeTestData(), 'col3')

  def testRaisesErrorWhenAggregateNotAvailable(self):
    with self.assertRaises(ValueError):
      util.get_slicing_metrics([(self.column_a, self.metrics_a),
                                (self.column_b, self.metrics_b)])

  def testRaisesErrorWhenMoreThanOneAggregateAvailable(self):
    with self.assertRaises(ValueError):
      util.get_slicing_metrics([('', self.metrics_a), ('', self.metrics_b)])

  def testFilterEvalResultsForTimeSeries(self):
    data = util.get_time_series(
        self._makeEvalResults(),
        SingleSliceSpec(features=[(self.column_2, self.slice_c)]),
        display_full_path=False)
    self.assertEqual(len(data), 2)
    self.assertEqual(data[0]['metrics'], self.metrics_c)
    self.assertEqual(
        data[0]['config'], {
            'dataIdentifier': self.data_location_1,
            'modelIdentifier': self.model_location_1
        })

    self.assertEqual(data[1]['metrics'], self.metrics_c2)
    self.assertEqual(
        data[1]['config'], {
            'dataIdentifier': self.base_data_location_2,
            'modelIdentifier': self.base_model_location_2
        })

  def testDisplayFullPathForTimeSeries(self):
    data = util.get_time_series(
        self._makeEvalResults(),
        SingleSliceSpec(features=[(self.column_2, self.slice_c)]),
        display_full_path=True)
    self.assertEqual(len(data), 2)

    self.assertEqual(
        data[1]['config'], {
            'dataIdentifier': self.full_data_location_2,
            'modelIdentifier': self.full_model_location_2
        })

  def testRaisesErrorWhenNoMatchAvailableInTimeSeries(self):
    with self.assertRaises(ValueError):
      util.get_time_series(
          self._makeEvalResults(),
          SingleSliceSpec(features=[(self.column_2, self.slice_a)]),
          display_full_path=False)

  def testRaisesErrorWhenToomManyMatchesAvailableInTimeSeries(self):
    with self.assertRaises(ValueError):
      util.get_time_series(
          self._makeEvalResults(),
          SingleSliceSpec(columns=[(self.column_1)]),
          display_full_path=False)

  def testGetPlotDataAndConfig(self):
    data, config = util.get_plot_data_and_config(
        self._makeTestPlotsData(),
        SingleSliceSpec(features=[(self.column_1, self.slice_a)]))

    self.assertEquals(data, self.plots_data_a)
    self.assertEquals(config['sliceName'], self.column_a)

  def testRaisesErrorWhenNoMatchAvailableInPlotData(self):
    with self.assertRaises(ValueError):
      util.get_plot_data_and_config(
          self._makeTestPlotsData(),
          SingleSliceSpec(features=[(self.column_2, self.slice_a)]))

  def testRaisesErrorWhenMultipleSlicesMatchInPlotData(self):
    with self.assertRaises(ValueError):
      util.get_plot_data_and_config(
          self._makeTestPlotsData(),
          SingleSliceSpec(features=[(self.column_1, self.slice_b)]))

  def testReplaceNaNInPlotWithNone(self):
    data, _ = util.get_plot_data_and_config(
        self._makeTestPlotsData(),
        SingleSliceSpec(features=[(self.column_1, self.slice_c)]))

    self.assertEquals(data, {
        'calibrationHistogramBuckets': {
            'buckets': [{
                'v': 0.5
            }, {
                'v': None
            }],
        }
    })


if __name__ == '__main__':
  tf.test.main()
