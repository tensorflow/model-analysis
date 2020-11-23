# Lint as: python3
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
from tensorflow_model_analysis import config
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.proto import metrics_for_slice_pb2
from tensorflow_model_analysis.slicer.slicer_lib import OVERALL_SLICE_NAME
from tensorflow_model_analysis.slicer.slicer_lib import SingleSliceSpec
from tensorflow_model_analysis.view import util
from tensorflow_model_analysis.view import view_types

from google.protobuf import text_format


def _add_to_nested_dict(metrics):
  return {
      '': {
          '': metrics,
      },
  }


class UtilTest(testutil.TensorflowModelAnalysisTest):
  column_1 = 'col1'
  column_2 = 'col2'

  metrics_a = _add_to_nested_dict({'a': 1, 'b': 2, 'example_weight': 3})
  slice_a = 'a'
  column_a = column_1 + ':' + slice_a
  result_a = ([(column_1, slice_a)], metrics_a)

  slice_b = 'b'
  metrics_b = _add_to_nested_dict({'a': 4, 'b': 5, 'example_weight': 6})
  column_b = column_1 + ':' + slice_b
  result_b = ([(column_1, slice_b)], metrics_b)

  slice_c = 'c'
  metrics_c = _add_to_nested_dict({'a': 1, 'b': 3, 'example_weight': 5})
  column_c = column_2 + ':' + slice_c
  result_c = ([(column_2, slice_c)], metrics_c)

  slice_d = 'd'
  metrics_d = _add_to_nested_dict({'a': 2, 'b': 4, 'example_weight': 6})
  column_d = column_1 + '_X_' + column_2 + ':' + slice_a + '_X_' + slice_d
  result_d = ([(column_1, slice_a), (column_2, slice_d)], metrics_d)

  metrics_aggregate = _add_to_nested_dict({
      'a': 10,
      'b': 20,
      'example_weight': 30
  })
  result_aggregate = ([], metrics_aggregate)

  metrics_c2 = _add_to_nested_dict({'a': 11, 'b': 33, 'example_weight': 55})
  result_c2 = ([(column_2, slice_c)], metrics_c2)

  data_location_1 = 'a.data'
  base_data_location_2 = 'b.data'
  full_data_location_2 = os.path.join('full', 'path', 'to', 'data',
                                      base_data_location_2)
  model_location_1 = 'a.model'
  base_model_location_2 = 'b.model'
  full_model_location_2 = os.path.join('full', 'path', 'to', 'model',
                                       base_model_location_2)

  key = 'plot_key'

  plots_data_a = {
      'calibrationHistogramBuckets': {
          'buckets': [{
              'v': 0
          }, {
              'v': 1
          }],
      }
  }
  plots_a = ([(column_1, slice_a)], _add_to_nested_dict(plots_data_a))

  plots_data_b = {
      key: {
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
  }
  plots_b = ([(column_1, slice_b)], _add_to_nested_dict(plots_data_b))

  plots_data_b2 = {
      key: {
          'calibrationHistogramBuckets': {
              'buckets': [{
                  'v': 0.5
              }],
          }
      }
  }
  plots_b2 = ([(column_1, slice_b)], _add_to_nested_dict(plots_data_b))

  plots_data_c = {
      'calibrationHistogramBuckets': {
          'buckets': [{
              'v': 0.5
          }, {
              'v': 'NaN'
          }],
      }
  }
  plots_c = ([(column_1, slice_c)], _add_to_nested_dict(plots_data_c))

  plots_data_c2 = {
      'label/head_a': {
          'calibrationHistogramBuckets': {
              'buckets': [{
                  'v': 0.5
              }],
          }
      },
      'label/head_b': {
          'calibrationHistogramBuckets': {
              'buckets': [{
                  'v': 0.5
              }],
          }
      }
  }
  plots_c2 = ([(column_2, slice_c)], _add_to_nested_dict(plots_data_c2))

  plots_data_0 = {
      'calibrationHistogramBuckets': {
          'buckets': [{
              'v': 0
          }, {
              'v': 1
          }],
      }
  }
  plots_data_1 = {
      'calibrationHistogramBuckets': {
          'buckets': [{
              'v': 0
          }, {
              'v': 1
          }],
      }
  }
  plots_multi_class = ([(column_2, slice_a)], {
      '': {
          'classId:0': plots_data_0,
          'classId:1': plots_data_1
      }
  })
  column_2a = column_2 + ':' + slice_a

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
        self.plots_c2,
        self.plots_multi_class,
    ]

  def _makeEvalResults(self):
    result_a = view_types.EvalResult(
        slicing_metrics=self._makeTestData(),
        plots=None,
        attributions=None,
        config=config.EvalConfig(),
        data_location=self.data_location_1,
        file_format='tfrecords',
        model_location=self.model_location_1)
    result_b = view_types.EvalResult(
        slicing_metrics=[self.result_c2],
        plots=None,
        attributions=None,
        config=config.EvalConfig(),
        data_location=self.full_data_location_2,
        file_format='tfrecords',
        model_location=self.full_model_location_2)
    return view_types.EvalResults([result_a, result_b],
                                  constants.MODEL_CENTRIC_MODE)

  def _makeEvalConfig(self):
    eval_config = config.EvalConfig(
        model_specs=[config.ModelSpec(example_weight_key='testing_key')])
    return eval_config

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
            slicing_spec=SingleSliceSpec(
                features=[(self.column_1,
                           self.slice_a), (self.column_2, self.slice_d)])), [{
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
    data, eval_config = util.get_plot_data_and_config(
        self._makeTestPlotsData(),
        SingleSliceSpec(features=[(self.column_1, self.slice_a)]))

    self.assertEqual(data, self.plots_data_a)
    self.assertEqual(eval_config['sliceName'], self.column_a)

  def testGetPlotDataAndConfigForMultiClass(self):
    data, eval_config = util.get_plot_data_and_config(
        self._makeTestPlotsData(),
        SingleSliceSpec(features=[(self.column_2, self.slice_a)]),
        class_id=0)

    self.assertEqual(data, self.plots_data_0)
    self.assertEqual(eval_config['sliceName'], self.column_2a)

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

    self.assertEqual(data, {
        'calibrationHistogramBuckets': {
            'buckets': [{
                'v': 0.5
            }, {
                'v': None
            }],
        }
    })

  def testGetPlotUsingLabel(self):
    data, _ = util.get_plot_data_and_config(
        self._makeTestPlotsData(),
        SingleSliceSpec(features=[(self.column_2, self.slice_c)]),
        label='head_a')

    self.assertEqual(data, self.plots_data_c2['label/head_a'])

  def testRaisesErrorWhenLabelNotProvided(self):
    with self.assertRaises(ValueError):
      util.get_plot_data_and_config(
          self._makeTestPlotsData(),
          SingleSliceSpec(features=[(self.column_2, self.slice_c)]))

  def testRaisesErrorWhenNoMatchForLabel(self):
    with self.assertRaises(ValueError):
      util.get_plot_data_and_config(
          self._makeTestPlotsData(),
          SingleSliceSpec(features=[(self.column_1, self.slice_c)]),
          label='head_a')

  def testRaisesErrorWhenMultipleMatchForLabel(self):
    with self.assertRaises(ValueError):
      util.get_plot_data_and_config(
          self._makeTestPlotsData(),
          SingleSliceSpec(features=[(self.column_2, self.slice_c)]),
          label='head_')

  def testRaiseErrorWhenBothLabelAndPlotKeyAreProvided(self):
    with self.assertRaises(ValueError):
      util.get_plot_data_and_config(
          self._makeTestPlotsData(),
          SingleSliceSpec(features=[(self.column_2, self.slice_c)]),
          label='head_a',
          output_name='')

  def testRaiseErrorWhenMoreThanOneMultiClassKeyAreProvided(self):
    with self.assertRaises(ValueError):
      util.get_plot_data_and_config(
          self._makeTestPlotsData(),
          SingleSliceSpec(features=[(self.column_2, self.slice_c)]),
          top_k=3,
          class_id=0)

  def testGetSlicingConfig(self):
    eval_config = self._makeEvalConfig()
    slicing_config = util.get_slicing_config(eval_config)
    self.assertEqual(
        slicing_config,
        {'weightedExamplesColumn': 'post_export_metrics/example_weight'})

  def testOverrideWeightColumnForSlicingMetricsView(self):
    overriding_weight_column = 'override'
    eval_config = self._makeEvalConfig()
    slicing_config = util.get_slicing_config(
        eval_config, weighted_example_column_to_use=overriding_weight_column)
    self.assertEqual(slicing_config['weightedExamplesColumn'],
                     overriding_weight_column)

  def testConvertAttributionsProto(self):
    attributions_for_slice = text_format.Parse(
        """
      slice_key {}
      attributions_keys_and_values {
        key {
          name: "total_attributions"
        }
        values {
          key: "feature1"
          value: {
            double_value {
              value: 1.0
            }
          }
        }
        values {
          key: "feature2"
          value: {
            double_value {
              value: 2.0
            }
          }
        }
      }
      attributions_keys_and_values {
        key {
          name: "total_attributions"
          output_name: "output1"
          sub_key: {
            class_id: { value: 1 }
          }
        }
        values {
          key: "feature1"
          value: {
            double_value {
              value: 1.0
            }
          }
        }
      }""", metrics_for_slice_pb2.AttributionsForSlice())

    got = util.convert_attributions_proto_to_dict(attributions_for_slice, None)
    self.assertEqual(got, ((), {
        '': {
            '': {
                'total_attributions': {
                    'feature2': {
                        'doubleValue': 2.0
                    },
                    'feature1': {
                        'doubleValue': 1.0
                    }
                }
            }
        },
        'output1': {
            'classId:1': {
                'total_attributions': {
                    'feature1': {
                        'doubleValue': 1.0
                    }
                }
            }
        }
    }))


if __name__ == '__main__':
  tf.test.main()
