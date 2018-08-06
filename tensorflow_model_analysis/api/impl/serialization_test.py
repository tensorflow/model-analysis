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
"""Test for using the serialization library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import string

import apache_beam as beam
import numpy as np

import tensorflow as tf
from tensorflow_model_analysis.api.impl import api_types
from tensorflow_model_analysis.api.impl import serialization
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.eval_saved_model.post_export_metrics import metric_keys
from tensorflow_model_analysis.eval_saved_model.post_export_metrics import post_export_metrics
from tensorflow_model_analysis.proto import metrics_for_slice_pb2
from tensorflow_model_analysis.slicer import slicer
from google.protobuf import text_format


def _make_slice_key(*args):
  if len(args) % 2 != 0:
    raise ValueError('number of arguments should be even')

  result = []
  for i in range(0, len(args), 2):
    result.append((args[i], args[i + 1]))

  return tuple(result)


class SerializationTest(testutil.TensorflowModelAnalysisTest):

  def setUp(self):
    self.longMessage = True

  def assertMetricsAlmostEqual(self, expected_metrics, got_metrics):
    self.assertItemsEqual(
        expected_metrics.keys(),
        got_metrics.keys(),
        msg='keys do not match. expected_metrics: %s, got_metrics: %s' %
        (expected_metrics, got_metrics))
    for key in expected_metrics.keys():
      self.assertAlmostEqual(
          expected_metrics[key],
          got_metrics[key],
          msg='value for key %s does not match' % key)

  def assertSliceListEqual(self, expected_list, got_list, value_assert_fn):
    self.assertEqual(
        len(expected_list),
        len(got_list),
        msg='expected_list: %s, got_list: %s' % (expected_list, got_list))
    for index, (expected, got) in enumerate(zip(expected_list, got_list)):
      (expected_key, expected_value) = expected
      (got_key, got_value) = got
      self.assertEqual(
          expected_key, got_key, msg='key mismatch at index %d' % index)
      value_assert_fn(expected_value, got_value)

  def assertSlicePlotsListEqual(self, expected_list, got_list):
    self.assertSliceListEqual(expected_list, got_list, self.assertProtoEquals)

  def assertSliceMetricsListEqual(self, expected_list, got_list):
    self.assertSliceListEqual(expected_list, got_list,
                              self.assertMetricsAlmostEqual)

  def testDeserializeSliceKey(self):
    slice_metrics = text_format.Parse(
        """
          single_slice_keys {
            column: 'age'
            int64_value: 5
          }
          single_slice_keys {
            column: 'language'
            bytes_value: 'english'
          }
          single_slice_keys {
            column: 'price'
            float_value: 1.0
          }
        """, metrics_for_slice_pb2.SliceKey())

    got_slice_key = serialization.deserialize_slice_key(slice_metrics)
    self.assertItemsEqual([('age', 5), ('language', 'english'), ('price', 1.0)],
                          got_slice_key)

  def testSerializePlots(self):
    slice_key = _make_slice_key('fruit', 'apple')
    tfma_plots = {
        metric_keys.CALIBRATION_PLOT_MATRICES:
            np.array([
                [0.0, 0.0, 0.0],
                [0.3, 1.0, 1.0],
                [0.7, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ]),
        metric_keys.CALIBRATION_PLOT_BOUNDARIES:
            np.array([0.0, 0.5, 1.0]),
    }
    expected_plot_data = """
      slice_key {
        single_slice_keys {
          column: 'fruit'
          bytes_value: 'apple'
        }
      }
      plot_data {
        calibration_histogram_buckets {
          buckets {
            lower_threshold_inclusive: -inf
            upper_threshold_exclusive: 0.0
            num_weighted_examples { value: 0.0 }
            total_weighted_label { value: 0.0 }
            total_weighted_refined_prediction { value: 0.0 }
          }
          buckets {
            lower_threshold_inclusive: 0.0
            upper_threshold_exclusive: 0.5
            num_weighted_examples { value: 1.0 }
            total_weighted_label { value: 1.0 }
            total_weighted_refined_prediction { value: 0.3 }
          }
          buckets {
            lower_threshold_inclusive: 0.5
            upper_threshold_exclusive: 1.0
            num_weighted_examples { value: 1.0 }
            total_weighted_label { value: 0.0 }
            total_weighted_refined_prediction { value: 0.7 }
          }
          buckets {
            lower_threshold_inclusive: 1.0
            upper_threshold_exclusive: inf
            num_weighted_examples { value: 0.0 }
            total_weighted_label { value: 0.0 }
            total_weighted_refined_prediction { value: 0.0 }
          }
        }
      }
    """
    calibration_plot = (
        post_export_metrics.calibration_plot_and_prediction_histogram())
    serialized = serialization._serialize_plots((slice_key, tfma_plots),
                                                [calibration_plot])
    self.assertProtoEquals(
        expected_plot_data,
        metrics_for_slice_pb2.PlotsForSlice.FromString(serialized))

  def testSerializeMetrics(self):
    slice_key = _make_slice_key('age', 5, 'language', 'english', 'price', 0.3)
    slice_metrics = {
        'accuracy': 0.8,
        metric_keys.AUPRC: 0.1,
        metric_keys.lower_bound(metric_keys.AUPRC): 0.05,
        metric_keys.upper_bound(metric_keys.AUPRC): 0.17,
        metric_keys.AUC: 0.2,
        metric_keys.lower_bound(metric_keys.AUC): 0.1,
        metric_keys.upper_bound(metric_keys.AUC): 0.3
    }
    expected_metrics_for_slice = text_format.Parse(
        string.Template("""
        slice_key {
          single_slice_keys {
            column: 'age'
            int64_value: 5
          }
          single_slice_keys {
            column: 'language'
            bytes_value: 'english'
          }
          single_slice_keys {
            column: 'price'
            float_value: 0.3
          }
        }
        metrics {
          key: "accuracy"
          value {
            double_value {
              value: 0.8
            }
          }
        }
        metrics {
          key: "$auc"
          value {
            bounded_value {
              lower_bound {
                value: 0.1
              }
              upper_bound {
                value: 0.3
              }
              value {
                value: 0.2
              }
            }
          }
        }
        metrics {
          key: "$auprc"
          value {
            bounded_value {
              lower_bound {
                value: 0.05
              }
              upper_bound {
                value: 0.17
              }
              value {
                value: 0.1
              }
            }
          }
        }""").substitute(auc=metric_keys.AUC, auprc=metric_keys.AUPRC),
        metrics_for_slice_pb2.MetricsForSlice())

    got = serialization._serialize_metrics(
        (slice_key, slice_metrics),
        [post_export_metrics.auc(),
         post_export_metrics.auc(curve='PR')])
    self.assertProtoEquals(
        expected_metrics_for_slice,
        metrics_for_slice_pb2.MetricsForSlice.FromString(got))

  def testSerializeDeserializeEvalConfig(self):
    eval_config = api_types.EvalConfig(
        model_location='/path/to/model',
        data_location='/path/to/data',
        slice_spec=[
            slicer.SingleSliceSpec(
                features=[('age', 5), ('gender', 'f')], columns=['country']),
            slicer.SingleSliceSpec(
                features=[('age', 6), ('gender', 'm')], columns=['interest'])
        ],
        example_weight_metric_key='key')
    serialized = serialization._serialize_eval_config(eval_config)
    deserialized = serialization._deserialize_eval_config_raw(serialized)
    got_eval_config = deserialized[serialization._EVAL_CONFIG_KEY]
    self.assertEqual(eval_config, got_eval_config)

  def testSerializeDeserializeToFile(self):
    metrics_slice_key = _make_slice_key('fruit', 'pear', 'animal', 'duck')
    metrics_for_slice = text_format.Parse(
        """
        slice_key {
          single_slice_keys {
            column: "fruit"
            bytes_value: "pear"
          }
          single_slice_keys {
            column: "animal"
            bytes_value: "duck"
          }
        }
        metrics {
          key: "accuracy"
          value {
            double_value {
              value: 0.8
            }
          }
        }
        metrics {
          key: "example_weight"
          value {
            double_value {
              value: 10.0
            }
          }
        }
        metrics {
          key: "auc"
          value {
            bounded_value {
              lower_bound {
                value: 0.1
              }
              upper_bound {
                value: 0.3
              }
              value {
                value: 0.2
              }
            }
          }
        }
        metrics {
          key: "auprc"
          value {
            bounded_value {
              lower_bound {
                value: 0.05
              }
              upper_bound {
                value: 0.17
              }
              value {
                value: 0.1
              }
            }
          }
        }""", metrics_for_slice_pb2.MetricsForSlice())
    plots_for_slice = text_format.Parse(
        """
        slice_key {
          single_slice_keys {
            column: "fruit"
            bytes_value: "peach"
          }
          single_slice_keys {
            column: "animal"
            bytes_value: "cow"
          }
        }
        plot_data {
          calibration_histogram_buckets {
            buckets {
              lower_threshold_inclusive: -inf
              upper_threshold_exclusive: 0.0
              num_weighted_examples { value: 0.0 }
              total_weighted_label { value: 0.0 }
              total_weighted_refined_prediction { value: 0.0 }
            }
            buckets {
              lower_threshold_inclusive: 0.0
              upper_threshold_exclusive: 0.5
              num_weighted_examples { value: 1.0 }
              total_weighted_label { value: 1.0 }
              total_weighted_refined_prediction { value: 0.3 }
            }
            buckets {
              lower_threshold_inclusive: 0.5
              upper_threshold_exclusive: 1.0
              num_weighted_examples { value: 1.0 }
              total_weighted_label { value: 0.0 }
              total_weighted_refined_prediction { value: 0.7 }
            }
            buckets {
              lower_threshold_inclusive: 1.0
              upper_threshold_exclusive: inf
              num_weighted_examples { value: 0.0 }
              total_weighted_label { value: 0.0 }
              total_weighted_refined_prediction { value: 0.0 }
            }
          }
        }""", metrics_for_slice_pb2.PlotsForSlice())
    plots_slice_key = _make_slice_key('fruit', 'peach', 'animal', 'cow')
    eval_config = api_types.EvalConfig(
        model_location='/path/to/model',
        data_location='/path/to/data',
        slice_spec=[
            slicer.SingleSliceSpec(
                features=[('age', 5), ('gender', 'f')], columns=['country']),
            slicer.SingleSliceSpec(
                features=[('age', 6), ('gender', 'm')], columns=['interest'])
        ],
        example_weight_metric_key='key')

    output_path = self._getTempDir()
    with beam.Pipeline() as pipeline:
      metrics = (
          pipeline
          | 'CreateMetrics' >> beam.Create(
              [metrics_for_slice.SerializeToString()]))
      plots = (
          pipeline
          | 'CreatePlots' >> beam.Create([plots_for_slice.SerializeToString()]))

      _ = ((metrics, plots)
           | 'WriteMetricsPlotsAndConfig' >>
           serialization.WriteMetricsPlotsAndConfig(
               output_path=output_path,
               eval_config=eval_config))

    metrics, plots = serialization.load_plots_and_metrics(output_path)
    self.assertSliceMetricsListEqual(
        [(metrics_slice_key, metrics_for_slice.metrics)], metrics)
    self.assertSlicePlotsListEqual(
        [(plots_slice_key, plots_for_slice.plot_data)], plots)
    got_eval_config = serialization.load_eval_config(output_path)
    self.assertEqual(eval_config, got_eval_config)


if __name__ == '__main__':
  tf.test.main()
