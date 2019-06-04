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
"""Test for using the MetricsAndPlotsEvaluator API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import string

# Standard Imports

import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import types
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.post_export_metrics import metric_keys
from tensorflow_model_analysis.post_export_metrics import post_export_metrics
from tensorflow_model_analysis.proto import metrics_for_slice_pb2
from tensorflow_model_analysis.slicer import slicer
from tensorflow_model_analysis.writers import metrics_and_plots_serialization

from google.protobuf import text_format


def _make_slice_key(*args):
  if len(args) % 2 != 0:
    raise ValueError('number of arguments should be even')

  result = []
  for i in range(0, len(args), 2):
    result.append((args[i], args[i + 1]))
  result = tuple(result)
  return result


class EvaluateMetricsAndPlotsTest(testutil.TensorflowModelAnalysisTest):

  def setUp(self):
    self.longMessage = True  # pylint: disable=invalid-name

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
      plots {
        key: "post_export_metrics"
        value {
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
      }
    """
    calibration_plot = (
        post_export_metrics.calibration_plot_and_prediction_histogram())
    serialized = metrics_and_plots_serialization._serialize_plots(
        (slice_key, tfma_plots), [calibration_plot])
    self.assertProtoEquals(
        expected_plot_data,
        metrics_for_slice_pb2.PlotsForSlice.FromString(serialized))

  def testSerializePlots_emptyPlot(self):
    slice_key = _make_slice_key('fruit', 'apple')
    tfma_plots = {metric_keys.ERROR_METRIC: 'error_message'}

    calibration_plot = (
        post_export_metrics.calibration_plot_and_prediction_histogram())
    actual_plot = metrics_and_plots_serialization._serialize_plots(
        (slice_key, tfma_plots), [calibration_plot])
    expected_plot = metrics_for_slice_pb2.PlotsForSlice()
    expected_plot.slice_key.CopyFrom(slicer.serialize_slice_key(slice_key))
    expected_plot.plots[
        metric_keys.ERROR_METRIC].debug_message = 'error_message'
    self.assertProtoEquals(
        expected_plot,
        metrics_for_slice_pb2.PlotsForSlice.FromString(actual_plot))

  def testSerializeConfusionMatrices(self):
    slice_key = _make_slice_key()

    thresholds = [0.25, 0.75, 1.00]
    matrices = [[0.0, 1.0, 0.0, 2.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0, 1.0, 0.5],
                [2.0, 1.0, 0.0, 0.0, float('nan'), 0.0]]

    slice_metrics = {
        metric_keys.CONFUSION_MATRIX_AT_THRESHOLDS_MATRICES: matrices,
        metric_keys.CONFUSION_MATRIX_AT_THRESHOLDS_THRESHOLDS: thresholds,
    }
    expected_metrics_for_slice = text_format.Parse(
        """
        slice_key {}
        metrics {
          key: "post_export_metrics/confusion_matrix_at_thresholds"
          value {
            confusion_matrix_at_thresholds {
              matrices {
                threshold: 0.25
                false_negatives: 0.0
                true_negatives: 1.0
                false_positives: 0.0
                true_positives: 2.0
                precision: 1.0
                recall: 1.0
                bounded_false_negatives {
                  value {
                    value: 0.0
                  }
                }
                bounded_true_negatives {
                  value {
                    value: 1.0
                  }
                }
                bounded_true_positives {
                  value {
                    value: 2.0
                  }
                }
                bounded_false_positives {
                  value {
                    value: 0.0
                  }
                }
                bounded_precision {
                  value {
                    value: 1.0
                  }
                }
                bounded_recall {
                  value {
                    value: 1.0
                  }
                }
                t_distribution_false_negatives {
                  unsampled_value {
                    value: 0.0
                  }
                }
                t_distribution_true_negatives {
                  unsampled_value {
                    value: 1.0
                  }
                }
                t_distribution_true_positives {
                  unsampled_value {
                    value: 2.0
                  }
                }
                t_distribution_false_positives {
                  unsampled_value {
                    value: 0.0
                  }
                }
                t_distribution_precision {
                  unsampled_value {
                    value: 1.0
                  }
                }
                t_distribution_recall {
                  unsampled_value {
                    value: 1.0
                  }
                }
              }
              matrices {
                threshold: 0.75
                false_negatives: 1.0
                true_negatives: 1.0
                false_positives: 0.0
                true_positives: 1.0
                precision: 1.0
                recall: 0.5
                bounded_false_negatives {
                  value {
                    value: 1.0
                  }
                }
                bounded_true_negatives {
                  value {
                    value: 1.0
                  }
                }
                bounded_true_positives {
                  value {
                    value: 1.0
                  }
                }
                bounded_false_positives {
                  value {
                    value: 0.0
                  }
                }
                bounded_precision {
                  value {
                    value: 1.0
                  }
                }
                bounded_recall {
                  value {
                    value: 0.5
                  }
                }
                t_distribution_false_negatives {
                  unsampled_value {
                    value: 1.0
                  }
                }
                t_distribution_true_negatives {
                  unsampled_value {
                    value: 1.0
                  }
                }
                t_distribution_true_positives {
                  unsampled_value {
                    value: 1.0
                  }
                }
                t_distribution_false_positives {
                  unsampled_value {
                    value: 0.0
                  }
                }
                t_distribution_precision {
                  unsampled_value {
                    value: 1.0
                  }
                }
                t_distribution_recall {
                  unsampled_value {
                    value: 0.5
                  }
                }
              }
              matrices {
                threshold: 1.00
                false_negatives: 2.0
                true_negatives: 1.0
                false_positives: 0.0
                true_positives: 0.0
                precision: nan
                recall: 0.0
                bounded_false_negatives {
                  value {
                    value: 2.0
                  }
                }
                bounded_true_negatives {
                  value {
                    value: 1.0
                  }
                }
                bounded_true_positives {
                  value {
                    value: 0.0
                  }
                }
                bounded_false_positives {
                  value {
                    value: 0.0
                  }
                }
                bounded_precision {
                  value {
                    value: nan
                  }
                }
                bounded_recall {
                  value {
                    value: 0.0
                  }
                }
                t_distribution_false_negatives {
                  unsampled_value {
                    value: 2.0
                  }
                }
                t_distribution_true_negatives {
                  unsampled_value {
                    value: 1.0
                  }
                }
                t_distribution_true_positives {
                  unsampled_value {
                    value: 0.0
                  }
                }
                t_distribution_false_positives {
                  unsampled_value {
                    value: 0.0
                  }
                }
                t_distribution_precision {
                  unsampled_value {
                    value: nan
                  }
                }
                t_distribution_recall {
                  unsampled_value {
                    value: 0.0
                  }
                }
              }
            }
          }
        }
        """, metrics_for_slice_pb2.MetricsForSlice())

    got = metrics_and_plots_serialization._serialize_metrics(
        (slice_key, slice_metrics),
        [post_export_metrics.confusion_matrix_at_thresholds(thresholds)])
    self.assertProtoEquals(
        expected_metrics_for_slice,
        metrics_for_slice_pb2.MetricsForSlice.FromString(got))

  def testSerializeMetricsRanges(self):
    slice_key = _make_slice_key('age', 5, 'language', 'english', 'price', 0.3)
    slice_metrics = {
        'accuracy': types.ValueWithTDistribution(0.8, 0.1, 9, 0.8),
        metric_keys.AUPRC: 0.1,
        metric_keys.lower_bound_key(metric_keys.AUPRC): 0.05,
        metric_keys.upper_bound_key(metric_keys.AUPRC): 0.17,
        metric_keys.AUC: 0.2,
        metric_keys.lower_bound_key(metric_keys.AUC): 0.1,
        metric_keys.upper_bound_key(metric_keys.AUC): 0.3
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
            bounded_value {
              value {
                value: 0.8
              }
              lower_bound {
                value: 0.7284643
              }
              upper_bound {
                value: 0.8715357
              }
              methodology: POISSON_BOOTSTRAP
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
              methodology: RIEMANN_SUM
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
              methodology: RIEMANN_SUM
            }
          }
        }""").substitute(auc=metric_keys.AUC, auprc=metric_keys.AUPRC),
        metrics_for_slice_pb2.MetricsForSlice())

    got = metrics_and_plots_serialization._serialize_metrics(
        (slice_key, slice_metrics),
        [post_export_metrics.auc(),
         post_export_metrics.auc(curve='PR')])
    self.assertProtoEquals(
        expected_metrics_for_slice,
        metrics_for_slice_pb2.MetricsForSlice.FromString(got))

  def testSerializeMetrics(self):
    slice_key = _make_slice_key('age', 5, 'language', 'english', 'price', 0.3)
    slice_metrics = {
        'accuracy': 0.8,
        metric_keys.AUPRC: 0.1,
        metric_keys.lower_bound_key(metric_keys.AUPRC): 0.05,
        metric_keys.upper_bound_key(metric_keys.AUPRC): 0.17,
        metric_keys.AUC: 0.2,
        metric_keys.lower_bound_key(metric_keys.AUC): 0.1,
        metric_keys.upper_bound_key(metric_keys.AUC): 0.3
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
              methodology: RIEMANN_SUM
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
              methodology: RIEMANN_SUM
            }
          }
        }""").substitute(auc=metric_keys.AUC, auprc=metric_keys.AUPRC),
        metrics_for_slice_pb2.MetricsForSlice())

    got = metrics_and_plots_serialization._serialize_metrics(
        (slice_key, slice_metrics),
        [post_export_metrics.auc(),
         post_export_metrics.auc(curve='PR')])
    self.assertProtoEquals(
        expected_metrics_for_slice,
        metrics_for_slice_pb2.MetricsForSlice.FromString(got))

  def testSerializeMetrics_emptyMetrics(self):
    slice_key = _make_slice_key('age', 5, 'language', 'english', 'price', 0.3)
    slice_metrics = {metric_keys.ERROR_METRIC: 'error_message'}

    actual_metrics = metrics_and_plots_serialization._serialize_metrics(
        (slice_key, slice_metrics),
        [post_export_metrics.auc(),
         post_export_metrics.auc(curve='PR')])

    expected_metrics = metrics_for_slice_pb2.MetricsForSlice()
    expected_metrics.slice_key.CopyFrom(slicer.serialize_slice_key(slice_key))
    expected_metrics.metrics[
        metric_keys.ERROR_METRIC].debug_message = 'error_message'
    self.assertProtoEquals(
        expected_metrics,
        metrics_for_slice_pb2.MetricsForSlice.FromString(actual_metrics))

  def testStringMetrics(self):
    slice_key = _make_slice_key()
    slice_metrics = {
        'valid_ascii': b'test string',
        'valid_unicode': b'\xF0\x9F\x90\x84',  # U+1F404, Cow
        'invalid_unicode': b'\xE2\x28\xA1',
    }
    expected_metrics_for_slice = metrics_for_slice_pb2.MetricsForSlice()
    expected_metrics_for_slice.slice_key.SetInParent()
    expected_metrics_for_slice.metrics[
        'valid_ascii'].bytes_value = slice_metrics['valid_ascii']
    expected_metrics_for_slice.metrics[
        'valid_unicode'].bytes_value = slice_metrics['valid_unicode']
    expected_metrics_for_slice.metrics[
        'invalid_unicode'].bytes_value = slice_metrics['invalid_unicode']

    got = metrics_and_plots_serialization._serialize_metrics(
        (slice_key, slice_metrics), [])
    self.assertProtoEquals(
        expected_metrics_for_slice,
        metrics_for_slice_pb2.MetricsForSlice.FromString(got))

  def testUncertaintyValuedMetrics(self):
    slice_key = _make_slice_key()
    slice_metrics = {
        'one_dim':
            types.ValueWithTDistribution(2.0, 1.0, 3, 2.0),
        'nans':
            types.ValueWithTDistribution(
                float('nan'), float('nan'), -1, float('nan')),
    }
    expected_metrics_for_slice = text_format.Parse(
        """
        slice_key {}
        metrics {
          key: "one_dim"
          value {
            bounded_value {
              value {
                value: 2.0
              }
              lower_bound {
                value: 0.4087768
              }
              upper_bound {
                value: 3.5912232
              }
              methodology: POISSON_BOOTSTRAP
            }
          }
        }
        metrics {
          key: "nans"
          value {
            bounded_value {
              value {
                value: nan
              }
              lower_bound {
                value: nan
              }
              upper_bound {
                value: nan
              }
              methodology: POISSON_BOOTSTRAP
            }
          }
        }
        """, metrics_for_slice_pb2.MetricsForSlice())
    got = metrics_and_plots_serialization._serialize_metrics(
        (slice_key, slice_metrics), [])
    self.assertProtoEquals(
        expected_metrics_for_slice,
        metrics_for_slice_pb2.MetricsForSlice.FromString(got))

  def testTensorValuedMetrics(self):
    slice_key = _make_slice_key()
    slice_metrics = {
        'one_dim':
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        'two_dims':
            np.array([['two', 'dims', 'test'], ['TWO', 'DIMS', 'TEST']]),
        'three_dims':
            np.array([[[100, 200, 300]], [[500, 600, 700]]], dtype=np.int64),
    }
    expected_metrics_for_slice = text_format.Parse(
        """
        slice_key {}
        metrics {
          key: "one_dim"
          value {
            array_value {
              data_type: FLOAT32
              shape: 4
              float32_values: [1.0, 2.0, 3.0, 4.0]
            }
          }
        }
        metrics {
          key: "two_dims"
          value {
            array_value {
              data_type: BYTES
              shape: [2, 3]
              bytes_values: ["two", "dims", "test", "TWO", "DIMS", "TEST"]
            }
          }
        }
        metrics {
          key: "three_dims"
          value {
            array_value {
              data_type: INT64
              shape: [2, 1, 3]
              int64_values: [100, 200, 300, 500, 600, 700]
            }
          }
        }
        """, metrics_for_slice_pb2.MetricsForSlice())
    got = metrics_and_plots_serialization._serialize_metrics(
        (slice_key, slice_metrics), [])
    self.assertProtoEquals(
        expected_metrics_for_slice,
        metrics_for_slice_pb2.MetricsForSlice.FromString(got))


if __name__ == '__main__':
  tf.test.main()
