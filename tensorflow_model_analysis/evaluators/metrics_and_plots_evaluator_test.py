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

import os
import string

# Standard Imports

import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import types
from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.api import tfma_unit
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.eval_saved_model.example_trainers import fixed_prediction_estimator
from tensorflow_model_analysis.eval_saved_model.example_trainers import fixed_prediction_estimator_no_labels
from tensorflow_model_analysis.eval_saved_model.example_trainers import linear_classifier
from tensorflow_model_analysis.evaluators import metrics_and_plots_evaluator
from tensorflow_model_analysis.extractors import predict_extractor
from tensorflow_model_analysis.extractors import slice_key_extractor
from tensorflow_model_analysis.post_export_metrics import metric_keys
from tensorflow_model_analysis.post_export_metrics import post_export_metrics
from tensorflow_model_analysis.proto import metrics_for_slice_pb2
from tensorflow_model_analysis.slicer import slicer

from google.protobuf import text_format


def _addExampleCountMetricCallback(  # pylint: disable=invalid-name
    features_dict, predictions_dict, labels_dict):
  del features_dict
  del labels_dict
  metric_ops = {}
  value_op, update_op = tf.contrib.metrics.count(predictions_dict['logits'])
  metric_ops['added_example_count'] = (value_op, update_op)
  return metric_ops


def _addPyFuncMetricCallback(  # pylint: disable=invalid-name
    features_dict, predictions_dict, labels_dict):
  del features_dict
  del predictions_dict

  total_value = tf.Variable(
      initial_value=0.0,
      dtype=tf.float64,
      trainable=False,
      collections=[tf.GraphKeys.METRIC_VARIABLES, tf.GraphKeys.LOCAL_VARIABLES],
      validate_shape=True,
      name='total')

  def my_func(x):
    return np.sum(x, dtype=np.float64)

  update_op = tf.assign_add(total_value,
                            tf.py_func(my_func, [labels_dict], tf.float64))
  value_op = tf.identity(total_value)
  metric_ops = {}
  metric_ops['py_func_label_sum'] = (value_op, update_op)
  return metric_ops


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

  def _getEvalExportDir(self):
    return os.path.join(self._getTempDir(), 'eval_export_dir')

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
    serialized = metrics_and_plots_evaluator._serialize_plots(
        (slice_key, tfma_plots), [calibration_plot])
    self.assertProtoEquals(
        expected_plot_data,
        metrics_for_slice_pb2.PlotsForSlice.FromString(serialized))

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
              }
            }
          }
        }
        """, metrics_for_slice_pb2.MetricsForSlice())

    got = metrics_and_plots_evaluator._serialize_metrics(
        (slice_key, slice_metrics),
        [post_export_metrics.confusion_matrix_at_thresholds(thresholds)])
    self.assertProtoEquals(
        expected_metrics_for_slice,
        metrics_for_slice_pb2.MetricsForSlice.FromString(got))

  def testSerializeMetricsRanges(self):
    slice_key = _make_slice_key('age', 5, 'language', 'english', 'price', 0.3)
    slice_metrics = {
        'accuracy': types.ValueWithConfidenceInterval(0.8, 0.7, 0.9),
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
                value: 0.7
              }
              upper_bound {
                value: 0.9
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

    got = metrics_and_plots_evaluator._serialize_metrics(
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

    got = metrics_and_plots_evaluator._serialize_metrics(
        (slice_key, slice_metrics),
        [post_export_metrics.auc(),
         post_export_metrics.auc(curve='PR')])
    self.assertProtoEquals(
        expected_metrics_for_slice,
        metrics_for_slice_pb2.MetricsForSlice.FromString(got))

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

    got = metrics_and_plots_evaluator._serialize_metrics(
        (slice_key, slice_metrics), [])
    self.assertProtoEquals(
        expected_metrics_for_slice,
        metrics_for_slice_pb2.MetricsForSlice.FromString(got))

  def testUncertaintyValuedMetrics(self):
    slice_key = _make_slice_key()
    slice_metrics = {
        'one_dim':
            types.ValueWithConfidenceInterval(2.0, 1.0, 3.0),
        'nans':
            types.ValueWithConfidenceInterval(
                float('nan'), float('nan'), float('nan')),
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
                value: 1.0
              }
              upper_bound {
                value: 3.0
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
    got = metrics_and_plots_evaluator._serialize_metrics(
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
    got = metrics_and_plots_evaluator._serialize_metrics(
        (slice_key, slice_metrics), [])
    self.assertProtoEquals(
        expected_metrics_for_slice,
        metrics_for_slice_pb2.MetricsForSlice.FromString(got))

  def testEvaluateNoSlicing(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = linear_classifier.simple_linear_classifier(
        None, temp_eval_export_dir)
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=eval_export_dir,
        add_metrics_callbacks=[_addExampleCountMetricCallback])
    extractors = [
        predict_extractor.PredictExtractor(eval_shared_model),
        slice_key_extractor.SliceKeyExtractor()
    ]

    with beam.Pipeline() as pipeline:
      example1 = self._makeExample(age=3.0, language='english', label=1.0)
      example2 = self._makeExample(age=3.0, language='chinese', label=0.0)
      example3 = self._makeExample(age=4.0, language='english', label=1.0)
      example4 = self._makeExample(age=5.0, language='chinese', label=0.0)

      metrics, _ = (
          pipeline
          | 'Create' >> beam.Create([
              example1.SerializeToString(),
              example2.SerializeToString(),
              example3.SerializeToString(),
              example4.SerializeToString()
          ])
          | 'InputsToExtracts' >> model_eval_lib.InputsToExtracts()
          | 'Extract' >> tfma_unit.Extract(extractors=extractors)  # pylint: disable=no-value-for-parameter
          | 'ComputeMetricsAndPlots' >> metrics_and_plots_evaluator
          .ComputeMetricsAndPlots(eval_shared_model=eval_shared_model))

      def check_result(got):
        try:
          self.assertEqual(1, len(got), 'got: %s' % got)
          (slice_key, value) = got[0]
          self.assertEqual((), slice_key)
          self.assertDictElementsAlmostEqual(
              value, {
                  'accuracy': 1.0,
                  'label/mean': 0.5,
                  'my_mean_age': 3.75,
                  'my_mean_age_times_label': 1.75,
                  'added_example_count': 4.0
              })
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(metrics, check_result)

  def testEvaluateWithSlicingAndDifferentBatchSizes(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = linear_classifier.simple_linear_classifier(
        None, temp_eval_export_dir)
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=eval_export_dir,
        add_metrics_callbacks=[_addExampleCountMetricCallback])
    extractors = [
        predict_extractor.PredictExtractor(eval_shared_model),
        slice_key_extractor.SliceKeyExtractor([
            slicer.SingleSliceSpec(),
            slicer.SingleSliceSpec(columns=['slice_key'])
        ])
    ]

    for batch_size in [1, 2, 4, 8]:

      with beam.Pipeline() as pipeline:
        example1 = self._makeExample(
            age=3.0, language='english', label=1.0, slice_key='first_slice')
        example2 = self._makeExample(
            age=3.0, language='chinese', label=0.0, slice_key='first_slice')
        example3 = self._makeExample(
            age=4.0, language='english', label=0.0, slice_key='second_slice')
        example4 = self._makeExample(
            age=5.0, language='chinese', label=1.0, slice_key='second_slice')
        example5 = self._makeExample(
            age=5.0, language='chinese', label=1.0, slice_key='second_slice')

        metrics, plots = (
            pipeline
            | 'Create' >> beam.Create([
                example1.SerializeToString(),
                example2.SerializeToString(),
                example3.SerializeToString(),
                example4.SerializeToString(),
                example5.SerializeToString(),
            ])
            | 'InputsToExtracts' >> model_eval_lib.InputsToExtracts()
            | 'Extract' >> tfma_unit.Extract(extractors=extractors)  # pylint: disable=no-value-for-parameter
            | 'ComputeMetricsAndPlots' >>
            metrics_and_plots_evaluator.ComputeMetricsAndPlots(
                eval_shared_model=eval_shared_model,
                desired_batch_size=batch_size))

        def check_result(got):
          try:
            self.assertEqual(3, len(got), 'got: %s' % got)
            slices = {}
            for slice_key, value in got:
              slices[slice_key] = value
            overall_slice = ()
            first_slice = (('slice_key', b'first_slice'),)
            second_slice = (('slice_key', b'second_slice'),)
            self.assertItemsEqual(
                list(slices.keys()), [overall_slice, first_slice, second_slice])
            self.assertDictElementsAlmostEqual(
                slices[overall_slice], {
                    'accuracy': 0.4,
                    'label/mean': 0.6,
                    'my_mean_age': 4.0,
                    'my_mean_age_times_label': 2.6,
                    'added_example_count': 5.0
                })
            self.assertDictElementsAlmostEqual(
                slices[first_slice], {
                    'accuracy': 1.0,
                    'label/mean': 0.5,
                    'my_mean_age': 3.0,
                    'my_mean_age_times_label': 1.5,
                    'added_example_count': 2.0
                })
            self.assertDictElementsAlmostEqual(
                slices[second_slice], {
                    'accuracy': 0.0,
                    'label/mean': 2.0 / 3.0,
                    'my_mean_age': 14.0 / 3.0,
                    'my_mean_age_times_label': 10.0 / 3.0,
                    'added_example_count': 3.0
                })

          except AssertionError as err:
            # This function is redefined every iteration, so it will have the
            # right value of batch_size.
            raise util.BeamAssertException(
                'batch_size = %d, error: %s' % (batch_size, err))  # pylint: disable=cell-var-from-loop

        util.assert_that(metrics, check_result, label='metrics')
        util.assert_that(plots, util.is_empty(), label='plots')

  def testEvaluateWithSlicingAndUncertainty(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = linear_classifier.simple_linear_classifier(
        None, temp_eval_export_dir)
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=eval_export_dir,
        add_metrics_callbacks=[_addExampleCountMetricCallback])
    extractors = [
        predict_extractor.PredictExtractor(eval_shared_model),
        slice_key_extractor.SliceKeyExtractor([
            slicer.SingleSliceSpec(),
            slicer.SingleSliceSpec(columns=['slice_key'])
        ])
    ]

    for batch_size in [1, 2, 4, 8]:

      with beam.Pipeline() as pipeline:
        example1 = self._makeExample(
            age=3.0, language='english', label=1.0, slice_key='first_slice')
        example2 = self._makeExample(
            age=3.0, language='chinese', label=0.0, slice_key='first_slice')
        example3 = self._makeExample(
            age=4.0, language='english', label=0.0, slice_key='second_slice')
        example4 = self._makeExample(
            age=5.0, language='chinese', label=1.0, slice_key='second_slice')
        example5 = self._makeExample(
            age=5.0, language='chinese', label=1.0, slice_key='second_slice')

        metrics, _ = (
            pipeline
            | 'Create' >> beam.Create([
                example1.SerializeToString(),
                example2.SerializeToString(),
                example3.SerializeToString(),
                example4.SerializeToString(),
                example5.SerializeToString(),
            ])
            | 'InputsToExtracts' >> model_eval_lib.InputsToExtracts()
            | 'Extract' >> tfma_unit.Extract(extractors=extractors)  # pylint: disable=no-value-for-parameter
            | 'ComputeMetricsAndPlots' >>
            metrics_and_plots_evaluator.ComputeMetricsAndPlots(
                eval_shared_model=eval_shared_model,
                desired_batch_size=batch_size,
                num_bootstrap_samples=10))

        def check_result(got):
          try:
            self.assertEqual(3, len(got), 'got: %s' % got)
            slices = {}
            for slice_key, value in got:
              slices[slice_key] = value
            overall_slice = ()
            first_slice = (('slice_key', b'first_slice'),)
            second_slice = (('slice_key', b'second_slice'),)
            self.assertItemsEqual(
                list(slices.keys()), [overall_slice, first_slice, second_slice])
            self.assertDictElementsWithIntervalsAlmostEqual(
                slices[overall_slice], {
                    'accuracy': 0.4,
                    'label/mean': 0.6,
                    'my_mean_age': 4.0,
                    'my_mean_age_times_label': 2.6,
                    'added_example_count': 5.0
                })
            self.assertDictElementsWithIntervalsAlmostEqual(
                slices[first_slice], {
                    'accuracy': 1.0,
                    'label/mean': 0.5,
                    'my_mean_age': 3.0,
                    'my_mean_age_times_label': 1.5,
                    'added_example_count': 2.0
                })
            self.assertDictElementsWithIntervalsAlmostEqual(
                slices[second_slice], {
                    'accuracy': 0.0,
                    'label/mean': 2.0 / 3.0,
                    'my_mean_age': 14.0 / 3.0,
                    'my_mean_age_times_label': 10.0 / 3.0,
                    'added_example_count': 3.0
                })
            # Ensure that serialization of the key at the end of
            # ComputeMetricsAndPlots works.
            for slice_key, value in got:
              metrics_and_plots_evaluator._serialize_metrics((slice_key, value),
                                                             [])

          except AssertionError as err:
            # This function is redefined every iteration, so it will have the
            # right value of batch_size.
            raise util.BeamAssertException(
                'batch_size = %d, error: %s' % (batch_size, err))  # pylint: disable=cell-var-from-loop

        util.assert_that(metrics, check_result, label='metrics')

  def testEvaluateNoSlicingAddPostExportAndCustomMetrics(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = linear_classifier.simple_linear_classifier(
        None, temp_eval_export_dir)
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=eval_export_dir,
        add_metrics_callbacks=[
            _addExampleCountMetricCallback,
            # Note that since everything runs in-process this doesn't
            # actually test that the py_func can be correctly recreated
            # on workers in a distributed context.
            _addPyFuncMetricCallback,
            post_export_metrics.example_count(),
            post_export_metrics.example_weight(example_weight_key='age')
        ])
    extractors = [
        predict_extractor.PredictExtractor(eval_shared_model),
        slice_key_extractor.SliceKeyExtractor()
    ]

    with beam.Pipeline() as pipeline:
      example1 = self._makeExample(age=3.0, language='english', label=1.0)
      example2 = self._makeExample(age=3.0, language='chinese', label=0.0)
      example3 = self._makeExample(age=4.0, language='english', label=1.0)
      example4 = self._makeExample(age=5.0, language='chinese', label=0.0)

      metrics, plots = (
          pipeline
          | 'Create' >> beam.Create([
              example1.SerializeToString(),
              example2.SerializeToString(),
              example3.SerializeToString(),
              example4.SerializeToString()
          ])
          | 'InputsToExtracts' >> model_eval_lib.InputsToExtracts()
          | 'Extract' >> tfma_unit.Extract(extractors=extractors)  # pylint: disable=no-value-for-parameter
          | 'ComputeMetricsAndPlots' >> metrics_and_plots_evaluator
          .ComputeMetricsAndPlots(eval_shared_model=eval_shared_model))

      def check_result(got):
        try:
          self.assertEqual(1, len(got), 'got: %s' % got)
          (slice_key, value) = got[0]
          self.assertEqual((), slice_key)
          self.assertDictElementsAlmostEqual(
              got_values_dict=value,
              expected_values_dict={
                  'accuracy': 1.0,
                  'label/mean': 0.5,
                  'my_mean_age': 3.75,
                  'my_mean_age_times_label': 1.75,
                  'added_example_count': 4.0,
                  'py_func_label_sum': 2.0,
                  metric_keys.EXAMPLE_COUNT: 4.0,
                  metric_keys.EXAMPLE_WEIGHT: 15.0
              })
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(metrics, check_result, label='metrics')
      util.assert_that(plots, util.is_empty(), label='plots')

  def testEvaluateNoSlicingAddPostExportAndCustomMetricsUnsupervisedModel(self):
    # Mainly for testing that the ExampleCount post export metric works with
    # unsupervised models.
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        fixed_prediction_estimator_no_labels
        .simple_fixed_prediction_estimator_no_labels(None,
                                                     temp_eval_export_dir))
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=eval_export_dir,
        add_metrics_callbacks=[
            post_export_metrics.example_count(),
            post_export_metrics.example_weight(example_weight_key='prediction')
        ])
    extractors = [
        predict_extractor.PredictExtractor(eval_shared_model),
        slice_key_extractor.SliceKeyExtractor()
    ]

    with beam.Pipeline() as pipeline:
      example1 = self._makeExample(prediction=1.0)
      example2 = self._makeExample(prediction=2.0)

      metrics, plots = (
          pipeline
          | 'Create' >> beam.Create([
              example1.SerializeToString(),
              example2.SerializeToString(),
          ])
          | 'InputsToExtracts' >> model_eval_lib.InputsToExtracts()
          | 'Extract' >> tfma_unit.Extract(extractors=extractors)  # pylint: disable=no-value-for-parameter
          | 'ComputeMetricsAndPlots' >> metrics_and_plots_evaluator
          .ComputeMetricsAndPlots(eval_shared_model=eval_shared_model))

      def check_result(got):
        try:
          self.assertEqual(1, len(got), 'got: %s' % got)
          (slice_key, value) = got[0]
          self.assertEqual((), slice_key)
          self.assertDictElementsAlmostEqual(
              got_values_dict=value,
              expected_values_dict={
                  'average_loss': 2.5,
                  metric_keys.EXAMPLE_COUNT: 2.0,
                  metric_keys.EXAMPLE_WEIGHT: 3.0
              })
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(metrics, check_result, label='metrics')
      util.assert_that(plots, util.is_empty(), label='plots')

  def testEvaluateWithPlots(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        fixed_prediction_estimator.simple_fixed_prediction_estimator(
            None, temp_eval_export_dir))
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=eval_export_dir,
        add_metrics_callbacks=[
            post_export_metrics.example_count(),
            post_export_metrics.auc_plots()
        ])
    extractors = [
        predict_extractor.PredictExtractor(eval_shared_model),
        slice_key_extractor.SliceKeyExtractor()
    ]

    with beam.Pipeline() as pipeline:
      example1 = self._makeExample(prediction=0.0, label=1.0)
      example2 = self._makeExample(prediction=0.7, label=0.0)
      example3 = self._makeExample(prediction=0.8, label=1.0)
      example4 = self._makeExample(prediction=1.0, label=1.0)

      metrics, plots = (
          pipeline
          | 'Create' >> beam.Create([
              example1.SerializeToString(),
              example2.SerializeToString(),
              example3.SerializeToString(),
              example4.SerializeToString()
          ])
          | 'InputsToExtracts' >> model_eval_lib.InputsToExtracts()
          | 'Extract' >> tfma_unit.Extract(extractors=extractors)  # pylint: disable=no-value-for-parameter
          | 'ComputeMetricsAndPlots' >> metrics_and_plots_evaluator
          .ComputeMetricsAndPlots(eval_shared_model=eval_shared_model))

      def check_metrics(got):
        try:
          self.assertEqual(1, len(got), 'got: %s' % got)
          (slice_key, value) = got[0]
          self.assertEqual((), slice_key)
          self.assertDictElementsAlmostEqual(
              got_values_dict=value,
              expected_values_dict={
                  metric_keys.EXAMPLE_COUNT: 4.0,
              })
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(metrics, check_metrics, label='metrics')

      def check_plots(got):
        try:
          self.assertEqual(1, len(got), 'got: %s' % got)
          (slice_key, value) = got[0]
          self.assertEqual((), slice_key)
          self.assertDictMatrixRowsAlmostEqual(
              got_values_dict=value,
              expected_values_dict={
                  metric_keys.AUC_PLOTS_MATRICES: [(8001, [
                      2, 1, 0, 1, 1.0 / 1.0, 1.0 / 3.0
                  ])],
              })
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(plots, check_plots, label='plots')


if __name__ == '__main__':
  tf.test.main()
