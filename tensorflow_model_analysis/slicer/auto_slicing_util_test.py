# Lint as: python3
# Copyright 2020 Google LLC
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
"""Tests for auto slicing utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
from pandas._testing import assert_frame_equal

import tensorflow as tf
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.proto import metrics_for_slice_pb2
from tensorflow_model_analysis.slicer import auto_slicing_util
from tensorflow_model_analysis.slicer import slicer_lib

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import statistics_pb2


class AutoSlicingUtilTest(tf.test.TestCase):

  def _get_metrics(self):
    return [
        text_format.Parse(
            """
        slice_key {
        }
        metric_keys_and_values {
          key { name: "accuracy" }
          value {
            bounded_value {
              value { value: 0.8 }
              lower_bound { value: 0.5737843 }
              upper_bound { value: 1.0262157 }
              methodology: POISSON_BOOTSTRAP
            }
            confidence_interval {
              lower_bound { value: 0.5737843 }
              upper_bound { value: 1.0262157 }
              t_distribution_value {
                sample_mean { value: 0.8 }
                sample_standard_deviation { value: 0.1 }
                sample_degrees_of_freedom { value: 9 }
                unsampled_value { value: 0.8 }
              }
            }
          }
        }
        metric_keys_and_values {
          key { name: "example_count" }
          value {
            bounded_value {
              value { value: 1500 }
              lower_bound { value: 1500 }
              upper_bound { value: 1500 }
              methodology: POISSON_BOOTSTRAP
            }
            confidence_interval {
              lower_bound { value: 1500 }
              upper_bound { value: 1500 }
              t_distribution_value {
                sample_mean { value: 1500 }
                sample_standard_deviation { value: 0 }
                sample_degrees_of_freedom { value: 9 }
                unsampled_value { value: 1500 }
              }
            }
          }
        }
        """, metrics_for_slice_pb2.MetricsForSlice()),
        text_format.Parse(
            """
        slice_key {
          single_slice_keys {
            column: 'age'
            bytes_value: '[1.0, 6.0)'
          }
        }
        metric_keys_and_values {
          key { name: "accuracy" }
          value {
            bounded_value {
              value { value: 0.4 }
              lower_bound { value: 0.3737843 }
              upper_bound { value: 0.6262157 }
              methodology: POISSON_BOOTSTRAP
            }
            confidence_interval {
              lower_bound { value: 0.3737843 }
              upper_bound { value: 0.6262157 }
              t_distribution_value {
                sample_mean { value: 0.4 }
                sample_standard_deviation { value: 0.1 }
                sample_degrees_of_freedom { value: 9 }
                unsampled_value { value: 0.4 }
              }
            }
          }
        }
        metric_keys_and_values {
          key { name: "example_count" }
          value {
            bounded_value {
              value { value: 500 }
              lower_bound { value: 500 }
              upper_bound { value: 500 }
              methodology: POISSON_BOOTSTRAP
            }
            confidence_interval {
              lower_bound { value: 500 }
              upper_bound { value: 500 }
              t_distribution_value {
                sample_mean { value: 500 }
                sample_standard_deviation { value: 0 }
                sample_degrees_of_freedom { value: 9 }
                unsampled_value { value: 500 }
              }
            }
          }
        }
        """, metrics_for_slice_pb2.MetricsForSlice()),
        text_format.Parse(
            """
        slice_key {
          single_slice_keys {
            column: 'age'
            bytes_value: '[6.0, 12.0)'
          }
        }
        metric_keys_and_values {
          key { name: "accuracy" }
          value {
            bounded_value {
              value { value: 0.79 }
              lower_bound { value: 0.5737843 }
              upper_bound { value: 1.0262157 }
              methodology: POISSON_BOOTSTRAP
            }
            confidence_interval {
              lower_bound { value: 0.5737843 }
              upper_bound { value: 1.0262157 }
              t_distribution_value {
                sample_mean { value: 0.79 }
                sample_standard_deviation { value: 0.1 }
                sample_degrees_of_freedom { value: 9 }
                unsampled_value { value: 0.79 }
              }
            }
          }
        }
        metric_keys_and_values {
          key { name: "example_count" }
          value {
            bounded_value {
              value { value: 500 }
              lower_bound { value: 500 }
              upper_bound { value: 500 }
              methodology: POISSON_BOOTSTRAP
            }
            confidence_interval {
              lower_bound { value: 500 }
              upper_bound { value: 500 }
              t_distribution_value {
                sample_mean { value: 500 }
                sample_standard_deviation { value: 0 }
                sample_degrees_of_freedom { value: 9 }
                unsampled_value { value: 500}
              }
            }
          }
        }
        """, metrics_for_slice_pb2.MetricsForSlice()),
        text_format.Parse(
            """
        slice_key {
          single_slice_keys {
            column: 'age'
            bytes_value: '[12.0, 18.0)'
          }
        }
        metric_keys_and_values {
          key { name: "accuracy" }
          value {
            bounded_value {
              value { value: 0.9 }
              lower_bound { value: 0.5737843 }
              upper_bound { value: 1.0262157 }
              methodology: POISSON_BOOTSTRAP
            }
            confidence_interval {
              lower_bound { value: 0.5737843 }
              upper_bound { value: 1.0262157 }
              t_distribution_value {
                sample_mean { value: 0.9 }
                sample_standard_deviation { value: 0.1 }
                sample_degrees_of_freedom { value: 9 }
                unsampled_value { value: 0.9 }
              }
            }
          }
        }
        metric_keys_and_values {
          key { name: "example_count" }
          value {
            bounded_value {
              value { value: 500 }
              lower_bound { value: 500 }
              upper_bound { value: 500 }
              methodology: POISSON_BOOTSTRAP
            }
            confidence_interval {
              lower_bound { value: 500 }
              upper_bound { value: 500 }
              t_distribution_value {
                sample_mean { value: 500 }
                sample_standard_deviation { value: 0 }
                sample_degrees_of_freedom { value: 9 }
                unsampled_value { value: 500}
              }
            }
          }
        }
        """, metrics_for_slice_pb2.MetricsForSlice()),
        text_format.Parse(
            """
        slice_key {
          single_slice_keys {
            column: 'country'
            bytes_value: 'USA'
          }
        }
        metric_keys_and_values {
          key { name: "accuracy" }
          value {
            bounded_value {
              value { value: 0.9 }
              lower_bound { value: 0.5737843 }
              upper_bound { value: 1.0262157 }
              methodology: POISSON_BOOTSTRAP
            }
            confidence_interval {
              lower_bound { value: 0.5737843 }
              upper_bound { value: 1.0262157 }
              t_distribution_value {
                sample_mean { value: 0.9 }
                sample_standard_deviation { value: 0.1 }
                sample_degrees_of_freedom { value: 9 }
                unsampled_value { value: 0.9 }
              }
            }
          }
        }
        metric_keys_and_values {
          key { name: "example_count" }
          value {
            bounded_value {
              value { value: 500 }
              lower_bound { value: 500 }
              upper_bound { value: 500 }
              methodology: POISSON_BOOTSTRAP
            }
            confidence_interval {
              lower_bound { value: 500 }
              upper_bound { value: 500 }
              t_distribution_value {
                sample_mean { value: 500 }
                sample_standard_deviation { value: 0 }
                sample_degrees_of_freedom { value: 9 }
                unsampled_value { value: 500}
              }
            }
          }
        }
        """, metrics_for_slice_pb2.MetricsForSlice()),
        text_format.Parse(
            """
        slice_key {
          single_slice_keys {
            column: 'country'
            bytes_value: 'USA'
          }
          single_slice_keys {
            column: 'age'
            bytes_value: '[12.0, 18.0)'
          }
        }
        metric_keys_and_values {
          key { name: "accuracy" }
          value {
            bounded_value {
              value { value: 0.9 }
              lower_bound { value: 0.5737843 }
              upper_bound { value: 1.0262157 }
              methodology: POISSON_BOOTSTRAP
            }
            confidence_interval {
              lower_bound { value: 0.5737843 }
              upper_bound { value: 1.0262157 }
              t_distribution_value {
                sample_mean { value: 0.9 }
                sample_standard_deviation { value: 0.1 }
                sample_degrees_of_freedom { value: 9 }
                unsampled_value { value: 0.9 }
              }
            }
          }
        }
        metric_keys_and_values {
          key { name: "example_count" }
          value {
            bounded_value {
              value { value: 500 }
              lower_bound { value: 500 }
              upper_bound { value: 500 }
              methodology: POISSON_BOOTSTRAP
            }
            confidence_interval {
              lower_bound { value: 500 }
              upper_bound { value: 500 }
              t_distribution_value {
                sample_mean { value: 500 }
                sample_standard_deviation { value: 0 }
                sample_degrees_of_freedom { value: 9 }
                unsampled_value { value: 500}
              }
            }
          }
        }
        """, metrics_for_slice_pb2.MetricsForSlice())
    ]

  def test_partition_slices_without_metric_sub_key(self):
    metrics = self._get_metrics()
    result = auto_slicing_util.partition_slices(
        metrics,
        metric_key=metric_types.MetricKey(name='accuracy'),
        comparison_type='LOWER')
    self.assertCountEqual([s.slice_key for s in result[0]],
                          [(('age', '[1.0, 6.0)'),)])
    self.assertCountEqual([s.slice_key for s in result[1]],
                          [(('age', '[6.0, 12.0)'),),
                           (('age', '[12.0, 18.0)'),), (('country', 'USA'),),
                           (('country', 'USA'), ('age', '[12.0, 18.0)'))])

    result = auto_slicing_util.partition_slices(
        metrics,
        metric_key=metric_types.MetricKey(name='accuracy'),
        comparison_type='HIGHER')
    self.assertCountEqual([s.slice_key for s in result[0]],
                          [(('age', '[12.0, 18.0)'),), (('country', 'USA'),),
                           (('country', 'USA'), ('age', '[12.0, 18.0)'))])
    self.assertCountEqual([s.slice_key for s in result[1]],
                          [(('age', '[1.0, 6.0)'),), (('age', '[6.0, 12.0)'),)])

  def test_partition_slices_with_metric_sub_key(self):
    metrics = self._get_metrics()
    # Set sub_key.
    for metric in metrics:
      for kv in metric.metric_keys_and_values:
        kv.key.sub_key.MergeFrom(metric_types.SubKey(class_id=0).to_proto())
    result = auto_slicing_util.partition_slices(
        metrics,
        metric_key=metric_types.MetricKey(
            name='accuracy', sub_key=metric_types.SubKey(class_id=0)),
        comparison_type='LOWER')
    self.assertCountEqual([s.slice_key for s in result[0]],
                          [(('age', '[1.0, 6.0)'),)])
    self.assertCountEqual([s.slice_key for s in result[1]],
                          [(('age', '[6.0, 12.0)'),),
                           (('age', '[12.0, 18.0)'),), (('country', 'USA'),),
                           (('country', 'USA'), ('age', '[12.0, 18.0)'))])

    result = auto_slicing_util.partition_slices(
        metrics,
        metric_key=metric_types.MetricKey(
            name='accuracy', sub_key=metric_types.SubKey(class_id=0)),
        comparison_type='HIGHER')
    self.assertCountEqual([s.slice_key for s in result[0]],
                          [(('age', '[12.0, 18.0)'),), (('country', 'USA'),),
                           (('country', 'USA'), ('age', '[12.0, 18.0)'))])
    self.assertCountEqual([s.slice_key for s in result[1]],
                          [(('age', '[1.0, 6.0)'),), (('age', '[6.0, 12.0)'),)])

  def test_find_top_slices(self):
    input_slices = [
        auto_slicing_util.SliceComparisonResult(
            slice_key=(('age', '[6.0, 12.0)'),),
            num_examples=1500.0,
            slice_metric=0.9,
            base_metric=0.8,
            p_value=0,
            effect_size=0.8,
            raw_slice_metrics=None),
        auto_slicing_util.SliceComparisonResult(
            slice_key=(('age', '[12.0, 18.0)'),),
            num_examples=500.0,
            slice_metric=0.9,
            base_metric=0.8,
            p_value=0.00001,
            effect_size=0.9,
            raw_slice_metrics=None),
        auto_slicing_util.SliceComparisonResult(
            slice_key=(('age', '[12.0, 18.0)'), ('country', 'USA')),
            num_examples=500.0,
            slice_metric=0.91,
            base_metric=0.8,
            p_value=0.000011,
            effect_size=0.91,
            raw_slice_metrics=None)
    ]
    self.assertCountEqual(
        auto_slicing_util.find_top_slices(
            input_slices, rank_by='EFFECT_SIZE', prune_subset_slices=False), [
                auto_slicing_util.SliceComparisonResult(
                    slice_key=(('age', '[12.0, 18.0)'), ('country', 'USA')),
                    num_examples=500.0,
                    slice_metric=0.91,
                    base_metric=0.8,
                    p_value=0.000011,
                    effect_size=0.91,
                    raw_slice_metrics=None),
                auto_slicing_util.SliceComparisonResult(
                    slice_key=(('age', '[12.0, 18.0)'),),
                    num_examples=500.0,
                    slice_metric=0.9,
                    base_metric=0.8,
                    p_value=0.00001,
                    effect_size=0.9,
                    raw_slice_metrics=None),
                auto_slicing_util.SliceComparisonResult(
                    slice_key=(('age', '[6.0, 12.0)'),),
                    num_examples=1500.0,
                    slice_metric=0.9,
                    base_metric=0.8,
                    p_value=0,
                    effect_size=0.8,
                    raw_slice_metrics=None),
            ])
    self.assertCountEqual(
        auto_slicing_util.find_top_slices(
            input_slices, rank_by='EFFECT_SIZE', prune_subset_slices=True), [
                auto_slicing_util.SliceComparisonResult(
                    slice_key=(('age', '[12.0, 18.0)'),),
                    num_examples=500.0,
                    slice_metric=0.9,
                    base_metric=0.8,
                    p_value=0.00001,
                    effect_size=0.9,
                    raw_slice_metrics=None),
                auto_slicing_util.SliceComparisonResult(
                    slice_key=(('age', '[6.0, 12.0)'),),
                    num_examples=1500.0,
                    slice_metric=0.9,
                    base_metric=0.8,
                    p_value=0,
                    effect_size=0.8,
                    raw_slice_metrics=None),
            ])
    self.assertCountEqual(
        auto_slicing_util.find_top_slices(input_slices, rank_by='PVALUE'), [
            auto_slicing_util.SliceComparisonResult(
                slice_key=(('age', '[6.0, 12.0)'),),
                num_examples=1500.0,
                slice_metric=0.9,
                base_metric=0.8,
                p_value=0,
                effect_size=0.8,
                raw_slice_metrics=None),
            auto_slicing_util.SliceComparisonResult(
                slice_key=(('age', '[12.0, 18.0)'),),
                num_examples=500.0,
                slice_metric=0.9,
                base_metric=0.8,
                p_value=0.00001,
                effect_size=0.9,
                raw_slice_metrics=None),
        ])
    self.assertCountEqual(
        auto_slicing_util.find_top_slices(
            input_slices, min_num_examples=1000, rank_by='EFFECT_SIZE'), [
                auto_slicing_util.SliceComparisonResult(
                    slice_key=(('age', '[6.0, 12.0)'),),
                    num_examples=1500.0,
                    slice_metric=0.9,
                    base_metric=0.8,
                    p_value=0,
                    effect_size=0.8,
                    raw_slice_metrics=None),
            ])

  def test_revert_slice_keys_for_transformed_features(self):
    statistics = text_format.Parse(
        """
        datasets{
          num_examples: 1500
          features {
            path { step: 'country' }
            type: STRING
            string_stats {
              unique: 10
            }
          }
          features {
            path { step: 'age' }
            type: INT
            num_stats {
              common_stats {
                num_non_missing: 1500
                min_num_values: 1
                max_num_values: 1
              }
              min: 1
              max: 18
              histograms {
                buckets {
                  low_value: 1
                  high_value: 6.0
                  sample_count: 500
                }
                buckets {
                  low_value: 6.0
                  high_value: 12.0
                  sample_count: 500
                }
                buckets {
                  low_value: 12.0
                  high_value: 18.0
                  sample_count: 500
                }
                type: QUANTILES
              }
            }
          }
        }
        """, statistics_pb2.DatasetFeatureStatisticsList())
    slices = [
        auto_slicing_util.SliceComparisonResult(
            slice_key=(),
            num_examples=500.0,
            slice_metric=0.9,
            base_metric=0.8,
            p_value=0,
            effect_size=0.9,
            raw_slice_metrics=None),
        auto_slicing_util.SliceComparisonResult(
            slice_key=(('transformed_age', 1),),
            num_examples=500.0,
            slice_metric=0.9,
            base_metric=0.8,
            p_value=0,
            effect_size=0.9,
            raw_slice_metrics=None),
        auto_slicing_util.SliceComparisonResult(
            slice_key=(('transformed_age', 2),),
            num_examples=500.0,
            slice_metric=0.9,
            base_metric=0.8,
            p_value=0,
            effect_size=0.9,
            raw_slice_metrics=None),
        auto_slicing_util.SliceComparisonResult(
            slice_key=(('country', 'USA'),),
            num_examples=500.0,
            slice_metric=0.9,
            base_metric=0.8,
            p_value=0,
            effect_size=0.9,
            raw_slice_metrics=None),
    ]
    expected_slices = [
        auto_slicing_util.SliceComparisonResult(
            slice_key=(),
            num_examples=500.0,
            slice_metric=0.9,
            base_metric=0.8,
            p_value=0,
            effect_size=0.9,
            raw_slice_metrics=None),
        auto_slicing_util.SliceComparisonResult(
            slice_key=(('age', '(1.0, 6.0]'),),
            num_examples=500.0,
            slice_metric=0.9,
            base_metric=0.8,
            p_value=0,
            effect_size=0.9,
            raw_slice_metrics=None),
        auto_slicing_util.SliceComparisonResult(
            slice_key=(('age', '(6.0, 12.0]'),),
            num_examples=500.0,
            slice_metric=0.9,
            base_metric=0.8,
            p_value=0,
            effect_size=0.9,
            raw_slice_metrics=None),
        auto_slicing_util.SliceComparisonResult(
            slice_key=(('country', 'USA'),),
            num_examples=500.0,
            slice_metric=0.9,
            base_metric=0.8,
            p_value=0,
            effect_size=0.9,
            raw_slice_metrics=None),
    ]
    actual = auto_slicing_util.revert_slice_keys_for_transformed_features(
        slices, statistics)
    self.assertEqual(actual, expected_slices)

  def test_remove_subset_slices(self):
    input_slices = [
        auto_slicing_util.SliceComparisonResult(
            slice_key=(('age', '[12.0, 18.0)'),),
            num_examples=500.0,
            slice_metric=0.9,
            base_metric=0.8,
            p_value=0,
            effect_size=0.9,
            raw_slice_metrics=None),
        auto_slicing_util.SliceComparisonResult(
            slice_key=(('age', '[12.0, 18.0)'), ('country', 'USA')),
            num_examples=500.0,
            slice_metric=0.9,
            base_metric=0.8,
            p_value=0,
            effect_size=0.9,
            raw_slice_metrics=None),
        auto_slicing_util.SliceComparisonResult(
            slice_key=(('age', '[6.0, 12.0)'), ('country', 'UK')),
            num_examples=500.0,
            slice_metric=0.9,
            base_metric=0.8,
            p_value=0,
            effect_size=0.9,
            raw_slice_metrics=None),
        auto_slicing_util.SliceComparisonResult(
            slice_key=(('age', '[6.0, 12.0)'), ('country', 'UK'), ('sex', 'M')),
            num_examples=500.0,
            slice_metric=0.9,
            base_metric=0.8,
            p_value=0,
            effect_size=0.9,
            raw_slice_metrics=None),
    ]
    expected_slices = [
        auto_slicing_util.SliceComparisonResult(
            slice_key=(('age', '[12.0, 18.0)'),),
            num_examples=500.0,
            slice_metric=0.9,
            base_metric=0.8,
            p_value=0,
            effect_size=0.9,
            raw_slice_metrics=None),
        auto_slicing_util.SliceComparisonResult(
            slice_key=(('age', '[6.0, 12.0)'), ('country', 'UK')),
            num_examples=500.0,
            slice_metric=0.9,
            base_metric=0.8,
            p_value=0,
            effect_size=0.9,
            raw_slice_metrics=None),
    ]
    self.assertCountEqual(
        auto_slicing_util.remove_subset_slices(input_slices), expected_slices)
    self.assertCountEqual(auto_slicing_util.remove_subset_slices([]), [])

  def test_is_significant_slice(self):
    self.assertEqual(
        auto_slicing_util._is_significant_slice(0.8, 0.001, 100, 0.9, 0.001,
                                                1000, 'LOWER', 0.01)[0], True)
    # Test zero std. deviation for overall dataset.
    with self.assertRaises(AssertionError):
      self.assertEqual(
          auto_slicing_util._is_significant_slice(0.8, 0.001, 100, 0.9, 0, 1000,
                                                  'LOWER', 0.01)[0], True)
    # Test zero std. deviation for slice.
    with self.assertRaises(AssertionError):
      auto_slicing_util._is_significant_slice(0.8, 0, 100, 0.9, 0, 1000,
                                              'LOWER', 0.01)

  def test_get_slices_as_dataframe(self):
    input_slices = [
        auto_slicing_util.SliceComparisonResult(
            slice_key=(('native-country', 'United-States'),),
            num_examples=29170,
            slice_metric=0.09,
            base_metric=0.087,
            p_value=0,
            effect_size=0.46,
            raw_slice_metrics=text_format.Parse(
                """
                slice_key {
                  single_slice_keys {
                    column: "native-country"
                    bytes_value: "United-States"
                    }
                }
                metric_keys_and_values {
                  key { name: "false_positives" }
                  value {
                    bounded_value {
                      lower_bound { value: 1754.6514199722158 }
                      upper_bound { value: 2092.488580027784 }
                      value { value: 1923.57 }
                      methodology: POISSON_BOOTSTRAP
                    }
                    confidence_interval {
                      lower_bound { value: 1754.6514199722158 }
                      upper_bound { value: 2092.488580027784 }
                      t_distribution_value {
                        sample_mean { value: 1923.57 }
                        sample_standard_deviation { value: 85.13110418664061 }
                        sample_degrees_of_freedom { value: 99 }
                        unsampled_value { value: 1943.0 }
                      }
                    }
                  }
                }
                metric_keys_and_values {
                  key { name: "false_negatives" }
                  value {
                    bounded_value {
                      lower_bound { value: 3595.413107983637 }
                      upper_bound { value: 4195.886892016363 }
                      value { value: 3895.65 }
                      methodology: POISSON_BOOTSTRAP
                    }
                    confidence_interval {
                      lower_bound { value: 3595.413107983637 }
                      upper_bound { value: 4195.886892016363 }
                      t_distribution_value {
                        sample_mean { value: 3895.65 }
                        sample_standard_deviation { value: 151.31253252729257 }
                        sample_degrees_of_freedom { value: 99 }
                        unsampled_value { value: 3935.0 }
                      }
                    }
                  }
                }""", metrics_for_slice_pb2.MetricsForSlice())),
        auto_slicing_util.SliceComparisonResult(
            slice_key=(('age', '[58.0, 90.0)'),),
            num_examples=2999,
            slice_metric=0.09,
            base_metric=0.0875,
            p_value=7.8,
            effect_size=0.98,
            raw_slice_metrics=text_format.Parse(
                """
                slice_key {
                  single_slice_keys {
                    column: "age"
                    bytes_value: "[58.0, 90.0)"
                  }
                }
                metric_keys_and_values {
                  key { name: "false_positives" }
                  value {
                    bounded_value {
                      lower_bound { value: 167.54646972321814 }
                      upper_bound { value: 236.37353027678188 }
                      value { value: 201.96 }
                      methodology: POISSON_BOOTSTRAP
                    }
                    confidence_interval {
                      lower_bound { value: 167.54646972321814 }
                      upper_bound { value: 236.37353027678188 }
                      t_distribution_value {
                        sample_mean { value: 201.96 }
                        sample_standard_deviation { value: 17.343632837435358 }
                        sample_degrees_of_freedom { value: 99 }
                        unsampled_value { value: 204.0 }
                      }
                    }
                  }
                }
                metric_keys_and_values {
                  key { name: "false_negatives" }
                  value {
                    bounded_value {
                      lower_bound { value: 486.4402337348782 }
                      upper_bound { value: 610.479766265122 }
                      value { value: 548.46 }
                      methodology: POISSON_BOOTSTRAP
                    }
                    confidence_interval {
                      lower_bound { value: 486.4402337348782 }
                      upper_bound { value: 610.479766265122 }
                      t_distribution_value {
                        sample_mean { value: 548.46 }
                        sample_standard_deviation { value: 31.256544914589938 }
                        sample_degrees_of_freedom { value: 99 }
                        unsampled_value { value: 554.0 }
                      }
                    }
                  }
                }""", metrics_for_slice_pb2.MetricsForSlice()))
    ]
    additional_metric_keys = [
        metric_types.MetricKey('false_positives'),
        metric_types.MetricKey('false_negatives')
    ]
    expected_dataframe_data = [{
        'Slice': slicer_lib.stringify_slice_key(input_slices[0].slice_key),
        'Size': input_slices[0].num_examples,
        'Slice metric': input_slices[0].slice_metric,
        'Base metric': input_slices[0].base_metric,
        'P-Value': input_slices[0].p_value,
        'Effect size': input_slices[0].effect_size,
        str(additional_metric_keys[0]): 1923.57,
        str(additional_metric_keys[1]): 3895.65
    }, {
        'Slice': slicer_lib.stringify_slice_key(input_slices[1].slice_key),
        'Size': input_slices[1].num_examples,
        'Slice metric': input_slices[1].slice_metric,
        'Base metric': input_slices[1].base_metric,
        'P-Value': input_slices[1].p_value,
        'Effect size': input_slices[1].effect_size,
        str(additional_metric_keys[0]): 201.96,
        str(additional_metric_keys[1]): 548.46
    }]
    expected_dataframe_column_labels = [
        'Slice', 'Size', 'Slice metric', 'Base metric', 'P-Value',
        'Effect size',
        str(additional_metric_keys[0]),
        str(additional_metric_keys[1])
    ]
    expected_dataframe = pd.DataFrame(
        expected_dataframe_data, columns=expected_dataframe_column_labels)
    expected_dataframe.set_index('Slice', inplace=True)

    actual_dataframe = auto_slicing_util.get_slices_as_dataframe(
        input_slices, additional_metric_keys)

    assert_frame_equal(actual_dataframe, expected_dataframe)


if __name__ == '__main__':
  tf.test.main()
