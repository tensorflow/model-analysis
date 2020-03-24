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

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tensorflow_model_analysis.proto import metrics_for_slice_pb2
from tensorflow_model_analysis.slicer import auto_slicing_util

from google.protobuf import text_format


class AutoSlicingUtilTest(tf.test.TestCase):

  def test_find_top_slices(self):
    metrics = [
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
            int64_value: 5
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
            int64_value: 10
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
            int64_value: 15
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
    self.assertEqual(
        auto_slicing_util.find_top_slices(
            metrics, metric_key='accuracy', comparison_type='LOWER'), [
                auto_slicing_util.SliceComparisonResult(
                    slice_key=u'age:5',
                    num_examples=500.0,
                    pvalue=0.0,
                    effect_size=4.0)
            ])
    self.assertEqual(
        auto_slicing_util.find_top_slices(
            metrics, metric_key='accuracy', comparison_type='HIGHER'), [
                auto_slicing_util.SliceComparisonResult(
                    slice_key=u'age:15',
                    num_examples=500.0,
                    pvalue=7.356017854191938e-70,
                    effect_size=0.9999999999999996)
            ])


if __name__ == '__main__':
  tf.test.main()
