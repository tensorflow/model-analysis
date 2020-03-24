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
"""Auto slicing utilities."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import math
import operator
from typing import List, NamedTuple, Text
from scipy import stats
from tensorflow_model_analysis import types
from tensorflow_model_analysis.proto import metrics_for_slice_pb2
from tensorflow_model_analysis.slicer import slicer_lib


# Reference: https://stats.idre.ucla.edu/other/mult-pkg/faq/general/faq-what-are-the-differences-between-one-tailed-and-two-tailed-tests/  pylint: disable=line-too-long
def _two_sided_to_one_sided_pvalue(pvalue: float, delta: float,
                                   comparison_type: Text) -> float:
  """Convert pvalue of a two sided t-test to one sided test.

  Args:
    pvalue: Two sided p value.
    delta: Metric diff (slice_metric - overall_metric).
    comparison_type: `HIGHER` if the alternate hypothesis is slice_metric >
      overall_metric else 'LOWER'.

  Returns:
    One sided p value.
  """
  if delta >= 0:
    if comparison_type == 'HIGHER':
      return 0.5 * pvalue
    else:
      return 1 - 0.5 * pvalue
  elif delta < 0:
    if comparison_type == 'HIGHER':
      return 1 - 0.5 * pvalue
    else:
      return 0.5 * pvalue
  return pvalue


class SliceComparisonResult(
    NamedTuple('SliceComparisonResult', [
        ('slice_key', Text),
        ('num_examples', int),
        ('pvalue', float),
        ('effect_size', float),
    ])):
  """Represents the result of comparing a slice with the base dataset.

  It includes the slice key, number of examples in the slice, the pvalue
  computed when doing significance testing and the effect size.
  """

  def __new__(
      cls,
      slice_key: Text,
      num_examples: int,
      pvalue: float,
      effect_size: float,
  ):
    return super(SliceComparisonResult,
                 cls).__new__(cls, slice_key, num_examples, pvalue, effect_size)


# Perform one sided Welch's t-test.
# Reference: https://en.wikipedia.org/wiki/Welch%27s_t-test
# When comparison_type is HIGHER,
#   Null hypothesis: slice_metric <= overall_metric
#   Alternate hypothesis: slice_metric > overall_metric
# When comparison_type is LOWER,
#   Null hypothesis: slice_metric >= overall_metric
#   Alternate hypothesis: slice_metric < overall_metric
def _is_significant_slice(slice_metric: float,
                          slice_std_dev: float,
                          slice_weight: float,
                          base_metric: float,
                          base_std_dev: float,
                          base_weight: float,
                          comparison_type: Text,
                          alpha: float = 0.01):
  """Perform statistical significance testing."""
  _, p_value_two_sided = stats.ttest_ind_from_stats(
      slice_metric,
      slice_std_dev,
      slice_weight,
      base_metric,
      base_std_dev,
      base_weight,
      equal_var=False)
  metric_diff = slice_metric - base_metric
  one_sided_p_value = _two_sided_to_one_sided_pvalue(
      p_value_two_sided, metric_diff, comparison_type=comparison_type)
  return (one_sided_p_value < alpha, one_sided_p_value)


# We use effect size to capture the magnitude of the metric difference.
# Reference: https://en.wikipedia.org/wiki/Effect_size#Difference_family:_Effect_sizes_based_on_differences_between_means
def _compute_effect_size(slice_metric: float, slice_std_dev: float,
                         base_metric: float, base_std_dev: float):
  """Computes effect size."""
  metric_diff = abs(slice_metric - base_metric)
  slice_var = slice_std_dev * slice_std_dev
  base_var = base_std_dev * base_std_dev
  return math.sqrt(2) * metric_diff / math.sqrt(slice_var + base_var)


def _get_metrics_as_dict(metrics):
  """Convert slice metrics to a Dict of types.ValueWithTDistribution."""
  result = {}
  for metric in metrics.metric_keys_and_values:
    t_distribution_value = metric.value.confidence_interval.t_distribution_value
    result[metric.key.name] = types.ValueWithTDistribution(
        sample_mean=t_distribution_value.sample_mean.value,
        sample_standard_deviation=t_distribution_value.sample_standard_deviation
        .value,
        sample_degrees_of_freedom=t_distribution_value.sample_degrees_of_freedom
        .value,
        unsampled_value=t_distribution_value.unsampled_value.value)
  return result


def find_top_slices(metrics: List[metrics_for_slice_pb2.MetricsForSlice],
                    metric_key: Text,
                    comparison_type: Text = 'HIGHER',
                    min_num_examples: int = 10,
                    num_top_slices: int = 10,
                    rank_by: Text = 'EFFECT_SIZE'):
  """Finds top-k slices.

  Args:
    metrics: List of slice metrics protos. We assume that the metrics have
    MetricValue.confidence_interval field populated. This will be populated when
      the metrics computed with confidence intervals enabled.
    metric_key: Name of the metric based on which significance testing is done.
    comparison_type: Type of comparison indicating if we are looking for slices
      whose metric is higher (`HIGHER`) or lower (`LOWER`) than the metric
      of the base slice (overall dataset).
    min_num_examples: Minimum number of examples that a slice should have.
    num_top_slices: Number of top slices to return.
    rank_by: Indicates how the slices should be ordered in the result.

  Returns:
    List of ordered slices.
  """
  assert comparison_type in ['HIGHER', 'LOWER']
  assert min_num_examples > 0
  assert 0 < num_top_slices
  assert rank_by in ['EFFECT_SIZE', 'PVALUE']

  metrics_dict = {
      slicer_lib.deserialize_slice_key(slice_metrics.slice_key): slice_metrics
      for slice_metrics in metrics
  }
  overall_slice_metrics = metrics_dict[()]
  del metrics_dict[()]

  overall_metrics_dict = _get_metrics_as_dict(overall_slice_metrics)
  to_be_sorted_slices = []
  for slice_key, slice_metrics in metrics_dict.items():
    slice_metrics_dict = _get_metrics_as_dict(slice_metrics)
    num_examples = slice_metrics_dict['example_count'].unsampled_value
    if num_examples < min_num_examples:
      continue
    # Prune non-interesting slices.
    if comparison_type == 'HIGHER':
      comparison_fn = operator.le
    else:
      comparison_fn = operator.ge
    if comparison_fn(slice_metrics_dict[metric_key].unsampled_value,
                     overall_metrics_dict[metric_key].unsampled_value):
      continue

    # Only consider statistically significant slices.
    is_significant, pvalue = _is_significant_slice(
        slice_metrics_dict[metric_key].unsampled_value,
        slice_metrics_dict[metric_key].sample_standard_deviation,
        slice_metrics_dict['example_count'].unsampled_value,
        overall_metrics_dict[metric_key].unsampled_value,
        overall_metrics_dict[metric_key].sample_standard_deviation,
        overall_metrics_dict['example_count'].unsampled_value, comparison_type)
    if not is_significant:
      continue
    # Format the slice info (feature names, values) in the proto into a
    # slice key.
    slice_key = slicer_lib.stringify_slice_key(slice_key)
    # Compute effect size for the slice.
    effect_size = _compute_effect_size(
        slice_metrics_dict[metric_key].unsampled_value,
        slice_metrics_dict[metric_key].sample_standard_deviation,
        overall_metrics_dict[metric_key].unsampled_value,
        overall_metrics_dict[metric_key].sample_standard_deviation)
    to_be_sorted_slices.append(
        SliceComparisonResult(slice_key, num_examples, pvalue, effect_size))
  # Rank the slices.
  ranking_fn, reverse = operator.attrgetter('effect_size'), True
  if rank_by == 'PVALUE':
    ranking_fn, reverse = operator.attrgetter('pvalue'), False
  result = sorted(
      to_be_sorted_slices, key=ranking_fn, reverse=reverse)[:num_top_slices]
  return result
