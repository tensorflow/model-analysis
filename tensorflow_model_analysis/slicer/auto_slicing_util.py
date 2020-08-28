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

import collections
import itertools
import math
import operator
from typing import Dict, List, NamedTuple, Optional, Text, Tuple
from absl import logging
import numpy as np
import pandas as pd
from scipy import stats
from tensorflow_model_analysis import types
from tensorflow_model_analysis.extractors import auto_slice_key_extractor
from tensorflow_model_analysis.proto import metrics_for_slice_pb2
from tensorflow_model_analysis.slicer import slicer_lib

from tensorflow_metadata.proto.v0 import statistics_pb2


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


# TODO(pachristopher): Replace the named tuple with a proto.
class SliceComparisonResult(
    NamedTuple('SliceComparisonResult',
               [('slice_key', slicer_lib.SliceKeyType), ('num_examples', int),
                ('slice_metric', float), ('base_metric', float),
                ('p_value', float), ('effect_size', float),
                ('raw_slice_metrics', metrics_for_slice_pb2.MetricsForSlice)])):
  """Represents the result of comparing a slice with the base dataset.

  It includes the slice key, number of examples in the slice, slice metric,
  base metric, the p_value computed when doing significance testing, the
  effect size, and the raw slice metrics.
  """

  def __new__(cls, slice_key: slicer_lib.SliceKeyType, num_examples: int,
              slice_metric: float, base_metric: float, p_value: float,
              effect_size: float,
              raw_slice_metrics: metrics_for_slice_pb2.MetricsForSlice):
    return super(SliceComparisonResult,
                 cls).__new__(cls, slice_key, num_examples, slice_metric,
                              base_metric, p_value, effect_size,
                              raw_slice_metrics)


# Perform one sided Welch's t-test.
# Reference: https://en.wikipedia.org/wiki/Welch%27s_t-test
# When comparison_type is HIGHER,
#   Null hypothesis: slice_metric <= overall_metric
#   Alternate hypothesis: slice_metric > overall_metric
# When comparison_type is LOWER,
#   Null hypothesis: slice_metric >= overall_metric
#   Alternate hypothesis: slice_metric < overall_metric
def _is_significant_slice(slice_metric: float, slice_std_dev: float,
                          slice_weight: float, base_metric: float,
                          base_std_dev: float, base_weight: float,
                          comparison_type: Text,
                          alpha: float) -> Tuple[bool, float]:
  """Perform statistical significance testing."""
  assert base_std_dev > 0, ('base_std_dev must be positive, but got '
                            '{}.'.format(base_std_dev))
  assert slice_std_dev > 0, ('slice_std_dev must be positive, but got '
                             '{}.'.format(slice_std_dev))
  assert base_weight > 1, ('base_weight must be greater than 1, but got '
                           '{}.'.format(base_weight))
  assert slice_weight > 1, ('slice_weight must be greater than 1, but got '
                            '{}.'.format(slice_weight))

  try:
    _, p_value_two_sided = stats.ttest_ind_from_stats(
        slice_metric,
        slice_std_dev,
        slice_weight,
        base_metric,
        base_std_dev,
        base_weight,
        equal_var=False)
  except ZeroDivisionError:
    raise ZeroDivisionError(
        'invalid ttest for params: slice_metric={}, '
        'slice_std_dev={}, slice_weight={}, '
        'base_metric={}, base_std_dev={}, base_weight={}, '.format(
            slice_metric, slice_std_dev, slice_weight, base_metric,
            base_std_dev, base_weight))

  metric_diff = slice_metric - base_metric
  one_sided_p_value = _two_sided_to_one_sided_pvalue(
      p_value_two_sided, metric_diff, comparison_type=comparison_type)
  return (one_sided_p_value < alpha, one_sided_p_value)


# We use effect size to capture the magnitude of the metric difference.
# We define effect size to be the difference in means of the slice metric
# and the base metric divided by the standard error.
def _compute_effect_size(slice_metric: float, slice_std_dev: float,
                         base_metric: float, base_std_dev: float) -> float:
  """Computes effect size."""
  metric_diff = abs(slice_metric - base_metric)
  slice_var = slice_std_dev * slice_std_dev
  base_var = base_std_dev * base_std_dev
  return math.sqrt(2) * metric_diff / math.sqrt(slice_var + base_var)


def _get_metrics_as_dict(
    metrics: metrics_for_slice_pb2.MetricsForSlice
) -> Dict[Text, types.ValueWithTDistribution]:
  """Convert slice metrics to a Dict of types.ValueWithTDistribution.

  For metrics missing the confidence interval message, an empty
  ValueWithTDistribution will be created and the double_value or
  bounded_value.value will be set as the unsampled value. Any metrics which are
  not represented as double_values or bounded_values will be ommitted from the
  result.

  Args:
    metrics: The MetricsForSlice proto to be converted.

  Returns:
    A dict from metric keys names to ValueWithTDistributions.
  """
  result = {}
  for metric in metrics.metric_keys_and_values:
    value_type = metric.value.WhichOneof('type')
    unsampled_value = float('nan')
    if value_type == 'bounded_value':
      unsampled_value = metric.value.bounded_value.value.value
    elif value_type == 'double_value':
      unsampled_value = metric.value.double_value.value
    t_distribution_value = metric.value.confidence_interval.t_distribution_value
    result[metric.key.name] = types.ValueWithTDistribution(
        sample_mean=t_distribution_value.sample_mean.value,
        sample_standard_deviation=t_distribution_value.sample_standard_deviation
        .value,
        sample_degrees_of_freedom=t_distribution_value.sample_degrees_of_freedom
        .value,
        unsampled_value=unsampled_value)
  return result


def _format_boundary(start: float, end: float) -> Text:
  """Formats bucket boundary as a string."""
  return '[' + str(start) + ', ' + str(end) + ')'


def get_raw_feature(
    column: Text, value: slicer_lib.FeatureValueType,
    boundaries: Dict[Text, List[float]]
) -> Tuple[Text, slicer_lib.FeatureValueType]:
  """Get raw feature name and value.

  Args:
    column: Raw or transformed column name.
    value: Raw or transformed column value.
    boundaries: Dictionary containing quantile boundaries of features keyed by
      column name.

  Returns:
    Tuple of raw column name and raw column value.
  """
  if column.startswith(auto_slice_key_extractor.TRANSFORMED_FEATURE_PREFIX):
    raw_feature = column[len(auto_slice_key_extractor.TRANSFORMED_FEATURE_PREFIX
                            ):]
    (start, end) = auto_slice_key_extractor.get_bucket_boundary(
        value, boundaries[raw_feature])
    return (raw_feature, _format_boundary(start, end))
  return (column, value)


def revert_slice_keys_for_transformed_features(
    slices: List[SliceComparisonResult],
    statistics: statistics_pb2.DatasetFeatureStatisticsList
) -> List[SliceComparisonResult]:
  """Revert the slice keys for the transformed features.

  Args:
    slices: List of slices.
    statistics: Data statistics used to configure AutoSliceKeyExtractor.

  Returns:
    List of slice metrics protos where transformed features are mapped back to
    raw features in the slice keys.
  """
  result = []
  boundaries = auto_slice_key_extractor.get_quantile_boundaries(statistics)
  for s in slices:
    transformed_slice_key = []
    for column, value in s.slice_key:
      raw_feature_name, raw_feature_value = get_raw_feature(
          column, value, boundaries)
      transformed_slice_key.append((raw_feature_name, raw_feature_value))
    result.append(s._replace(slice_key=tuple(transformed_slice_key)))
  return result


def _is_subset_slice(
    slice_key: slicer_lib.SliceKeyType,
    selected_slices: Dict[slicer_lib.SliceKeyType,
                          SliceComparisonResult]) -> bool:
  """Checks if a slice is a subset of the already selected slices."""
  for i in range(1, len(slice_key)):
    for cross in itertools.combinations(slice_key, i):
      # TODO(pachristopher): Should we consider pruning a subset slice only if
      # it has a smaller effect size than the parent slice?
      if cross in selected_slices:
        return True
  return False


def remove_subset_slices(
    slices: List[SliceComparisonResult]) -> List[SliceComparisonResult]:
  """Prune slices that are subset of other slices."""
  if len(slices) < 2:
    return slices
  # Group slices based on the number of predicates.
  slices_per_length = collections.defaultdict(list)
  for slice_comparison_result in slices:
    slices_per_length[len(
        slice_comparison_result.slice_key)].append(slice_comparison_result)

  selected_slices = {}
  for length in sorted(slices_per_length.keys()):
    for slice_comparison_result in slices_per_length[length]:
      # Check if this slice is a subset of any of the already selected slices.
      # TODO(pachristopher): Also keep track of the subset slices which are
      # pruned as it can help drill down a problematic slice. Another idea is to
      # capture the subset relationships between slices as a graph.
      if (length == 1 or not _is_subset_slice(slice_comparison_result.slice_key,
                                              selected_slices)):
        selected_slices[
            slice_comparison_result.slice_key] = slice_comparison_result
  return list(selected_slices.values())


def partition_slices(
    metrics: List[metrics_for_slice_pb2.MetricsForSlice],
    metric_key: Text,
    comparison_type: Text = 'HIGHER',
    alpha: float = 0.01,
    min_num_examples: int = 1
) -> Tuple[List[SliceComparisonResult], List[SliceComparisonResult]]:
  """Partition slices into significant and non-significant slices.

  Args:
    metrics: List of slice metrics protos. We assume that the metrics have
      MetricValue.confidence_interval field populated. This will be populated
      when the metrics computed with confidence intervals enabled.
    metric_key: Name of the metric based on which significance testing is done.
    comparison_type: Type of comparison indicating if we are looking for slices
      whose metric is higher (`HIGHER`) or lower (`LOWER`) than the metric of
      the base slice (overall dataset).
    alpha: Significance-level for statistical significance testing.
    min_num_examples: Minimum number of examples that a slice should have. If it
      is set to zero, we don't do any filtering.

  Returns:
    Tuple containing list of statistically significant and non-significant
    slices.
  """
  assert comparison_type in ['HIGHER', 'LOWER']
  if min_num_examples == 0:
    min_num_examples = 1

  metrics_dict = {
      slicer_lib.deserialize_slice_key(slice_metrics.slice_key): slice_metrics
      for slice_metrics in metrics
  }
  overall_slice_metrics = metrics_dict[()]
  del metrics_dict[()]

  overall_metrics_dict = _get_metrics_as_dict(overall_slice_metrics)
  significant_slices, non_significant_slices = [], []
  for slice_key, slice_metrics in metrics_dict.items():
    slice_metrics_dict = _get_metrics_as_dict(slice_metrics)
    num_examples = int(slice_metrics_dict['example_count'].unsampled_value)
    if num_examples < min_num_examples:
      continue
    # Prune non-interesting slices.
    if np.isnan(slice_metrics_dict[metric_key].unsampled_value):
      continue
    if slice_metrics_dict[metric_key].sample_standard_deviation == 0:
      logging.warning('Ignoring slice: %s with standard deviation: %s ',
                      slice_key,
                      slice_metrics_dict[metric_key].sample_standard_deviation)
      continue
    # TODO(pachristopher): Should we use weighted example count?
    if slice_metrics_dict['example_count'].unsampled_value <= 1:
      logging.warning('Ignoring slice: %s with example count: %s ', slice_key,
                      slice_metrics_dict['example_count'].unsampled_value)
      continue
    # Only consider statistically significant slices.
    is_significant, p_value = _is_significant_slice(
        slice_metrics_dict[metric_key].unsampled_value,
        slice_metrics_dict[metric_key].sample_standard_deviation,
        slice_metrics_dict['example_count'].unsampled_value,
        overall_metrics_dict[metric_key].unsampled_value,
        overall_metrics_dict[metric_key].sample_standard_deviation,
        overall_metrics_dict['example_count'].unsampled_value, comparison_type,
        alpha)
    # Compute effect size for the slice.
    effect_size = _compute_effect_size(
        slice_metrics_dict[metric_key].unsampled_value,
        slice_metrics_dict[metric_key].sample_standard_deviation,
        overall_metrics_dict[metric_key].unsampled_value,
        overall_metrics_dict[metric_key].sample_standard_deviation)
    slice_info = SliceComparisonResult(
        slice_key, num_examples, slice_metrics_dict[metric_key].unsampled_value,
        overall_metrics_dict[metric_key].unsampled_value, p_value, effect_size,
        slice_metrics)
    if not is_significant:
      non_significant_slices.append(slice_info)
      continue
    significant_slices.append(slice_info)
  return significant_slices, non_significant_slices


def find_top_slices(
    slices: List[SliceComparisonResult],
    min_num_examples: int = 100,
    num_top_slices: int = 10,
    rank_by: Text = 'EFFECT_SIZE',
    prune_subset_slices: bool = True) -> List[SliceComparisonResult]:
  """Finds top-k slices.

  Args:
    slices: List of slices.
    min_num_examples: Minimum number of examples that a slice should have.
    num_top_slices: Number of top slices to return.
    rank_by: Indicates how the slices should be ordered in the result.
    prune_subset_slices: Boolean indicating if the slices which are subsets of
      other slices should be pruned from the result.

  Returns:
    List of ordered slices.
  """
  assert 0 < num_top_slices
  assert rank_by in ['EFFECT_SIZE', 'PVALUE']

  # Prune smaller slices.
  result = list(filter(lambda s: s.num_examples >= min_num_examples, slices))

  # Prune subset slices if enabled.
  if prune_subset_slices:
    result = remove_subset_slices(result)

  # Rank the slices.
  ranking_fn, reverse = operator.attrgetter('effect_size'), True
  if rank_by == 'PVALUE':
    ranking_fn, reverse = operator.attrgetter('p_value'), False
  result = sorted(result, key=ranking_fn, reverse=reverse)[:num_top_slices]
  return result


def get_slices_as_dataframe(
    slices: List[SliceComparisonResult],
    additional_metric_names: Optional[List[Text]] = None) -> pd.DataFrame:
  """Returns top slices as a dataframe.

  Args:
    slices: List of ordered slices.
    additional_metric_names: An optional list of additional metric names to
      display

  Returns:
    Dataframe containing information about the slices.
  """
  rows = []
  for slice_info in slices:
    slice_metrics = _get_metrics_as_dict(slice_info.raw_slice_metrics)
    row = {
        'Slice': slicer_lib.stringify_slice_key(slice_info.slice_key),
        'Size': slice_info.num_examples,
        'Slice metric': slice_info.slice_metric,
        'Base metric': slice_info.base_metric,
        'P-Value': slice_info.p_value,
        'Effect size': slice_info.effect_size
    }
    if additional_metric_names:
      for metric_key in additional_metric_names:
        row[metric_key] = slice_metrics[metric_key].unsampled_value
    rows.append(row)

  ordered_columns = [
      'Slice', 'Size', 'Slice metric', 'Base metric', 'P-Value', 'Effect size'
  ]
  if additional_metric_names:
    ordered_columns.extend(additional_metric_names)
  dataframe = pd.DataFrame(rows, columns=ordered_columns)
  dataframe.set_index('Slice', inplace=True)
  return dataframe
