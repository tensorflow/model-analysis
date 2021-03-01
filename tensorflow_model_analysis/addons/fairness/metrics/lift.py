# Lint as: python3
# Copyright 2021 Google LLC
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
"""Lift Metrics."""

from typing import Any, Dict, Optional, Text

from tensorflow_model_analysis import config
from tensorflow_model_analysis.metrics import calibration_histogram
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util

LIFT_METRICS_NAME = 'lift'

# Recommendation is to use prime number for number of buckets.
# For binary ground truth labels, set this to two.
DEFAULT_NUM_BUCKETS = 23


class Lift(metric_types.Metric):
  """Lift metrics.

  For a given slice, the goal of the Lift metric is to assess the difference
  between the average of predictions for in-slice items differs from the average
  of background items, conditioned on ground truth. The Lift metric can be used
  to see whether the predictions on a given slice of items are, on average,
  higher/lower than the background.

  Use config.CrossSlicingSpec to define background (baseline) and in-slice items
  (comparison).

  Raises an exception when config.CrossSlicingSpec is not provided.
  """

  def __init__(self,
               num_buckets: Optional[int] = None,
               left: Optional[float] = None,
               right: Optional[float] = None,
               name: Optional[Text] = None,
               ignore_out_of_bound_examples: bool = False):
    """Initializes lift metrics.

    Args:
      num_buckets: Number of buckets to use. Note that the actual number of
        buckets will be num_buckets + 2 to account for the edge cases.
      left: Start of labels interval.
      right: End of labels interval.
      name: Metric name.
      ignore_out_of_bound_examples: Whether to ignore examples with label values
        falling outside of provide label interval i.e. [left, right).
    """
    super(Lift, self).__init__(
        metric_util.merge_per_key_computations(_lift_metrics),
        num_buckets=num_buckets,
        left=left,
        right=right,
        name=name,
        ignore_out_of_bound_examples=ignore_out_of_bound_examples)


def _lift_metrics(
    num_buckets: Optional[int] = None,
    left: Optional[float] = None,
    right: Optional[float] = None,
    name: Optional[Text] = None,
    eval_config: Optional[config.EvalConfig] = None,
    model_name: Text = '',
    output_name: Text = '',
    aggregation_type: Optional[metric_types.AggregationType] = None,
    sub_key: Optional[metric_types.SubKey] = None,
    class_weights: Optional[Dict[int, float]] = None,
    ignore_out_of_bound_examples: bool = False,
) -> metric_types.MetricComputations:
  """Returns computations for lift metrics."""
  if eval_config is None or not eval_config.cross_slicing_specs:
    raise ValueError(
        'config.CrossSlicingSpec with a baseline and at least one comparison '
        'slicing spec must be provided for Lift metrics')

  if num_buckets is None:
    num_buckets = DEFAULT_NUM_BUCKETS

  if name is None:
    name = f'{LIFT_METRICS_NAME}@{num_buckets}'

  key = metric_types.MetricKey(
      name=name,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key)

  computations = calibration_histogram.calibration_histogram(
      eval_config=eval_config,
      num_buckets=num_buckets,
      left=left,
      right=right,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key,
      aggregation_type=aggregation_type,
      class_weights=class_weights,
      prediction_based_bucketing=False,
      fractional_labels=False)
  metric_key = computations[-1].keys[-1]

  def cross_slice_comparison(
      baseline_metrics: Dict[metric_types.MetricKey, Any],
      comparison_metrics: Dict[metric_types.MetricKey, Any],
  ) -> Dict[metric_types.MetricKey, Any]:
    """Returns lift metrics values."""
    baseline_histogram = baseline_metrics[metric_key]
    comparison_histogram = comparison_metrics[metric_key]

    baseline_bucket = {}
    comparison_bucket = {}
    bucket_ids = set()

    for bucket in baseline_histogram:
      baseline_bucket[bucket.bucket_id] = bucket
      bucket_ids.add(bucket.bucket_id)

    for bucket in comparison_histogram:
      comparison_bucket[bucket.bucket_id] = bucket
      bucket_ids.add(bucket.bucket_id)

    baseline_pred_values = 0.0
    comparison_pred_values = 0.0
    comparison_num_examples = 0.0

    for bucket_id in bucket_ids:
      if ignore_out_of_bound_examples:
        # Ignore buckets having examples with out of bound label values.
        if bucket_id <= 0 or bucket_id > num_buckets:
          continue
      num_examples = 0.0
      if bucket_id in comparison_bucket:
        num_examples = comparison_bucket[bucket_id].weighted_examples
        comparison_pred_values += comparison_bucket[
            bucket_id].weighted_predictions
        comparison_num_examples += num_examples

      if bucket_id in baseline_bucket:
        # To compute background/baseline re-weighted average prediction values.
        # Background re-weighting is done by dividing the in-slice ground truth
        # density by the background density so that the marginal ground truth
        # distributions of in-slice items and background items appear similar.
        weight = num_examples / baseline_bucket[bucket_id].weighted_examples
        baseline_pred_values += weight * baseline_bucket[
            bucket_id].weighted_predictions

    lift_value = (comparison_pred_values -
                  baseline_pred_values) / comparison_num_examples
    return {key: lift_value}

  cross_slice_computation = metric_types.CrossSliceMetricComputation(
      keys=[key], cross_slice_comparison=cross_slice_comparison)

  computations.append(cross_slice_computation)
  return computations


metric_types.register_metric(Lift)
