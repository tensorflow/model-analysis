# Lint as: python3
# Copyright 2019 Google LLC
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
"""Calibration histogram."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import bisect
import heapq
import itertools
import operator

from typing import Dict, List, Optional, NamedTuple, Text

import apache_beam as beam
from tensorflow_model_analysis import config
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util

CALIBRATION_HISTOGRAM_NAME = '_calibration_histogram'

DEFAULT_NUM_BUCKETS = 10000

Bucket = NamedTuple('Bucket', [('bucket_id', int), ('weighted_labels', float),
                               ('weighted_predictions', float),
                               ('weighted_examples', float)])

Histogram = List[Bucket]


def calibration_histogram(
    num_buckets: Optional[int] = None,
    left: Optional[float] = None,
    right: Optional[float] = None,
    name: Text = None,
    eval_config: Optional[config.EvalConfig] = None,
    model_name: Text = '',
    output_name: Text = '',
    sub_key: Optional[metric_types.SubKey] = None,
    aggregation_type: Optional[metric_types.AggregationType] = None,
    class_weights: Optional[Dict[int, float]] = None
) -> metric_types.MetricComputations:
  """Returns metric computations for calibration histogram.

  Args:
    num_buckets: Number of buckets to use. Note that the actual number of
      buckets will be num_buckets + 2 to account for the edge cases.
    left: Start of predictions interval.
    right: End of predictions interval.
    name: Metric name.
    eval_config: Eval config.
    model_name: Optional model name (if multi-model evaluation).
    output_name: Optional output name (if multi-output model type).
    sub_key: Optional sub key.
    aggregation_type: Optional aggregation type.
    class_weights: Optional class weights to apply to multi-class / multi-label
      labels and predictions prior to flattening (when micro averaging is used).

  Returns:
    MetricComputations for computing the histogram(s).
  """
  if num_buckets is None:
    num_buckets = DEFAULT_NUM_BUCKETS
  if left is None:
    left = 0.0
  if right is None:
    right = 1.0
  if name is None:
    name = '{}_{}'.format(CALIBRATION_HISTOGRAM_NAME, num_buckets)
  key = metric_types.PlotKey(
      name=name,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key)
  return [
      metric_types.MetricComputation(
          keys=[key],
          preprocessor=None,
          combiner=_CalibrationHistogramCombiner(
              key=key,
              eval_config=eval_config,
              aggregation_type=aggregation_type,
              class_weights=class_weights,
              num_buckets=num_buckets,
              left=left,
              right=right))
  ]


class _CalibrationHistogramCombiner(beam.CombineFn):
  """Creates histogram from labels, predictions, and example weights."""

  def __init__(self, key: metric_types.PlotKey,
               eval_config: Optional[config.EvalConfig],
               aggregation_type: Optional[metric_types.AggregationType],
               class_weights: Optional[Dict[int, float]], num_buckets: int,
               left: float, right: float):
    self._key = key
    self._eval_config = eval_config
    self._aggregation_type = aggregation_type
    self._class_weights = class_weights
    self._num_buckets = num_buckets
    self._left = left
    self._range = right - left
    self._is_unit_interval = (left == 0.0 and right == 1.0)

  def _bucket_index(self, prediction: float) -> int:
    """Returns bucket index given prediction value. Values are truncated."""
    bucket_index = (
        (prediction - self._left) / self._range * self._num_buckets) + 1
    if bucket_index < 0:
      return 0
    if bucket_index >= self._num_buckets + 1:
      return self._num_buckets + 1
    return int(bucket_index)

  def create_accumulator(self) -> Histogram:
    # The number of accumulator (histogram) buckets is variable and depends on
    # the number of distinct intervals that are matched during calls to
    # add_inputs. This allows the historam size to start small and gradually
    # grow size during calls to merge until reaching the final histogram.
    return []

  def add_input(self, accumulator: Histogram,
                element: metric_types.StandardMetricInputs) -> Histogram:
    # Note that in the case of top_k, if the aggregation type is not set then
    # the non-top_k predictions will be set to float('-inf'), but the labels
    # will remain unchanged. If aggregation type is set then both the
    # predictions and labels will be truncated to only the top_k values.
    for label, prediction, example_weight in (
        metric_util.to_label_prediction_example_weight(
            element,
            eval_config=self._eval_config,
            model_name=self._key.model_name,
            output_name=self._key.output_name,
            sub_key=self._key.sub_key,
            fractional_labels=self._is_unit_interval,
            flatten=True,
            aggregation_type=self._aggregation_type,
            class_weights=self._class_weights)):
      example_weight = float(example_weight)
      label = float(label)
      prediction = float(prediction)
      weighted_label = label * example_weight
      weighted_prediction = prediction * example_weight
      bucket_index = self._bucket_index(prediction)
      # Check if bucket exists, all bucket values are > 0, so -1 are always less
      insert_index = bisect.bisect_left(accumulator,
                                        Bucket(bucket_index, -1, -1, -1))
      if (insert_index == len(accumulator) or
          accumulator[insert_index].bucket_id != bucket_index):
        accumulator.insert(
            insert_index,
            Bucket(bucket_index, weighted_label, weighted_prediction,
                   example_weight))
      else:
        existing_bucket = accumulator[insert_index]
        accumulator[insert_index] = Bucket(
            bucket_index, existing_bucket.weighted_labels + weighted_label,
            existing_bucket.weighted_predictions + weighted_prediction,
            existing_bucket.weighted_examples + example_weight)
    return accumulator

  def merge_accumulators(self, accumulators: List[Histogram]) -> Histogram:
    result = []
    for bucket_id, buckets in itertools.groupby(
        heapq.merge(*accumulators), key=operator.attrgetter('bucket_id')):
      total_weighted_labels = 0.0
      total_weighted_predictions = 0.0
      total_weighted_examples = 0.0
      for bucket in buckets:
        total_weighted_labels += bucket.weighted_labels
        total_weighted_predictions += bucket.weighted_predictions
        total_weighted_examples += bucket.weighted_examples
      result.append(
          Bucket(bucket_id, total_weighted_labels, total_weighted_predictions,
                 total_weighted_examples))
    return result

  def extract_output(
      self, accumulator: Histogram) -> Dict[metric_types.PlotKey, Histogram]:
    return {self._key: accumulator}


def rebin(thresholds: List[float],
          histogram: Histogram,
          num_buckets: int = DEFAULT_NUM_BUCKETS,
          left: float = 0.0,
          right: float = 1.0) -> Histogram:
  """Applies new thresholds to an existing calibration histogram.

  Args:
    thresholds: New thresholds to apply to the histogram. Must be in sorted
      order, but need not be evenly spaced.
    histogram: Existing calibration histogram.
    num_buckets: Number of buckets in existing histogram.
    left: Left boundary for existing histogram.
    right: Right boundary for existing histogram.

  Returns:
    A histogram of len(thresholds) where the buckets with IDs (0, 1, 2, ...)
    correspond to the intervals:
      [thresholds[0], thresholds[1]), ... [thresholds[i], thresholds[i+1])
    Any values in buckets -inf or +inf will be added to the start and end
    thresholds respectively. Unlike the input histogram empty buckets will be
    returned.
  """
  buckets = []
  offset = 0
  total_weighted_labels = 0.0
  total_weighted_predictions = 0.0
  total_weighted_examples = 0.0
  for bucket in histogram:
    if bucket.bucket_id == 0:
      pred = float('-inf')
    elif bucket.bucket_id >= num_buckets + 1:
      pred = float('inf')
    else:
      pred = (bucket.bucket_id - 1) / num_buckets * (right - left) + left
    if offset + 1 < len(thresholds) and pred >= thresholds[offset + 1]:
      buckets.append(
          Bucket(offset, total_weighted_labels, total_weighted_predictions,
                 total_weighted_examples))
      offset += 1
      total_weighted_labels = 0.0
      total_weighted_predictions = 0.0
      total_weighted_examples = 0.0
      while offset + 1 < len(thresholds) and pred >= thresholds[offset + 1]:
        buckets.append(Bucket(offset, 0.0, 0.0, 0.0))
        offset += 1
    total_weighted_labels += bucket.weighted_labels
    total_weighted_predictions += bucket.weighted_predictions
    total_weighted_examples += bucket.weighted_examples
  buckets.append(
      Bucket(offset, total_weighted_labels, total_weighted_predictions,
             total_weighted_examples))
  offset += 1
  while offset < len(thresholds):
    buckets.append(Bucket(offset, 0.0, 0.0, 0.0))
    offset += 1
  return buckets
