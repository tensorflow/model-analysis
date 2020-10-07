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
"""Binary confusion matrices."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

from typing import Any, Dict, List, NamedTuple, Optional, Text

from tensorflow_model_analysis import config
from tensorflow_model_analysis.metrics import calibration_histogram
from tensorflow_model_analysis.metrics import metric_types

DEFAULT_NUM_THRESHOLDS = calibration_histogram.DEFAULT_NUM_BUCKETS

BINARY_CONFUSION_MATRICES_NAME = '_binary_confusion_matrices'

Matrices = NamedTuple('Matrices', [('thresholds', List[float]),
                                   ('tp', List[float]), ('tn', List[float]),
                                   ('fp', List[float]), ('fn', List[float])])

_EPSILON = 1e-7


def binary_confusion_matrices(
    num_thresholds: Optional[int] = None,
    thresholds: Optional[List[float]] = None,
    name: Text = BINARY_CONFUSION_MATRICES_NAME,
    eval_config: Optional[config.EvalConfig] = None,
    model_name: Text = '',
    output_name: Text = '',
    sub_key: Optional[metric_types.SubKey] = None,
    aggregation_type: Optional[metric_types.AggregationType] = None,
    class_weights: Optional[Dict[int, float]] = None
) -> metric_types.MetricComputations:
  """Returns metric computations for computing binary confusion matrices.

  Args:
    num_thresholds: Number of thresholds to use. Thresholds will be calculated
      using linear interpolation between 0.0 and 1.0 with equidistant values and
      bondardaries at -epsilon and 1.0+epsilon. Values must be > 0. Only one of
      num_thresholds or thresholds should be used. If used, num_thresholds must
      be > 1.
    thresholds: A specific set of thresholds to use. The caller is responsible
      for marking the boundaries with +/-epsilon if desired. Only one of
      num_thresholds or thresholds should be used. For metrics computed at top k
      this may be a single negative threshold value (i.e. -inf).
    name: Metric name.
    eval_config: Eval config.
    model_name: Optional model name (if multi-model evaluation).
    output_name: Optional output name (if multi-output model type).
    sub_key: Optional sub key.
    aggregation_type: Optional aggregation type.
    class_weights: Optional class weights to apply to multi-class / multi-label
      labels and predictions prior to flattening (when micro averaging is used).

  Raises:
    ValueError: If both num_thresholds and thresholds are set at the same time.
  """
  key = metric_types.MetricKey(
      name=name,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key)

  if num_thresholds is not None and thresholds is not None:
    raise ValueError(
        'only one of thresholds or num_thresholds can be set at a time')
  if num_thresholds is None and thresholds is None:
    num_thresholds = DEFAULT_NUM_THRESHOLDS
  if num_thresholds is not None:
    if num_thresholds <= 1:
      raise ValueError('num_thresholds must be > 1')
    # The interpolation strategy used here matches that used by keras for AUC.
    thresholds = [
        (i + 1) * 1.0 / (num_thresholds - 1) for i in range(num_thresholds - 2)
    ]
    thresholds = [-_EPSILON] + thresholds + [1.0 + _EPSILON]

  # Use calibration histogram to calculate matrices. For efficiency (unless all
  # predictions are matched - i.e. thresholds <= 0) we will assume that other
  # metrics will make use of the calibration histogram and re-use the default
  # histogram for the given model_name/output_name/sub_key. This is also
  # required to get accurate counts at the threshold boundaries. If this becomes
  # an issue, then calibration histogram can be updated to support non-linear
  # boundaries.
  histogram_computations = calibration_histogram.calibration_histogram(
      eval_config=eval_config,
      num_buckets=(
          # For precision/recall_at_k were a single large negative threshold is
          # used, we only need one bucket. Note that the histogram will actually
          # have 2 buckets: one that we set (which handles predictions > -1.0)
          # and a default catch-all bucket (i.e. bucket 0) that the histogram
          # creates for large negative predictions (i.e. predictions <= -1.0).
          1 if len(thresholds) == 1 and thresholds[0] <= 0 else None),
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key,
      aggregation_type=aggregation_type,
      class_weights=class_weights)
  histogram_key = histogram_computations[-1].keys[-1]

  def result(
      metrics: Dict[metric_types.MetricKey, Any]
  ) -> Dict[metric_types.MetricKey, Matrices]:
    """Returns binary confusion matrices."""
    if len(thresholds) == 1 and thresholds[0] < 0:
      # This case is used when all positive prediction values are considered
      # matches (e.g. when calculating top_k for precision/recall where the
      # non-top_k values are expected to have been set to float('-inf')).
      histogram = metrics[histogram_key]
    else:
      # Calibration histogram uses intervals of the form [start, end) where the
      # prediction >= start. The confusion matrices want intervals of the form
      # (start, end] where the prediction > start. Add a small epsilon so that
      # >= checks don't match. This correction shouldn't be needed in practice
      # but allows for correctness in small tests.
      rebin_thresholds = [t + _EPSILON if t != 0 else t for t in thresholds]
      if thresholds[0] >= 0:
        # Add -epsilon bucket to account for differences in histogram vs
        # confusion matrix intervals mentioned above. If the epsilon bucket is
        # missing the false negatives and false positives will be 0 for the
        # first threshold.
        rebin_thresholds = [-_EPSILON] + rebin_thresholds
      if thresholds[-1] < 1.0:
        # If the last threshold < 1.0, then add a fence post at 1.0 + epsilon
        # othewise true negatives and true positives will be overcounted.
        rebin_thresholds = rebin_thresholds + [1.0 + _EPSILON]
      histogram = calibration_histogram.rebin(rebin_thresholds,
                                              metrics[histogram_key])
    matrices = _to_binary_confusion_matrices(thresholds, histogram)
    return {key: matrices}

  derived_computation = metric_types.DerivedMetricComputation(
      keys=[key], result=result)
  computations = histogram_computations
  computations.append(derived_computation)
  return computations


def _to_binary_confusion_matrices(
    thresholds: List[float],
    histogram: calibration_histogram.Histogram) -> Matrices:
  """Converts histogram to binary confusion matrices."""
  # tp(i) - sum of positive labels >= bucket i
  # fp(i) - sum of negative labels >= bucket i
  # fn(i) - sum of positive labels < bucket i
  # tn(i) - sum of negative labels < bucket i
  n = len(histogram)
  tp = [0.0] * n
  fp = [0.0] * n
  tn = [0.0] * n
  fn = [0.0] * n
  for i in range(n):
    start = i
    end = n - i - 1
    start_pos = histogram[start].weighted_labels
    start_neg = (
        histogram[start].weighted_examples - histogram[start].weighted_labels)
    end_pos = histogram[end].weighted_labels
    end_neg = (
        histogram[end].weighted_examples - histogram[end].weighted_labels)
    tp[end] = tp[end + 1] + end_pos if end < n - 1 else end_pos
    fp[end] = fp[end + 1] + end_neg if end < n - 1 else end_neg
    if start + 1 < n:
      tn[start + 1] = tn[start] + start_neg
      fn[start + 1] = fn[start] + start_pos
  # Check if need to remove -epsilon bucket (or reset back to 1 bucket).
  threshold_offset = 0
  if (thresholds[0] >= 0 or len(thresholds) == 1) and len(histogram) > 1:
    threshold_offset = 1
  tp = tp[threshold_offset:threshold_offset + len(thresholds)]
  fp = fp[threshold_offset:threshold_offset + len(thresholds)]
  tn = tn[threshold_offset:threshold_offset + len(thresholds)]
  fn = fn[threshold_offset:threshold_offset + len(thresholds)]
  # We sum all values >= bucket i, but TP/FP values greater that 1.0 + EPSILON
  # should be 0.0. The FN/TN above 1.0 + _EPSILON should also be adjusted to
  # match the TP/FP values at the start.
  for i, t in enumerate(thresholds):
    if t >= 1.0 + _EPSILON:
      tp[i] = 0.0
      fp[i] = 0.0
      fn[i] = tp[0]
      tn[i] = fp[0]
  return Matrices(thresholds, tp, tn, fp, fn)
