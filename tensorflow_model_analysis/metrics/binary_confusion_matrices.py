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

from tensorflow_model_analysis import config
from tensorflow_model_analysis.metrics import calibration_histogram
from tensorflow_model_analysis.metrics import metric_types
from typing import Any, Dict, List, NamedTuple, Optional, Text

DEFAULT_NUM_THRESHOLDS = calibration_histogram.DEFAULT_NUM_BUCKETS

BINARY_CONFUSION_MATRICES_NAME = '_binary_confusion_matrices'

Matrices = NamedTuple('Matrices', [('thresholds', List[float]),
                                   ('tp', List[int]), ('tn', List[int]),
                                   ('fp', List[int]), ('fn', List[int])])

_EPSILON = 1e-7


def binary_confusion_matrices(
    num_thresholds: Optional[int] = None,
    thresholds: Optional[List[float]] = None,
    name: Text = BINARY_CONFUSION_MATRICES_NAME,
    eval_config: Optional[config.EvalConfig] = None,
    model_name: Text = '',
    output_name: Text = '',
    sub_key: Optional[metric_types.SubKey] = None
) -> metric_types.MetricComputations:
  """Returns metric computations for computing binary confusion matrices.

  Args:
    num_thresholds: Number of thresholds to use. Thresholds will be calculated
      using linear interpolation between 0.0 and 1.0 with equidistant values and
      bondardaries at -epsilon and 1.0+epsilon. Values must be > 0. Only one of
      num_thresholds or thresholds should be used.
    thresholds: A specific set of thresholds to use. The caller is responsible
      for marking the bondaires with +/-epsilon if desired. Only one of
      num_thresholds or thresholds should be used.
    name: Metric name.
    eval_config: Eval config.
    model_name: Optional model name (if multi-model evaluation).
    output_name: Optional output name (if multi-output model type).
    sub_key: Optional sub key.

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
  num_buckets = 1 if len(thresholds) == 1 and thresholds[0] <= 0 else None
  histogram_computations = calibration_histogram.calibration_histogram(
      eval_config=eval_config,
      num_buckets=num_buckets,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key)
  histogram_key = histogram_computations[-1].keys[-1]

  def result(
      metrics: Dict[metric_types.MetricKey, Any]
  ) -> Dict[metric_types.MetricKey, Matrices]:
    """Returns binary confusion matrices."""
    # Calibration histogram uses intervals of the form [start, end) where the
    # prediction >= start. The confusion matrices want intervals of the form
    # (start, end] where the prediction > start. Add a small epsilon so that >=
    # checks don't match. This correction shouldn't be needed in practice but
    # allows for correctness in small tests.
    if len(thresholds) == 1:
      # When there is only one threshold, we need to make adjustments so that
      # we have proper boundaries around the threshold for <, >= comparions.
      if thresholds[0] < 0:
        # This case is used when all prediction values are considered matches
        # (e.g. when calculating top_k for precision/recall).
        rebin_thresholds = [thresholds[0], thresholds[0] + _EPSILON]
      else:
        # This case is used for a single threshold within [0, 1] (e.g. 0.5).
        rebin_thresholds = [-_EPSILON, thresholds[0] + _EPSILON, 1.0 + _EPSILON]
    else:
      rebin_thresholds = ([thresholds[0]] +
                          [t + _EPSILON for t in thresholds[1:]])
    histogram = calibration_histogram.rebin(rebin_thresholds,
                                            metrics[histogram_key])
    matrices = _to_binary_confusion_matrices(thresholds, histogram)
    if len(thresholds) == 1:
      # Reset back to 1 bucket
      matrices = Matrices(
          thresholds,
          tp=[matrices.tp[1]],
          fp=[matrices.fp[1]],
          tn=[matrices.tn[1]],
          fn=[matrices.fn[1]])
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
    else:
      tn[start] += start_neg
      fn[start] += start_pos
  return Matrices(thresholds, tp, tn, fp, fn)
