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
"""Calibration plot."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

from tensorflow_model_analysis import config
from tensorflow_model_analysis.metrics import calibration_histogram
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.proto import metrics_for_slice_pb2
from typing import Any, Dict, List, Optional, Text

DEFAULT_NUM_BUCKETS = 1000

CALIBRATION_PLOT_NAME = 'calibration_plot'


class CalibrationPlot(metric_types.Metric):
  """Calibration plot."""

  def __init__(self,
               num_buckets: int = DEFAULT_NUM_BUCKETS,
               left: float = 0.0,
               right: float = 1.0,
               name: Text = CALIBRATION_PLOT_NAME):
    """Initializes calibration plot.

    Args:
      num_buckets: Number of buckets to use when creating the plot. Defaults to
        1000.
      left: Left boundary of plot. Defaults to 0.0.
      right: Right boundary of plot. Defaults to 1.0.
      name: Plot name.
    """
    super(CalibrationPlot, self).__init__(
        metric_util.merge_per_key_computations(_calibration_plot),
        num_buckets=num_buckets,
        left=left,
        right=right,
        name=name)


metric_types.register_metric(CalibrationPlot)


def _calibration_plot(
    num_buckets: int = DEFAULT_NUM_BUCKETS,
    left: float = 0.0,
    right: float = 1.0,
    name: Text = CALIBRATION_PLOT_NAME,
    eval_config: Optional[config.EvalConfig] = None,
    model_name: Text = '',
    output_name: Text = '',
    sub_key: Optional[metric_types.SubKey] = None
) -> metric_types.MetricComputations:
  """Returns metric computations for calibration plot."""
  key = metric_types.PlotKey(
      name=name,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key)

  # Make sure calibration histogram is calculated. Note we are using the default
  # number of buckets assigned to the histogram instead of the value used for
  # the plots just in case the computation is shared with other metrics and
  # plots that need higher preicion. It will be downsampled later.
  computations = calibration_histogram.calibration_histogram(
      eval_config=eval_config,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key,
      left=left,
      right=right)
  histogram_key = computations[-1].keys[-1]

  def result(
      metrics: Dict[metric_types.MetricKey, Any]
  ) -> Dict[metric_types.MetricKey, Any]:
    thresholds = [
        left + i * (right - left) / num_buckets for i in range(num_buckets + 1)
    ]
    thresholds = [float('-inf')] + thresholds
    histogram = calibration_histogram.rebin(
        thresholds, metrics[histogram_key], left=left, right=right)
    return {key: _to_proto(thresholds, histogram)}

  derived_computation = metric_types.DerivedMetricComputation(
      keys=[key], result=result)
  computations.append(derived_computation)
  return computations


def _to_proto(
    thresholds: List[float], histogram: calibration_histogram.Histogram
) -> metrics_for_slice_pb2.CalibrationHistogramBuckets:
  """Converts histogram into CalibrationHistogramBuckets proto.

  Args:
    thresholds: Thresholds associated with histogram buckets.
    histogram: Calibration histogram.

  Returns:
    A histogram in CalibrationHistogramBuckets proto format.
  """
  pb = metrics_for_slice_pb2.CalibrationHistogramBuckets()
  lower_threshold = float('-inf')
  for i, bucket in enumerate(histogram):
    if i >= len(thresholds) - 1:
      upper_threshold = float('inf')
    else:
      upper_threshold = thresholds[i + 1]
    pb.buckets.add(
        lower_threshold_inclusive=lower_threshold,
        upper_threshold_exclusive=upper_threshold,
        total_weighted_label={'value': bucket.weighted_labels},
        total_weighted_refined_prediction={
            'value': bucket.weighted_predictions
        },
        num_weighted_examples={'value': bucket.weighted_examples})
    lower_threshold = upper_threshold
  return pb
