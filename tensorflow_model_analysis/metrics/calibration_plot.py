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
"""Calibration plot."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

from typing import Any, Dict, List, Optional, Text, Tuple, Union

from tensorflow_model_analysis import config
from tensorflow_model_analysis import model_util
from tensorflow_model_analysis.metrics import calibration_histogram
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.proto import metrics_for_slice_pb2

from tensorflow_metadata.proto.v0 import schema_pb2

DEFAULT_NUM_BUCKETS = 1000

CALIBRATION_PLOT_NAME = 'calibration_plot'


class CalibrationPlot(metric_types.Metric):
  """Calibration plot."""

  def __init__(self,
               num_buckets: int = DEFAULT_NUM_BUCKETS,
               left: float = None,
               right: float = None,
               name: Text = CALIBRATION_PLOT_NAME):
    """Initializes calibration plot.

    Args:
      num_buckets: Number of buckets to use when creating the plot. Defaults to
        1000.
      left: Left boundary of plot. Defaults to 0.0 when a schema is not
        provided.
      right: Right boundary of plot. Defaults to 1.0 when a schema is not
        provided.
      name: Plot name.
    """
    super(CalibrationPlot, self).__init__(
        metric_util.merge_per_key_computations(_calibration_plot),
        num_buckets=num_buckets,
        left=left,
        right=right,
        name=name)


metric_types.register_metric(CalibrationPlot)


def _find_label_domain(
    eval_config: config.EvalConfig, schema: schema_pb2.Schema, model_name: Text,
    output_name: Text
) -> Tuple[Optional[Union[int, float]], Optional[Union[int, float]]]:
  """Find the min and max value for the label_key for this model / output."""
  model_spec = model_util.get_model_spec(eval_config, model_name)
  if not model_spec:
    return None, None
  label_key = model_util.get_label_key(model_spec, output_name)
  if not label_key:
    return None, None
  label_schema = None
  for feature_schema in schema.feature:
    if feature_schema.name == label_key:
      label_schema = feature_schema
      break
  if label_schema is None:
    return None, None

  # Find the domain
  if label_schema.HasField('int_domain'):
    label_domain = label_schema.int_domain
  elif label_schema.HasField('float_domain'):
    label_domain = label_schema.float_domain
  else:
    return None, None

  left, right = None, None
  if label_domain.HasField('min'):
    left = float(label_domain.min)
  if label_domain.HasField('max'):
    right = float(label_domain.max)
  return left, right


def _calibration_plot(
    num_buckets: int = DEFAULT_NUM_BUCKETS,
    left: Optional[float] = None,
    right: Optional[float] = None,
    name: Text = CALIBRATION_PLOT_NAME,
    eval_config: Optional[config.EvalConfig] = None,
    schema: Optional[schema_pb2.Schema] = None,
    model_name: Text = '',
    output_name: Text = '',
    sub_key: Optional[metric_types.SubKey] = None,
    aggregation_type: Optional[metric_types.AggregationType] = None,
    class_weights: Optional[Dict[int, float]] = None
) -> metric_types.MetricComputations:
  """Returns metric computations for calibration plot."""
  key = metric_types.PlotKey(
      name=name,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key)

  label_left, label_right = None, None
  if (left is None or right is None) and eval_config and schema:
    label_left, label_right = _find_label_domain(eval_config, schema,
                                                 model_name, output_name)
  if left is None:
    left = label_left if label_left is not None else 0.0
  if right is None:
    right = label_right if label_right is not None else 1.0

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
      right=right,
      aggregation_type=aggregation_type,
      class_weights=class_weights)
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
