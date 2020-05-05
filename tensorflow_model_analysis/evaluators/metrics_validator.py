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
"""Metrics validator."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

from typing import Any, Dict, Tuple
import numpy as np

from tensorflow_model_analysis import config
from tensorflow_model_analysis import model_util
from tensorflow_model_analysis.metrics import metric_specs
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.proto import validation_result_pb2
from tensorflow_model_analysis.slicer import slicer_lib as slicer


# TODO(b/142683826): Beam type check error in
# //third_party/tfx/components/evaluator:executor_test.python3
# _EvaluateMetricsAndPlots is passing str instead of MetricKey, remove quotes
# around metric_types.MetricKey below when fixed.
def validate_metrics(
    sliced_metrics: Tuple[slicer.SliceKeyType, Dict['metric_types.MetricKey',
                                                    Any]],
    eval_config: config.EvalConfig) -> validation_result_pb2.ValidationResult:
  """Check the metrics and check whether they should be validated."""
  # Find out which model is baseline.
  baseline_spec = model_util.get_baseline_model_spec(eval_config)
  baseline_model_name = baseline_spec.name if baseline_spec else None

  sliced_key, metrics = sliced_metrics
  thresholds = metric_specs.metric_thresholds_from_metrics_specs(
      eval_config.metrics_specs)  # pytype: disable=wrong-arg-types

  def _check_threshold(key: metric_types.MetricKey, metric: Any) -> bool:
    """Verify a metric given its metric key and metric value."""
    threshold = thresholds[key]
    if isinstance(threshold, config.GenericValueThreshold):
      lower_bound, upper_bound = -np.inf, np.inf
      if threshold.HasField('lower_bound'):
        lower_bound = threshold.lower_bound.value
      if threshold.HasField('upper_bound'):
        upper_bound = threshold.upper_bound.value
      return metric > lower_bound and metric < upper_bound
    elif isinstance(threshold, config.GenericChangeThreshold):
      diff = metric
      ratio = diff / metrics[key.make_baseline_key(baseline_model_name)]
      if threshold.direction == config.MetricDirection.LOWER_IS_BETTER:
        absolute, relative = np.inf, np.inf
      elif threshold.direction == config.MetricDirection.HIGHER_IS_BETTER:
        absolute, relative = -np.inf, -np.inf
      else:
        raise ValueError('"UNKNOWN" direction for change threshold.')
      if threshold.HasField('absolute'):
        absolute = threshold.absolute.value
      if threshold.HasField('relative'):
        relative = threshold.relative.value
      if threshold.direction == config.MetricDirection.LOWER_IS_BETTER:
        return diff < absolute and ratio < relative
      elif threshold.direction == config.MetricDirection.HIGHER_IS_BETTER:
        return diff > absolute and ratio > relative

  def _copy_metric(metric, to):
    # Will add more types when more MetricValue are supported.
    to.double_value.value = float(metric)

  def _copy_threshold(threshold, to):
    if isinstance(threshold, config.GenericValueThreshold):
      to.value_threshold.CopyFrom(threshold)
    if isinstance(threshold, config.GenericChangeThreshold):
      to.change_threshold.CopyFrom(threshold)

  # Empty metrics per slice is considered validated.
  result = validation_result_pb2.ValidationResult(validation_ok=True)
  validation_for_slice = validation_result_pb2.MetricsValidationForSlice()
  for metric_key, metric in metrics.items():
    # Not meaningful to check threshold for baseline model, thus always return
    # True if such threshold is configured. We also do not compare Message type
    # metrics.
    if (metric_key.model_name == baseline_model_name or
        metric_key not in thresholds):
      continue
    msg = ''
    # We try to convert to float values.
    try:
      metric = float(metric)
    except (TypeError, ValueError):
      msg = """
        Invalid threshold config: This metric is not comparable to the
        threshold. The type of the threshold is: {}, and the metric value is:
        \n{}""".format(type(metric), metric)
    if not _check_threshold(metric_key, metric):
      failure = validation_for_slice.failures.add()
      failure.metric_key.CopyFrom(metric_key.to_proto())
      _copy_metric(metric, failure.metric_value)
      _copy_threshold(thresholds[metric_key], failure.metric_threshold)
      failure.message = msg
  # Any failure leads to overall failure.
  if validation_for_slice.failures:
    validation_for_slice.slice_key.CopyFrom(
        slicer.serialize_slice_key(sliced_key))
    result.validation_ok = False
    result.metric_validations_per_slice.append(validation_for_slice)
  return result
