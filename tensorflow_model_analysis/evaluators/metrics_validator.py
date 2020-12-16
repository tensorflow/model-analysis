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

from typing import Any, Dict, Iterable, List, Tuple, Union
import numpy as np

from tensorflow_model_analysis import config
from tensorflow_model_analysis import model_util
from tensorflow_model_analysis.metrics import metric_specs
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.proto import validation_result_pb2
from tensorflow_model_analysis.slicer import slicer_lib as slicer

_ThresholdType = Union[config.GenericValueThreshold,
                       config.GenericChangeThreshold]


# TODO(b/142683826): Beam type check error in
# //third_party/py/tfx/components/evaluator:executor_test.python3
# _EvaluateMetricsAndPlots is passing str instead of MetricKey, remove quotes
# around metric_types.MetricKey below when fixed.
def validate_metrics(
    sliced_metrics: Tuple[Union[slicer.SliceKeyType, slicer.CrossSliceKeyType],
                          Dict['metric_types.MetricKey', Any]],
    eval_config: config.EvalConfig) -> validation_result_pb2.ValidationResult:
  """Check the metrics and check whether they should be validated."""
  # Find out which model is baseline.
  baseline_spec = model_util.get_baseline_model_spec(eval_config)
  baseline_model_name = baseline_spec.name if baseline_spec else None

  sliced_key, metrics = sliced_metrics
  thresholds = metric_specs.metric_thresholds_from_metrics_specs(
      eval_config.metrics_specs)
  is_cross_slice = slicer.is_cross_slice_key(sliced_key)

  def _check_threshold(key: metric_types.MetricKey, threshold: _ThresholdType,
                       metric: Any) -> bool:
    """Verify a metric given its metric key and metric value."""
    metric = float(metric)
    if isinstance(threshold, config.GenericValueThreshold):
      lower_bound, upper_bound = -np.inf, np.inf
      if threshold.HasField('lower_bound'):
        lower_bound = threshold.lower_bound.value
      if threshold.HasField('upper_bound'):
        upper_bound = threshold.upper_bound.value
      return metric > lower_bound and metric < upper_bound
    elif isinstance(threshold, config.GenericChangeThreshold):
      diff = metric
      metric_baseline = float(
          metrics[key.make_baseline_key(baseline_model_name)])
      ratio = diff / metric_baseline
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
    else:
      raise ValueError('Unknown threshold: {}'.format(threshold))

  def _copy_metric(metric, to):
    # Will add more types when more MetricValue are supported.
    to.double_value.value = float(metric)

  def _copy_threshold(threshold, to):
    if isinstance(threshold, config.GenericValueThreshold):
      to.value_threshold.CopyFrom(threshold)
    if isinstance(threshold, config.GenericChangeThreshold):
      to.change_threshold.CopyFrom(threshold)

  def _add_to_set(s, v):
    """Adds value to set. Returns true if didn't exist."""
    if v in s:
      return False
    else:
      s.add(v)
      return True

  # Empty metrics per slice is considered validated.
  result = validation_result_pb2.ValidationResult(validation_ok=True)
  validation_for_slice = validation_result_pb2.MetricsValidationForSlice()
  unchecked_thresholds = dict(thresholds)
  for metric_key, metric in metrics.items():
    if metric_key not in thresholds:
      continue
    del unchecked_thresholds[metric_key]
    # Not meaningful to check threshold for baseline model, thus always return
    # True if such threshold is configured. We also do not compare Message type
    # metrics.
    if metric_key.model_name == baseline_model_name:
      continue
    msg = ''
    existing_failures = set()
    for slice_spec, threshold in thresholds[metric_key]:
      if slice_spec is not None:
        if (isinstance(slice_spec, config.SlicingSpec) and
            (is_cross_slice or not slicer.SingleSliceSpec(
                spec=slice_spec).is_slice_applicable(sliced_key))):
          continue
        if (isinstance(slice_spec, config.CrossSlicingSpec) and
            (not is_cross_slice or not slicer.is_cross_slice_applicable(
                cross_slice_key=sliced_key, cross_slicing_spec=slice_spec))):
          continue
      elif is_cross_slice:
        continue
      try:
        check_result = _check_threshold(metric_key, threshold, metric)
      except ValueError:
        msg = """
          Invalid metrics or threshold for comparison: The type of the metric
          is: {}, the metric value is: {}, and the threshold is: {}.
          """.format(type(metric), metric, threshold)
        check_result = False
      else:
        msg = ''
      if not check_result:
        # The same threshold values could be set for multiple matching slice
        # specs. Only store the first match.
        #
        # Note that hashing by SerializeToString() is only safe if used within
        # the same process.
        if not _add_to_set(existing_failures, threshold.SerializeToString()):
          continue
        failure = validation_for_slice.failures.add()
        failure.metric_key.CopyFrom(metric_key.to_proto())
        _copy_metric(metric, failure.metric_value)
        _copy_threshold(threshold, failure.metric_threshold)
        failure.message = msg
      # Track we have completed a validation check for slice spec and metric
      slicing_details = result.validation_details.slicing_details.add()
      if slice_spec is not None:
        if isinstance(slice_spec, config.SlicingSpec):
          slicing_details.slicing_spec.CopyFrom(slice_spec)
        else:
          slicing_details.cross_slicing_spec.CopyFrom(slice_spec)
      else:
        slicing_details.slicing_spec.CopyFrom(config.SlicingSpec())
      slicing_details.num_matching_slices = 1
  # All unchecked thresholds are considered failures.
  for metric_key, thresholds in unchecked_thresholds.items():
    if metric_key.model_name == baseline_model_name:
      continue
    existing_failures = set()
    for slice_spec, threshold in thresholds:
      if slice_spec is not None:
        if is_cross_slice != isinstance(slice_spec, config.CrossSlicingSpec):
          continue
        if (is_cross_slice and not slicer.is_cross_slice_applicable(
            cross_slice_key=sliced_key, cross_slicing_spec=slice_spec)):
          continue
      elif is_cross_slice:
        continue
      # The same threshold values could be set for multiple matching slice
      # specs. Only store the first match.
      #
      # Note that hashing by SerializeToString() is only safe if used within
      # the same process.
      if not _add_to_set(existing_failures, threshold.SerializeToString()):
        continue
      failure = validation_for_slice.failures.add()
      failure.metric_key.CopyFrom(metric_key.to_proto())
      _copy_threshold(threshold, failure.metric_threshold)
      failure.message = 'Metric not found.'
  # Any failure leads to overall failure.
  if validation_for_slice.failures:
    if not is_cross_slice:
      validation_for_slice.slice_key.CopyFrom(
          slicer.serialize_slice_key(sliced_key))
    else:
      validation_for_slice.cross_slice_key.CopyFrom(
          slicer.serialize_cross_slice_key(sliced_key))
    result.validation_ok = False
    result.metric_validations_per_slice.append(validation_for_slice)
  return result


def _hashed_slicing_details(
    slicing_details: Iterable[validation_result_pb2.SlicingDetails]
) -> Dict[bytes, validation_result_pb2.SlicingDetails]:
  """Returns hash table of slicing details keyed by serialized slice spec."""
  # Note that hashing by SerializeToString() is only safe if used within the
  # same process.
  hashed_details = {}
  for details in slicing_details:
    slice_hash = details.slicing_spec.SerializeToString()
    if slice_hash not in hashed_details:
      hashed_details[slice_hash] = details
  return hashed_details


def merge_details(a: validation_result_pb2.ValidationResult,
                  b: validation_result_pb2.ValidationResult):
  """Merges validation details in ValidationtResult b into ValidationResult a."""
  hashed_details = _hashed_slicing_details(b.validation_details.slicing_details)
  # Combine a with matching values from b
  for details in a.validation_details.slicing_details:
    slice_hash = details.slicing_spec.SerializeToString()
    if slice_hash in hashed_details:
      details.num_matching_slices = (
          details.num_matching_slices +
          hashed_details[slice_hash].num_matching_slices)
      del hashed_details[slice_hash]
  # Add any values from b not matched in a
  for details in hashed_details.values():
    a.validation_details.slicing_details.append(details)


def get_missing_slices(
    slicing_details: Iterable[validation_result_pb2.SlicingDetails],
    eval_config: config.EvalConfig
) -> List[Union[config.SlicingSpec, config.CrossSlicingSpec]]:
  """Returns specs that are defined in the EvalConfig but not found in details.

  Args:
    slicing_details: Slicing details.
    eval_config: Eval config.

  Returns:
    List of missing slices or empty list if none are missing.
  """
  hashed_details = _hashed_slicing_details(slicing_details)
  thresholds = metric_specs.metric_thresholds_from_metrics_specs(
      eval_config.metrics_specs)
  baseline_spec = model_util.get_baseline_model_spec(eval_config)
  baseline_model_name = baseline_spec.name if baseline_spec else None
  missing_slices = []
  for metric_key, sliced_thresholds in thresholds.items():
    # Skip baseline.
    if metric_key.model_name == baseline_model_name:
      continue
    for slice_spec, _ in sliced_thresholds:
      if not slice_spec:
        slice_spec = config.SlicingSpec()
      slice_hash = slice_spec.SerializeToString()
      if slice_hash not in hashed_details:
        missing_slices.append(slice_spec)
        # Same slice may be used by other metrics/thresholds, only add once
        hashed_details[slice_hash] = validation_result_pb2.SlicingDetails()
  return missing_slices
