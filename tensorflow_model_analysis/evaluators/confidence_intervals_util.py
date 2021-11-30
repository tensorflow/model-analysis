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
"""Utilities shared by the jackknife and bootstrap CI methodologies."""

import collections
import numbers
from typing import Iterable, NamedTuple, Optional, Set, Sequence, Tuple
import apache_beam as beam
import numpy as np
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.post_export_metrics import metric_keys

SampleMetrics = NamedTuple('SampleMetrics',
                           [('metrics', metric_types.MetricsDict),
                            ('sample_id', int)])


def mean_and_std(
    values: Sequence[types.MetricValueType],
    ddof: int) -> Tuple[types.MetricValueType, types.MetricValueType]:
  """Computes mean and standard deviation for (structued) metric values.

  Args:
    values: An iterable of values for which to compute the mean and standard
      deviation
    ddof: The difference in degrees of freedom to use for the standard deviation
      computation, relative to the length of values. For example, if len(values)
      == 10, and ddof is 1, the standard deviation will be computed with 9
      degreees of freedom

  Returns:
     A 2-tuple in which the first element is the mean and the second element is
     the standard deviation. The types of the mean and standard deviation will
     be the same as the type of each element in values.
  """
  total = None
  for value in values:
    if total is None:
      total = value
    else:
      total = total + value
  mean = total / len(values)
  squared_residual_total = None
  for value in values:
    squared_residual = (value - mean)**2
    if squared_residual_total is None:
      squared_residual_total = squared_residual
    else:
      squared_residual_total = squared_residual_total + squared_residual
  std = (squared_residual_total / (len(values) - ddof))**0.5
  return mean, std


class SampleCombineFn(beam.CombineFn):
  """Computes the standard deviation for each metric from samples."""

  class SampleAccumulator:

    __slots__ = ['point_estimates', 'num_samples', 'metric_samples']

    def __init__(self):
      self.point_estimates = None
      self.num_samples = 0
      self.metric_samples = collections.defaultdict(list)

  def __init__(
      self,
      num_samples: int,
      full_sample_id: int,
      skip_ci_metric_keys: Optional[Set[metric_types.MetricKey]] = None):
    """Initializes a SampleCombineFn.

    Args:
      num_samples: The number of samples computed per slice.
      full_sample_id: The sample_id corresponding to the unsampled metrics.
      skip_ci_metric_keys: Set of metric keys for which to skip confidence
        interval computation. For metric keys in this set, just the unsampled
        value will be returned.
    """
    self._num_samples = num_samples
    self._full_sample_id = full_sample_id
    self._skip_ci_metric_keys = skip_ci_metric_keys
    self._num_slices_counter = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, 'num_slices')
    self._missing_samples_counter = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, 'num_slices_missing_samples')
    self._missing_metric_samples_counter = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, 'num_slices_missing_metric_samples')

  def create_accumulator(self) -> 'SampleCombineFn.SampleAccumulator':
    return SampleCombineFn.SampleAccumulator()

  def add_input(self, accumulator: 'SampleCombineFn.SampleAccumulator',
                sample: SampleMetrics) -> 'SampleCombineFn.SampleAccumulator':
    sample_id = sample.sample_id
    sample = sample.metrics
    if sample_id == self._full_sample_id:
      accumulator.point_estimates = sample
    else:
      accumulator.num_samples += 1
      for metric_key, value in sample.items():
        if (not (isinstance(value,
                            (numbers.Number, types.StructuredMetricValue)) or
                 (isinstance(value, np.ndarray) and
                  np.issubdtype(value.dtype, np.number)))):
          # skip non-numeric values
          continue
        if (self._skip_ci_metric_keys and
            metric_key in self._skip_ci_metric_keys):
          continue
        accumulator.metric_samples[metric_key].append(value)
    return accumulator

  def merge_accumulators(
      self, accumulators: Iterable['SampleCombineFn.SampleAccumulator']
  ) -> 'SampleCombineFn.SampleAccumulator':
    # treat as iterator to enforce streaming processing
    accumulators = iter(accumulators)
    result = next(accumulators)
    for accumulator in accumulators:
      if accumulator.point_estimates is not None:
        result.point_estimates = accumulator.point_estimates
      result.num_samples += accumulator.num_samples
      for metric_key, sample_values in accumulator.metric_samples.items():
        result.metric_samples[metric_key].extend(sample_values)
    return result

  def _validate_accumulator(
      self, accumulator: 'SampleCombineFn.SampleAccumulator'
  ) -> 'SampleCombineFn.SampleAccumulator':
    self._num_slices_counter.inc(1)
    error_metric_key = metric_types.MetricKey(metric_keys.ERROR_METRIC)
    if accumulator.num_samples < self._num_samples:
      self._missing_samples_counter.inc(1)
      accumulator.point_estimates[error_metric_key] = (
          f'CI not computed because only {accumulator.num_samples} samples '
          f'were non-empty. Expected {self._num_samples}.')
      # If we are missing samples, clear samples for all metrics as they are all
      # unusable.
      accumulator.metric_samples = {}
    # Check that all metrics were present in all samples
    metric_incorrect_sample_counts = {}
    for metric_key in accumulator.point_estimates:
      if metric_key in accumulator.metric_samples:
        actual_num_samples = len(accumulator.metric_samples[metric_key])
        if actual_num_samples != self._num_samples:
          metric_incorrect_sample_counts[metric_key] = actual_num_samples
    if metric_incorrect_sample_counts:
      accumulator.point_estimates[error_metric_key] = (
          'CI not computed for the following metrics due to incorrect number '
          f'of samples: "{metric_incorrect_sample_counts}".'
          f'Expected {self._num_samples}.')
    return accumulator

  # TODO(b/195132951): replace with @abc.abstractmethod
  def extract_output(
      self, accumulator: 'SampleCombineFn.SampleAccumulator'
  ) -> metric_types.MetricsDict:
    raise NotImplementedError('Must be implemented in subclasses.')
