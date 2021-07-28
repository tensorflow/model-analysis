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
"""Utilities shared by the jackknife and bootstrap CI methodologies."""


import collections
import numbers
from typing import Any, Iterable, NamedTuple, Optional, Set, Sequence, TypeVar

import apache_beam as beam
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.post_export_metrics import metric_keys

SampleMetrics = NamedTuple('SampleMetrics',
                           [('metrics', metric_types.MetricsDict),
                            ('sample_id', int)])

_RawAccumulatorType = TypeVar('_RawAccumulatorType')
AccumulatorType = Sequence[_RawAccumulatorType]


class CombineFnWrapper(beam.CombineFn):
  """CombineFn which wraps and delegates to another CombineFn.

  This is useful as a base case for other CombineFn wrappers that need to do
  instance-level overriding. By subclassing CombineFnWrapper and overriding only
  the methods which need to be changed, a subclass can rely on CombineFnWrapper
  to appropriately delegate all of the other CombineFn API methods.

  Note that it is important that this class handle all of the CombineFn APIs,
  otherwise a combine_fn wrapped by any subclass which uses an un-delegated API
  would break. For example if setup() weren't delegated, then on calls to
  add_input(), the wrapped CombineFn would find that none of the setup had
  happened.

  TODO(b/194704747): Find ways to mitigate risk of future CombineFn API changes.
  """

  def __init__(self, combine_fn: beam.CombineFn):
    self._combine_fn = combine_fn

  def setup(self, *args, **kwargs):
    self._combine_fn.setup(*args, **kwargs)

  def create_accumulator(self):
    return self._combine_fn.create_accumulator()

  def add_input(self, accumulator: AccumulatorType,
                element: Any) -> AccumulatorType:
    return self._combine_fn.add_input(accumulator, element)

  def merge_accumulators(
      self, accumulators: Iterable[AccumulatorType]) -> AccumulatorType:
    return self._combine_fn.merge_accumulators(accumulators)

  def compact(self, accumulator: AccumulatorType) -> AccumulatorType:
    return self._combine_fn.compact(accumulator)

  def extract_output(self, accumulator: AccumulatorType) -> AccumulatorType:
    return self._combine_fn.extract_output(accumulator)

  def teardown(self, *args, **kwargs):
    return self._combine_fn.teardown(*args, **kwargs)


class SampleCombineFn(beam.CombineFn):
  """Computes the jackknife standard error for each metric from samples."""

  class _SampleAccumulator(object):

    __slots__ = ['sums', 'sums_of_squares', 'num_samples', 'unsampled_values']

    def __init__(self):
      self.sums = collections.defaultdict(float)
      self.sums_of_squares = collections.defaultdict(float)
      self.num_samples = 0
      self.unsampled_values = None

  def __init__(
      self,
      num_samples: int,
      full_sample_id: int,
      skip_ci_metric_keys: Optional[Set[metric_types.MetricKey]] = None,
      allow_missing_samples: bool = False):
    """Initializes a _MergeJackknifeSamples CombineFn.

    Args:
      num_samples: The number of samples computed per slice.
      full_sample_id: The sample_id corresponding to the unsampled metrics.
      skip_ci_metric_keys: Set of metric keys for which to skip confidence
        interval computation. For metric keys in this set, just the unsampled
        value will be returned.
      allow_missing_samples: Whether to return standard error estimates in cases
        where fewer than the expected number of samples are present. This can
        happen when operating on a small slice and the sampling method generates
        an empty sample.
    """
    self._num_samples = num_samples
    self._full_sample_id = full_sample_id
    self._skip_ci_metric_keys = skip_ci_metric_keys
    self._allow_missing_samples = allow_missing_samples
    self._num_slices_counter = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, 'num_slices')
    self._missing_samples_counter = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, 'num_slices_missing_samples')
    self._negative_variance_counter = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, 'num_negative_variance_metric_slices')
    self._zero_variance_counter = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, 'num_zero_variance_metric_slices')

  def create_accumulator(self) -> 'SampleCombineFn._SampleAccumulator':
    return SampleCombineFn._SampleAccumulator()

  def add_input(self, accumulator: 'SampleCombineFn._SampleAccumulator',
                sample: SampleMetrics) -> 'SampleCombineFn._SampleAccumulator':
    sample_id = sample.sample_id
    sample = sample.metrics
    if sample_id == self._full_sample_id:
      accumulator.unsampled_values = sample
    else:
      accumulator.num_samples += 1
      for metric_key, value in sample.items():
        if not isinstance(value, numbers.Number):
          # skip non-numeric values
          continue
        # Numpy int64 and int32 types can overflow without warning. To prevent
        # this we always cast to python floats prior to doing any operations.
        value = float(value)
        accumulator.sums[metric_key] += value
        accumulator.sums_of_squares[metric_key] += value * value
    return accumulator

  def merge_accumulators(
      self, accumulators: Iterable['SampleCombineFn._SampleAccumulator']
  ) -> 'SampleCombineFn._SampleAccumulator':
    # treat as iterator to enforce streaming processing
    accumulators = iter(accumulators)
    result = next(accumulators)
    for accumulator in accumulators:
      if accumulator.unsampled_values:
        result.unsampled_values = accumulator.unsampled_values
      result.num_samples += accumulator.num_samples
      for metric_key, sum_value in accumulator.sums.items():
        result.sums[metric_key] += sum_value
        result.sums_of_squares[metric_key] += (
            accumulator.sums_of_squares[metric_key])
    return result

  def extract_output(
      self, accumulator: 'SampleCombineFn._SampleAccumulator'
  ) -> metric_types.MetricsDict:
    self._num_slices_counter.inc(1)
    result = {}
    missing_samples = False
    # If we don't get at least one example in each sample, don't compute CI.
    if (not self._allow_missing_samples and
        accumulator.num_samples < self._num_samples):
      self._missing_samples_counter.inc(1)
      missing_samples = True
      error_metric_key = metric_types.MetricKey(metric_keys.ERROR_METRIC),
      result[error_metric_key] = (
          f'CI not computed because only {accumulator.num_samples} samples '
          f'were non-empty. Expected {self._num_samples}.')

    dof = accumulator.num_samples - 1

    assert accumulator.unsampled_values is not None, (
        'Expected unsampled value to be present in final accumulator')
    for metric_key, unsampled_value in accumulator.unsampled_values.items():
      if (missing_samples or metric_key not in accumulator.sums or
          (self._skip_ci_metric_keys and
           metric_key in self._skip_ci_metric_keys)):
        result[metric_key] = unsampled_value
      else:
        mean = accumulator.sums[metric_key] / accumulator.num_samples
        sum_of_squares = accumulator.sums_of_squares[metric_key]
        # one-pass variance formula with num_samples degrees of freedom
        sample_variance = (
            sum_of_squares / float(accumulator.num_samples) - mean * mean)
        # we want to use num_samples - 1 degrees of freedom, so we rescale
        sample_variance = sample_variance * accumulator.num_samples / dof
        standard_error = sample_variance**0.5
        if standard_error == 0:
          self._zero_variance_counter.inc()
        elif standard_error < 0:
          self._negative_variance_counter.inc()
        result[metric_key] = types.ValueWithTDistribution(
            sample_mean=mean,
            sample_standard_deviation=standard_error,
            sample_degrees_of_freedom=dof,
            unsampled_value=unsampled_value)
    # TODO(b/194750790): remove this once the typing issue is resolved.
    return result  # pytype: disable=bad-return-type
