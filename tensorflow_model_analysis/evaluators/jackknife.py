# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Helper methods for computing jackknife standard error estimates on metrics.

To compute jackknife estimates for a set of metrics, this module provides two
PTransforms which should be applied in order: JackknifeCombinePerKey and
MergeJackknifeSamples. The first computes metrics on jackknife samples of the
data (leaving out 1 partition of the data per sample) and the second merges
these samples into estimates of the standard error and populates a
ValueWithTDistribution.

Given a combiner which takes in elements of type K and returns outputs of type
T, when JackKnifeCombinePerKey(combiner, ...) is applied to a collection of
key value pairs of types <K, V>, this PTransform returns a PCollection of the
form <K+S, T> where K+S denotes that the jackknife sample ID has been added to
the input key.

Then, calling MergeJackknifeSamples will drop the sample ID from the key and
transform a collection of type <K+S, T> into per-input-key values, of type
<K, ValueWithTDistribution<T>>. ValueWithTDistribution objects contain both the
exact metric values when computed over all examples matching a given key, and
an estimate of the standard error of each metric.

Putting it all together, this should look like:

  # Tuple[slicer.SliceKeyType, Dict[MetricKey, ValueWithTDistribution]]
  combined_values_per_key_w_stderr = (
    sliced_extracts
    | 'JackknifeCombinePerKey' >> jackknife.JackknifeCombinePerKey(combiner)
    ...
    | 'MergeJackknifeSamples' >> jackknife.MergeJackknifeSamples())

In which ValueWithTDistribution.unsampled_value is the same as the value in

  # Tuple[slicer.SliceKeyType, Dict[MetricKey, Any]]
  combined_values_per_key = (
    sliced_extracts
    | 'CombinePerKey' >> beam.CombinePerKey(combiner))
"""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import collections
import copy
import numbers
from typing import Any, Iterator, List, NamedTuple, Optional, Sequence, Set, Tuple, TypeVar

import apache_beam as beam
import numpy as np
import six

from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.post_export_metrics import metric_keys
from tensorflow_model_analysis.slicer import slicer_lib as slicer

_JACKKNIFE_SAMPLE_ID_KEY = u'_sample_id'
_JACKKNIFE_EXAMPLE_COUNT_METRIC_KEY = metric_types.MetricKey(
    u'__jackknife_example_count')
_JACKKNIFE_FULL_SAMPLE_ID = -1

_AccumulatorType = TypeVar('_AccumulatorType')

_PartitionInfo = NamedTuple('_PartitionInfo', [('accumulator', Any),
                                               ('size', int),
                                               ('partition_id', int)])


@beam.typehints.with_input_types(Any)
@beam.typehints.with_output_types(_AccumulatorType)
class _AccumulateOnlyCombiner(beam.CombineFn):
  """A combiner wrapper which returns the accumulator as the output value.

  This is intended to allow invoking CombineFns in settings where you might need
  to subsequently merge the results before calling extract_output. A typical use
  of a _AccumulateOnlyCombiner might look like:

      c = OtherCombiner()

      # combine per key, but don't call extract_output()
      accumulators = [p | beam.CombinePerKey(_AccumulateOnlyCombiner(c)
                      for i in range(3)]

      # do some merging of per-key combiners
      even_output = c.extract_output(c.merge_accumulators(
          [a for i, a in enumerate(accumulators) if i % 2 == 0]))
      odd_output = c.extract_output(c.merge_accumulators(
          [a for i, a in enumerate(accumulators) if i % 2 == 1])
  """

  def __init__(self, combiner: beam.CombineFn):
    self._combiner = combiner

  def create_accumulator(self) -> Any:
    return self._combiner.create_accumulator()

  def add_input(self, accumulator: _AccumulatorType,
                element: Any) -> _AccumulatorType:
    return self._combiner.add_input(accumulator, element)

  def merge_accumulators(
      self, accumulators: Sequence[_AccumulatorType]) -> _AccumulatorType:
    return self._combiner.merge_accumulators(accumulators)

  def compact(self, accumulator: _AccumulatorType) -> _AccumulatorType:
    return self._combiner.compact(accumulator)

  def extract_output(self, accumulator: _AccumulatorType) -> _AccumulatorType:
    return accumulator


def _make_loo_accumulators(
    accumulators: List[_AccumulatorType],
    combiner: beam.CombineFn) -> Iterator[_AccumulatorType]:
  """Yields accumulators which each leave out one value in accumulators.

  Args:
    accumulators: Tuple of values for which to compute complements
    combiner: A combiner to use for creating and merging accumulators.

  Yields:
    Leave-one-out accumulators for each element in `accumulators`. The ith
    accumulator will be the result of merging all accumulators but the ith,
    along with the accumulator passed as `complement`.
  """

  def make_loo_accumulators_rec(
      accumulators: List[_AccumulatorType], complement: _AccumulatorType,
      combiner: beam.CombineFn) -> Iterator[_AccumulatorType]:
    """Recursive helper to compute leave one out accumulators."""
    if len(accumulators) == 1:
      yield complement
    else:
      split_idx = int(len(accumulators) / 2)
      left, right = accumulators[:split_idx], accumulators[split_idx:]
      left_c = copy.deepcopy(complement)
      left_c = combiner.merge_accumulators([left_c] + right)
      for c in make_loo_accumulators_rec(left, left_c, combiner):
        yield c
      # reuse the complement accumulator on the right recursion.
      right_c = combiner.merge_accumulators([complement] + left)
      for c in make_loo_accumulators_rec(right, right_c, combiner):
        yield c

  # TODO(b/151445942) use `yield from` when we stop supporting python < 3.3
  for acc in make_loo_accumulators_rec(accumulators,
                                       combiner.create_accumulator(), combiner):
    yield acc


# TODO(b/152812821): Disble Beam annotations support due to failure in:
# //third_party/py/tensorflow_model_analysis/evaluators:jackknife_test.python3
# Output type hint violation at JackknifeCombinePerKey: expected Tuple[Union[
# Tuple[Tuple[str, Union[float, int, str]], ...], Tuple[]], Tuple[Dict[
# MetricKey, Any], ...]], got Tuple[Union[Tuple[Tuple[str, Union[float, int,
# str]], ...], Tuple[]], Dict[MetricKey, Any]]
#
# Since @beam.typehints.no_annotations is not available yet, part of the output
# type is put in quotes, which currently makes Beam ignore the hint.
def _make_jackknife_samples(
    slice_partitions: Tuple[slicer.SliceKeyType,
                            Sequence[_PartitionInfo]], combiner: beam.CombineFn
) -> Iterator[Tuple[slicer.SliceKeyType, 'metric_types.MetricsDict']]:
  """Computes leave-one-out and unsampled ouputs for the combiner.

  This function creates leave-one-out combiner outputs by combining all but one
  accumulator and extracting the output. Second, it creates an unsampled output
  using all of the accumulators and extracts an unsampled output. The keys
  yielded by thus function are augmented versions of the input slice key in
  which the sample ID (or a special placeholder ID for the unsampled value) has
  been added.

  Args:
    slice_partitions: The result of GroupByKey in which the key is a slice_key,
      and the grouped stream consists of per-partition _PartitionInfo tuples in
      which the first element is an accumulator for that partition, the second
      element is the size of that partition, and the third element is the
      partition ID.
    combiner: The combiner to be used for converting accumulators to outputs.

  Yields:
    Tuples of the form (slice_key, metrics), for each jackknife sample and for
    the unsampled value.
  """
  slice_key, accumulators_sizes_and_ids = slice_partitions
  accumulators, sizes, partition_ids = zip(*accumulators_sizes_and_ids)
  unsampled_accumulator = None
  for i, loo_accumulator in enumerate(
      _make_loo_accumulators(list(accumulators), combiner)):
    # yield sampled output with sample_id of the leftout partition
    sample_id_key = (_JACKKNIFE_SAMPLE_ID_KEY, partition_ids[i])
    yield slice_key + (sample_id_key,), combiner.extract_output(loo_accumulator)
    if i == 0:
      # Create the unsampled accumulator from sample 0 and its complement.
      unsampled_accumulator = combiner.merge_accumulators(
          [loo_accumulator, accumulators[0]])

  # yield unsampled output along with total count as a special metric
  count_dict = {_JACKKNIFE_EXAMPLE_COUNT_METRIC_KEY: sum(sizes)}
  sample_id_key = ((_JACKKNIFE_SAMPLE_ID_KEY, _JACKKNIFE_FULL_SAMPLE_ID),)
  unsampled_output = combiner.extract_output(unsampled_accumulator)
  unsampled_key = slice_key + sample_id_key
  unsampled_val = unsampled_output + (count_dict,)
  yield unsampled_key, unsampled_val


@beam.typehints.with_input_types(Tuple[slicer.SliceKeyType, types.Extracts])
@beam.typehints.with_output_types(Tuple[slicer.SliceKeyType,
                                        Tuple[metric_types.MetricsDict, ...]])
class JackknifeCombinePerKey(beam.PTransform):
  """Computes per-key delete-d jackknife combiner outputs.

  Rather than returning a single combiner output per key, as you would expect
  when calling beam.CombinePerKey, this PTransform returns a combiner output
  value per input key and per jacknife sample. This means that if the key has
  cardinality C, the returned PCollection will have size C *
  num_jackknife_samples. Each jackknife sample for a given input key represents
  the combiner value computed over all but one partition of values with that
  key.
  """

  def __init__(self,
               combiner: beam.CombineFn,
               num_jackknife_samples: int,
               random_seed: Optional[int] = None):
    self._combiner = combiner
    self._num_jackknife_samples = num_jackknife_samples
    self._random_state = np.random.RandomState(random_seed)

  def expand(self, sliced_extracts):

    def partition_fn(_, num_partitions):
      return self._random_state.randint(num_partitions)

    # Partition the data
    # List[PCollection[Tuple[slicer.SliceKeyType, types.Extracts]]]
    partitions = (
        sliced_extracts
        | 'Partition' >> beam.Partition(partition_fn,
                                        self._num_jackknife_samples))

    def add_partition_index(slice_key,
                            accumulator_and_size,
                            partition_index=None):
      accumulator, size = accumulator_and_size
      return slice_key, _PartitionInfo(accumulator, size, partition_index)

    # Within each partition, partially combine per slice key to get accumulators
    # and partition sizes; add partition_id for determinism.
    # List[PCollection[slicer.SliceKeyType, _PartitionInfo]]
    partition_accumulators = []
    for i, partition in enumerate(partitions):
      partition_accumulators.append(
          partition
          | 'CombinePartition[{}]'.format(i) >> beam.CombinePerKey(
              beam.transforms.combiners.SingleInputTupleCombineFn(
                  _AccumulateOnlyCombiner(combiner=self._combiner),
                  beam.transforms.combiners.CountCombineFn()))
          | 'AddPartitionId[{}]'.format(i) >> beam.MapTuple(
              add_partition_index, i))

    # Group partitions for the same slice, compute LOO metrics, and flatten back
    # into per-partition LOO metrics.
    # (slicer.SliceKeyType, Tuple[metric_types.MetricsDict])
    return (partition_accumulators
            | 'FlattenPartitionAccumulators' >> beam.Flatten()
            | 'CollectPerSlicePartitions' >> beam.GroupByKey()
            | 'MakeJackknifeSamples' >> beam.FlatMap(
                _make_jackknife_samples, combiner=self._combiner))


def _move_jackknife_sample_id_to_value(
    slice_key: slicer.SliceKeyType, sample: metric_types.MetricsDict
) -> Tuple[slicer.SliceKeyType, metric_types.MetricsDict]:
  """Moves the jackknife sample ID from the key to value.

  Args:
    slice_key: The slice key which contains the sample_id.
    sample: The per-sample metrics computed by the _JackknifeCombiner.

  Returns:
    The updated slice key and value.
  """
  index_and_sample_id = [(i, subkey[1])
                         for (i, subkey) in enumerate(slice_key)
                         if subkey[0] == _JACKKNIFE_SAMPLE_ID_KEY]
  assert len(index_and_sample_id) == 1, ('Expected 1 "_sample_id" subkey in '
                                         'slice key, found: {}'.format(
                                             len(index_and_sample_id)))
  subkey_index, sample_id = index_and_sample_id[0]
  original_slice_key = slice_key[:subkey_index] + slice_key[subkey_index + 1:]
  sample = sample.copy()
  sample[metric_types.MetricKey(_JACKKNIFE_SAMPLE_ID_KEY)] = sample_id
  return original_slice_key, sample


class _MergeJacknifeAccumulator(object):

  __slots__ = ['sums', 'sums_of_squares', 'num_samples', 'unsampled_values']

  def __init__(self):
    self.sums = collections.defaultdict(float)
    self.sums_of_squares = collections.defaultdict(float)
    self.num_samples = 0
    self.unsampled_values = None


@beam.typehints.with_input_types(metric_types.MetricsDict)
@beam.typehints.with_output_types(metric_types.MetricsDict)
class _JackknifeSampleCombiner(beam.CombineFn):
  """Computes the jackknife standard error for each metric from samples."""

  def __init__(
      self,
      num_jackknife_samples: int,
      skip_ci_metric_keys: Optional[Set[metric_types.MetricKey]] = None):
    """Initializes a _MergeJackknifeSamples CombineFn.

    Args:
      num_jackknife_samples: The number of samples computed per slice.
      skip_ci_metric_keys: Set of metric keys for which to skip confidence
        interval computation. For metric keys in this set, just the unsampled
        value will be returned.
    """
    self._num_jackknife_samples = num_jackknife_samples
    self._skip_ci_metric_keys = skip_ci_metric_keys
    self._num_slices_counter = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, 'num_slices')
    self._missing_samples_counter = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, 'num_slices_missing_jackknife_samples')
    self._small_samples_counter = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, 'num_slices_with_small_jackknife_samples')
    self._negative_variance_dist = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE, 'negative_variance_metric_slice_size_dist')
    self._zero_variance_dist = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE, 'zero_variance_metric_slice_size_dist')
    self._sample_id_key = metric_types.MetricKey(_JACKKNIFE_SAMPLE_ID_KEY)

  def create_accumulator(self) -> _MergeJacknifeAccumulator:
    return _MergeJacknifeAccumulator()

  def add_input(self, accumulator: _MergeJacknifeAccumulator,
                sample: metric_types.MetricsDict) -> _MergeJacknifeAccumulator:
    if sample[self._sample_id_key] == _JACKKNIFE_FULL_SAMPLE_ID:
      full_sample = sample.copy()
      full_sample.pop(self._sample_id_key)
      accumulator.unsampled_values = full_sample
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

  def merge_accumulators(self, accumulators):
    # treat as iterator to enforce streaming processing
    accumulators = iter(accumulators)
    result = six.next(accumulators)
    for accumulator in accumulators:
      if accumulator.unsampled_values:
        result.unsampled_values = accumulator.unsampled_values
      result.num_samples += accumulator.num_samples
      for metric_key, sum_value in accumulator.sums.items():
        result.sums[metric_key] += sum_value
        result.sums_of_squares[metric_key] += (
            accumulator.sums_of_squares[metric_key])
    return result

  def extract_output(self, accumulator):
    # Compute the jackknife standard error for each metric.
    # See delete-d bootstrap method described in:
    # https://www.stat.berkeley.edu/~hhuang/STAT152/Jackknife-Bootstrap.pdf
    # Rather than normalize by all possible n-choose-d samples, we normalize by
    # the actual number of samples.
    self._num_slices_counter.inc(1)
    unsampled_values = accumulator.unsampled_values
    assert _JACKKNIFE_EXAMPLE_COUNT_METRIC_KEY in unsampled_values, (
        'Expected unsampled jackknife values to contain the example count key: '
        '"{}". Instead, found keys: {}'.format(
            _JACKKNIFE_EXAMPLE_COUNT_METRIC_KEY, unsampled_values.keys()))
    n = unsampled_values.pop(_JACKKNIFE_EXAMPLE_COUNT_METRIC_KEY)

    result = {}
    missing_samples = False
    # If we don't get at least one example in each sample, don't compute CI.
    if accumulator.num_samples < self._num_jackknife_samples:
      self._missing_samples_counter.inc(1)
      missing_samples = True
      result[metric_types.MetricKey(metric_keys.ERROR_METRIC)] = (
          'CI not computed because only {num_samples} samples were non-empty. '
          'Expected {num_jackknife_samples}.'.format(
              num_samples=accumulator.num_samples,
              num_jackknife_samples=self._num_jackknife_samples))

    # set d to expected size of a sample holdout
    d = n / float(accumulator.num_samples)
    if d < n**0.5:
      # if d < sqrt(n) the jackknife standard error will behave poorly for some
      # metrics (including the median).
      self._small_samples_counter.inc(1)

    jackknife_scaling_factor = (n - d) / d
    dof = accumulator.num_samples - 1
    num_samples = accumulator.num_samples

    for metric_key, unsampled_value in unsampled_values.items():
      if (missing_samples or metric_key not in accumulator.sums or
          (self._skip_ci_metric_keys and
           metric_key in self._skip_ci_metric_keys)):
        result[metric_key] = unsampled_value
      else:
        mean = accumulator.sums[metric_key] / num_samples
        sum_of_squares = accumulator.sums_of_squares[metric_key]
        # one-pass variance formula with num_samples degrees of freedom
        sample_variance = sum_of_squares / float(num_samples) - mean * mean
        if sample_variance < 0:
          self._negative_variance_dist.update(n)
        standard_error = (jackknife_scaling_factor * sample_variance)**0.5
        if standard_error == 0:
          self._zero_variance_dist.update(n)
        result[metric_key] = types.ValueWithTDistribution(
            sample_mean=mean,
            sample_standard_deviation=standard_error,
            sample_degrees_of_freedom=dof,
            unsampled_value=unsampled_value)
    return result


@beam.typehints.with_input_types(Tuple[slicer.SliceKeyType,
                                       metric_types.MetricsDict])
@beam.typehints.with_output_types(Tuple[slicer.SliceKeyType,
                                        metric_types.MetricsDict])
class MergeJackknifeSamples(beam.PTransform):
  """Merges per-slice per sample values into per-slice point estimates and SEs.

  When applied to a PCollection of per-key and per-sample values, this
  PTransform merges all values for a given key into a single
  ValueWithTDistribution containing both the unsampled value and an estimate of
  the standard error of that unsampled value.
  """

  def __init__(
      self,
      num_jackknife_samples: int,
      skip_ci_metric_keys: Optional[Set[metric_types.MetricKey]] = None):
    self._num_jackknife_samples = num_jackknife_samples
    self._skip_ci_metric_keys = skip_ci_metric_keys

  def expand(self, sliced_derived_values_and_diffs):
    return (sliced_derived_values_and_diffs
            | 'MoveJackknifeSampleIdToValue' >>
            beam.MapTuple(_move_jackknife_sample_id_to_value)
            | 'CombineJackknifeSamplesPerSlice' >> beam.CombinePerKey(
                _JackknifeSampleCombiner(
                    num_jackknife_samples=self._num_jackknife_samples,
                    skip_ci_metric_keys=self._skip_ci_metric_keys)))
