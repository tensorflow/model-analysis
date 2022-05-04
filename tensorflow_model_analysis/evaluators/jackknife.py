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
"""Helper methods for computing jackknife std error estimates on metrics."""

from typing import Optional, Set, Tuple, TypeVar

import apache_beam as beam
import numpy as np

from tensorflow_model_analysis import types
from tensorflow_model_analysis.evaluators import confidence_intervals_util
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.slicer import slicer_lib as slicer
from tensorflow_model_analysis.utils import beam_util

_FULL_SAMPLE_ID = -1

_AccumulatorType = TypeVar('_AccumulatorType')


class _AccumulateOnlyCombineFn(beam_util.DelegatingCombineFn):
  """A combine_fn wrapper which returns the accumulator as the output value.

  This is intended to allow invoking CombineFns in settings where you might need
  to subsequently merge the results before calling extract_output. A typical use
  of a _AccumulateOnlyCombineFn might look like:

      c = OtherCombineFn()

      # combine per key, but don't call extract_output()
      accumulators = [p | beam.CombinePerKey(_AccumulateOnlyCombineFn(c)
                      for i in range(3)]

      # create two different lists of accumulator PCollections
      even_accumulators = [a for i, a in enumerate(accumulators) if i % 2 == 0]

      # extract output on the two different sets of accumulators without
      # recomputing the even accumulators
      even_output = (even_accumulators | beam.Flatten()
          | beam.CombinePerKey(_AccumulatorCombineFn(c)))
      all_output = (accumulators | beam.Flatten()
          | beam.CombinePerKey(_AccumulatorCombineFn(c)))
  """

  def extract_output(self, accumulator: _AccumulatorType) -> _AccumulatorType:
    return accumulator


class _AccumulatorCombineFn(beam_util.DelegatingCombineFn):
  """A CombineFn wrapper that takes accumulators as add_input elements.

  In combination with _AccumulateOnlyCombineFn, this makes it possible to
  operate on a CombineFn's accumulators prior to calling extract input. See the
  _AccumulateOnlyCombineFn docstring for more details.
  """

  def add_input(self, accumulator: _AccumulatorType,
                element: _AccumulatorType) -> _AccumulatorType:
    return self._combine_fn.merge_accumulators([accumulator, element])


class _JackknifeSampleCombineFn(confidence_intervals_util.SampleCombineFn):
  """Computes the jackknife standard error for each metric from samples."""

  def __init__(
      self,
      num_jackknife_samples: int,
      skip_ci_metric_keys: Optional[Set[metric_types.MetricKey]] = None):
    """Initializes a _JackknifeSampleCombineFn.

    Args:
      num_jackknife_samples: The expected number of samples computed per slice.
      skip_ci_metric_keys: Set of metric keys for which to skip confidence
        interval computation. For metric keys in this set, just the unsampled
        value will be returned.
    """
    super().__init__(
        num_samples=num_jackknife_samples,
        full_sample_id=_FULL_SAMPLE_ID,
        skip_ci_metric_keys=skip_ci_metric_keys)

  def extract_output(
      self,
      accumulator: confidence_intervals_util.SampleCombineFn.SampleAccumulator
  ) -> metric_types.MetricsDict:
    accumulator = self._validate_accumulator(accumulator)
    result = {}
    num_buckets = self._num_samples
    for key, point_estimate in accumulator.point_estimates.items():
      if key not in accumulator.metric_samples:
        result[key] = point_estimate
      else:
        # See jackknife cookie bucket method described in:
        # go/rasta-confidence-intervals
        pseudo_values = []
        total = None
        for sample_value in accumulator.metric_samples[key]:
          if total is None:
            total = sample_value
          else:
            total = total + sample_value
          pseudo_values.append(point_estimate * num_buckets - sample_value *
                               (num_buckets - 1))
        _, std_dev = confidence_intervals_util.mean_and_std(
            pseudo_values, ddof=1)
        # Here we use Student's t-distribution to estimate the standard
        # error with n - 1 degrees of freedom as S.E. = S.D. / sqrt(n)a
        # In the case of the delete-d jackknife, the standard error is inversely
        # proprotional to the square root of the number of data partitions.
        std_error = std_dev / (num_buckets**0.5)
        mean = total / num_buckets
        result[key] = types.ValueWithTDistribution(
            sample_mean=mean,
            sample_standard_deviation=std_error,
            unsampled_value=point_estimate,
            sample_degrees_of_freedom=num_buckets - 1)
    return result


def _add_sample_id(slice_key,
                   metrics_dict: metric_types.MetricsDict,
                   sample_id: int = 0):
  # sample_id has a default value in order to satisfy requirement of MapTuple
  return slice_key, confidence_intervals_util.SampleMetrics(
      metrics=metrics_dict, sample_id=sample_id)


@beam.ptransform_fn
def _ComputeJackknifeSample(  # pylint: disable=invalid-name
    sample_accumulators: beam.PCollection[
        confidence_intervals_util.SampleCombineFn.SampleAccumulator],
    sample_id: int, computations_combine_fn: beam.CombineFn,
    derived_metrics_ptransform: beam.PTransform
) -> beam.PCollection[confidence_intervals_util.SampleMetrics]:
  """Computes a single jackknife delete-d sample from partition accumulators.

  Args:
    sample_accumulators: A PCollections of combiner accumulators to be used for
      a given sample.
    sample_id: The sample_id to generate. This is used to determine which
      partition accumulators to skip.
    computations_combine_fn: a beam.CombineFn instance that takes input elements
      of type Extracts and returns a MetricsDict. This will be invoked as part
      of a CombinePerKey in which the key is a slice key.
    derived_metrics_ptransform: A PTransform which adds derived metrics to the
      results of the computations_combine_fn. This PTransform should both input
      and output a single PCollection with elements of type MetricsDict where
      the output MetricsDict includes additional derived metrics.

  Returns:
    A single sample tuple containing the sample_id and the metric dictionary for
    that sample
  """

  return (sample_accumulators
          | 'MergePartitionsPerSlice' >> beam.CombinePerKey(
              _AccumulatorCombineFn(computations_combine_fn))
          | 'AddDerivedMetrics' >> derived_metrics_ptransform
          | 'AddSampleIdToValue' >> beam.MapTuple(
              _add_sample_id, sample_id=sample_id))


@beam.ptransform_fn
def ComputeWithConfidenceIntervals(  # pylint: disable=invalid-name
    sliced_extracts: beam.pvalue.PCollection[Tuple[slicer.SliceKeyType,
                                                   types.Extracts]],
    computations_combine_fn: beam.CombineFn,
    derived_metrics_ptransform: beam.PTransform,
    num_jackknife_samples: int,
    skip_ci_metric_keys: Optional[Set[metric_types.MetricKey]] = None,
    random_seed_for_testing: Optional[int] = None) -> beam.pvalue.PCollection[
        Tuple[slicer.SliceKeyOrCrossSliceKeyType, metric_types.MetricsDict]]:
  """Computes base metrics and derived metrics and adds std error estimates.

  Args:
    sliced_extracts: Incoming PCollection consisting of slice key and extracts.
    computations_combine_fn: a beam.CombineFn instance that takes input elements
      of type Extracts and returns a MetricsDict. This will be invoked as part
      of a CombinePerKey in which the key is a slice key.
    derived_metrics_ptransform: A PTransform which adds derived metrics to the
      results of the computations_combine_fn. This PTransform should both input
      and output a single PCollection with elements of type MetricsDict where
      the output MetricsDict includes additional derived metrics.
    num_jackknife_samples: The number of jackknife replicates to use in
      computing the jackknife standard error.
    skip_ci_metric_keys: Set of metric keys for which to skip confidence
      interval computation. For metric keys in this set, just the unsampled
      value will be returned.
    random_seed_for_testing: Seed to use for unit testing, because
      nondeterministic tests stink. Each partition will use this value + i.

  Returns:
    A PCollection of sliced metrics containing standard error estimates for
    each numeric metric.
  """

  random_state = np.random.RandomState(random_seed_for_testing)

  def partition_fn(_, num_partitions):
    return random_state.randint(num_partitions)

  # Partition the data
  # List[PCollection[Tuple[slicer.SliceKeyType, types.Extracts]]]
  partitions = (
      sliced_extracts
      | f'Partition({num_jackknife_samples})' >> beam.Partition(
          partition_fn, num_jackknife_samples))

  # Within each partition, partially combine per slice key to get accumulators
  # and partition sizes; add partition_id for determinism.
  # List[PCollection[Tuple[slicer.SliceKeyType, AccumulatorType]]]
  partition_accumulators = []
  for i, partition in enumerate(partitions):
    partition_accumulators.append(
        partition
        | f'CombinePartitionPerSlice[{i}]' >> beam.CombinePerKey(
            _AccumulateOnlyCombineFn(computations_combine_fn)))

  unsampled_metrics = (
      partition_accumulators
      | 'FlattenPartitions' >> beam.Flatten()
      | 'MergePartitionsPerSlice' >> beam.CombinePerKey(
          _AccumulatorCombineFn(computations_combine_fn))
      | 'AddDerivedMetrics' >> derived_metrics_ptransform
      | 'AddSampleIdToValue' >> beam.MapTuple(
          _add_sample_id, sample_id=_FULL_SAMPLE_ID))

  # Compute the combine_fn output for the delete-d samples by merging all but
  # one partitions.
  # List[PCollection[Tuple[slicer.SliceKeyType, SampleMetrics]]]
  delete_d_samples = []
  for sample_id in range(num_jackknife_samples):
    # TODO(b/194732335): Push filter and Flatten into _ComputeJackknifeSample.
    # TODO(b/130032676): Replace the 'ExcludePartition' step with for-loop
    # exclusion after cl/435922775 (or equivalent) is submitted.
    sample_accumulators = [
        acc | f'ExcludePartition[{sample_id}]' >> beam.Filter(lambda _: False)
        if i == sample_id else acc
        for i, acc in enumerate(partition_accumulators)
    ]
    delete_d_samples.append(
        sample_accumulators
        | f'FlattenPartitions[{sample_id}]' >> beam.Flatten()
        | f'ComputeJackknifeSample[{sample_id}]' >> _ComputeJackknifeSample(  # pylint: disable=no-value-for-parameter
            sample_id=sample_id,
            computations_combine_fn=computations_combine_fn,
            derived_metrics_ptransform=derived_metrics_ptransform))

  # PCollection[Tuple[slicer.SliceKeyType, metric_types.MetricsDict]]
  return (delete_d_samples + [unsampled_metrics]
          | 'FlattenSamples' >> beam.Flatten()
          | 'CombineJackknifeSamplesPerSlice' >> beam.CombinePerKey(
              _JackknifeSampleCombineFn(num_jackknife_samples,
                                        skip_ci_metric_keys)))
