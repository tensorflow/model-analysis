# Copyright 2018 Google LLC
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
"""Utils for performing poisson bootstrapping."""

from typing import Any, Optional, Set, Tuple, TypeVar

import apache_beam as beam
import numpy as np

from tensorflow_model_analysis import types
from tensorflow_model_analysis.evaluators import confidence_intervals_util
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.slicer import slicer_lib as slicer
from tensorflow_model_analysis.utils import beam_util

DEFAULT_NUM_BOOTSTRAP_SAMPLES = 20
_FULL_SAMPLE_ID = -1

_AccumulatorType = TypeVar('_AccumulatorType')


class _BootstrapCombineFn(beam_util.DelegatingCombineFn):
  """CombineFn wrapper which adds poisson resampling to input elements."""

  def __init__(self,
               combine_fn: beam.CombineFn,
               random_seed: Optional[int] = None):
    super().__init__(combine_fn)
    self._random_seed = random_seed

  def setup(self):
    super().setup()
    self._random_state = np.random.RandomState(self._random_seed)

  def add_input(self, accumulator: _AccumulatorType,
                element: Any) -> _AccumulatorType:
    for sampled_element in [element] * int(self._random_state.poisson(1, 1)):
      accumulator = self._combine_fn.add_input(accumulator, sampled_element)
    return accumulator


def _add_sample_id(  # pylint: disable=invalid-name
    slice_key,
    metrics_dict: metric_types.MetricsDict,
    sample_id: int = 0):
  # sample_id has a default value in order to satisfy requirement of MapTuple
  return slice_key, confidence_intervals_util.SampleMetrics(
      metrics=metrics_dict, sample_id=sample_id)


@beam.ptransform_fn
def _ComputeBootstrapSample(  # pylint disable=invalid-name
    sliced_extracts: beam.pvalue.PCollection[Tuple[slicer.SliceKeyType,
                                                   types.Extracts]],
    sample_id: int, computations_combine_fn: beam.CombineFn,
    derived_metrics_ptransform: beam.PTransform, seed: int, hot_key_fanout: int
) -> beam.PCollection[confidence_intervals_util.SampleMetrics]:
  """Computes a single bootstrap sample from SlicedExtracts.

  Args:
    sliced_extracts: Incoming PCollection consisting of slice key and extracts.
    sample_id: The sample_id to attach to the computed metrics as part of the
      returned SampleMetrics objects.
    computations_combine_fn: a beam.CombineFn instance that takes input elements
      of type Extracts and returns a MetricsDict. This will be invoked as part
      of a CombinePerKey in which the key is a slice key.
    derived_metrics_ptransform: A PTransform which adds derived metrics to the
      results of the computations_combine_fn. This PTransform should both input
      and output a single PCollection with elements of type MetricsDict where
      the output MetricsDict includes additional derived metrics.
    seed: The seed to use when doing resampling. Note that this is only useful
      in testing or when using a single worker, as otherwise Beam will introduce
      non-determinism in when using distributed computation.
    hot_key_fanout: The hot key fanout factor to use when calling
      beam.CombinePerKey with the computations_combine_fn on replicates. Note
      that these replicates will in expectation have the same size as the input
      PCollection of extracts and will use the normal set of slices keys.

  Returns:
    A PCollection of sliced SampleMetrics objects, containing the metrics dicts
    for a given slice, computed from the resampled extracts, along with the
    provided sample_id.
  """
  return (
      sliced_extracts
      | 'CombineSampledMetricsPerSlice' >> beam.CombinePerKey(
          _BootstrapCombineFn(computations_combine_fn,
                              seed)).with_hot_key_fanout(hot_key_fanout)
      |
      'AddSampledDerivedCrossSliceAndDiffMetrics' >> derived_metrics_ptransform
      | 'AddSampleIdToValue' >> beam.MapTuple(
          _add_sample_id, sample_id=sample_id))


class _BootstrapSampleCombineFn(confidence_intervals_util.SampleCombineFn):
  """Computes the bootstrap standard error for each metric from samples."""

  def __init__(
      self,
      num_bootstrap_samples: int,
      skip_ci_metric_keys: Optional[Set[metric_types.MetricKey]] = None):
    """Initializes a _BootstrapSampleCombineFn.

    Args:
      num_bootstrap_samples: The expected number of samples computed per slice.
      skip_ci_metric_keys: Set of metric keys for which to skip confidence
        interval computation. For metric keys in this set, just the point
        estimate will be returned.
    """
    super().__init__(
        num_samples=num_bootstrap_samples,
        full_sample_id=_FULL_SAMPLE_ID,
        skip_ci_metric_keys=skip_ci_metric_keys)

  def extract_output(
      self,
      accumulator: confidence_intervals_util.SampleCombineFn.SampleAccumulator
  ) -> metric_types.MetricsDict:
    accumulator = self._validate_accumulator(accumulator)
    result = {}
    dof = self._num_samples - 1
    for key, point_estimate in accumulator.point_estimates.items():
      if key not in accumulator.metric_samples:
        result[key] = point_estimate
      else:
        mean, std_error = confidence_intervals_util.mean_and_std(
            accumulator.metric_samples[key], ddof=1)
        result[key] = types.ValueWithTDistribution(
            sample_mean=mean,
            sample_standard_deviation=std_error,
            unsampled_value=point_estimate,
            sample_degrees_of_freedom=dof)
    return result


@beam.ptransform_fn
def ComputeWithConfidenceIntervals(  # pylint: disable=invalid-name
    sliced_extracts: beam.pvalue.PCollection[Tuple[slicer.SliceKeyType,
                                                   types.Extracts]],
    computations_combine_fn: beam.CombineFn,
    derived_metrics_ptransform: beam.PTransform,
    num_bootstrap_samples: int,
    hot_key_fanout: Optional[int] = None,
    skip_ci_metric_keys: Optional[Set[metric_types.MetricKey]] = None,
    random_seed_for_testing: Optional[int] = None) -> beam.pvalue.PCollection[
        Tuple[slicer.SliceKeyOrCrossSliceKeyType, metric_types.MetricsDict]]:
  """PTransform for computing metrics using T-Distribution values.

  Args:
    sliced_extracts: Incoming PCollection consisting of slice key and extracts.
    computations_combine_fn: a beam.CombineFn instance that takes input elements
      of type Extracts and returns a MetricsDict. This will be invoked as part
      of a CombinePerKey in which the key is a slice key.
    derived_metrics_ptransform: A PTransform which adds derived metrics to the
      results of the computations_combine_fn. This PTransform should both input
      and output a single PCollection with elements of type MetricsDict where
      the output MetricsDict includes additional derived metrics.
    num_bootstrap_samples: The number of bootstrap replicates to use in
      computing the bootstrap standard error.
    hot_key_fanout: The hot key fanout factor to use when calling
      beam.CombinePerKey with the computations_combine_fn on replicates. Note
      that these replicates will in expectation have the same size as the input
      PCollection of extracts and will use the normal set of slices keys.
    skip_ci_metric_keys: Set of metric keys for which to skip confidence
      interval computation. For metric keys in this set, just the unsampled
      value will be returned.
    random_seed_for_testing: Seed to use for unit testing, because
      nondeterministic tests stink. Each partition will use this value + i.

  Returns:
    PCollection of (slice key, dict of metrics)
  """
  if num_bootstrap_samples < 1:
    raise ValueError('num_bootstrap_samples should be > 0, got %d' %
                     num_bootstrap_samples)

  unsampled_metrics = (
      sliced_extracts
      | 'CombineUnsampledMetricsPerSlice' >> beam.CombinePerKey(
          computations_combine_fn).with_hot_key_fanout(hot_key_fanout)
      | 'AddDerivedMetrics' >> derived_metrics_ptransform
      |
      'AddUnsampledSampleId' >> beam.MapTuple(_add_sample_id, _FULL_SAMPLE_ID))

  sampled_metrics = []
  for sample_id in range(num_bootstrap_samples):
    seed = (None if random_seed_for_testing is None else
            random_seed_for_testing + sample_id)
    sampled_metrics.append(
        sliced_extracts
        | f'ComputeBootstrapSample[{sample_id}]' >> _ComputeBootstrapSample(  # pylint: disable=no-value-for-parameter
            sample_id=sample_id,
            computations_combine_fn=computations_combine_fn,
            derived_metrics_ptransform=derived_metrics_ptransform,
            seed=seed,
            hot_key_fanout=hot_key_fanout))

  return (sampled_metrics + [unsampled_metrics]
          | 'FlattenBootstrapPartitions' >> beam.Flatten()
          | 'CombineSamplesPerSlice' >> beam.CombinePerKey(
              _BootstrapSampleCombineFn(
                  num_bootstrap_samples=num_bootstrap_samples,
                  skip_ci_metric_keys=skip_ci_metric_keys)))
