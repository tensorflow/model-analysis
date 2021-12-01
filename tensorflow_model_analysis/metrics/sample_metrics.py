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
"""A collection of metrics which sample per-example values."""

from typing import Any, List, Optional, Text, Tuple

import apache_beam as beam
import numpy as np
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.utils import beam_util
from tensorflow_model_analysis.utils import util

FIXED_SIZE_SAMPLE_NAME = 'fixed_size_sample'

# This corresponds to the comments in apache_beam/transforms/combiners.py
_HeapType = Tuple[bool, List[Any]]


class FixedSizeSample(metric_types.Metric):
  """Computes a fixed-size sample per slice."""

  def __init__(self,
               sampled_key: Text,
               size: int,
               name: Text = FIXED_SIZE_SAMPLE_NAME,
               random_seed: Optional[int] = None):
    """Initializes a FixedSizeSample metric.

    Args:
      sampled_key: The key whose values should be sampled
      size: The number of samples to collect (per slice)
      name: Metric name.
      random_seed: The random_seed to be used for intializing the per worker
        np.random.RandomGenerator in the CombineFn setup. Note that when more
        than one worker is used, setting this is not sufficient to guarantee
        determinism.
    """
    super().__init__(
        _fixed_size_sample,
        sampled_key=sampled_key,
        size=size,
        name=name,
        random_seed=random_seed)


metric_types.register_metric(FixedSizeSample)


def _fixed_size_sample(
    sampled_key: Text,
    size: int,
    name: Text,
    random_seed: Optional[int],
    model_names: Optional[List[Text]] = None,
    output_names: Optional[List[Text]] = None,
    sub_keys: Optional[List[metric_types.SubKey]] = None,
    example_weighted: bool = False) -> metric_types.MetricComputations:
  """Returns metrics computations for FixedSizeSample metrcs."""
  keys = []
  for model_name in model_names or ['']:
    for output_name in output_names or ['']:
      for sub_key in sub_keys or [None]:
        keys.append(
            metric_types.MetricKey(
                name,
                model_name=model_name,
                output_name=output_name,
                sub_key=sub_key,
                example_weighted=example_weighted))
  return [
      metric_types.MetricComputation(
          keys=keys,
          preprocessor=metric_types.FeaturePreprocessor(
              feature_keys=[sampled_key]),
          combiner=_FixedSizeSampleCombineFn(
              metric_keys=keys,
              sampled_key=sampled_key,
              size=size,
              example_weighted=example_weighted,
              random_seed=random_seed))
  ]


class _FixedSizeSampleCombineFn(beam_util.DelegatingCombineFn):
  """A fixed size sample combiner which samples values of a specified key.

  This CombineFn is similar to beam.combiners.SampleCombineFn except it makes
  use of the numpy random generator which means that it accepts a seed for use
  with deterministic testing.
  """

  def __init__(self, metric_keys: List[metric_types.MetricKey],
               sampled_key: Text, size: int, example_weighted: bool,
               random_seed: Optional[int]):
    self._metric_keys = metric_keys
    self._sampled_key = sampled_key
    self._example_weighted = example_weighted
    self._random_seed = random_seed
    # We delegate to the TopCombineFn rather than subclass because the use of a
    # TopCombineFn is an implementation detail.
    super().__init__(combine_fn=beam.combiners.TopCombineFn(n=size))

  def setup(self):
    self._random_generator = np.random.default_rng(self._random_seed)

  def add_input(self, heap: _HeapType,
                element: metric_types.StandardMetricInputs) -> _HeapType:
    # TODO(b/206546545): add support for sampling derived features
    sampled_value = util.get_by_keys(element.features, [self._sampled_key])
    random_tag = self._random_generator.random()
    if self._example_weighted:
      # For details, see Weighted Random Sampling over Data Streams:
      # https://arxiv.org/abs/1012.0256
      weight = element.example_weight
      random_tag = random_tag**(1 / weight)
    return super().add_input(heap, (random_tag, sampled_value))

  def extract_output(self, heap: _HeapType) -> metric_types.MetricsDict:
    # drop random numbers used for sampling
    sampled_values = np.array([v for _, v in super().extract_output(heap)])
    return {k: sampled_values for k in self._metric_keys}
