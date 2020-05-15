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
"""Example count metric."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

from typing import Optional, Dict, Iterable, List, Text

import apache_beam as beam
from tensorflow_model_analysis import types
from tensorflow_model_analysis.metrics import metric_types

EXAMPLE_COUNT_NAME = 'example_count'


class ExampleCount(metric_types.Metric):
  """Example count.

  Note that although the example_count is independent of the model, this metric
  will be associated with a model for consistency with other metrics.
  """

  def __init__(self, name: Text = EXAMPLE_COUNT_NAME):
    """Initializes example count.

    Args:
      name: Metric name.
    """

    super(ExampleCount, self).__init__(_example_count, name=name)

  @property
  def compute_confidence_interval(self) -> bool:
    """Always disable confidence intervals for ExampleCount.

    Confidence intervals capture uncertainty in a metric if it were computed on
    more examples. For ExampleCount, this sort of uncertainty is not meaningful,
    so confidence intervals are disabled.

    Returns:
      Whether to compute confidence intervals.
    """
    return False


metric_types.register_metric(ExampleCount)


def _example_count(
    name: Text = EXAMPLE_COUNT_NAME,
    model_names: Optional[List[Text]] = None,
    output_names: Optional[List[Text]] = None,
    sub_keys: Optional[List[metric_types.SubKey]] = None
) -> metric_types.MetricComputations:
  """Returns metric computations for computing example counts."""
  keys = []
  for model_name in model_names or ['']:
    for output_name in output_names or ['']:
      for sub_key in sub_keys or [None]:
        key = metric_types.MetricKey(
            name=name,
            model_name=model_name,
            output_name=output_name,
            sub_key=sub_key)
        keys.append(key)
  return [
      metric_types.MetricComputation(
          keys=keys,
          preprocessor=_ExampleCountPreprocessor(),
          combiner=_ExampleCountCombiner(keys))
  ]


class _ExampleCountPreprocessor(beam.DoFn):
  """Computes example count."""

  def process(self, extracts: types.Extracts) -> Iterable[int]:
    yield 1


class _ExampleCountCombiner(beam.CombineFn):
  """Computes example count."""

  def __init__(self, metric_keys: List[metric_types.MetricKey]):
    self._metric_keys = metric_keys

  def create_accumulator(self) -> int:
    return 0

  def add_input(self, accumulator: int, state: int) -> int:
    return accumulator + state

  def merge_accumulators(self, accumulators: List[int]) -> int:
    result = 0
    for accumulator in accumulators:
      result += accumulator
    return result

  def extract_output(self,
                     accumulator: int) -> Dict[metric_types.MetricKey, int]:
    return {k: accumulator for k in self._metric_keys}
