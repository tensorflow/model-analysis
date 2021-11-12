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

from typing import Optional, Dict, Iterable, List

import apache_beam as beam
import numpy as np
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.utils import util

EXAMPLE_COUNT_NAME = 'example_count'


class ExampleCount(metric_types.Metric):
  """Example count.

  Note that although the example_count is independent of the model, this metric
  will be associated with a model for consistency with other metrics.
  """

  def __init__(self, name: str = EXAMPLE_COUNT_NAME):
    """Initializes example count.

    Args:
      name: Metric name.
    """

    super().__init__(example_count, name=name)

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


def example_count(
    name: str = EXAMPLE_COUNT_NAME,
    model_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    sub_keys: Optional[List[metric_types.SubKey]] = None,
    example_weighted: bool = False) -> metric_types.MetricComputations:
  """Returns metric computations for example count."""
  computations = []
  for model_name in model_names or ['']:
    for output_name in output_names or ['']:
      keys = []
      for sub_key in sub_keys or [None]:
        key = metric_types.MetricKey(
            name=name,
            model_name=model_name,
            output_name=output_name,
            sub_key=sub_key,
            example_weighted=example_weighted)
        keys.append(key)

      # Note: This cannot be implemented based on the weight stored in
      # calibration because weighted example count is used with multi-class, etc
      # models that do not use calibration metrics.
      computations.append(
          metric_types.MetricComputation(
              keys=keys,
              preprocessor=None,
              combiner=_ExampleCountCombiner(model_name, output_name, keys,
                                             example_weighted)))
  return computations


class _ExampleCountCombiner(beam.CombineFn):
  """Computes example count."""

  def __init__(self, model_name: str, output_name: str,
               keys: List[metric_types.MetricKey], example_weighted):
    self._model_name = model_name
    self._output_name = output_name
    self._keys = keys
    self._example_weighted = example_weighted

  def create_accumulator(self) -> float:
    return 0.0

  def add_input(self, accumulator: float,
                element: metric_types.StandardMetricInputs) -> float:
    if not self._example_weighted or element.example_weight is None:
      example_weight = np.array(1.0)
    else:
      example_weight = element.example_weight
    if isinstance(example_weight, dict) and self._model_name:
      value = util.get_by_keys(
          example_weight, [self._model_name], optional=True)
      if value is not None:
        example_weight = value
    if isinstance(example_weight, dict) and self._output_name:
      example_weight = util.get_by_keys(example_weight, [self._output_name],
                                        np.array(1.0))
    if isinstance(example_weight, dict):
      raise ValueError(
          f'example_count cannot be calculated on a dict {example_weight}: '
          f'model_name={self._model_name}, output_name={self._output_name}.\n\n'
          'This is most likely a configuration error (for multi-output models'
          'a separate metric is needed for each output).')
    return accumulator + np.sum(example_weight)

  def merge_accumulators(self, accumulators: Iterable[float]) -> float:
    result = 0.0
    for accumulator in accumulators:
      result += accumulator
    return result

  def extract_output(self,
                     accumulator: float) -> Dict[metric_types.MetricKey, float]:
    return {k: accumulator for k in self._keys}
