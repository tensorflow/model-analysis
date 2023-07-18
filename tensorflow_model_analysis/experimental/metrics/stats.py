# Copyright 2023 Google LLC
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
"""Stats metrics."""

import dataclasses
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import apache_beam as beam
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.utils import util


_KeyPath = Tuple[str]

_MEAN_METRIC_BASE_NAME = 'mean'


# TODO(b/287700355): Add __slots__ to _Accumulator
@dataclasses.dataclass
class _Accumulator:
  count: float = 0.0
  total: float = 0.0


class _MeanCombiner(beam.CombineFn):
  """Computes mean metric."""

  def __init__(
      self,
      key: metric_types.MetricKey,
      feature_key_path: _KeyPath,
      example_weights_key_path: Optional[_KeyPath],
  ):
    self._key = key
    self._feature_key_path = feature_key_path
    self._example_weights_key_path = example_weights_key_path

  def create_accumulator(self) -> _Accumulator:
    return _Accumulator()

  def add_input(
      self,
      accumulator: _Accumulator,
      element: metric_types.StandardMetricInputs,
  ) -> _Accumulator:
    # Get feature value
    features = util.get_by_keys(element, self._feature_key_path)
    assert len(features) == 1, (
        'Mean() is only supported for scalar features, but found features = '
        f'{features}'
    )

    # Get example weight
    if self._example_weights_key_path is None:
      example_weight = 1.0
    else:
      example_weights = util.get_by_keys(
          element, self._example_weights_key_path
      )
      assert len(example_weights) == 1, (
          'Expected 1 (scalar) example weight for each example, '
          f'but found example weight = {example_weights}'
      )
      example_weight = example_weights[0]

    # Update accumulator
    accumulator.count += example_weight
    accumulator.total += example_weight * features[0]

    return accumulator

  def merge_accumulators(
      self, accumulators: Iterable[_Accumulator]
  ) -> _Accumulator:
    accumulators = iter(accumulators)
    result = next(accumulators)
    for accumulator in accumulators:
      result.count += accumulator.count
      result.total += accumulator.total
    return result

  def extract_output(
      self, accumulator: _Accumulator
  ) -> Dict[metric_types.MetricKey, float]:
    if accumulator.count == 0.0:
      return {self._key: float('nan')}
    return {self._key: accumulator.total / accumulator.count}


def _convert_key_path_to_dict(
    key_path: Union[_KeyPath, Tuple[()]]
) -> Dict[str, Any]:
  """Recursively converts _KeyPath to nested dict."""
  return (
      {key_path[0]: _convert_key_path_to_dict(key_path[1:])} if key_path else {}
  )


def _mean_metric(
    feature_key_path: _KeyPath,
    example_weights_key_path: Optional[_KeyPath],
    name: Optional[str],
) -> metric_types.MetricComputations:
  """Returns metric computation for mean metric."""
  if name:
    key_name = name
  else:
    key_name = f"{_MEAN_METRIC_BASE_NAME}_{'.'.join(feature_key_path)}"
  key = metric_types.MetricKey(
      name=key_name, example_weighted=example_weights_key_path is not None
  )

  include_filter = _convert_key_path_to_dict(feature_key_path)
  if example_weights_key_path:
    include_filter = util.merge_filters(
        include_filter,
        _convert_key_path_to_dict(example_weights_key_path),
    )

  return [
      metric_types.MetricComputation(
          keys=[key],
          preprocessors=[
              metric_types.StandardMetricInputsPreprocessor(
                  include_filter=include_filter,
                  include_default_inputs=False,
              )
          ],
          combiner=_MeanCombiner(
              key, feature_key_path, example_weights_key_path
          ),
      )
  ]


class Mean(metric_types.Metric):
  """Mean metric."""

  def __init__(
      self,
      feature_key_path: _KeyPath,
      example_weights_key_path: Optional[_KeyPath] = None,
      name: Optional[str] = None,
  ):
    """Initializes mean metric.

    Args:
      feature_key_path: key path to feature to calculate the mean of.
      example_weights_key_path: key path to example weights.
      name: Metric base name.
    """
    super().__init__(
        _mean_metric,
        feature_key_path=feature_key_path,
        example_weights_key_path=example_weights_key_path,
        name=name,
    )


metric_types.register_metric(Mean)
