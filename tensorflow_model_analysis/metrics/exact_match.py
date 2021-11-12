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
"""Exact match metric."""
import json
from typing import Dict, Iterable, Optional

import apache_beam as beam
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.proto import config_pb2

EXACT_MATCH_NAME = 'exact_match'
_JSON = 'json'
_CONVERT_TO_VALUES = frozenset([_JSON])


class ExactMatch(metric_types.Metric):
  """Exact Match Metric."""

  def __init__(self,
               name: str = EXACT_MATCH_NAME,
               convert_to: Optional[str] = None):
    """Initializes exact match metric.

    Args:
      name: The name of the metric to use.
      convert_to: The conversion to perform before checking equality.
    """

    super().__init__(
        metric_util.merge_per_key_computations(_exact_match),
        name=name,
        convert_to=convert_to)
    if convert_to and convert_to not in _CONVERT_TO_VALUES:
      raise ValueError('convert_to can only be one of the following: %s' %
                       str(convert_to))


metric_types.register_metric(ExactMatch)


def _exact_match(
    name: str,
    eval_config: Optional[config_pb2.EvalConfig] = None,
    model_name: str = '',
    output_name: str = '',
    sub_key: Optional[metric_types.SubKey] = None,
    aggregation_type: Optional[metric_types.AggregationType] = None,
    class_weights: Optional[Dict[int, float]] = None,
    example_weighted: bool = False,
    convert_to: Optional[str] = None) -> metric_types.MetricComputations:
  """Returns metric computations for computing the exact match score."""
  key = metric_types.MetricKey(
      name=name,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key,
      example_weighted=example_weighted)
  return [
      metric_types.MetricComputation(
          keys=[key],
          preprocessor=None,
          combiner=_ExactMatchCombiner(key, eval_config, aggregation_type,
                                       class_weights, example_weighted,
                                       convert_to))
  ]


class _ExactMatchAccumulator:
  """Exact match accumulator."""
  __slots__ = ['total_weighted_exact_match_scores', 'total_weighted_examples']

  def __init__(self):
    self.total_weighted_exact_match_scores = 0.0
    self.total_weighted_examples = 0.0

  def __iadd__(self, other):
    self.total_weighted_exact_match_scores += other.total_weighted_exact_match_scores
    self.total_weighted_examples += other.total_weighted_examples
    return self


class _ExactMatchCombiner(beam.CombineFn):
  """Combines Exact Match scores."""

  def __init__(self, key: metric_types.MetricKey,
               eval_config: Optional[config_pb2.EvalConfig],
               aggregation_type: Optional[metric_types.AggregationType],
               class_weights: Optional[Dict[int, float]],
               exampled_weighted: bool, convert_to: Optional[str]):
    self._key = key
    self._eval_config = eval_config
    self._aggregation_type = aggregation_type
    self._class_weights = class_weights
    self._example_weighted = exampled_weighted
    self._convert_to = convert_to

  def create_accumulator(self) -> _ExactMatchAccumulator:
    return _ExactMatchAccumulator()

  def add_input(
      self, accumulator: _ExactMatchAccumulator,
      element: metric_types.StandardMetricInputs) -> _ExactMatchAccumulator:
    for label, prediction, example_weight in (
        metric_util.to_label_prediction_example_weight(
            element,
            eval_config=self._eval_config,
            model_name=self._key.model_name,
            output_name=self._key.output_name,
            aggregation_type=self._aggregation_type,
            class_weights=self._class_weights,
            example_weighted=self._example_weighted)):
      label = label.tolist()
      prediction = prediction.tolist()
      if self._convert_to == _JSON:
        label = [json.loads(l) for l in label]
        prediction = [json.loads(p) for p in prediction]
      match = [p == l for p, l in zip(prediction, label)]
      score = int(all(match))
      example_weight = example_weight.item()
      accumulator.total_weighted_exact_match_scores += score * example_weight
      accumulator.total_weighted_examples += example_weight
    return accumulator

  def merge_accumulators(
      self,
      accumulators: Iterable[_ExactMatchAccumulator]) -> _ExactMatchAccumulator:
    accumulators = iter(accumulators)
    result = next(accumulators)
    for accumulator in accumulators:
      result += accumulator
    return result

  def extract_output(
      self, accumulator: _ExactMatchAccumulator
  ) -> Dict[metric_types.MetricKey, float]:
    score = accumulator.total_weighted_exact_match_scores / accumulator.total_weighted_examples
    return {self._key: score}
