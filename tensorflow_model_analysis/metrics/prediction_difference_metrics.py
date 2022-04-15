# Copyright 2022 Google LLC
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
"""PredictionDifference metrics."""

import dataclasses
from typing import Optional, Dict, Iterable, List

import apache_beam as beam
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.utils import model_util

SYMMETRIC_PREDICITON_DIFFERENCE_NAME = 'symmetric_prediction_difference'
_K_EPSILON = 1e-7


class SymmetricPredictionDifference(metric_types.Metric):
  """PredictionDifference computes the avg pointwise diff between models."""

  def __init__(self, name: str = SYMMETRIC_PREDICITON_DIFFERENCE_NAME):
    """Initializes PredictionDifference metric.

    Args:
      name: Metric name.
    """

    super().__init__(symmetric_prediction_difference_computations, name=name)


metric_types.register_metric(SymmetricPredictionDifference)


def symmetric_prediction_difference_computations(
    name: str = SYMMETRIC_PREDICITON_DIFFERENCE_NAME,
    eval_config: Optional[config_pb2.EvalConfig] = None,
    model_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    sub_keys: Optional[List[metric_types.SubKey]] = None,
    example_weighted: bool = False) -> metric_types.MetricComputations:
  """Returns metric computations for SymmetricPredictionDifference.

  This is not meant to be used with merge_per_key_computations because we
  don't want to create computations for the baseline model, and we want to
  provide the baseline model name to each Combiner

  Args:
    name: The name of the metric returned by the computations.
    eval_config: The EvalConfig for this TFMA evaluation.
    model_names: The set of models for which to compute this metric.
    output_names: The set of output names for which to compute this metric.
    sub_keys: The set of sub_key settings for which to compute this metric.
    example_weighted: Whether to compute this metric using example weights.
  """
  computations = []
  baseline_spec = model_util.get_baseline_model_spec(eval_config)
  baseline_model_name = baseline_spec.name if baseline_spec else None
  for model_name in model_names or ['']:
    if model_name == baseline_model_name:
      continue
    for output_name in output_names or ['']:
      for sub_key in sub_keys or [None]:
        key = metric_types.MetricKey(
            name=name,
            model_name=model_name,
            output_name=output_name,
            sub_key=sub_key,
            example_weighted=example_weighted,
            is_diff=True)
        computations.append(
            metric_types.MetricComputation(
                keys=[key],
                preprocessor=None,
                combiner=_SymmetricPredictionDifferenceCombiner(
                    eval_config, baseline_model_name, model_name, output_name,
                    key, example_weighted)))
  return computations


@dataclasses.dataclass
class SymmetricPredictionDifferenceAccumulator:
  num_weighted_examples: float = 0.0
  total_pointwise_sym_diff: float = 0.0

  def merge(self, other: 'SymmetricPredictionDifferenceAccumulator'):
    self.num_weighted_examples += other.num_weighted_examples
    self.total_pointwise_sym_diff += other.total_pointwise_sym_diff


class _SymmetricPredictionDifferenceCombiner(beam.CombineFn):
  """Computes PredictionDifference."""

  def __init__(self, eval_config: config_pb2.EvalConfig,
               baseline_model_name: str, model_name: str, output_name: str,
               key: metric_types.MetricKey, example_weighted: bool):
    self._eval_config = eval_config
    self._baseline_model_name = baseline_model_name
    self._model_name = model_name
    self._output_name = output_name
    self._key = key
    self._example_weighted = example_weighted

  def create_accumulator(self) -> SymmetricPredictionDifferenceAccumulator:
    return SymmetricPredictionDifferenceAccumulator()

  def add_input(
      self, accumulator: SymmetricPredictionDifferenceAccumulator,
      element: metric_types.StandardMetricInputs
  ) -> SymmetricPredictionDifferenceAccumulator:

    _, base_prediction, base_example_weight = next(
        metric_util.to_label_prediction_example_weight(
            element,
            eval_config=self._eval_config,
            model_name=self._baseline_model_name,
            output_name=self._output_name,
            flatten=True,
            example_weighted=self._example_weighted))

    _, model_prediction, _ = next(
        metric_util.to_label_prediction_example_weight(
            element,
            eval_config=self._eval_config,
            model_name=self._key.model_name,
            output_name=self._output_name,
            flatten=True,
            example_weighted=self._example_weighted))

    accumulator.num_weighted_examples += base_example_weight
    numerator = 2 * abs(base_prediction - model_prediction)
    denominator = abs(base_prediction + model_prediction)
    if numerator < _K_EPSILON and denominator < _K_EPSILON:
      sym_pd = 0.0
    else:
      sym_pd = numerator / denominator
    accumulator.total_pointwise_sym_diff += sym_pd * base_example_weight
    return accumulator

  def merge_accumulators(
      self, accumulators: Iterable[SymmetricPredictionDifferenceAccumulator]
  ) -> SymmetricPredictionDifferenceAccumulator:
    result = next(iter(accumulators))
    for accumulator in accumulators:
      result.merge(accumulator)
    return result

  def extract_output(
      self, accumulator: SymmetricPredictionDifferenceAccumulator
  ) -> Dict[metric_types.MetricKey, float]:
    return {
        self._key:
            accumulator.total_pointwise_sym_diff /
            accumulator.num_weighted_examples
    }
