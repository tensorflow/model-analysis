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
"""Flip rate metrics."""

import dataclasses
from typing import Optional, Iterable, List

import apache_beam as beam
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.utils import model_util

FLIP_RATE_NAME = 'flip_rate'
NEG_TO_NEG_FLIP_RATE_NAME = 'neg_to_neg_flip_rate'
NEG_TO_POS_FLIP_RATE_NAME = 'neg_to_pos_flip_rate'
POS_TO_NEG_FLIP_RATE_NAME = 'pos_to_neg_flip_rate'
POS_TO_POS_FLIP_RATE_NAME = 'pos_to_pos_flip_rate'

_DEFAULT_FLIP_RATE_THRESHOLD = 0.5


class BooleanFlipRates(metric_types.Metric):
  """FlipeRate computes the rate at which predicitons between models switch.

  Given a pair of models and a threshold for converting continuous model outputs
  into boolean predictions, this metric will produce three numbers (keyed by
  separate MetricKeys):

  - (symmetric) flip rate: The number of times the boolean predictions don't
      match, regardless of the direction of the flip.
  - negative-to-positive flip rate: The rate at which the baseline model's
      boolean prediction is negative but the candidate model's is positive.
  - positive-to-negative flip rate: The rate at which the baseline model's
      boolean prediction is positive but the candidate model's is negative.
  """

  def __init__(self,
               threshold: float = _DEFAULT_FLIP_RATE_THRESHOLD,
               flip_rate_name: str = FLIP_RATE_NAME,
               neg_to_neg_flip_rate_name: str = NEG_TO_NEG_FLIP_RATE_NAME,
               neg_to_pos_flip_rate_name: str = NEG_TO_POS_FLIP_RATE_NAME,
               pos_to_neg_flip_rate_name: str = POS_TO_NEG_FLIP_RATE_NAME,
               pos_to_pos_flip_rate_name: str = POS_TO_POS_FLIP_RATE_NAME):
    """Initializes FlipRate metric.

    Args:
      threshold: The threshold to use for converting the model prediction into a
        boolean value that can be used for comparison between models.
      flip_rate_name: Metric name for symmetric flip rate.
      neg_to_neg_flip_rate_name: Metric name for the negative-to-negative flip
        rate.
      neg_to_pos_flip_rate_name: Metric name for the negative-to-positive flip
        rate.
      pos_to_neg_flip_rate_name: Metric name for the positive-to-negative flip
        rate.
      pos_to_pos_flip_rate_name: Metric name for the positive-to-positive flip
        rate.
    """

    super().__init__(
        _boolean_flip_rates_computations,
        threshold=threshold,
        flip_rate_name=flip_rate_name,
        neg_to_neg_flip_rate_name=neg_to_neg_flip_rate_name,
        neg_to_pos_flip_rate_name=neg_to_pos_flip_rate_name,
        pos_to_neg_flip_rate_name=pos_to_neg_flip_rate_name,
        pos_to_pos_flip_rate_name=pos_to_pos_flip_rate_name)


metric_types.register_metric(BooleanFlipRates)


def _boolean_flip_rates_computations(
    threshold: float,
    flip_rate_name: str,
    neg_to_neg_flip_rate_name: str,
    neg_to_pos_flip_rate_name: str,
    pos_to_neg_flip_rate_name: str,
    pos_to_pos_flip_rate_name: str,
    eval_config: Optional[config_pb2.EvalConfig],
    model_names: Optional[List[str]],
    output_names: Optional[List[str]] = None,
    sub_keys: Optional[List[metric_types.SubKey]] = None,
    example_weighted: bool = False) -> metric_types.MetricComputations:
  """Returns metric computations for SymmetricPredictionDifference.

  This is not meant to be used with merge_per_key_computations because we
  don't want to create computations for the baseline model, and we want to
  provide the baseline model name to each Combiner

  Args:
    threshold: The threshold to use for converting both the baseline and
      candidate predictions into boolean values that can be compared.
    flip_rate_name: Metric name for symmetric flip rate.
    neg_to_neg_flip_rate_name: Metric name for the negative-to-negative flip
      rate.
    neg_to_pos_flip_rate_name: Metric name for the negative-to-positive flip
      rate.
    pos_to_neg_flip_rate_name: Metric name for the positive-to-negative flip
      rate.
    pos_to_pos_flip_rate_name: Metric name for the positive-to-positive flip
      rate.
    eval_config: The EvalConfig for this TFMA evaluation. This is used to
      identify which model is the baseline.
    model_names: The set of models for which to compute this metric.
    output_names: The set of output names for which to compute this metric.
    sub_keys: The set of sub_key settings for which to compute this metric.
    example_weighted: Whether to compute this metric using example weights.
  """
  computations = []
  baseline_spec = model_util.get_baseline_model_spec(eval_config)
  baseline_model_name = baseline_spec.name if baseline_spec else None
  for model_name in model_names:
    if model_name == baseline_model_name:
      continue
    for output_name in output_names or ['']:
      for sub_key in sub_keys or [None]:
        flip_rate_key = metric_types.MetricKey(
            name=flip_rate_name,
            model_name=model_name,
            output_name=output_name,
            sub_key=sub_key,
            example_weighted=example_weighted,
            is_diff=True)
        neg_to_neg_key = metric_types.MetricKey(
            name=neg_to_neg_flip_rate_name,
            model_name=model_name,
            output_name=output_name,
            sub_key=sub_key,
            example_weighted=example_weighted,
            is_diff=True)
        neg_to_pos_key = metric_types.MetricKey(
            name=neg_to_pos_flip_rate_name,
            model_name=model_name,
            output_name=output_name,
            sub_key=sub_key,
            example_weighted=example_weighted,
            is_diff=True)
        pos_to_neg_key = metric_types.MetricKey(
            name=pos_to_neg_flip_rate_name,
            model_name=model_name,
            output_name=output_name,
            sub_key=sub_key,
            example_weighted=example_weighted,
            is_diff=True)
        pos_to_pos_key = metric_types.MetricKey(
            name=pos_to_pos_flip_rate_name,
            model_name=model_name,
            output_name=output_name,
            sub_key=sub_key,
            example_weighted=example_weighted,
            is_diff=True)
        computations.append(
            metric_types.MetricComputation(
                keys=[
                    flip_rate_key, neg_to_neg_key, neg_to_pos_key,
                    pos_to_neg_key, pos_to_pos_key
                ],
                preprocessors=None,
                combiner=_BooleanFlipRatesCombiner(
                    threshold, eval_config, baseline_model_name, model_name,
                    output_name, flip_rate_key, neg_to_neg_key, neg_to_pos_key,
                    pos_to_neg_key, pos_to_pos_key, example_weighted)))
  return computations


@dataclasses.dataclass
class _BooleanFlipRatesAccumulator:
  """Accumulator for computing BooleanFlipRates."""
  num_weighted_examples: float = 0.0
  num_weighted_neg_to_neg: float = 0.0
  num_weighted_neg_to_pos: float = 0.0
  num_weighted_pos_to_neg: float = 0.0
  num_weighted_pos_to_pos: float = 0.0

  def merge(self, other: '_BooleanFlipRatesAccumulator'):
    self.num_weighted_examples += other.num_weighted_examples
    self.num_weighted_neg_to_neg += other.num_weighted_neg_to_neg
    self.num_weighted_neg_to_pos += other.num_weighted_neg_to_pos
    self.num_weighted_pos_to_neg += other.num_weighted_pos_to_neg
    self.num_weighted_pos_to_pos += other.num_weighted_pos_to_pos


class _BooleanFlipRatesCombiner(beam.CombineFn):
  """A combiner which computes boolean flip rate metrics."""

  def __init__(self, threshold: float, eval_config: config_pb2.EvalConfig,
               baseline_model_name: str, model_name: str, output_name: str,
               flip_rate_key: metric_types.MetricKey,
               neg_to_neg_key: metric_types.MetricKey,
               neg_to_pos_key: metric_types.MetricKey,
               pos_to_neg_key: metric_types.MetricKey,
               pos_to_pos_key: metric_types.MetricKey, example_weighted: bool):
    self._threshold = threshold
    self._eval_config = eval_config
    self._baseline_model_name = baseline_model_name
    self._model_name = model_name
    self._output_name = output_name
    self._flip_rate_key = flip_rate_key
    self._neg_to_neg_key = neg_to_neg_key
    self._neg_to_pos_key = neg_to_pos_key
    self._pos_to_neg_key = pos_to_neg_key
    self._pos_to_pos_key = pos_to_pos_key
    self._example_weighted = example_weighted

  def create_accumulator(self) -> _BooleanFlipRatesAccumulator:
    return _BooleanFlipRatesAccumulator()

  def add_input(
      self, accumulator: _BooleanFlipRatesAccumulator,
      element: metric_types.StandardMetricInputs
  ) -> _BooleanFlipRatesAccumulator:
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
            model_name=self._model_name,
            output_name=self._output_name,
            flatten=True,
            example_weighted=self._example_weighted))
    base_example_weight = base_example_weight.item()
    accumulator.num_weighted_examples += base_example_weight
    base_prediciton_bool = base_prediction > self._threshold
    model_prediction_bool = model_prediction > self._threshold
    accumulator.num_weighted_neg_to_neg += base_example_weight * int(
        not base_prediciton_bool and not model_prediction_bool)
    accumulator.num_weighted_neg_to_pos += base_example_weight * int(
        not base_prediciton_bool and model_prediction_bool)
    accumulator.num_weighted_pos_to_neg += base_example_weight * int(
        base_prediciton_bool and not model_prediction_bool)
    accumulator.num_weighted_pos_to_pos += base_example_weight * int(
        base_prediciton_bool and model_prediction_bool)
    return accumulator

  def merge_accumulators(
      self, accumulators: Iterable[_BooleanFlipRatesAccumulator]
  ) -> _BooleanFlipRatesAccumulator:
    result = next(iter(accumulators))
    for accumulator in accumulators:
      result.merge(accumulator)
    return result

  def extract_output(
      self,
      accumulator: _BooleanFlipRatesAccumulator) -> metric_types.MetricsDict:
    return {
        self._flip_rate_key: (accumulator.num_weighted_neg_to_pos +
                              accumulator.num_weighted_pos_to_neg) /
                             accumulator.num_weighted_examples,
        self._neg_to_neg_key: (accumulator.num_weighted_neg_to_neg /
                               accumulator.num_weighted_examples),
        self._neg_to_pos_key: (accumulator.num_weighted_neg_to_pos /
                               accumulator.num_weighted_examples),
        self._pos_to_neg_key: (accumulator.num_weighted_pos_to_neg /
                               accumulator.num_weighted_examples),
        self._pos_to_pos_key: (accumulator.num_weighted_pos_to_pos /
                               accumulator.num_weighted_examples),
    }
