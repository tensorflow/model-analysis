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

import abc
from collections.abc import Iterable
import dataclasses
import functools
from typing import Any, Callable, Optional

import apache_beam as beam
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.utils import model_util


# Flip Metrics Names
FLIP_RATE_NAME = 'flip_rate'  # Symmetric Flip Rate Name in BooleanFlipRates().
SYMMETRIC_FLIP_RATE_NAME = 'symmetric_flip_rate'  # In SymmetricFlipRates().
NEG_TO_NEG_FLIP_RATE_NAME = 'neg_to_neg_flip_rate'
NEG_TO_POS_FLIP_RATE_NAME = 'neg_to_pos_flip_rate'
POS_TO_NEG_FLIP_RATE_NAME = 'pos_to_neg_flip_rate'
POS_TO_POS_FLIP_RATE_NAME = 'pos_to_pos_flip_rate'

_FLIP_COUNTS_BASE_NAME = '_flip_counts'  # flip_counts_computation name.


_DEFAULT_FLIP_RATE_THRESHOLD = 0.5


@dataclasses.dataclass
class _BooleanFlipCountsAccumulator:
  """Accumulator for computing BooleanFlipRates."""

  num_weighted_examples: float = 0.0
  num_weighted_neg_to_neg: float = 0.0
  num_weighted_neg_to_pos: float = 0.0
  num_weighted_pos_to_neg: float = 0.0
  num_weighted_pos_to_pos: float = 0.0

  def merge(self, other: '_BooleanFlipCountsAccumulator'):
    self.num_weighted_examples += other.num_weighted_examples
    self.num_weighted_neg_to_neg += other.num_weighted_neg_to_neg
    self.num_weighted_neg_to_pos += other.num_weighted_neg_to_pos
    self.num_weighted_pos_to_neg += other.num_weighted_pos_to_neg
    self.num_weighted_pos_to_pos += other.num_weighted_pos_to_pos


class _BooleanFlipCountsCombiner(beam.CombineFn):
  """A combiner that computes the counts needed to calculate the Flip Rates."""

  def __init__(
      self,
      key: metric_types.MetricKey,
      eval_config: config_pb2.EvalConfig,
      baseline_model_name: str,
      model_name: str,
      output_name: str,
      example_weighted: bool,
      threshold: float,
  ):
    self._key = key
    self._eval_config = eval_config
    self._baseline_model_name = baseline_model_name
    self._model_name = model_name
    self._output_name = output_name
    self._example_weighted = example_weighted
    self._threshold = threshold

  def create_accumulator(self) -> _BooleanFlipCountsAccumulator:
    return _BooleanFlipCountsAccumulator()

  def add_input(
      self,
      accumulator: _BooleanFlipCountsAccumulator,
      element: metric_types.StandardMetricInputs,
  ) -> _BooleanFlipCountsAccumulator:
    _, base_prediction, base_example_weight = next(
        metric_util.to_label_prediction_example_weight(
            inputs=element,
            eval_config=self._eval_config,
            model_name=self._baseline_model_name,
            output_name=self._output_name,
            example_weighted=self._example_weighted,
            flatten=True,
            allow_none=True,
        )
    )

    _, model_prediction, _ = next(
        metric_util.to_label_prediction_example_weight(
            inputs=element,
            eval_config=self._eval_config,
            model_name=self._model_name,
            output_name=self._output_name,
            example_weighted=self._example_weighted,
            flatten=True,
            allow_none=True,
        )
    )

    base_example_weight = metric_util.safe_to_scalar(base_example_weight)
    base_prediciton_bool = base_prediction > self._threshold
    model_prediction_bool = model_prediction > self._threshold

    accumulator.merge(
        _BooleanFlipCountsAccumulator(
            num_weighted_examples=base_example_weight,
            num_weighted_neg_to_neg=base_example_weight
            * int(not base_prediciton_bool and not model_prediction_bool),
            num_weighted_neg_to_pos=base_example_weight
            * int(not base_prediciton_bool and model_prediction_bool),
            num_weighted_pos_to_neg=base_example_weight
            * int(base_prediciton_bool and not model_prediction_bool),
            num_weighted_pos_to_pos=base_example_weight
            * int(base_prediciton_bool and model_prediction_bool),
        )
    )

    return accumulator

  def merge_accumulators(
      self, accumulators: Iterable[_BooleanFlipCountsAccumulator]
  ) -> _BooleanFlipCountsAccumulator:
    result = next(iter(accumulators))

    for accumulator in accumulators:
      result.merge(accumulator)

    return result

  def extract_output(
      self, accumulator: _BooleanFlipCountsAccumulator
  ) -> dict[metric_types.MetricKey, _BooleanFlipCountsAccumulator]:
    # We return a _BooleanFlipCountsAccumulator here, not a metric value.
    return {self._key: accumulator}


def _flip_counts(
    model_name: str,
    output_name: str,
    example_weighted: bool,
    eval_config: config_pb2.EvalConfig,
    baseline_model_name: str,
    threshold: float,
) -> metric_types.MetricComputation:
  """Returns the metric computations for calculating the boolean flip rates.

  Args:
    model_name: The model for which to compute this metric.
    output_name: The output name for which to compute this metric.
    example_weighted: Whether to compute this metric using example weights.
    eval_config: The EvalConfig for this TFMA evaluation. This is used to
      identify which model is the baseline.
    baseline_model_name: The baseline model to compare the model to.
    threshold: The threshold to use for converting both the baseline and
      candidate predictions into boolean values that can be compared.
  """
  key = metric_types.MetricKey(
      name=metric_util.generate_private_name_from_arguments(
          name=_FLIP_COUNTS_BASE_NAME,
          eval_config=eval_config,
          baseline_model_name=baseline_model_name,
          threshold=threshold,
      ),
      model_name=model_name,
      output_name=output_name,
      example_weighted=example_weighted,
  )

  return metric_types.MetricComputation(
      keys=[key],
      preprocessors=None,
      combiner=_BooleanFlipCountsCombiner(
          key=key,
          eval_config=eval_config,
          baseline_model_name=baseline_model_name,
          model_name=model_name,
          output_name=output_name,
          example_weighted=example_weighted,
          threshold=threshold,
      ),
  )


class _FlipRateBase(metric_types.Metric, abc.ABC):
  """Base class to generate the computations for all individual flip rates."""

  def __init__(self, name, threshold):
    super().__init__(
        self._metric_computations,
        name=name,
        threshold=threshold,
    )

  @abc.abstractmethod
  def result(self, flip_counts: _BooleanFlipCountsAccumulator) -> float:
    """This method will be overriden in each Individual Metric Class.

    Args:
      flip_counts: A _BooleanFlipCountsAccumulator containing the necessary
        counts to calculate the individual flip rate.

    Returns:
      The individual flip rate.
    """
    pass

  def _get_derived_metric_result_fn(
      self,
      metrics: dict[metric_types.MetricKey, Any],
      flip_counts_key: metric_types.MetricKey,
      metric_key: metric_types.MetricKey,
      calculate_flip_rate_fn: Callable[[_BooleanFlipCountsAccumulator], float],
  ) -> dict[metric_types.MetricKey, float]:
    """Generates the result() function for the Derived Metric Computations.

    Args:
      metrics: All the metrics (computed by any computation) including the
        individual flip rate metric.
      flip_counts_key: The key of the flip counts computation in "metrics".
      metric_key: The key of this metric computation.
      calculate_flip_rate_fn: A function that calculates the necessary flip
        rate.

    Returns:
      The result() function.
    """

    def _result() -> dict[metric_types.MetricKey, float]:
      # We only need the accumulator to calculate the result.
      flip_counts = metrics[flip_counts_key]

      return {metric_key: calculate_flip_rate_fn(flip_counts)}

    return _result()

  def _metric_computations(
      self,
      name: str,
      eval_config: config_pb2.EvalConfig,
      example_weighted: bool,
      threshold: float,
      model_names: Iterable[str],
      output_names: Optional[Iterable[str]] = ('',),
      sub_keys: Optional[Iterable[metric_types.SubKey]] = None,
  ) -> metric_types.MetricComputations:
    """Returns metric computations for an individual boolean flip rate.

    This is not meant to be used with merge_per_key_computations because we
    don't want to create computations for the baseline model, and we want to
    provide the baseline model name to each Combiner

    Args:
      name: Metric name for individual flip rate.
      eval_config: The EvalConfig for this TFMA evaluation. This is used to
        identify which model is the baseline.
      example_weighted: Whether to compute this metric using example weights.
      threshold: The threshold to use for converting both the baseline and
        candidate predictions into boolean values that can be compared.
      model_names: The name of the baseline model and the candidate model.
      output_names: The set of output names for which to compute this metric.
      sub_keys: The set of sub_key settings for which to compute this metric.
    """
    computations = []

    # Get the baseline model name.
    baseline_spec = model_util.get_baseline_model_spec(eval_config)
    baseline_model_name = baseline_spec.name if baseline_spec else None

    for candidate_model_name in model_names:
      if candidate_model_name == baseline_model_name:
        continue
      for output_name in output_names:
        for sub_key in sub_keys or (None,):
          # Define the metric key.
          metric_key = metric_types.MetricKey(
              name=name,
              model_name=candidate_model_name,
              output_name=output_name,
              sub_key=sub_key,
              example_weighted=example_weighted,
              is_diff=True,
          )

          flip_counts_computation = _flip_counts(
              model_name=candidate_model_name,
              output_name=output_name,
              example_weighted=example_weighted,
              eval_config=eval_config,
              baseline_model_name=baseline_model_name,
              threshold=threshold,
          )

          # Append flip counts to computations.
          computations.append(flip_counts_computation)

          # Append flip rate (derived metric computation) to computations.
          computations.append(
              metric_types.DerivedMetricComputation(
                  keys=[metric_key],
                  result=functools.partial(
                      self._get_derived_metric_result_fn,
                      flip_counts_key=flip_counts_computation.keys[0],
                      metric_key=metric_key,
                      calculate_flip_rate_fn=self.result,
                  ),
              )
          )

    return computations


class SymmetricFlipRate(_FlipRateBase):
  """FlipRate is the rate at which predictions between models switch.

  Given a pair of models and a threshold for converting continuous model outputs
  into boolean predictions, this metric will produce the symmetric flip rate
  (i.e. the number of times the boolean predictions don't match, regardless of
  the direction of the flip).
  """

  def __init__(
      self,
      name: str = SYMMETRIC_FLIP_RATE_NAME,
      threshold: float = _DEFAULT_FLIP_RATE_THRESHOLD,
  ):
    """Initializes BooleanFlipRates metric.

    Args:
      name: Metric name for the symmetric flip rate.
      threshold: The threshold to use for converting the model prediction into a
        boolean value that can be used for comparison between models.
    """

    super().__init__(
        name=name,
        threshold=threshold,
    )

  def result(self, flip_counts: _BooleanFlipCountsAccumulator) -> float:
    return (
        flip_counts.num_weighted_neg_to_pos
        + flip_counts.num_weighted_pos_to_neg
    ) / flip_counts.num_weighted_examples


class NegToNegFlipRate(_FlipRateBase):
  """FlipRate is the rate at which predictions between models switch.

  Given a pair of models and a threshold for converting continuous model outputs
  into boolean predictions, this metric will produce the neg-to-neg flip rate
  (i.e. the rate at which the baseline model's and the candidate model's
  predictions are both negative).
  """

  def __init__(
      self,
      name: str = NEG_TO_NEG_FLIP_RATE_NAME,
      threshold: float = _DEFAULT_FLIP_RATE_THRESHOLD,
  ):
    """Initializes BooleanFlipRates metric.

    Args:
      name: Metric name for the neg-to-neg flip rate.
      threshold: The threshold to use for converting the model prediction into a
        boolean value that can be used for comparison between models.
    """

    super().__init__(
        name=name,
        threshold=threshold,
    )

  def result(self, flip_counts: _BooleanFlipCountsAccumulator) -> float:
    return (
        flip_counts.num_weighted_neg_to_neg / flip_counts.num_weighted_examples
    )


class NegToPosFlipRate(_FlipRateBase):
  """FlipRate is the rate at which predictions between models switch.

  Given a pair of models and a threshold for converting continuous model outputs
  into boolean predictions, this metric will produce the neg-to-pos flip rate
  (i.e. the rate at which the baseline model's boolean prediction is negative
  and the candidate model's is positive).
  """

  def __init__(
      self,
      name: str = NEG_TO_POS_FLIP_RATE_NAME,
      threshold: float = _DEFAULT_FLIP_RATE_THRESHOLD,
  ):
    """Initializes BooleanFlipRates metric.

    Args:
      name: Metric name for the neg-to-pos flip rate.
      threshold: The threshold to use for converting the model prediction into a
        boolean value that can be used for comparison between models.
    """

    super().__init__(
        name=name,
        threshold=threshold,
    )

  def result(self, flip_counts: _BooleanFlipCountsAccumulator) -> float:
    return (
        flip_counts.num_weighted_neg_to_pos / flip_counts.num_weighted_examples
    )


class PosToNegFlipRate(_FlipRateBase):
  """FlipRate is the rate at which predictions between models switch.

  Given a pair of models and a threshold for converting continuous model outputs
  into boolean predictions, this metric will produce the pos-to-neg flip rate
  (i.e. the rate at which the baseline model's boolean prediction is positive
  and the candidate model's is negative).
  """

  def __init__(
      self,
      name: str = POS_TO_NEG_FLIP_RATE_NAME,
      threshold: float = _DEFAULT_FLIP_RATE_THRESHOLD,
  ):
    """Initializes BooleanFlipRates metric.

    Args:
      name: Metric name for the pos-to-neg flip rate.
      threshold: The threshold to use for converting the model prediction into a
        boolean value that can be used for comparison between models.
    """

    super().__init__(
        name=name,
        threshold=threshold,
    )

  def result(self, flip_counts: _BooleanFlipCountsAccumulator) -> float:
    return (
        flip_counts.num_weighted_pos_to_neg / flip_counts.num_weighted_examples
    )


class PosToPosFlipRate(_FlipRateBase):
  """FlipRate is the rate at which predictions between models switch.

  Given a pair of models and a threshold for converting continuous model outputs
  into boolean predictions, this metric will produce the pos-to-pos flip rate
  (i.e. the rate at which the baseline model's and the candidate model's
  predictions are both positive).
  """

  def __init__(
      self,
      name: str = POS_TO_POS_FLIP_RATE_NAME,
      threshold: float = _DEFAULT_FLIP_RATE_THRESHOLD,
  ):
    """Initializes BooleanFlipRates metric.

    Args:
      name: Metric name for the pos-to-pos flip rate.
      threshold: The threshold to use for converting the model prediction into a
        boolean value that can be used for comparison between models.
    """

    super().__init__(
        name=name,
        threshold=threshold,
    )

  def result(self, flip_counts: _BooleanFlipCountsAccumulator) -> float:
    return (
        flip_counts.num_weighted_pos_to_pos / flip_counts.num_weighted_examples
    )


def _boolean_flip_rates_computations(
    symmetric_flip_rate_name: str,
    neg_to_neg_flip_rate_name: str,
    neg_to_pos_flip_rate_name: str,
    pos_to_neg_flip_rate_name: str,
    pos_to_pos_flip_rate_name: str,
    threshold: float,
    eval_config: Optional[config_pb2.EvalConfig] = None,
    model_names: Optional[list[str]] = None,
    output_names: Optional[list[str]] = None,
    sub_keys: Optional[list[Optional[metric_types.SubKey]]] = None,
    example_weighted: bool = False,
) -> metric_types.MetricComputations:
  """Returns metric computations for all boolean flip rates.

  This is not meant to be used with merge_per_key_computations because we
  don't want to create computations for the baseline model, and we want to
  provide the baseline model name to each Combiner

  Args:
    symmetric_flip_rate_name: Metric name for symmetric flip rate.
    neg_to_neg_flip_rate_name: Metric name for the negative-to-negative flip
      rate.
    neg_to_pos_flip_rate_name: Metric name for the negative-to-positive flip
      rate.
    pos_to_neg_flip_rate_name: Metric name for the positive-to-negative flip
      rate.
    pos_to_pos_flip_rate_name: Metric name for the positive-to-positive flip
      rate.
    threshold: The threshold to use for converting both the baseline and
      candidate predictions into boolean values that can be compared.
    eval_config: The EvalConfig for this TFMA evaluation. This is used to
      identify which model is the baseline.
    model_names: The name of the baseline model and the candidate model.
    output_names: The set of output names for which to compute this metric.
    sub_keys: The set of sub_key settings for which to compute this metric.
    example_weighted: Whether to compute this metric using example weights.
      candidate predictions into boolean values that can be compared.
  """
  symmetric_metric = SymmetricFlipRate(
      name=symmetric_flip_rate_name, threshold=threshold
  )
  neg_to_neg_metric = NegToNegFlipRate(
      name=neg_to_neg_flip_rate_name, threshold=threshold
  )
  neg_to_pos_metric = NegToPosFlipRate(
      name=neg_to_pos_flip_rate_name, threshold=threshold
  )
  pos_to_neg_metric = PosToNegFlipRate(
      name=pos_to_neg_flip_rate_name, threshold=threshold
  )
  pos_to_pos_metric = PosToPosFlipRate(
      name=pos_to_pos_flip_rate_name, threshold=threshold
  )

  all_metrics = (
      symmetric_metric,
      neg_to_neg_metric,
      neg_to_pos_metric,
      pos_to_neg_metric,
      pos_to_pos_metric,
  )

  computations = []
  for metric in all_metrics:
    computations += metric.computations(
        eval_config=eval_config,
        model_names=model_names,
        output_names=output_names,
        example_weighted=example_weighted,
        sub_keys=sub_keys,
    )

  return computations


class BooleanFlipRates(metric_types.Metric):
  """FlipRate is the rate at which predictions between models switch.

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

  def __init__(
      self,
      threshold: float = _DEFAULT_FLIP_RATE_THRESHOLD,
      flip_rate_name: str = FLIP_RATE_NAME,
      neg_to_neg_flip_rate_name: str = NEG_TO_NEG_FLIP_RATE_NAME,
      neg_to_pos_flip_rate_name: str = NEG_TO_POS_FLIP_RATE_NAME,
      pos_to_neg_flip_rate_name: str = POS_TO_NEG_FLIP_RATE_NAME,
      pos_to_pos_flip_rate_name: str = POS_TO_POS_FLIP_RATE_NAME,
  ):
    """Initializes BooleanFlipRates metric.

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
        symmetric_flip_rate_name=flip_rate_name,
        neg_to_neg_flip_rate_name=neg_to_neg_flip_rate_name,
        neg_to_pos_flip_rate_name=neg_to_pos_flip_rate_name,
        pos_to_neg_flip_rate_name=pos_to_neg_flip_rate_name,
        pos_to_pos_flip_rate_name=pos_to_pos_flip_rate_name,
        threshold=threshold,
    )


# Register Individual Metrics.
metric_types.register_metric(SymmetricFlipRate)
metric_types.register_metric(NegToNegFlipRate)
metric_types.register_metric(NegToPosFlipRate)
metric_types.register_metric(PosToNegFlipRate)
metric_types.register_metric(PosToPosFlipRate)

# Register BooleanFlipRates.
metric_types.register_metric(BooleanFlipRates)
