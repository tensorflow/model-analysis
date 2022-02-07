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
"""TJUR discrimination metrics.

TJUR discrimination metrics are used for logistic regression problems and are
designed for class imbalance problems.
"""

from typing import Any, Dict, Iterable, Optional

import apache_beam as beam
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.proto import config_pb2

COEFFICIENT_OF_DISCRIMINATION_NAME = 'coefficient_of_discimination'
RELATIVE_COEFFICIENT_OF_DISCRIMINATION_NAME = (
    'relative_coefficient_of_discimination')
_TJUR_DISCRIMINATION_NAME = '_tjur_discimination'


class CoefficientOfDiscrimination(metric_types.Metric):
  """Coefficient of discrimination metric.

  The coefficient of discrimination measures the differences between the average
  prediction for the positive examples and the average prediction for the
  negative examples.

  The formula is: AVG(pred | label = 1) - AVG(pred | label = 0)
  More details can be found in the following paper:
  https://www.tandfonline.com/doi/abs/10.1198/tast.2009.08210
  """

  def __init__(self, name: str = COEFFICIENT_OF_DISCRIMINATION_NAME):
    """Initializes coefficient of discrimination metric.

    Args:
      name: Metric name.
    """
    super().__init__(
        metric_util.merge_per_key_computations(_coefficient_of_discrimination),
        name=name)


metric_types.register_metric(CoefficientOfDiscrimination)


def _coefficient_of_discrimination(
    name: str = COEFFICIENT_OF_DISCRIMINATION_NAME,
    eval_config: Optional[config_pb2.EvalConfig] = None,
    model_name: str = '',
    output_name: str = '',
    sub_key: Optional[metric_types.SubKey] = None,
    aggregation_type: Optional[metric_types.AggregationType] = None,
    class_weights: Optional[Dict[int, float]] = None,
    example_weighted: bool = False) -> metric_types.MetricComputations:
  """Returns metric computations for coefficient of discrimination."""
  key = metric_types.MetricKey(
      name=name,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key,
      example_weighted=example_weighted)

  # Compute shared tjur discimination metrics.
  computations = _tjur_discrimination(
      eval_config=eval_config,
      model_name=model_name,
      output_name=output_name,
      aggregation_type=aggregation_type,
      class_weights=class_weights,
      example_weighted=example_weighted)
  # Shared metrics are based on a single computation and key.
  tjur_discrimination_key = computations[0].keys[0]

  def result(
      metrics: Dict[metric_types.MetricKey, Any]
  ) -> Dict[metric_types.MetricKey, float]:
    """Returns coefficient of discrimination."""
    metric = metrics[tjur_discrimination_key]
    if (metric.total_negative_weighted_labels == 0 or
        metric.total_positive_weighted_labels == 0):
      value = float('nan')
    else:
      avg_pos_label = (
          metric.total_positive_weighted_predictions /
          metric.total_positive_weighted_labels)
      avg_neg_label = (
          metric.total_negative_weighted_predictions /
          metric.total_negative_weighted_labels)
      value = avg_pos_label - avg_neg_label
    return {key: value}

  derived_computation = metric_types.DerivedMetricComputation(
      keys=[key], result=result)
  computations.append(derived_computation)
  return computations


class RelativeCoefficientOfDiscrimination(metric_types.Metric):
  """Relative coefficient of discrimination metric.

  The relative coefficient of discrimination measures the ratio between the
  average prediction for the positive examples and the average prediction for
  the negative examples. This has a very simple intuitive explanation, measuring
  how much higher is the prediction going to be for a positive example than for
  a negative example.
  """

  def __init__(self, name: str = RELATIVE_COEFFICIENT_OF_DISCRIMINATION_NAME):
    """Initializes relative coefficient of discrimination metric.

    Args:
      name: Metric name.
    """
    super().__init__(
        metric_util.merge_per_key_computations(
            _relative_coefficient_of_discrimination),
        name=name)


metric_types.register_metric(RelativeCoefficientOfDiscrimination)


def _relative_coefficient_of_discrimination(
    name: str = RELATIVE_COEFFICIENT_OF_DISCRIMINATION_NAME,
    eval_config: Optional[config_pb2.EvalConfig] = None,
    model_name: str = '',
    output_name: str = '',
    aggregation_type: Optional[metric_types.AggregationType] = None,
    class_weights: Optional[Dict[float, int]] = None,
    example_weighted: bool = False) -> metric_types.MetricComputations:
  """Returns metric computations for coefficient of discrimination."""
  key = metric_types.MetricKey(
      name=name,
      model_name=model_name,
      output_name=output_name,
      example_weighted=example_weighted)

  # Compute shared tjur discimination metrics.
  computations = _tjur_discrimination(
      eval_config=eval_config,
      model_name=model_name,
      output_name=output_name,
      aggregation_type=aggregation_type,
      class_weights=class_weights,
      example_weighted=example_weighted)
  # Shared metrics are based on a single computation and key.
  tjur_discrimination_key = computations[0].keys[0]

  def result(
      metrics: Dict[metric_types.MetricKey, Any]
  ) -> Dict[metric_types.MetricKey, float]:
    """Returns coefficient of discrimination."""
    metric = metrics[tjur_discrimination_key]
    if (metric.total_negative_weighted_labels == 0 or
        metric.total_positive_weighted_labels == 0 or
        metric.total_negative_weighted_predictions == 0):
      value = float('nan')
    else:
      avg_pos_label = (
          metric.total_positive_weighted_predictions /
          metric.total_positive_weighted_labels)
      avg_neg_label = (
          metric.total_negative_weighted_predictions /
          metric.total_negative_weighted_labels)
      value = avg_pos_label / avg_neg_label
    return {key: value}

  derived_computation = metric_types.DerivedMetricComputation(
      keys=[key], result=result)
  computations.append(derived_computation)
  return computations


def _tjur_discrimination(
    name: str = _TJUR_DISCRIMINATION_NAME,
    eval_config: Optional[config_pb2.EvalConfig] = None,
    model_name: str = '',
    output_name: str = '',
    aggregation_type: Optional[metric_types.AggregationType] = None,
    class_weights: Optional[Dict[int, float]] = None,
    example_weighted: bool = False) -> metric_types.MetricComputations:
  """Returns metric computations for TJUR discrimination."""
  key = metric_types.MetricKey(
      name=name,
      model_name=model_name,
      output_name=output_name,
      example_weighted=example_weighted)
  return [
      metric_types.MetricComputation(
          keys=[key],
          preprocessor=None,
          combiner=_TJURDiscriminationCombiner(key, eval_config,
                                               aggregation_type, class_weights,
                                               example_weighted))
  ]


class _TJURDiscriminationAccumulator:
  """TJUR discrimination accumulator."""
  __slots__ = [
      'total_negative_weighted_predictions', 'total_negative_weighted_labels',
      'total_positive_weighted_predictions', 'total_positive_weighted_labels'
  ]

  def __init__(self):
    self.total_negative_weighted_predictions = 0.0
    self.total_negative_weighted_labels = 0.0
    self.total_positive_weighted_predictions = 0.0
    self.total_positive_weighted_labels = 0.0


class _TJURDiscriminationCombiner(beam.CombineFn):
  """Computes min label position metric."""

  def __init__(self, key: metric_types.MetricKey,
               eval_config: Optional[config_pb2.EvalConfig],
               aggregation_type: Optional[metric_types.AggregationType],
               class_weights: Optional[Dict[int,
                                            float]], example_weighted: bool):
    self._key = key
    self._eval_config = eval_config
    self._aggregation_type = aggregation_type
    self._class_weights = class_weights
    self._example_weighted = example_weighted

  def create_accumulator(self) -> _TJURDiscriminationAccumulator:
    return _TJURDiscriminationAccumulator()

  def add_input(
      self, accumulator: _TJURDiscriminationAccumulator,
      element: metric_types.StandardMetricInputs
  ) -> _TJURDiscriminationAccumulator:
    for label, prediction, example_weight in (
        metric_util.to_label_prediction_example_weight(
            element,
            eval_config=self._eval_config,
            model_name=self._key.model_name,
            output_name=self._key.output_name,
            aggregation_type=self._aggregation_type,
            class_weights=self._class_weights,
            example_weighted=self._example_weighted)):
      label = float(label)
      prediction = float(prediction)
      example_weight = float(example_weight)
      accumulator.total_negative_weighted_labels += ((1.0 - label) *
                                                     example_weight)
      accumulator.total_positive_weighted_labels += label * example_weight
      accumulator.total_negative_weighted_predictions += ((1.0 - label) *
                                                          prediction *
                                                          example_weight)
      accumulator.total_positive_weighted_predictions += (
          label * prediction * example_weight)
    return accumulator

  def merge_accumulators(
      self, accumulators: Iterable[_TJURDiscriminationAccumulator]
  ) -> _TJURDiscriminationAccumulator:
    accumulators = iter(accumulators)
    result = next(accumulators)
    for accumulator in accumulators:
      result.total_negative_weighted_predictions += (
          accumulator.total_negative_weighted_predictions)
      result.total_negative_weighted_labels += (
          accumulator.total_negative_weighted_labels)
      result.total_positive_weighted_predictions += (
          accumulator.total_positive_weighted_predictions)
      result.total_positive_weighted_labels += (
          accumulator.total_positive_weighted_labels)
    return result

  def extract_output(
      self, accumulator: _TJURDiscriminationAccumulator
  ) -> Dict[metric_types.MetricKey, _TJURDiscriminationAccumulator]:
    return {self._key: accumulator}
