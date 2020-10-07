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
"""Squared pearson correlation (r^2) metric."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

from typing import Dict, List, Optional, Text

import apache_beam as beam
from tensorflow_model_analysis import config
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util

SQUARED_PEARSON_CORRELATION_NAME = 'squared_pearson_correlation'


class SquaredPearsonCorrelation(metric_types.Metric):
  """Squared pearson correlation (r^2) metric."""

  def __init__(self, name=SQUARED_PEARSON_CORRELATION_NAME):
    """Initializes squared pearson correlation (r^2) metric.

    Args:
      name: Metric name.
    """
    super(SquaredPearsonCorrelation, self).__init__(
        metric_util.merge_per_key_computations(_squared_pearson_correlation),
        name=name)


metric_types.register_metric(SquaredPearsonCorrelation)


def _squared_pearson_correlation(
    name: Text = SQUARED_PEARSON_CORRELATION_NAME,
    eval_config: Optional[config.EvalConfig] = None,
    model_name: Text = '',
    output_name: Text = '',
    sub_key: Optional[metric_types.SubKey] = None,
    aggregation_type: Optional[metric_types.AggregationType] = None,
    class_weights: Optional[Dict[int, float]] = None
) -> metric_types.MetricComputations:
  """Returns metric computations for squared pearson correlation (r^2)."""
  key = metric_types.MetricKey(
      name=name,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key)
  return [
      metric_types.MetricComputation(
          keys=[key],
          preprocessor=None,
          combiner=_SquaredPearsonCorrelationCombiner(key, eval_config,
                                                      aggregation_type,
                                                      class_weights))
  ]


class _SquaredPearsonCorrelationAccumulator(object):
  """Squared pearson correlation (r^2) accumulator."""
  __slots__ = [
      'total_weighted_labels', 'total_weighted_predictions',
      'total_weighted_squared_labels', 'total_weighted_squared_predictions',
      'total_weighted_labels_times_predictions', 'total_weighted_examples'
  ]

  def __init__(self):
    self.total_weighted_labels = 0.0
    self.total_weighted_predictions = 0.0
    self.total_weighted_squared_labels = 0.0
    self.total_weighted_squared_predictions = 0.0
    self.total_weighted_labels_times_predictions = 0.0
    self.total_weighted_examples = 0.0


class _SquaredPearsonCorrelationCombiner(beam.CombineFn):
  """Computes squared pearson correlation (r^2) metric."""

  def __init__(self, key: metric_types.MetricKey,
               eval_config: Optional[config.EvalConfig],
               aggregation_type: Optional[metric_types.AggregationType],
               class_weights: Optional[Dict[int, float]]):
    self._key = key
    self._eval_config = eval_config
    self._aggregation_type = aggregation_type
    self._class_weights = class_weights

  def create_accumulator(self) -> _SquaredPearsonCorrelationAccumulator:
    return _SquaredPearsonCorrelationAccumulator()

  def add_input(
      self, accumulator: _SquaredPearsonCorrelationAccumulator,
      element: metric_types.StandardMetricInputs
  ) -> _SquaredPearsonCorrelationAccumulator:
    for label, prediction, example_weight in (
        metric_util.to_label_prediction_example_weight(
            element,
            eval_config=self._eval_config,
            model_name=self._key.model_name,
            output_name=self._key.output_name,
            aggregation_type=self._aggregation_type,
            class_weights=self._class_weights)):
      example_weight = float(example_weight)
      label = float(label)
      prediction = float(prediction)
      accumulator.total_weighted_labels += example_weight * label
      accumulator.total_weighted_predictions += example_weight * prediction
      accumulator.total_weighted_squared_labels += example_weight * label**2
      accumulator.total_weighted_squared_predictions += (
          example_weight * prediction**2)
      accumulator.total_weighted_labels_times_predictions += (
          example_weight * label * prediction)
      accumulator.total_weighted_examples += example_weight
    return accumulator

  def merge_accumulators(
      self, accumulators: List[_SquaredPearsonCorrelationAccumulator]
  ) -> _SquaredPearsonCorrelationAccumulator:
    result = self.create_accumulator()
    for accumulator in accumulators:
      result.total_weighted_labels += accumulator.total_weighted_labels
      result.total_weighted_predictions += (
          accumulator.total_weighted_predictions)
      result.total_weighted_squared_labels += (
          accumulator.total_weighted_squared_labels)
      result.total_weighted_squared_predictions += (
          accumulator.total_weighted_squared_predictions)
      result.total_weighted_labels_times_predictions += (
          accumulator.total_weighted_labels_times_predictions)
      result.total_weighted_examples += accumulator.total_weighted_examples
    return result

  def extract_output(
      self, accumulator: _SquaredPearsonCorrelationAccumulator
  ) -> Dict[metric_types.MetricKey, float]:
    result = float('nan')

    if accumulator.total_weighted_examples > 0.0:
      # See https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
      # r^2 = Cov(X, Y)^2 / VAR(X) * VAR(Y)
      #     = (E[XY] - E[X]E[Y])^2 / (E[X^2] - E[X]^2) * (E[Y^2] - E[Y]^2)
      #     = [SUM(xy) - n*mean(x)*mean(y)]^2 /
      #         [SUM(x^2) - n*mean(x)^2 * SUM(y^2) - n*mean(y)^2]
      # n = total_weighted_examples
      # SUM(x) = total_weighted_labels
      # SUM(y) = total_weighted_predictions
      # SUM(xy) = total_weighted_labels_times_predictions
      # SUM(x^2) = total_weighted_squared_labels
      # SUM(y^2) = total_weighted_squared_predictions

      # numerator = [SUM(xy) - n*mean(x)*mean(y)]^2
      #           = [SUM(xy) - n*SUM(x)/n*SUM(y)/n]^2
      #           = [SUM(xy) - SUM(x)*SUM(y)/n]^2
      numerator = (accumulator.total_weighted_labels_times_predictions -
                   accumulator.total_weighted_labels *
                   accumulator.total_weighted_predictions /
                   accumulator.total_weighted_examples)**2
      # denominator_y = SUM(y^2) - n*mean(y)^2
      #               = SUM(y^2) - n*(SUM(y)/n)^2
      #               = SUM(y^2) - SUM(y)^2/n
      denominator_y = (
          accumulator.total_weighted_squared_predictions -
          accumulator.total_weighted_predictions**2 /
          accumulator.total_weighted_examples)

      # denominator_x = SUM(x^2) - n*mean(x)^2
      #               = SUM(x^2) - n*(SUM(x)/n)^2
      #               = SUM(x^2) - SUM(x)^2/n
      denominator_x = (
          accumulator.total_weighted_squared_labels -
          accumulator.total_weighted_labels**2 /
          accumulator.total_weighted_examples)
      denominator = denominator_x * denominator_y
      if denominator > 0.0:
        result = numerator / denominator

    return {self._key: result}
