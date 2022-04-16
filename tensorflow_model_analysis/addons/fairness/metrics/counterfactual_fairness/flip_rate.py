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
"""Flip Rate Metrics."""

from typing import Dict, Optional, Sequence, Union

import numpy as np
from tensorflow_model_analysis.addons.fairness.metrics.counterfactual_fairness import flip_count
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.proto import config_pb2

FLIP_RATE_NAME = 'flip_rate'

# List of sub-metrics computed.
_NEGATIVE_EXAMPLES_COUNT = 'negative_examples_count'
_NEGATIVE_TO_POSITIVE = 'negative_to_positive'
_NEGATIVE_TO_POSITIVE_EXAMPLE_IDS = 'negative_to_positive_examples_ids'
_OVERALL = 'overall'
_POSITIVE_EXAMPLES_COUNT = 'positive_examples_count'
_POSITIVE_TO_NEGATIVE = 'positive_to_negative'
_POSITIVE_TO_NEGATIVE_EXAMPLE_IDS = 'positive_to_negative_examples_ids'
_SAMPLE_EXAMPLES_IDS = 'sample_examples_ids'

_METRICS_LIST = [
    _NEGATIVE_TO_POSITIVE,
    _NEGATIVE_TO_POSITIVE_EXAMPLE_IDS,
    _OVERALL,
    _POSITIVE_TO_NEGATIVE,
    _POSITIVE_TO_NEGATIVE_EXAMPLE_IDS,
    _SAMPLE_EXAMPLES_IDS,
]


class FlipRate(metric_types.Metric):
  """Flip Rate metrics.

  Proportion of flip counts (total, positive to negative and negative to
  positive) to number of examples.
  It calculates:
  - Total Flip Count / Total Example Count
  - Positive to Negative Flip Count / Total Example Count
  - Negative to Positive Flip Count / Total Example Count

  """

  def __init__(
      self,
      counterfactual_prediction_key: Optional[str] = None,
      name: str = FLIP_RATE_NAME,
      thresholds: Sequence[float] = flip_count.DEFAULT_THRESHOLDS,
      example_id_key: Optional[str] = None,
      example_ids_count: int = flip_count.DEFAULT_NUM_EXAMPLE_IDS,
  ):
    """Initializes flip rate metrics.

    Args:
      counterfactual_prediction_key: Prediction label key for counterfactual
        example to be used to measure the flip count. Defaults to None which
          indicates that the counterfactual_prediction_key should be extracted
          from the baseline model spec.
      name: Metric name.
      thresholds: Thresholds to be used to measure flips.
      example_id_key: Feature key containing example id.
      example_ids_count: Max number of example ids to be extracted for false
        positives and false negatives.
    """
    super().__init__(
        metric_util.merge_per_key_computations(_flip_rate),
        counterfactual_prediction_key=counterfactual_prediction_key,
        thresholds=thresholds,
        name=name,
        example_id_key=example_id_key,
        example_ids_count=example_ids_count)


def _flip_rate(
    counterfactual_prediction_key: Optional[str] = None,
    example_id_key: Optional[str] = None,
    example_ids_count: int = flip_count.DEFAULT_NUM_EXAMPLE_IDS,
    name: str = FLIP_RATE_NAME,
    thresholds: Sequence[float] = flip_count.DEFAULT_THRESHOLDS,
    model_name: str = '',
    output_name: str = '',
    eval_config: Optional[config_pb2.EvalConfig] = None,
    example_weighted: bool = False) -> metric_types.MetricComputations:
  """Returns computations for flip rate."""
  keys, metric_key_by_name_by_threshold = flip_count.create_metric_keys(
      thresholds, _METRICS_LIST, name, model_name, output_name,
      example_weighted)

  computations = flip_count.flip_count(
      thresholds=thresholds,
      counterfactual_prediction_key=counterfactual_prediction_key,
      example_id_key=example_id_key,
      example_ids_count=example_ids_count,
      model_name=model_name,
      output_name=output_name,
      eval_config=eval_config,
      example_weighted=example_weighted)

  _, flip_count_metric_key_by_name_by_threshold = flip_count.create_metric_keys(
      thresholds, flip_count.METRICS_LIST, flip_count.FLIP_COUNT_NAME,
      model_name, output_name, example_weighted)

  def pick_overall_flip_examples(ntp_examples: np.ndarray,
                                 ptn_examples: np.ndarray) -> np.ndarray:
    output_size = min(example_ids_count, ntp_examples.size + ptn_examples.size)
    examples = np.vstack([ntp_examples, ptn_examples])
    return np.random.choice(examples.flatten(), size=output_size, replace=False)

  def result(
      metrics: Dict[metric_types.MetricKey, Union[float, np.ndarray]]
  ) -> Dict[metric_types.MetricKey, Union[float, np.ndarray]]:
    """Returns flip rate metrics values."""
    output = {}
    for threshold in thresholds:
      ptn = flip_count_metric_key_by_name_by_threshold[threshold][
          _POSITIVE_TO_NEGATIVE]
      ntp = flip_count_metric_key_by_name_by_threshold[threshold][
          _NEGATIVE_TO_POSITIVE]
      pos_examples = flip_count_metric_key_by_name_by_threshold[threshold][
          _POSITIVE_TO_NEGATIVE_EXAMPLE_IDS]
      neg_examples = flip_count_metric_key_by_name_by_threshold[threshold][
          _NEGATIVE_TO_POSITIVE_EXAMPLE_IDS]
      pos = flip_count_metric_key_by_name_by_threshold[threshold][
          _POSITIVE_EXAMPLES_COUNT]
      neg = flip_count_metric_key_by_name_by_threshold[threshold][
          _NEGATIVE_EXAMPLES_COUNT]
      output[metric_key_by_name_by_threshold[threshold]
             [_OVERALL]] = (metrics[ntp] + metrics[ptn]) / (
                 (metrics[pos] + metrics[neg]) or float('NaN'))
      output[metric_key_by_name_by_threshold[threshold]
             [_POSITIVE_TO_NEGATIVE]] = metrics[ptn] / (
                 (metrics[pos] + metrics[neg]) or float('NaN'))
      output[metric_key_by_name_by_threshold[threshold]
             [_NEGATIVE_TO_POSITIVE]] = metrics[ntp] / (
                 (metrics[pos] + metrics[neg]) or float('NaN'))
      output[metric_key_by_name_by_threshold[threshold]
             [_POSITIVE_TO_NEGATIVE_EXAMPLE_IDS]] = metrics[pos_examples]
      output[metric_key_by_name_by_threshold[threshold]
             [_NEGATIVE_TO_POSITIVE_EXAMPLE_IDS]] = metrics[neg_examples]
      # TODO(sokeefe): Should this depend on  of example_weighted?
      if not example_weighted:
        assert isinstance(metrics[neg_examples], np.ndarray)
        assert isinstance(metrics[pos_examples], np.ndarray)
        output[metric_key_by_name_by_threshold[threshold]
               [_SAMPLE_EXAMPLES_IDS]] = pick_overall_flip_examples(
                   ntp_examples=metrics[neg_examples],
                   ptn_examples=metrics[pos_examples])

    return output

  derived_computation = metric_types.DerivedMetricComputation(
      keys=keys, result=result)

  computations.append(derived_computation)
  return computations


metric_types.register_metric(FlipRate)
