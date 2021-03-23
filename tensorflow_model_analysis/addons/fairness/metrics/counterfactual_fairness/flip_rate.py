# Lint as: python3
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

from typing import Dict, Optional, Sequence

from tensorflow_model_analysis import config
from tensorflow_model_analysis.addons.fairness.metrics.counterfactual_fairness import flip_count
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util

FLIP_RATE_NAME = 'flip_rate'

# List of sub-metrics computed.
_METRICS_LIST = [
    'negative_to_positive',
    'negative_to_positive_examples_ids',
    'overall',
    'positive_to_negative',
    'positive_to_negative_examples_ids',
]


class FlipRate(metric_types.Metric):
  """Flip Rate metrics.

  Proportion of flip counts (total, positive to negative and negative to
  positive) to number of examples.
  It calculates:
  - Total Flip Count / Total Example Count
  - Positive to Negative Flip Count / Positive Examples Count
  - Negative to Positive Flip Count / Negative Examples Count

  """

  def __init__(
      self,
      counterfactual_prediction_key: str,
      name: str = FLIP_RATE_NAME,
      thresholds: Sequence[float] = flip_count.DEFAULT_THRESHOLDS,
      example_id_key: Optional[str] = None,
      example_ids_count: int = flip_count.DEFAULT_NUM_EXAMPLE_IDS,
  ):
    """Initializes flip rate metrics.

    Args:
      counterfactual_prediction_key: Prediction label key for counterfactual
        example to be used to measure the flip count.
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
    counterfactual_prediction_key: str,
    example_id_key: Optional[str] = None,
    example_ids_count: int = flip_count.DEFAULT_NUM_EXAMPLE_IDS,
    name: str = FLIP_RATE_NAME,
    thresholds: Sequence[float] = flip_count.DEFAULT_THRESHOLDS,
    model_name: str = '',
    output_name: str = '',
    eval_config: Optional[config.EvalConfig] = None,
) -> metric_types.MetricComputations:
  """Returns computations for flip rate."""
  keys, metric_key_by_name_by_threshold = flip_count.create_metric_keys(
      thresholds, _METRICS_LIST, name, model_name, output_name)

  computations = flip_count.flip_count(
      thresholds=thresholds,
      counterfactual_prediction_key=counterfactual_prediction_key,
      example_id_key=example_id_key,
      example_ids_count=example_ids_count,
      model_name=model_name,
      output_name=output_name,
      eval_config=eval_config)

  _, flip_count_metric_key_by_name_by_threshold = flip_count.create_metric_keys(
      thresholds, flip_count.METRICS_LIST, flip_count.FLIP_COUNT_NAME,
      model_name, output_name)

  def result(
      metrics: Dict[metric_types.MetricKey, float]
  ) -> Dict[metric_types.MetricKey, float]:
    """Returns flip rate metrics values."""
    output = {}
    for threshold in thresholds:
      ptn = flip_count_metric_key_by_name_by_threshold[threshold][
          'positive_to_negative']
      ntp = flip_count_metric_key_by_name_by_threshold[threshold][
          'negative_to_positive']
      pos_examples = flip_count_metric_key_by_name_by_threshold[threshold][
          'positive_to_negative_examples_ids']
      neg_examples = flip_count_metric_key_by_name_by_threshold[threshold][
          'negative_to_positive_examples_ids']
      pos = flip_count_metric_key_by_name_by_threshold[threshold][
          'positive_examples_count']
      neg = flip_count_metric_key_by_name_by_threshold[threshold][
          'negative_examples_count']
      output[metric_key_by_name_by_threshold[threshold]
             ['overall']] = (metrics[ntp] + metrics[ptn]) / (
                 metrics[pos] + metrics[neg])
      output[metric_key_by_name_by_threshold[threshold]
             ['positive_to_negative']] = metrics[ptn] / metrics[pos]
      output[metric_key_by_name_by_threshold[threshold]
             ['negative_to_positive']] = metrics[ntp] / metrics[neg]
      output[metric_key_by_name_by_threshold[threshold]
             ['positive_to_negative_examples_ids']] = metrics[pos_examples]
      output[metric_key_by_name_by_threshold[threshold]
             ['negative_to_positive_examples_ids']] = metrics[neg_examples]

    return output

  derived_computation = metric_types.DerivedMetricComputation(
      keys=keys, result=result)

  computations.append(derived_computation)
  return computations


metric_types.register_metric(FlipRate)
