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
"""NDCG (normalized discounted cumulative gain) metric."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import apache_beam as beam
import numpy as np
from tensorflow_model_analysis import config
from tensorflow_model_analysis import util
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util

from typing import Dict, List, Optional, Text, Tuple, Union

NDCG_NAME = 'ndcg'


class NDCG(metric_types.Metric):
  """NDCG (normalized discounted cumulative gain) metric.

  Calculates NDCG@k for the given value sub key value for top_k and the value of
  gain in the 'gain_key' feature. The value of NDCG@k returned is a weighted
  average of NDCG@k over the set of queries using the example weights.

  NDCG@k = (DCG@k for the given rank)/(DCG@k
  DCG@k = sum_{i=1}^k gain_i/log_2(i+1), where gain_i is the gain (relevance
          score) of the i^th ranked response, indexed from 1.

  This is a query/ranking based metric so a query_key must also be provided in
  the associated metrics spec.
  """

  def __init__(self, gain_key: Text, name: Text = NDCG_NAME):
    """Initializes NDCG.

    Args:
      gain_key: Key of feature in features dictionary that holds gain values.
      name: Metric name.
    """
    super(NDCG, self).__init__(_ndcg, gain_key=gain_key, name=name)


metric_types.register_metric(NDCG)


def _ndcg(gain_key: Text,
          name: Text = NDCG_NAME,
          eval_config: Optional[config.EvalConfig] = None,
          model_names: List[Text] = None,
          output_names: List[Text] = None,
          sub_keys: Optional[List[metric_types.SubKey]] = None,
          query_key: Text = '') -> metric_types.MetricComputations:
  """Returns metric computations for NDCG."""
  if not query_key:
    raise ValueError('a query_key is required to use NDCG metric')
  if sub_keys is None or any([sub_key.top_k is None for sub_key in sub_keys]):
    raise ValueError(
        'top_k values are required to use NDCG metric: {}'.format(sub_keys))
  computations = []
  for model_name in model_names if model_names else ['']:
    for output_name in output_names if output_names else ['']:
      keys = []
      for sub_key in sub_keys:
        keys.append(
            metric_types.MetricKey(
                name,
                model_name=model_name,
                output_name=output_name,
                sub_key=sub_key))
      computations.append(
          metric_types.MetricComputation(
              keys=keys,
              preprocessor=metric_types.FeaturePreprocessor(
                  feature_keys=[query_key, gain_key]),
              combiner=_NDCGCombiner(
                  metric_keys=keys,
                  eval_config=eval_config,
                  model_name=model_name,
                  output_name=output_name,
                  query_key=query_key,
                  gain_key=gain_key)))
  return computations


class _NDCGAccumulator(object):
  """NDCG accumulator."""
  __slots__ = ['ndcg', 'total_weighted_examples']

  def __init__(self, size: int):
    self.ndcg = [0.0] * size
    self.total_weighted_examples = 0.0


class _NDCGCombiner(beam.CombineFn):
  """Computes NDCG (normalized discounted cumulative gain)."""

  def __init__(self, metric_keys: List[metric_types.MetricKey],
               eval_config: config.EvalConfig, model_name: Text,
               output_name: Text, query_key: Text, gain_key: Text):
    """Initialize.

    Args:
      metric_keys: Metric keys.
      eval_config: Eval config.
      model_name: Model name.
      output_name: Output name.
      query_key: Query key.
      gain_key: Key of feature in features dictionary that holds gain values.
    """
    self._metric_keys = metric_keys
    self._eval_config = eval_config
    self._model_name = model_name
    self._output_name = output_name
    self._query_key = query_key
    self._gain_key = gain_key

  def _query(
      self, i: metric_types.StandardMetricInputs
  ) -> Optional[Union[float, int, Text]]:
    return metric_util.to_scalar(
        util.get_by_keys(i.features, [self._query_key]))

  def _gain(self, i: metric_types.StandardMetricInputs) -> float:
    gain = util.get_by_keys(i.features, [self._gain_key])
    if gain.size == 1:
      scalar = metric_util.to_scalar(gain)
      if scalar is not None:
        return scalar
    raise ValueError('expected {} to be scalar, but instead it has size = {}: '
                     'value={}, metric_keys={}, '
                     'StandardMetricInputs={}'.format(self._gain_key, gain.size,
                                                      gain, self._metric_keys,
                                                      i))

  def _to_gains_example_weight(
      self, inputs: List[metric_types.StandardMetricInputs]
  ) -> Tuple[List[float], float]:
    """Returns gains and example_weight sorted by prediction."""
    predictions = []
    example_weight = None
    for i in inputs:
      _, prediction, weight = (
          metric_util.to_label_prediction_example_weight(
              i,
              eval_config=self._eval_config,
              model_name=self._model_name,
              output_name=self._output_name,
              array_size=1))
      weight = float(weight)
      if example_weight is None:
        example_weight = weight
      elif example_weight != weight:
        raise ValueError(
            'all example weights for the same query value must use the '
            'same value {} != {}: query={}, StandardMetricInputs={}'.format(
                weight, example_weight, self._query(i), i))
      predictions.append(float(prediction))
    if example_weight is None:
      example_weight = 1.0
    sort_indices = np.argsort(predictions)[::-1]
    sorted_gains = []
    for i in sort_indices:
      sorted_gains.append(self._gain(inputs[i]))
    return (sorted_gains, example_weight)

  def _calculate_dcg_at_k(self, k: int, sorted_values: List[float]) -> float:
    """Calculate the value of DCG@k.

    Args:
      k: The last position to consider.
      sorted_values: A list of gain values assumed to be sorted in the desired
        ranking order.

    Returns:
      The value of DCG@k.
    """
    return np.sum(
        np.array(sorted_values)[:k] / np.log2(np.array(range(2, k + 2))))

  def _calculate_ndcg(self, values: List[Tuple[int, float]], k: int) -> float:
    """Calculate NDCG@k, based on given rank and gain values.

    Args:
      values: A list of tuples representing rank order and gain values.
      k: The maximum position to consider in calculating nDCG

    Returns:
      The value of NDCG@k, for the given list of values.
    """
    max_rank = min(k, len(values))
    ranked_values = [
        gain for _, gain in sorted(values, key=lambda x: x[0], reverse=False)
    ]
    optimal_values = [
        gain for _, gain in sorted(values, key=lambda x: x[1], reverse=True)
    ]
    dcg = self._calculate_dcg_at_k(max_rank, ranked_values)
    optimal_dcg = self._calculate_dcg_at_k(max_rank, optimal_values)
    if optimal_dcg > 0:
      return dcg / optimal_dcg
    else:
      return 0

  def create_accumulator(self):
    return _NDCGAccumulator(len(self._metric_keys))

  def add_input(
      self, accumulator: _NDCGAccumulator,
      elements: List[metric_types.StandardMetricInputs]) -> _NDCGAccumulator:
    gains, example_weight = self._to_gains_example_weight(elements)
    rank_gain = [(pos + 1, gain) for pos, gain in enumerate(gains)]
    for i, key in enumerate(self._metric_keys):
      accumulator.ndcg[i] += (
          self._calculate_ndcg(rank_gain, key.sub_key.top_k) * example_weight)
    accumulator.total_weighted_examples += float(example_weight)
    return accumulator

  def merge_accumulators(
      self, accumulators: List[_NDCGAccumulator]) -> _NDCGAccumulator:
    result = self.create_accumulator()
    for accumulator in accumulators:
      result.ndcg = [a + b for a, b in zip(result.ndcg, accumulator.ndcg)]
      result.total_weighted_examples += accumulator.total_weighted_examples
    return result

  def extract_output(self, accumulator: _NDCGAccumulator) -> Dict[Text, float]:
    output = {}
    for i, key in enumerate(self._metric_keys):
      if accumulator.total_weighted_examples > 0:
        output[key] = accumulator.ndcg[i] / accumulator.total_weighted_examples
      else:
        output[key] = float('nan')
    return output
