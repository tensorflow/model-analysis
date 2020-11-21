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
"""NDCG (normalized discounted cumulative gain) metric."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

from typing import Dict, List, Optional, Text, Tuple, Union

import apache_beam as beam
import numpy as np
from tensorflow_model_analysis import config
from tensorflow_model_analysis import util
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util

NDCG_NAME = 'ndcg'


class NDCG(metric_types.Metric):
  """NDCG (normalized discounted cumulative gain) metric.

  Calculates NDCG@k for a given set of top_k values calculated from a list of
  gains (relevance scores) that are sorted based on the associated predictions.
  The top_k_list can be passed as part of the NDCG metric config or using
  tfma.MetricsSpec.binarize.top_k_list if configuring multiple top_k metrics.
  The gain (relevance score) is determined from the value stored in the
  'gain_key' feature. The value of NDCG@k returned is a weighted average of
  NDCG@k over the set of queries using the example weights.

  NDCG@k = (DCG@k for the given rank)/(DCG@k
  DCG@k = sum_{i=1}^k gain_i/log_2(i+1), where gain_i is the gain (relevance
          score) of the i^th ranked response, indexed from 1.

  This is a query/ranking based metric so a query_key must also be provided in
  the associated tfma.MetricsSpec.
  """

  def __init__(self,
               gain_key: Text,
               top_k_list: Optional[List[int]] = None,
               name: Text = NDCG_NAME):
    """Initializes NDCG.

    Args:
      gain_key: Key of feature in features dictionary that holds gain values.
      top_k_list: Values for top k. This can also be set using the
        tfma.MetricsSpec.binarize.top_k_list associated with the metric.
      name: Metric name.
    """
    super(NDCG, self).__init__(
        _ndcg, gain_key=gain_key, top_k_list=top_k_list, name=name)


metric_types.register_metric(NDCG)


def _ndcg(gain_key: Text,
          top_k_list: Optional[List[int]] = None,
          name: Text = NDCG_NAME,
          eval_config: Optional[config.EvalConfig] = None,
          model_names: List[Text] = None,
          output_names: List[Text] = None,
          sub_keys: Optional[List[metric_types.SubKey]] = None,
          query_key: Text = '') -> metric_types.MetricComputations:
  """Returns metric computations for NDCG."""
  if not query_key:
    raise ValueError('a query_key is required to use NDCG metric')
  sub_keys = [k for k in sub_keys if k is not None]
  if top_k_list:
    if sub_keys is None:
      sub_keys = []
    for k in top_k_list:
      if not any([sub_key.top_k == k for sub_key in sub_keys]):
        sub_keys.append(metric_types.SubKey(top_k=k))
  if not sub_keys or any([sub_key.top_k is None for sub_key in sub_keys]):
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
               eval_config: Optional[config.EvalConfig], model_name: Text,
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
      self,
      element: metric_types.StandardMetricInputs) -> Union[float, int, Text]:
    query = util.get_by_keys(element.features, [self._query_key]).flatten()
    if query.size == 0 or not np.all(query == query[0]):
      raise ValueError(
          'missing query value or not all values are the same: value={}, '
          'metric_keys={}, StandardMetricInputs={}'.format(
              query, self._metric_keys, element))
    return query[0]

  def _to_gains_example_weight(
      self,
      element: metric_types.StandardMetricInputs) -> Tuple[np.ndarray, float]:
    """Returns gains and example_weight sorted by prediction."""
    _, predictions, example_weight = next(
        metric_util.to_label_prediction_example_weight(
            element,
            eval_config=self._eval_config,
            model_name=self._model_name,
            output_name=self._output_name,
            flatten=False))  # pytype: disable=wrong-arg-types
    gains = util.get_by_keys(element.features, [self._gain_key])
    if gains.size != predictions.size:
      raise ValueError('expected {} to be same size as predictions {} != {}: '
                       'gains={}, metric_keys={}, '
                       'StandardMetricInputs={}'.format(self._gain_key,
                                                        gains.size,
                                                        predictions.size, gains,
                                                        self._metric_keys,
                                                        element))
    gains = gains.reshape(predictions.shape)
    # Ignore non-positive gains.
    if gains.max() <= 0:
      example_weight = 0.0
    return (gains[np.argsort(predictions)[::-1]], float(example_weight))

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

  def add_input(self, accumulator: _NDCGAccumulator,
                element: metric_types.StandardMetricInputs) -> _NDCGAccumulator:
    gains, example_weight = self._to_gains_example_weight(element)
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

  def extract_output(
      self,
      accumulator: _NDCGAccumulator) -> Dict[metric_types.MetricKey, float]:
    output = {}
    for i, key in enumerate(self._metric_keys):
      if accumulator.total_weighted_examples > 0:
        output[key] = accumulator.ndcg[i] / accumulator.total_weighted_examples
      else:
        output[key] = float('nan')
    return output
