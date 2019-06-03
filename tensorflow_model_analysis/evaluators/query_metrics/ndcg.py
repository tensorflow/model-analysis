# Copyright 2018 Google LLC
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
r"""Normalized discounted cumulative gain.

Calculate nDCG@k for a all configured values of k and using the value of gain
in the 'gain_key' feature configured by the ndcg_metric proto. For each k in
at_vals, the value of nDCG@k returned is a weighted average of nDCG@k over the
set of queries using the assigned weights.

nDCG@k = (DCG@k for the given rank)/(DCG@k [optimally] ranked by the gain value)
DCG@k = \sum_{i=1}^k gain_i/log_2(i+1), where gain_i is the gain (relevance
score) of the i^th ranked response, indexed from 1.
"""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import apache_beam as beam
import numpy as np
from tensorflow_model_analysis.evaluators.query_metrics import query_types
from tensorflow_model_analysis.post_export_metrics import metric_keys

from typing import Any, Dict, List, NamedTuple, Text, Tuple

_State = NamedTuple('_State', [('ndcg', Dict[int, float]), ('weight', float)])


def _get_feature_value(fpl: query_types.FPL, key: Text) -> float:
  """Get value of the given feature from the features dictionary.

  The feature must have exactly one value.

  Args:
    fpl: FPL
    key: Key of feature to retrieve in features dictionary.

  Returns:
    The singular value of the feature.
  """
  feature = fpl['features'].get(key)
  if feature is None:
    raise ValueError('feature %s not found in features %s' %
                     (key, fpl['features']))
  if feature.size != 1:
    raise ValueError('feature %s did not contain exactly 1 value. '
                     'value was: %s' % (key, feature))
  return feature[0][0]


class NdcgMetricCombineFn(beam.CombineFn):
  """Computes normalized discounted cumulative gain."""

  def __init__(self, at_vals: List[int], gain_key: Text, weight_key: Text):
    """Initialize.

    Args:
      at_vals: A list containing the number of values to consider in calculating
        the values of nDCG (eg. nDCG@at).
      gain_key: The key in the features dictionary which holds the gain values.
      weight_key: The key in the features dictionary which holds the weights.
        Note that the weight value must be identical across all examples in the
        same query. If set to empty, uses 1.0 instead.
    """
    self._at_vals = at_vals
    self._gain_key = gain_key
    self._weight_key = weight_key

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
    """Calculate nDCG@k, based on given rank and gain values.

    Args:
      values: A list of tuples representing rank order and gain values.
      k: The maximum position to consider in calculating nDCG

    Returns:
      The value of nDCG@k, for the given list of values.
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

  def _new_ndcg_dict(self):
    return dict.fromkeys(self._at_vals, 0)

  def create_accumulator(self):
    return _State(ndcg=self._new_ndcg_dict(), weight=0.0)

  def _add_states(self, left: _State, right: _State) -> _State:
    ndcg_dict = self._new_ndcg_dict()
    for at in self._at_vals:
      ndcg_dict[at] = left.ndcg[at] + right.ndcg[at]
    return _State(ndcg_dict, left.weight + right.weight)

  def add_input(self, accumulator: _State,
                query_fpl: query_types.QueryFPL) -> _State:
    weight = 1.0
    if self._weight_key:
      weights = [
          float(_get_feature_value(fpl, self._weight_key))
          for fpl in query_fpl.fpls
      ]
      if weights:
        if min(weights) != max(weights):
          raise ValueError('weights were not identical for all examples in the '
                           'query. query_id was: %s, weights were: %s' %
                           (query_fpl.query_id, weights))
        weight = weights[0]

    ndcg_dict = {}
    for at in self._at_vals:
      rank_gain = [(pos + 1, float(_get_feature_value(fpl, self._gain_key)))
                   for pos, fpl in enumerate(query_fpl.fpls)]
      ndcg_dict[at] = self._calculate_ndcg(rank_gain, at) * weight

    return self._add_states(accumulator, _State(ndcg=ndcg_dict, weight=weight))

  def merge_accumulators(self, accumulators: List[_State]) -> _State:
    result = self.create_accumulator()
    for accumulator in accumulators:
      result = self._add_states(result, accumulator)
    return result

  def extract_output(self, accumulator: _State) -> Dict[Text, Any]:
    avg_dict = {}
    for at in self._at_vals:
      if accumulator.weight > 0:
        avg_ndcg = accumulator.ndcg[at] / accumulator.weight
      else:
        avg_ndcg = 0
      avg_dict[metric_keys.base_key('ndcg@%d' % at)] = avg_ndcg
    return avg_dict
