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
"""Minimum label position.

Calculates the least index in a query which has a positive label. The final
returned value is the weighted average over all queries in the evaluation set
which have at least one labeled entry. Note, ranking is indexed from one, so the
optimal value for this metric is one. If there are no labeled rows in the
evaluation set, the final output will be zero.
"""
from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import apache_beam as beam
from tensorflow_model_analysis.eval_saved_model import constants as eval_saved_model_constants
from tensorflow_model_analysis.eval_saved_model import util as eval_saved_model_util
from tensorflow_model_analysis.evaluators.query_metrics import query_types
from tensorflow_model_analysis.post_export_metrics import metric_keys

from typing import Any, Dict, List, NamedTuple, Text

_State = NamedTuple('_State', [('min_pos_sum', float), ('weight_sum', float)])


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


class MinLabelPositionCombineFn(beam.CombineFn):
  """Computes minimum label position."""

  def __init__(self, label_key: Text, weight_key: Text):
    """Initialize.

    Args:
      label_key: The key in the labels dictionary which holds the label. Set
        this to empty to if labels is a Tensor and not a dictionary.
      weight_key: The key in the features dictionary which holds the weights.
        Note that the weight value must be identical across all examples in the
        same query. If set to empty, uses 1.0 instead.
    """
    if not label_key:
      # If label_key is set to the empty string, the user is telling us
      # that their Estimator returns a labels Tensor rather than a
      # dictionary. Set the key to the magic key we use in that case.
      self._label_key = eval_saved_model_util.default_dict_key(
          eval_saved_model_constants.LABELS_NAME)
    else:
      self._label_key = label_key
    self._weight_key = weight_key

  def _get_label(self, fpl: query_types.FPL) -> float:
    result = fpl['labels'].get(self._label_key)
    if result is None:
      return 0.0
    return result

  def create_accumulator(self):
    return _State(min_pos_sum=0.0, weight_sum=0.0)

  def _add_states(self, left: _State, right: _State) -> _State:
    return _State(
        min_pos_sum=left.min_pos_sum + right.min_pos_sum,
        weight_sum=left.weight_sum + right.weight_sum)

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

    min_label_pos = None
    for pos, fpl in enumerate(query_fpl.fpls):
      if self._get_label(fpl) > 0:
        min_label_pos = pos + 1  # Use 1-indexed positions
        break

    state_to_add = _State(min_pos_sum=0.0, weight_sum=0.0)
    if min_label_pos:
      state_to_add = _State(min_pos_sum=min_label_pos, weight_sum=weight)

    return self._add_states(accumulator, state_to_add)

  def merge_accumulators(self, accumulators: List[_State]) -> _State:
    result = self.create_accumulator()
    for accumulator in accumulators:
      result = self._add_states(result, accumulator)
    return result

  def extract_output(self, accumulator: _State) -> Dict[Text, Any]:
    if accumulator.weight_sum > 0:
      return {
          metric_keys.base_key('average_min_label_position/%s' %
                               self._label_key):
              accumulator.min_pos_sum / accumulator.weight_sum
      }
    return {}
