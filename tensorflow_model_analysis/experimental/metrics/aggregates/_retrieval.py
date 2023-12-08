# Copyright 2023 Google LLC
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

"""Aggregates modules for all retrieval metrics."""

from collections.abc import Sequence
import dataclasses
from typing import Any

import numpy as np
from numpy import typing as npt
from tensorflow_model_analysis.experimental.lazytf import api as lazytf
from tensorflow_model_analysis.experimental.metrics.aggregates import _classification

StrEnum = _classification.StrEnum
InputType = _classification.InputType

MeanState = _classification.MeanState
MeanStatesPerMetric = dict[str, MeanState]

_NumbersT = npt.ArrayLike
_DefaultDType = np.float64


class RetrievalMetric(StrEnum):  # pylint: disable=invalid-enum-extension
  """Supported retrieval metrics."""

  CONFUSION_MATRIX = 'confusion_matrix'
  PRECISION = 'precision'


safe_divide = _classification.safe_divide


_SamplewiseMeanAggFnState = MeanStatesPerMetric


@dataclasses.dataclass(frozen=True, kw_only=True)
class TopKSamplewiseMeanAggFn(lazytf.AggregateFn):
  """TopK sample mean aggregate.

  Attributes:
    k_list: topk list used to truncate prediction. E.g., in an retrieval results
      of ['dog', 'cat', 'pig'], k=[2] means only ['dog', 'cat'] is used as the
      prediction. The default is None, which means all outputs in each
      prediction is included. Effectively k=None is same as k=[inf].
    metrics: The metrics to be computed.
    input_type: input encoding type, must be multiclass(-multioutput).
  """

  k_list: Sequence[int] | None = None
  metrics: Sequence[RetrievalMetric] = ()
  input_type: InputType = InputType.MULTICLASS_MULTIOUTPUT

  def __post_init__(self):
    if self.input_type not in (
        InputType.MULTICLASS_MULTIOUTPUT,
        InputType.MULTICLASS,
    ):
      raise NotImplementedError(f'"{str(self.input_type)}" is not supported.')

  def create_accumulator(self) -> _SamplewiseMeanAggFnState:
    """Creates an empty state."""
    return {metric: MeanState() for metric in self.metrics}

  def _compute_metric_states(self, *inputs, **kwargs):
    """The actual metric state calculation logic."""
    raise NotImplementedError()

  def add_inputs(
      self, mean_states: _SamplewiseMeanAggFnState, y_trues, y_preds
  ) -> _SamplewiseMeanAggFnState:
    """Updated the intermediate states by batches."""
    tp_metric = self._compute_metric_states(y_trues, y_preds)
    return self.merge_accumulators((mean_states, tp_metric))

  def merge_accumulators(
      self, states: Sequence[_SamplewiseMeanAggFnState]
  ) -> _SamplewiseMeanAggFnState:
    """Merge the intermediate metric states."""
    states_iter = iter(states)
    result = next(states_iter)
    for state in states_iter:
      for metric in self.metrics:
        result[metric] += state[metric]
    return result

  def extract_output(self, states: _SamplewiseMeanAggFnState) -> Any:
    """Extract the metric results from the intermediate states."""
    result = [states[metric].result() for metric in self.metrics]
    # Extends the remaining Ks from the last value.

    if self.k_list and result and len(self.k_list) > len(result[0]):
      for i, metric_result in enumerate(result):
        extra_ks = len(self.k_list) - len(metric_result)
        result[i] = list(metric_result) + [metric_result[-1]] * extra_ks
    return tuple(tuple(metric) for metric in result)


def _precision(tp_at_topks, k_list, y_pred_count):
  return tp_at_topks[:, k_list - 1] / np.minimum(
      k_list, y_pred_count[:, np.newaxis]
  )


@dataclasses.dataclass(frozen=True, kw_only=True)
class TopKRetrievalAggFn(TopKSamplewiseMeanAggFn):
  """TopKRetrievalAggFn aggregate.

  Attributes:
    k_list: topk list, default to None, which means all outputs are considered.
    metrics: The metrics to be computed.
    input_type: input encoding type, must be `multiclass` or
      `multiclass-multioutput`.
  """

  metrics: Sequence[RetrievalMetric] = (RetrievalMetric.PRECISION,)

  def _compute_metric_states(self, y_trues, y_preds):
    """Compute all true positive related metrics."""
    k_list = list(sorted(self.k_list)) if self.k_list else [float('inf')]
    y_pred_count = np.asarray([len(row) for row in y_preds])
    max_pred_count = max(y_pred_count)
    max_pred_count = min(max_pred_count, max(k_list))
    tp = []
    for y_pred, y_true in zip(y_preds, y_trues):
      tp.append([
          int(y_pred[i] in y_true) if i < len(y_pred) else 0
          for i in range(max_pred_count)
      ])
    tp = np.asarray(tp)
    # True positives at TopK is of a dimension of Examples x K as the following:
    # The first dimension is always batch dimension (# examples), the second
    # dimension can be either 0D (single-output) or 1D (multioutput) array.
    # E.g., provided one multioutput prediction and its true positive:
    # topk:         [1,    2,     3]
    # true-positive: [True, False, True]
    # We will get the true positives at topKs:
    # topk:     [top1, top2, top3]
    # tp_topks: [1,     1,      2]
    tp_at_topks = np.cumsum(tp, axis=1)
    # Truncates the k_list with maximum length of the predictions.
    k_list = np.asarray(
        [k for k in k_list if k < max_pred_count] + [max_pred_count]
    )

    result = {}
    if 'precision' in self.metrics:
      precision = _precision(tp_at_topks, k_list, y_pred_count)
      result['precision'] = MeanState(precision.sum(axis=0), precision.shape[0])

    return result
