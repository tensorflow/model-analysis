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
"""Aggregates modules for all classification metrics."""

from collections.abc import Iterable, Sequence
import dataclasses
import enum
import itertools
from typing import Any, Optional, Union

import numpy as np
from numpy import typing as npt
from tensorflow_model_analysis.experimental.lazytf import api as lazytf


_NumbersT = npt.ArrayLike
_DefaultDType = np.float64


# TODO: b/312290886 - move this to Python StrEnum when moved to Python 3.11.
class StrEnum(str, enum.Enum):
  """Enum where members also must be strings."""

  __str__ = str.__str__

  __repr__ = str.__repr__

  __format__ = str.__format__

  __iter__ = enum.Enum.__iter__


@dataclasses.dataclass
class MeanState:
  """Mergeable states for batch update in an aggregate function."""

  total: _NumbersT = 0.0
  count: _NumbersT = 0

  def __iadd__(self, other: 'MeanState'):
    self.total += other.total
    self.count += other.count
    return self

  def result(self):
    return safe_divide(self.total, self.count)


class ConfusionMatrixMetric(StrEnum):  # pylint: disable=invalid-enum-extension
  CONFUSION_MATRIX = 'confusion_matrix'
  PRECISION = 'precision'
  RECALL = 'recall'
  F1_SCORE = 'f1_score'
  ACCURACY = 'accuracy'
  MEAN_AVERAGE_PRECISION = 'mean_average_precision'


class _ConfusionMatrix:
  """Confusion Matrix accumulator with kwargs used to compute it.

  Confusion matrix for classification and retrieval tasks.
  See https://en.wikipedia.org/wiki/Confusion_matrix for more info.

  Attributes:
    tp: True positives count.
    tn: True negatives count.
    fp: False positives count.
    fn: False negative count.
    average: The average type of which this confusion matrix is computed.
    dtype: data type of the instance numpy array. None by default, dtype is
      deduced by numpy at construction.
  """

  def __init__(
      self,
      tp: _NumbersT,
      tn: _NumbersT,
      fp: _NumbersT,
      fn: _NumbersT,
      *,
      dtype: Optional[type[Any]] = None,
  ):
    self.tp = np.asarray(tp, dtype=dtype)
    self.tn = np.asarray(tn, dtype=dtype)
    self.fp = np.asarray(fp, dtype=dtype)
    self.fn = np.asarray(fn, dtype=dtype)
    self.dtype = dtype

  @property
  def t(self):
    """Labeled True count."""
    return self.tp + self.fn

  @property
  def p(self):
    """Predicted True count."""
    return self.tp + self.fp

  def __iadd__(self, other):
    self.tp += other.tp
    self.tn += other.tn
    self.fp += other.fp
    self.fn += other.fn
    return self

  def __add__(self, other):
    tp = self.tp + other.tp
    tn = self.tn + other.tn
    fp = self.fp + other.fp
    fn = self.fn + other.fn
    return _ConfusionMatrix(tp, tn, fp, fn)

  def __eq__(self, other):
    """Numerically equals."""
    return (
        np.allclose(self.tp, other.tp)
        and np.allclose(self.tn, other.tn)
        and np.allclose(self.fp, other.fp)
        and np.allclose(self.fn, other.fn)
    )

  def __repr__(self):
    return f'tp={self.tp}, tn={self.tn}, fp={self.fp}, fn={self.fn}'

  def derive_metric(
      self, metric: ConfusionMatrixMetric, average=None
  ) -> _NumbersT:
    """Helper to call the right metric function given a Metric Enum."""
    if metric == ConfusionMatrixMetric.PRECISION:
      result = _precision(self)
    elif metric == ConfusionMatrixMetric.RECALL:
      result = _recall(self)
    elif metric == ConfusionMatrixMetric.F1_SCORE:
      result = _f1(self)
    elif metric == ConfusionMatrixMetric.ACCURACY:
      result = _accuracy(self)
    else:
      raise NotImplementedError(f'"{metric}" metric is not supported.')
    assert (
        average != AverageType.SAMPLES
    ), 'Unexpected samplewise average for a derived metric.'
    if average is None or average in ('micro', 'binary'):
      return result
    elif average == 'macro':
      return np.mean(result, axis=0)
    else:
      raise NotImplementedError(f'"{average}" average is not supported.')


class _TopKConfusionMatrix(_ConfusionMatrix):
  """Confusion Matrix accumulator with kwargs used to compute it.

  Confusion matrix for classification and retrieval tasks.
  See https://en.wikipedia.org/wiki/Confusion_matrix for more info.

  Attributes:
    k: a sequence of topk, sequentially corresponds to the actual confusion
      matrix counts (tp, tn, fp, fn).
    tp: True positives count.
    tn: True negatives count.
    fp: False positives count.
    fn: False negative count.
    dtype: data type of the instance numpy array. None by default, dtype is
      deduced by numpy at construction.
  """

  def __init__(
      self,
      k: _NumbersT,
      tp: _NumbersT,
      tn: _NumbersT,
      fp: _NumbersT,
      fn: _NumbersT,
      *,
      dtype: Optional[type[Any]] = None,
  ):
    self.k = np.asarray(k, dtype=dtype)
    super().__init__(tp, tn, fp, fn, dtype=dtype)

  def __eq__(self, other):
    """Numerically equals."""
    return np.allclose(self.k, other.k) and super().__eq__(other)

  def __str__(self):
    return f'k={self.k}, tp={self.tp}, tn={self.tn}, fp={self.fp}, fn={self.fn}'


class InputType(StrEnum):  # pylint: disable=invalid-enum-extension
  """Label prediction encoding types."""

  # 1D array per batch, e.g., [0,1,0,1,0], [-1, 1, -1], or ['Y', 'N']
  BINARY = 'binary'
  # 1D array of floats typically is the probability for the binary
  # classification problem, e.g., [0.2, 0.3, 0.9]
  CONTINUOUS = 'continuous'
  # 2D array of the floats for the multilabel/multiclass classification problem.
  # Dimension: BatchSize x # Class
  # e.g., [[0.2, 0.8, 0.9], [0.1, 0.2, 0.7]].
  CONTINUOUS_MULTIOUTPUT = 'continuous-multioutput'
  # 1D array of class identifiers, e.g, ['a', 'b'] or [1, 29, 12].
  MULTICLASS = 'multiclass'
  # 2D lists of multiclass encodings of the classes, e.g., [[1,2,0], [3,2,0]]
  # The list can be ragged, e.g, [ ['a', 'b'], ['c'] ]
  MULTICLASS_MULTIOUTPUT = 'multiclass-multioutput'
  # 2D array of one-hot encoding of the classes, e.g., [[0,1,0], [0,0,1]]
  # This is a special case for "multilabel-indicator" except that only one
  # class is set to positive per example.
  MULTICLASS_INDICATOR = 'multiclass-indicator'


class AverageType(StrEnum):  # pylint: disable=invalid-enum-extension
  """Average type of the confusion matrix."""

  # Treats each class as one example and calculates the metrics on the total
  # aggregates of the result.
  MICRO = 'micro'
  # Macro calculates metrics for each class first, then average them across
  # classes.
  MACRO = 'macro'
  # Macro average with explicit weights per class.
  WEIGHTED = 'weighted'
  # Samples average calculates the metrics per example, and average them across
  # all examples.
  SAMPLES = 'samples'
  # Average for the positive label only.
  BINARY = 'binary'


def safe_divide(a, b):
  result = np.divide(
      a, b, out=np.zeros_like(a, dtype=_DefaultDType), where=(b != 0)
  )
  if result.ndim == 0:
    return result.item()
  return result


def _precision(cm: _ConfusionMatrix):
  return safe_divide(cm.tp, cm.p)


def _recall(cm: _ConfusionMatrix):
  return safe_divide(cm.tp, cm.t)


def _f1(cm: _ConfusionMatrix):
  precision = _precision(cm)
  recall = _recall(cm)
  return safe_divide(2 * precision * recall, precision + recall)


def _accuracy(cm: _ConfusionMatrix) -> _NumbersT:
  """Accuracy, only meaningful for samplewise ConfusionMatrixAtK."""
  return (cm.tp > 0).astype(int)


def _indicator_confusion_matrix(
    y_true: _NumbersT,
    y_pred: _NumbersT,
    *,
    pos_label: Union[int, str, bytes] = 1,
    multiclass: bool = False,
    average: AverageType = AverageType.MICRO,
) -> _ConfusionMatrix:
  """Calculates confusion matix.

  Args:
    y_true: ground truth classification labels. This has to be encoded in one of
      the `input_type`.
    y_pred: classification preditions. This has to be encoded in one of the
      `input_type`.
    pos_label: label to be recognized as positive class, only relevant when
      input type is "binary".
    multiclass: If True, input is encoded as 2D array-like of "multi-hot"
      encoding in a shape of Batch X NumClass.
    average: average type of the confusion matrix.

  Returns:
    Confusion matrix.
  """
  if average in (AverageType.MICRO, AverageType.BINARY):
    axis = None
  elif average is None or average == AverageType.MACRO:
    axis = 0
  elif average == AverageType.SAMPLES:
    axis = 1
  else:
    raise NotImplementedError(f'"{average}" average is not supported.')

  y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
  if multiclass and (y_true.ndim < 2 or y_pred.ndim < 2):
    raise ValueError(
        'Multiclass indicator input needs to be 2D array, actual dimensions:'
        f' y_true: {y_true.ndim}; y_pred: {y_pred.ndim}'
    )

  true = y_true == pos_label
  positive = y_pred == pos_label
  # Forces the input to binary if average type is binary.
  if multiclass and average == AverageType.BINARY:
    if true.shape[1] > 2:
      raise ValueError(
          'Non-binary multiclass indicator input is not supported for `binary`'
          f' average. #classes is {true.shape[1]}'
      )
    true = true[:, 0]
    positive = positive[:, 0]
  # Reshapes to a multi-class format to Batch X Classes
  elif not multiclass and average != AverageType.BINARY:
    true = np.vstack((true, ~true)).T
    positive = np.vstack((positive, ~positive)).T
  negative = ~positive
  tp = positive & true
  fn = negative & true

  positive_cnt = positive.sum(axis=axis)
  tp_cnt = tp.sum(axis=axis)
  fp_cnt = positive_cnt - tp_cnt
  fn_cnt = fn.sum(axis=axis)
  negative_cnt = negative.sum(axis=axis)
  tn_cnt = negative_cnt - fn_cnt
  return _ConfusionMatrix(tp_cnt, tn_cnt, fp_cnt, fn_cnt)


def _get_vocab(rows: Iterable[Any], multioutput: bool):
  """Constructs a vocabulary that maps hashables to an integer."""
  if multioutput:
    return {
        k: i for i, k in enumerate(set(itertools.chain.from_iterable(rows)))
    }
  else:
    return {k: i for i, k in enumerate(set(rows))}


def _apply_vocab(rows: Sequence[Any], vocab: dict[str, int], multioutput: bool):
  """Given a vocabulary, converts a multioutput input to a indicator output."""
  result = np.empty((len(rows), len(vocab)), dtype=np.bool_)
  result.fill(False)
  if multioutput:
    for i, row in enumerate(rows):
      for elem in row:
        result[i][vocab[elem]] = True
  else:
    for i, elem in enumerate(rows):
      result[i][vocab[elem]] = True
  return result


def _multiclass_confusion_matrix(
    y_true: _NumbersT,
    y_pred: _NumbersT,
    *,
    vocab: Optional[dict[str, int]] = None,
    multioutput: bool = False,
    average: AverageType = AverageType.MICRO,
) -> _ConfusionMatrix:
  """Calculates a confusion matrix for multiclass(-multioutput) input.

  Args:
    y_true: ground truth classification labels. This has to be encoded in one of
      the `multiclass` or `multiclass-output`.
    y_pred: classification preditions. This has to be encoded in one of the
      `multiclass` or `multiclass-output`.
    vocab: an external vocabulary that maps categorical value to integer class
      id. If not provided, one is deduced within this input.
    multioutput: encoding types of the y_true and y_pred, if True, the input is
      a nested list of class identifiers.
    average: average type of the confusion matrix.

  Returns:
    Confusion matrices with k in k_list.
  """
  vocab = vocab or _get_vocab(itertools.chain(y_true, y_pred), multioutput)
  y_true_dense = _apply_vocab(y_true, vocab, multioutput)
  y_pred_dense = _apply_vocab(y_pred, vocab, multioutput)
  return _indicator_confusion_matrix(
      y_true_dense,
      y_pred_dense,
      average=average,
      multiclass=True,
      pos_label=True,
  )


ConfusionMatrixAggState = _ConfusionMatrix


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConfusionMatrixAggFn(lazytf.AggregateFn):
  """ConfusionMatrix aggregate.

  Attributes:
    pos_label: the value considered as positive, default to 1.
    input_type: input encoding type, must be one of `InputType`.
    average: average type, must be one of `AverageType`.
    vocab: an external vocabulary that maps categorical value to integer class
      id. This is required if computed distributed (when merge_accumulators is
      called) and the average is macro where the class id mapping needs to be
      stable.
    dtype: dtype of the confusion matrix and all computations. Default to None
      as it is inferred.
  """

  pos_label: Union[bool, int, str, bytes] = 1
  input_type: InputType = InputType.BINARY
  # TODO(b/311208939): implements average = None.
  average: AverageType = AverageType.BINARY
  vocab: Optional[dict[str, int]] = None
  metrics: Sequence[ConfusionMatrixMetric] = ()
  dtype: Optional[type[Any]] = None

  def __post_init__(self):
    if self.average == AverageType.SAMPLES:
      raise ValueError(
          '"samples" average is unsupported, use the Samplewise version.'
      )

  def _calculate_confusion_matrix(
      self,
      y_true: _NumbersT,
      y_pred: _NumbersT,
  ) -> _ConfusionMatrix:
    if self.input_type in (InputType.MULTICLASS_INDICATOR, InputType.BINARY):
      return _indicator_confusion_matrix(
          y_true,
          y_pred,
          pos_label=self.pos_label,
          multiclass=(self.input_type == InputType.MULTICLASS_INDICATOR),
          average=self.average,
      )
    elif self.input_type in (
        InputType.MULTICLASS,
        InputType.MULTICLASS_MULTIOUTPUT,
    ):
      return _multiclass_confusion_matrix(
          y_true,
          y_pred,
          vocab=self.vocab,
          multioutput=(self.input_type == InputType.MULTICLASS_MULTIOUTPUT),
          average=self.average,
      )
    else:
      raise NotImplementedError(f'"{self.input_type}" input is not supported.')

  def add_inputs(
      self, state: Optional[ConfusionMatrixAggState], *inputs: Any
  ) -> ConfusionMatrixAggState:
    cm = self._calculate_confusion_matrix(*inputs)
    return (cm + state) if state else cm

  def merge_accumulators(
      self, states: list[ConfusionMatrixAggState]
  ) -> ConfusionMatrixAggState:
    if (
        self.average in (AverageType.WEIGHTED, AverageType.MACRO)
        and self.vocab is None
    ):
      raise ValueError(f'Global vocab is needed for "{self.average}" average.')
    iter_acc = iter(states)
    result = next(iter_acc)
    for accumulator in iter_acc:
      result += accumulator
    return result

  def extract_output(self, state: ConfusionMatrixAggState) -> Any:
    if self.metrics:
      return tuple(
          state.derive_metric(metric, average=self.average)
          for metric in self.metrics
      )
    return (state,)


def _apply_vocab_at_k(
    rows: _NumbersT,
    vocab: dict[str, int],
    multioutput: bool,
    k_list: Sequence[int],
):
  """Encodes a multiclass(-multioutput) input in multiclass-indicator format."""
  result = np.full((len(rows), len(vocab)), False, dtype=np.bool_)
  k_list = set(k_list)
  for j in range(max(k_list)):
    if multioutput:
      for i, row in enumerate(rows):
        if j < len(row):
          result[i][vocab[row[j]]] = True
    else:
      if j == 0:
        for i, elem in enumerate(rows):
          result[i][vocab[elem]] = True
    if j + 1 in k_list:
      yield j + 1, result


def _topk_confusion_matrix(
    y_true: _NumbersT,
    y_pred: _NumbersT,
    *,
    k_list: Sequence[int],
    multioutput: bool,
    average: AverageType,
    vocab: Optional[dict[str, int]],
) -> _TopKConfusionMatrix:
  """Calculates a confusion matrix for multiclass(-multioutput) input.

  Args:
    y_true: ground truth classification labels. This has to be encoded in one of
      the `multiclass` or `multiclass-output`.
    y_pred: classification predictions. This has to be encoded in one of the
      `multiclass` or `multiclass-output`.
    k_list: a list of topk each of which slices y_pred by y_pred[:topk] assuming
      the predictions are sorted in descending order.
    multioutput: encoding types of the y_true and y_pred, if True, the input is
      a nested list of class identifiers.
    average: average type of the confusion matrix.
    vocab: an external vocabulary, if not provided, one is deduced within this
      input.

  Returns:
    Confusion matrices with k in k_list.
  """
  vocab = vocab or _get_vocab(itertools.chain(y_true, y_pred), multioutput)
  y_true_dense = _apply_vocab(y_true, vocab, multioutput)
  cms = []
  for k, y_pred_dense in _apply_vocab_at_k(y_pred, vocab, multioutput, k_list):
    cm = _indicator_confusion_matrix(
        y_true_dense,
        y_pred_dense,
        average=average,
        multiclass=True,
        pos_label=True,
    )
    cms.append((k, cm.tp, cm.tn, cm.fp, cm.fn))
  return _TopKConfusionMatrix(*tuple(zip(*cms)))


@dataclasses.dataclass(kw_only=True, frozen=True)
class TopKConfusionMatrixAggFn(ConfusionMatrixAggFn):
  """ConfusionMatrixAtK aggregate.

  Attributes:
    pos_label: the value considered as positive, default to 1.
    input_type: input encoding type, must be either  `multiclass` or
      `multiclass-multioutput`.
    average: average type, must be one of the types under `AverageType`.
    vocab: an external vocabulary that maps categorical value to integer class
      id. This is required if computed distributed (when merge_accumulators is
      called) and the average is macro where the class id mapping needs to be
      stable.
    k_list: a list of topk each of which slices y_pred by y_pred[:topk] assuming
      the predictions are sorted in descending order.
  """

  k_list: Sequence[int] = ()

  def __post_init__(self):
    if self.input_type not in (
        InputType.MULTICLASS,
        InputType.MULTICLASS_MULTIOUTPUT,
    ):
      raise ValueError(
          f'"{self.input_type}" input is not supported for TopK Confusion'
          ' Matrix.'
      )

  def _calculate_confusion_matrix(
      self, y_true: _NumbersT, y_pred: _NumbersT
  ) -> _TopKConfusionMatrix:
    result = _topk_confusion_matrix(
        y_true,
        y_pred,
        k_list=self.k_list,
        vocab=self.vocab,
        multioutput=(self.input_type == InputType.MULTICLASS_MULTIOUTPUT),
        average=AverageType(self.average),
    )
    return result


SamplewiseConfusionMatrixAggState = dict[str, MeanState]


@dataclasses.dataclass(frozen=True, kw_only=True)
class SamplewiseConfusionMatrixAggFn(lazytf.AggregateFn):
  """ConfusionMatrix aggregate.

  Attributes:
    pos_label: the value considered as positive, default to 1.
    input_type: input encoding type, must be one of `InputType`.
    average: fixed as `samples` average.
    vocab: an external vocabulary that maps categorical value to integer class
      id. This is required if computed distributed (when merge_accumulators is
      called) and the average is macro where the class id mapping needs to be
      stable.
    dtype: dtype of the confusion matrix and all computations. Default to None
      as it is inferred.
  """

  metrics: Sequence[ConfusionMatrixMetric]
  pos_label: Union[bool, int, str, bytes] = 1
  input_type: InputType = InputType.BINARY
  average: AverageType = dataclasses.field(
      default=AverageType.SAMPLES, init=False
  )
  vocab: Optional[dict[str, int]] = None
  dtype: Optional[type[Any]] = None

  def __post_init__(self):
    if self.input_type == InputType.BINARY:
      raise ValueError(
          'Samples average is not available for Binary classification.'
      )

  def create_accumulator(self) -> SamplewiseConfusionMatrixAggState:
    """Creates the initial empty state."""
    return {}

  def _metric_states(self, cm: _ConfusionMatrix) -> dict[str, MeanState]:
    result = {}
    for metric in self.metrics:
      if (score := cm.derive_metric(metric)) is not None:
        result[metric] = MeanState(np.sum(score), len(score))
    return result

  def _calculate_confusion_matrix(
      self,
      y_true: _NumbersT,
      y_pred: _NumbersT,
  ) -> _ConfusionMatrix:
    if self.input_type in (InputType.MULTICLASS_INDICATOR, InputType.BINARY):
      return _indicator_confusion_matrix(
          y_true,
          y_pred,
          pos_label=self.pos_label,
          multiclass=(self.input_type == InputType.MULTICLASS_INDICATOR),
          average=self.average,
      )
    elif self.input_type in (
        InputType.MULTICLASS,
        InputType.MULTICLASS_MULTIOUTPUT,
    ):
      return _multiclass_confusion_matrix(
          y_true,
          y_pred,
          vocab=self.vocab,
          multioutput=(self.input_type == InputType.MULTICLASS_MULTIOUTPUT),
          average=self.average,
      )
    else:
      raise NotImplementedError(f'"{self.input_type}" input is not supported.')

  def add_inputs(
      self, state: SamplewiseConfusionMatrixAggState, *inputs: Any
  ) -> SamplewiseConfusionMatrixAggState:
    """Batch updates the states of the aggregate."""
    cm = self._calculate_confusion_matrix(*inputs)
    metric_states = self._metric_states(cm)
    return self.merge_accumulators((metric_states, state))

  def merge_accumulators(
      self, states: Sequence[SamplewiseConfusionMatrixAggState]
  ) -> SamplewiseConfusionMatrixAggState:
    states_iter = iter(states)
    result = next(states_iter)
    for state in states_iter:
      for key, value in state.items():
        result[key] += value
    return result

  def extract_output(self, state: SamplewiseConfusionMatrixAggState) -> Any:
    """Extracts the outputs from the aggregate states."""
    return tuple(state[metric].result() for metric in self.metrics)
