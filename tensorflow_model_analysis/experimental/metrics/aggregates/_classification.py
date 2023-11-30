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

from collections.abc import Sequence
import dataclasses
import enum
import itertools
from typing import Any

import numpy as np
from numpy import typing as npt
from tensorflow_model_analysis.experimental.lazytf import api as lazytf

_NumbersT = npt.ArrayLike


class ConfusionMatrix:
  """Confusion Matrix accumulator with kwargs used to compute it.

  Confusion matrix for classification and retrieval tasks.
  See https://en.wikipedia.org/wiki/Confusion_matrix for more info.

  Attributes:
    tp: True positives count.
    tn: True negatives count.
    fp: False positives count.
    fn: False negative count.
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
      dtype: type[Any] | None = None,
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
    return ConfusionMatrix(tp, tn, fp, fn)

  def __eq__(self, other):
    """Numerically equals."""
    return (
        np.allclose(self.tp, other.tp)
        and np.allclose(self.tn, other.tn)
        and np.allclose(self.fp, other.fp)
        and np.allclose(self.fn, other.fn)
    )

  @classmethod
  def concatenate(cls, cms: Sequence['ConfusionMatrix']):
    """Helper function to concatenate each nparray inside a ConfusionMatrix."""
    tp = np.concatenate([cm.tp for cm in cms], axis=0)
    tn = np.concatenate([cm.tn for cm in cms], axis=0)
    fp = np.concatenate([cm.fp for cm in cms], axis=0)
    fn = np.concatenate([cm.fn for cm in cms], axis=0)
    return cls(tp, tn, fp, fn)


# TODO(b/312290886): move this to Python StrEnum when moved to Python 3.11.
class _StrEnum(str, enum.Enum):
  """Enum where members are must be strings."""


class InputType(_StrEnum):
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


class AverageType(_StrEnum):
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


def _indicator_confusion_matrix(
    y_true: _NumbersT,
    y_pred: _NumbersT,
    *,
    pos_label=1,
    multiclass: bool = False,
    average: AverageType = AverageType.MICRO,
    unknown_tn: bool = False,
) -> ConfusionMatrix:
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
    unknown_tn: tn can be unknown if the total number of classes cannot be
      deduced from the inputs shape, this is most useful for the
      multiclass-multioutput case where only the trues and positives are known.

  Returns:
    Confusion matrix.
  """
  if average in (AverageType.MICRO, AverageType.BINARY):
    axis = None
  elif average == AverageType.MACRO:
    axis = 0
  elif average == AverageType.SAMPLES:
    axis = 1
  else:
    raise NotImplementedError(f'{average=} is not supported.')

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
  # Reshapes to a multi-class format if average type is not BINARY.
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
  if unknown_tn:
    tn_cnt = float('inf')
  else:
    negative_cnt = negative.sum(axis=axis)
    tn_cnt = negative_cnt - fn_cnt
  return ConfusionMatrix(tp_cnt, tn_cnt, fp_cnt, fn_cnt)


def _get_vocab(rows: Sequence[Any], multioutput: bool):
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
    vocab: dict[str, int] | None = None,
    multioutput: bool = False,
    average: AverageType = AverageType.MICRO,
    unknown_tn: bool = True,
) -> ConfusionMatrix:
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
    unknown_tn: tn can be unknown if the total number of classes cannot be
      deduced from the inputs shape, this is most useful for the
      multiclass-multioutput case where only the trues and positives are known.

  Returns:
    Confusion matrices with k in k_list.
  """
  vocab = vocab or (
      _get_vocab(y_true, multioutput) | _get_vocab(y_pred, multioutput)
  )
  y_true_dense = _apply_vocab(y_true, vocab, multioutput)
  y_pred_dense = _apply_vocab(y_pred, vocab, multioutput)
  return _indicator_confusion_matrix(
      y_true_dense,
      y_pred_dense,
      average=average,
      multiclass=True,
      pos_label=True,
      unknown_tn=unknown_tn,
  )


@dataclasses.dataclass(kw_only=True)
class ConfusionMatrixAggregate(lazytf.AggregateFn):
  """ConfusionMatrix aggregate.

  Attributes:
    pos_label: the value considered as positive, default to 1.
    input_type: input encoding type, must be one of `InputType`.
    average: average type, must be one of `AverageType`.
    unkown_tn: unkonw true negative, applicable when the number of classes is
      unknown, typically used for retrieval cases. If True, the true negative is
      set to inf.
    vocab: an external vocabulary that maps categorical value to integer class
      id. This is required if computed distributed (when merge_accumulators is
      called) and the average is macro where the class id mapping needs to be
      stable.
  """

  pos_label: bool | int | str | bytes = 1
  input_type: str | InputType = InputType.BINARY
  # TODO(b/311208939): implements average = None.
  average: str | AverageType = AverageType.BINARY
  unknown_tn: bool = False
  vocab: dict[str, int] | None = None

  def _calculate_confusion_matrix(
      self,
      y_true: _NumbersT,
      y_pred: _NumbersT,
  ) -> ConfusionMatrix:
    input_type = InputType(self.input_type)
    if input_type in (InputType.MULTICLASS_INDICATOR, InputType.BINARY):
      return _indicator_confusion_matrix(
          y_true,
          y_pred,
          pos_label=self.pos_label,
          multiclass=(input_type == InputType.MULTICLASS_INDICATOR),
          average=AverageType(self.average),
          unknown_tn=self.unknown_tn,
      )
    elif input_type in (
        InputType.MULTICLASS,
        InputType.MULTICLASS_MULTIOUTPUT,
    ):
      return _multiclass_confusion_matrix(
          y_true,
          y_pred,
          vocab=self.vocab,
          multioutput=(input_type == InputType.MULTICLASS_MULTIOUTPUT),
          average=self.average,
          unknown_tn=self.unknown_tn,
      )
    else:
      # TODO(b/311208939): implements continuous input type with thresholds.
      raise NotImplementedError(f'{input_type=} is not supported.')

  def add_inputs(self, accumulator: Any, *inputs: Any) -> Any:
    result = self._calculate_confusion_matrix(*inputs)
    if accumulator:
      # TODO(b/311208939): implements distributed efficient samplewise metric.
      if self.average == AverageType.SAMPLES:
        result = result.__class__.concatenate((accumulator, result))
      else:
        result += accumulator
    return result

  def merge_accumulators(
      self, accumulators: list[ConfusionMatrix]
  ) -> ConfusionMatrix:
    if self.average == AverageType.SAMPLES:
      return accumulators[0].__class__.concatenate(accumulators)
    if (
        self.average in (AverageType.WEIGHTED, AverageType.MACRO)
        and self.vocab is None
    ):
      raise ValueError(f'Global vocab is needed for {self.average} average.')
    iter_acc = iter(accumulators)
    result = next(iter_acc)
    for accumulator in iter_acc:
      result += accumulator
    return result
