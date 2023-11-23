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

import dataclasses
import enum
from typing import Any

import numpy as np
from numpy import typing as npt
from tensorflow_model_analysis.experimental.lazytf import api as lazytf

_NumbersT = npt.ArrayLike


@dataclasses.dataclass
class ConfusionMatrixAcc:
  """Confusion Matrix accumulator with kwargs used to compute it.

  Confusion matrix for classification and retrieval tasks.
  See https://en.wikipedia.org/wiki/Confusion_matrix for more info.

  Attributes:
    tp: True positives count.
    tn: True negatives count.
    fp: False positives count.
    fn: False negative count.
    config: The relevant configuration for downstream calculations, e.g, micro
      averaged confusion matrix is computed differently from macro averaged
      confusion matrix for precision, recall.
  """

  tp: _NumbersT
  tn: _NumbersT
  fp: _NumbersT
  fn: _NumbersT
  config: dict[str, Any] = dataclasses.field(default_factory=dict)

  @property
  def t(self):
    """Labeled True count."""
    return self.tp + self.fn

  @property
  def p(self):
    """Predicted True count."""
    return self.tp + self.fp

  def _verify_config(self, other):
    if self.config != other.config:
      raise ValueError(
          f'Incompatile config: {other.config}: \nOriginal:{self.config}'
      )

  def __iadd__(self, other):
    self._verify_config(other)
    self.tp += other.tp
    self.tn += other.tn
    self.fp += other.fp
    self.fn += other.fn
    return self

  def __add__(self, other):
    self._verify_config(other)
    tp = self.tp + other.tp
    tn = self.tn + other.tn
    fp = self.fp + other.fp
    fn = self.fn + other.fn
    return ConfusionMatrixAcc(tp, tn, fp, fn, self.config)

  def __eq__(self, other):
    """Numerically equals ignoring config."""
    return (
        np.allclose(self.tp, other.tp)
        and np.allclose(self.tn, other.tn)
        and np.allclose(self.fp, other.fp)
        and np.allclose(self.fn, other.fn)
    )


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


def confusion_matrix(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    *,
    pos_label=1,
    input_type: _StrEnum = InputType.BINARY,
    average: _StrEnum = AverageType.MICRO,
    unknown_tn: bool = False,
) -> ConfusionMatrixAcc:
  """Calculates confusion matix.

  Args:
    y_true: ground truth classification labels. This has to be encoded in one of
      the `input_type`.
    y_pred: classification preditions. This has to be encoded in one of the
      `input_type`.
    pos_label: label to be recognized as positive class, only relevant when
      input type is "binary".
    input_type: encoding types of the y_true and y_pred.
    average: average type of the confusion matrix.
    unknown_tn: tn can be unknown if the total number of classes cannot be
      deduced from the inputs shape, this is most useful for the
      multiclass-multioutput case where only the trues and positives are known.

  Returns:
    Confusion matrix.
  """
  # TODO(b/311208939): Implement other types of input type.
  if input_type not in (InputType.BINARY, InputType.MULTICLASS_INDICATOR):
    raise NotImplementedError(f'{input_type=} is not supported.')

  if average in (AverageType.MICRO, AverageType.BINARY):
    axis = None
  elif average == AverageType.MACRO:
    axis = 0
  elif average == AverageType.SAMPLES:
    axis = 1
  else:
    raise NotImplementedError(f'{average=} is not supported.')

  y_true, y_pred = np.asanyarray(y_true), np.asarray(y_pred)
  true = y_true == pos_label
  positive = y_pred == pos_label
  if input_type == InputType.BINARY and average != AverageType.BINARY:
    positive = np.vstack((positive, ~positive)).T
    true = np.vstack((true, ~true)).T
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
  return ConfusionMatrixAcc(
      tp_cnt,
      tn_cnt,
      fp_cnt,
      fn_cnt,
      config=dict(average=average),
  )


@dataclasses.dataclass(kw_only=True, slots=True)
class ConfusionMatrix(lazytf.AggregateFn):
  """ConfusionMatrix aggregate."""

  pos_label = 1
  input_type: _StrEnum = InputType.BINARY
  average: _StrEnum = AverageType.BINARY
  unknown_tn: bool = False

  def add_inputs(self, accumulator: Any, *inputs: Any, **kwargs: Any) -> Any:
    # Override by call signature.
    config = dataclasses.asdict(self)
    config.update(kwargs)
    result = confusion_matrix(*inputs, **config)
    if accumulator:
      result += accumulator
    return result

  def merge_accumulators(
      self, accumulators: list[ConfusionMatrixAcc]
  ) -> ConfusionMatrixAcc:
    iter_acc = iter(accumulators)
    result = next(iter_acc)
    for accumulator in iter_acc:
      result += accumulator
    return result
