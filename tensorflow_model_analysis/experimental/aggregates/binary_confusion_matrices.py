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
"""Binary confusion matrix."""

from typing import Dict, Iterable, List, NamedTuple

DEFAULT_NUM_EXAMPLE_IDS = 100

Matrix = NamedTuple(
    'Matrix', [('tp', float), ('tn', float), ('fp', float), ('fn', float)]
)

_ThresholdEntry = NamedTuple(
    '_ThresholdEntry',
    [
        ('matrix', Matrix),
        ('tp_examples', List[str]),
        ('tn_examples', List[str]),
        ('fp_examples', List[str]),
        ('fn_examples', List[str]),
    ],
)

MatrixAccumulator = Dict[float, _ThresholdEntry]


class BinaryConfusionMatrices:
  """Computes binary confusion matrix."""

  def __init__(
      self,
      thresholds: List[float],
      example_ids_count: int = DEFAULT_NUM_EXAMPLE_IDS,
      enable_fractional_labels: bool = True,
  ):
    """Initializes the class.

    Args:
      thresholds: A specific set of thresholds to use. The caller is responsible
        for marking the boundaries with +/-epsilon if desired. Only one of
        num_thresholds or thresholds should be used. For metrics computed at top
        k this may be a single negative threshold value (i.e. -inf).
      example_ids_count: Max number of example ids to be extracted for each
        result in the binary confusion matrix (tp, tn, fp, and fn).
      enable_fractional_labels: If false, labels will be compared to the
        threshold in the same way predictions are. If true, each incoming tuple
        of (label, prediction, and example weight) will be split into two tuples
        as follows (where l, p, w represent the resulting label, prediction, and
        example weight values): (1) l = 0.0, p = prediction, and w =
        example_weight * (1.0 - label) (2) l = 1.0, p = prediction, and w =
        example_weight * label. If enabled, an exception will be raised if
        labels are not within [0, 1]. The implementation is such that tuples
        associated with a weight of zero are not yielded. This means it is safe
        to enable fractional labels even when the labels only take on the values
        of 0.0 or 1.0.
    """
    self._thresholds = thresholds
    self._example_ids_count = example_ids_count
    self._enable_fractional_labels = enable_fractional_labels

  def create_accumulator(self) -> MatrixAccumulator:
    return {}

  def _merge_example_ids(
      self, list_1: List[str], list_2: List[str]
  ) -> List[str]:
    result = list_1[: self._example_ids_count]
    result.extend(list_2[: self._example_ids_count - len(result)])
    return result

  def _merge_entry(
      self,
      result: MatrixAccumulator,
      threshold: float,
      entry: _ThresholdEntry,
  ) -> _ThresholdEntry:
    if threshold not in result:
      return entry

    return _ThresholdEntry(
        matrix=Matrix(
            tp=result[threshold].matrix.tp + entry.matrix.tp,
            tn=result[threshold].matrix.tn + entry.matrix.tn,
            fp=result[threshold].matrix.fp + entry.matrix.fp,
            fn=result[threshold].matrix.fn + entry.matrix.fn,
        ),
        tp_examples=self._merge_example_ids(
            result[threshold].tp_examples, entry.tp_examples
        ),
        tn_examples=self._merge_example_ids(
            result[threshold].tn_examples, entry.tn_examples
        ),
        fp_examples=self._merge_example_ids(
            result[threshold].fp_examples, entry.fp_examples
        ),
        fn_examples=self._merge_example_ids(
            result[threshold].fn_examples, entry.fn_examples
        ),
    )

  def add_input(
      self,
      accumulator: MatrixAccumulator,
      labels,
      predictions,
      example_weights=None,
      example_id=None,
  ) -> MatrixAccumulator:
    """Adds input to the accumulator.

    Args:
      accumulator: Accumulator to add input to.
      labels: Expected values.
      predictions: Predicted values.
      example_weights: Weights for each example.
      example_id: ID For this example.

    Returns:
      Merged MatrixAccumulator of the original accumulator and the added inputs.
    """
    if not example_weights:
      example_weights = [1] * len(labels)

    result = accumulator
    for threshold in self._thresholds:
      tp = 0.0
      tn = 0.0
      fp = 0.0
      fn = 0.0
      tp_example = None
      tn_example = None
      fp_example = None
      fn_example = None
      for label, prediction, example_weight in zip(
          labels, predictions, example_weights
      ):
        if (
            label == 1.0
            if self._enable_fractional_labels
            else label > threshold
        ):
          if prediction > threshold:
            tp += example_weight
            tp_example = example_id
          else:
            fn += example_weight
            fn_example = example_id
        else:
          if prediction > threshold:
            fp += example_weight
            fp_example = example_id
          else:
            tn += example_weight
            tn_example = example_id

      result[threshold] = self._merge_entry(
          result=result,
          threshold=threshold,
          entry=_ThresholdEntry(
              Matrix(tp=tp, tn=tn, fp=fp, fn=fn),
              tp_examples=[tp_example] if tp_example else [],
              tn_examples=[tn_example] if tn_example else [],
              fp_examples=[fp_example] if fp_example else [],
              fn_examples=[fn_example] if fn_example else [],
          ),
      )

    return result

  def merge_accumulators(
      self,
      accumulators: Iterable[MatrixAccumulator],
  ) -> MatrixAccumulator:
    """Merges accumulators.

    Args:
      accumulators: Accumulators to be merged

    Returns:
      The merged accumulator.
    """
    accumulators = iter(accumulators)
    result = next(accumulators)
    for accumulator in accumulators:
      for threshold in self._thresholds:
        if threshold not in accumulator:
          continue
        result[threshold] = self._merge_entry(
            result=result, threshold=threshold, entry=accumulator[threshold]
        )
    return result

  def extract_output(self, accumulator: MatrixAccumulator) -> MatrixAccumulator:
    for threshold in self._thresholds:
      if threshold not in accumulator:
        accumulator[threshold] = _ThresholdEntry(
            Matrix(tp=0.0, tn=0.0, fp=0.0, fn=0.0),
            tp_examples=[],
            tn_examples=[],
            fp_examples=[],
            fn_examples=[],
        )
    return accumulator
