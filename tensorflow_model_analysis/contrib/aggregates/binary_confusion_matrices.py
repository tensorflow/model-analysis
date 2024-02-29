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
"""Binary confusion matrices."""

from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence

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
      thresholds: Sequence[float],
      example_ids_count: int = DEFAULT_NUM_EXAMPLE_IDS,
      enable_fractional_labels: bool = True,
  ):
    """Initializes the class.

    Args:
      thresholds: A specific set of thresholds to use. The caller is responsible
        for marking the boundaries with +/-epsilon if desired.
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
      accumulator: MatrixAccumulator,
      threshold: float,
      entry: _ThresholdEntry,
  ) -> _ThresholdEntry:
    if threshold not in accumulator:
      return entry

    return _ThresholdEntry(
        matrix=Matrix(
            tp=accumulator[threshold].matrix.tp + entry.matrix.tp,
            tn=accumulator[threshold].matrix.tn + entry.matrix.tn,
            fp=accumulator[threshold].matrix.fp + entry.matrix.fp,
            fn=accumulator[threshold].matrix.fn + entry.matrix.fn,
        ),
        tp_examples=self._merge_example_ids(
            accumulator[threshold].tp_examples, entry.tp_examples
        ),
        tn_examples=self._merge_example_ids(
            accumulator[threshold].tn_examples, entry.tn_examples
        ),
        fp_examples=self._merge_example_ids(
            accumulator[threshold].fp_examples, entry.fp_examples
        ),
        fn_examples=self._merge_example_ids(
            accumulator[threshold].fn_examples, entry.fn_examples
        ),
    )

  def add_input(
      self,
      accumulator: MatrixAccumulator,
      labels: Sequence[float],
      predictions: Sequence[float],
      example_weights: Optional[Sequence[float]],
      example_id: Optional[str],
  ) -> MatrixAccumulator:
    """Adds a single example input to the accumulator.

    Args:
      accumulator: Accumulator to add input to.
      labels: Expected values.
      predictions: Predicted values.
      example_weights: Weights for this example.
      example_id: ID for this example.

    Returns:
      Merged MatrixAccumulator of the original accumulator and the added inputs.
    """
    if example_weights is None or all(w is None for w in example_weights):
      example_weights = [1] * len(labels)

    for threshold in self._thresholds:
      tp = 0.0
      tn = 0.0
      fp = 0.0
      fn = 0.0
      tp_example = None
      tn_example = None
      fp_example = None
      fn_example = None
      # We need to iterate here even though it is one example because one
      # example can contain multiple labels/predictions/example_weights.
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

      accumulator[threshold] = self._merge_entry(
          accumulator=accumulator,
          threshold=threshold,
          entry=_ThresholdEntry(
              Matrix(tp=tp, tn=tn, fp=fp, fn=fn),
              tp_examples=[tp_example] if tp_example is not None else [],
              tn_examples=[tn_example] if tn_example is not None else [],
              fp_examples=[fp_example] if fp_example is not None else [],
              fn_examples=[fn_example] if fn_example is not None else [],
          ),
      )

    return accumulator

  def add_inputs(
      self,
      accumulator: MatrixAccumulator,
      labels: Sequence[Sequence[float]],
      predictions: Sequence[Sequence[float]],
      example_weights: Optional[Sequence[Sequence[float]]],
      example_ids: Optional[Sequence[str]],
  ) -> MatrixAccumulator:
    """Adds a batch of inputs to the accumulator.

    Args:
      accumulator: Accumulator to add input to.
      labels: Expected values.
      predictions: Predicted values.
      example_weights: Weights for each example.
      example_ids: IDs For each example.

    Returns:
      Merged MatrixAccumulator of the original accumulator and the added inputs.
    """
    make_iter = lambda ex: ex if hasattr(ex, '__iter__') else [ex]

    if example_weights is None:
      example_weights = [None] * len(labels)

    if example_ids is None:
      example_ids = [None] * len(labels)

    for label, prediction, example_weight, example_id in zip(
        labels, predictions, example_weights, example_ids
    ):
      # Calls self.add_input() for each example within the batch.
      accumulator = self.add_input(
          accumulator=accumulator,
          labels=make_iter(label),
          predictions=make_iter(prediction),
          example_weights=make_iter(example_weight),
          example_id=example_id,
      )

    return accumulator

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
        # We need to check if threshold is in the accumulator because the
        # accumulator can be empty (i.e. no input was been added).
        if threshold in accumulator:
          result[threshold] = self._merge_entry(
              accumulator=result,
              threshold=threshold,
              entry=accumulator[threshold],
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

  def __call__(self, *inputs, **named_inputs):
    """Directly apply aggregate on inputs."""
    return self.extract_output(
        self.add_inputs(self.create_accumulator(), *inputs, **named_inputs)
    )
