# Copyright 2020 Google LLC
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
"""Multi-class confusion matrix metrics at thresholds."""

from typing import Callable, Dict, Iterable, List, Optional, NamedTuple

import apache_beam as beam
import numpy as np
from tensorflow_model_analysis import types
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.proto import metrics_for_slice_pb2

MULTI_CLASS_CONFUSION_MATRIX_AT_THRESHOLDS_NAME = (
    'multi_class_confusion_matrix_at_thresholds')


class MultiClassConfusionMatrixAtThresholds(metric_types.Metric):
  """Multi-class confusion matrix metrics at thresholds.

  Computes weighted example counts for all combinations of actual / (top)
  predicted classes.

  The inputs are assumed to contain a single positive label per example (i.e.
  only one class can be true at a time) while the predictions are assumed to sum
  to 1.0.
  """

  def __init__(self,
               thresholds: Optional[List[float]] = None,
               name: str = MULTI_CLASS_CONFUSION_MATRIX_AT_THRESHOLDS_NAME):
    """Initializes multi-class confusion matrix.

    Args:
      thresholds: Optional thresholds, defaults to 0.5 if not specified. If the
        top prediction is less than a threshold then the associated example will
        be assumed to have no prediction associated with it (the
        predicted_class_id will be set to NO_PREDICTED_CLASS_ID).
      name: Metric name.
    """
    super().__init__(
        metric_util.merge_per_key_computations(
            _multi_class_confusion_matrix_at_thresholds),
        thresholds=thresholds,
        name=name)  # pytype: disable=wrong-arg-types


metric_types.register_metric(MultiClassConfusionMatrixAtThresholds)


def _multi_class_confusion_matrix_at_thresholds(
    thresholds: Optional[List[float]] = None,
    name: str = MULTI_CLASS_CONFUSION_MATRIX_AT_THRESHOLDS_NAME,
    eval_config: Optional[config_pb2.EvalConfig] = None,
    model_name: str = '',
    output_name: str = '',
    example_weighted: bool = False) -> metric_types.MetricComputations:
  """Returns computations for multi-class confusion matrix at thresholds."""
  if not thresholds:
    thresholds = [0.5]

  key = metric_types.MetricKey(
      name=name,
      model_name=model_name,
      output_name=output_name,
      example_weighted=example_weighted)

  # Make sure matrices are calculated.
  matrices_computations = multi_class_confusion_matrices(
      thresholds=thresholds,
      eval_config=eval_config,
      model_name=model_name,
      output_name=output_name,
      example_weighted=example_weighted)
  matrices_key = matrices_computations[-1].keys[-1]

  def result(
      metrics: Dict[metric_types.MetricKey,
                    metrics_for_slice_pb2.MultiClassConfusionMatrixAtThresholds]
  ) -> Dict[metric_types.MetricKey,
            metrics_for_slice_pb2.MultiClassConfusionMatrixAtThresholds]:
    return {key: metrics[matrices_key]}

  derived_computation = metric_types.DerivedMetricComputation(
      keys=[key], result=result)
  computations = matrices_computations
  computations.append(derived_computation)
  return computations


MULTI_CLASS_CONFUSION_MATRICES = '_multi_class_confusion_matrices'

_EPSILON = 1e-7

# Class ID used when no prediction was made because a threshold was given and
# the top prediction was less than the threshold.
NO_PREDICTED_CLASS_ID = -1


def multi_class_confusion_matrices(
    thresholds: Optional[List[float]] = None,
    num_thresholds: Optional[int] = None,
    name: str = MULTI_CLASS_CONFUSION_MATRICES,
    eval_config: Optional[config_pb2.EvalConfig] = None,
    model_name: str = '',
    output_name: str = '',
    example_weighted: bool = False) -> metric_types.MetricComputations:
  """Returns computations for multi-class confusion matrices.

  Args:
    thresholds: A specific set of thresholds to use. The caller is responsible
      for marking the bondaires with +/-epsilon if desired. Only one of
      num_thresholds or thresholds should be used.
    num_thresholds: Number of thresholds to use. Thresholds will be calculated
      using linear interpolation between 0.0 and 1.0 with equidistant values and
      bondardaries at -epsilon and 1.0+epsilon. Values must be > 0. Only one of
      num_thresholds or thresholds should be used.
    name: Metric name.
    eval_config: Eval config.
    model_name: Optional model name (if multi-model evaluation).
    output_name: Optional output name (if multi-output model type).
    example_weighted: True if example weights should be applied.

  Raises:
    ValueError: If both num_thresholds and thresholds are set at the same time.
  """
  if num_thresholds is not None and thresholds is not None:
    raise ValueError(
        'only one of thresholds or num_thresholds can be set at a time')
  if num_thresholds is None and thresholds is None:
    thresholds = [0.0]
  if num_thresholds is not None:
    thresholds = [
        (i + 1) * 1.0 / (num_thresholds - 1) for i in range(num_thresholds - 2)
    ]
    thresholds = [-_EPSILON] + thresholds + [1.0 + _EPSILON]

  key = metric_types.MetricKey(
      name=name,
      model_name=model_name,
      output_name=output_name,
      example_weighted=example_weighted)
  return [
      metric_types.MetricComputation(
          keys=[key],
          preprocessor=None,
          combiner=_MultiClassConfusionMatrixCombiner(
              key=key,
              eval_config=eval_config,
              example_weighted=example_weighted,
              thresholds=thresholds))
  ]


MatrixEntryKey = NamedTuple('MatrixEntryKey', [('actual_class_id', int),
                                               ('predicted_class_id', int)])


class Matrices(types.StructuredMetricValue, dict):
  """A Matrices object wraps a Dict[float, Dict[MatrixEntryKey, float]].

  A specific confusion matrix entry can be accessed for a threshold,
  actual_class and predicted_class with

      instance[threshold][MatrixEntryKey(actual_class_id, predicted_class_id)]
  """

  def _apply_binary_op_elementwise(
      self, other: 'Matrices', op: Callable[[float, float],
                                            float]) -> 'Matrices':
    result = Matrices()
    all_thresholds = set(self.keys()).union(other.keys())
    for threshold in all_thresholds:
      self_entries = self.get(threshold, {})
      other_entries = other.get(threshold, {})
      result[threshold] = {}
      all_entry_keys = set(self_entries.keys()).union(set(other_entries.keys()))
      for entry_key in all_entry_keys:
        self_count = self_entries.get(entry_key, 0)
        other_count = other_entries.get(entry_key, 0)
        result[threshold][entry_key] = op(self_count, other_count)
    return result

  def _apply_binary_op_broadcast(
      self, other: float, op: Callable[[float, float], float]) -> 'Matrices':
    result = Matrices()
    for threshold, self_entries in self.items():
      result[threshold] = {}
      for entry_key, self_count in self_entries.items():
        result[threshold][entry_key] = op(self_count, other)
    return result

  def to_proto(self) -> metrics_for_slice_pb2.MetricValue:
    result = metrics_for_slice_pb2.MetricValue()
    multi_class_confusion_matrices_at_thresholds_proto = (
        result.multi_class_confusion_matrix_at_thresholds)
    for threshold in sorted(self.keys()):
      # Convert -epsilon and 1.0+epsilon back to 0.0 and 1.0.
      if threshold == -_EPSILON:
        t = 0.0
      elif threshold == 1.0 + _EPSILON:
        t = 1.0
      else:
        t = threshold
      matrix = multi_class_confusion_matrices_at_thresholds_proto.matrices.add(
          threshold=t)
      for k in sorted(self[threshold].keys()):
        matrix.entries.add(
            actual_class_id=k.actual_class_id,
            predicted_class_id=k.predicted_class_id,
            num_weighted_examples=self[threshold][k])
    return result


class _MultiClassConfusionMatrixCombiner(beam.CombineFn):
  """Creates multi-class confusion matrix at thresholds from standard inputs."""

  def __init__(self, key: metric_types.MetricKey,
               eval_config: Optional[config_pb2.EvalConfig],
               example_weighted: bool, thresholds: List[float]):
    self._key = key
    self._eval_config = eval_config
    self._example_weighted = example_weighted
    self._thresholds = thresholds if thresholds else [0.0]

  def create_accumulator(self) -> Matrices:
    return Matrices()

  def add_input(self, accumulator: Matrices,
                element: metric_types.StandardMetricInputs) -> Matrices:
    label, predictions, example_weight = next(
        metric_util.to_label_prediction_example_weight(
            element,
            eval_config=self._eval_config,
            model_name=self._key.model_name,
            output_name=self._key.output_name,
            example_weighted=self._example_weighted,
            flatten=False,
            require_single_example_weight=True))  # pytype: disable=wrong-arg-types
    if not label.shape:
      raise ValueError(
          'Label missing from example: StandardMetricInputs={}'.format(element))
    if predictions.shape in ((), (1,)):
      raise ValueError(
          'Predictions shape must be > 1 for multi-class confusion matrix: '
          'shape={}, StandardMetricInputs={}'.format(predictions.shape,
                                                     element))
    if label.size > 1:
      actual_class_id = np.argmax(label)
    else:
      actual_class_id = int(label)
    predicted_class_id = np.argmax(predictions)
    example_weight = float(example_weight)
    for threshold in self._thresholds:
      if threshold not in accumulator:
        accumulator[threshold] = {}
      if predictions[predicted_class_id] <= threshold:
        predicted_class_id = NO_PREDICTED_CLASS_ID
      matrix_key = MatrixEntryKey(actual_class_id, predicted_class_id)
      if matrix_key in accumulator[threshold]:
        accumulator[threshold][matrix_key] += example_weight
      else:
        accumulator[threshold][matrix_key] = example_weight
    return accumulator

  def merge_accumulators(self, accumulators: Iterable[Matrices]) -> Matrices:
    accumulators = iter(accumulators)
    result = next(accumulators)
    for accumulator in accumulators:
      for threshold, matrix in accumulator.items():
        if threshold not in result:
          result[threshold] = {}
        for k, v in matrix.items():
          if k in result[threshold]:
            result[threshold][k] += v
          else:
            result[threshold][k] = v
    return result

  def extract_output(
      self, accumulator: Matrices) -> Dict[metric_types.MetricKey, Matrices]:
    return {self._key: accumulator}
