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
"""Multi-class confusion matrix at thresholds."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import apache_beam as beam
import numpy as np
from tensorflow_model_analysis import config
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.proto import metrics_for_slice_pb2
from typing import Dict, List, Optional, NamedTuple, Text

MULTI_CLASS_CONFUSION_MATRIX_AT_THRESHOLDS_NAME = (
    'multi_class_confusion_matrix_at_thresholds')

# Class ID used when no prediction was made because a threshold was given and
# the top prediction was less than the threshold.
NO_PREDICTED_CLASS_ID = -1


class MultiClassConfusionMatrixAtThresholds(metric_types.Metric):
  """Multi-class confusion matrix.

  Computes weighted example counts for all combinations of actual / (top)
  predicted classes.

  The inputs are assumed to contain a single positive label per example (i.e.
  only one class can be true at a time) while the predictions are assumed to sum
  to 1.0.
  """

  def __init__(self,
               thresholds: Optional[float] = None,
               name: Text = MULTI_CLASS_CONFUSION_MATRIX_AT_THRESHOLDS_NAME):
    """Initializes multi-class confusion matrix.

    Args:
      thresholds: Optional thresholds. If the top prediction is less than a
        threshold then the associated example will be assumed to have no
        prediction associated with it (the predicted_class_id will be set to
        NO_PREDICTED_CLASS_ID). Defaults to [0.0].
      name: Metric name.
    """
    super(MultiClassConfusionMatrixAtThresholds, self).__init__(
        metric_util.merge_per_key_computations(
            _multi_class_confusion_matrix_at_thresholds),
        thresholds=thresholds,
        name=name)


metric_types.register_metric(MultiClassConfusionMatrixAtThresholds)


def _multi_class_confusion_matrix_at_thresholds(
    thresholds: Optional[List[float]] = None,
    name: Text = MULTI_CLASS_CONFUSION_MATRIX_AT_THRESHOLDS_NAME,
    eval_config: Optional[config.EvalConfig] = None,
    model_name: Text = '',
    output_name: Text = '',
) -> metric_types.MetricComputations:
  """Returns computations for multi-class confusion matrix at thresholds."""
  key = metric_types.PlotKey(
      name=name, model_name=model_name, output_name=output_name)
  return [
      metric_types.MetricComputation(
          keys=[key],
          preprocessor=None,
          combiner=_MultiClassConfusionMatrixAtThresholdsCombiner(
              key=key, eval_config=eval_config, thresholds=thresholds))
  ]


_MatrixEntryKey = NamedTuple('_MatrixEntryKey', [('actual_class_id', int),
                                                 ('predicted_class_id', int)])
# Thresholds -> entry -> example_weights
_Matrices = Dict[float, Dict[_MatrixEntryKey, float]]


class _MultiClassConfusionMatrixAtThresholdsCombiner(beam.CombineFn):
  """Creates multi-class confusion matrix at thresholds from standard inputs."""

  def __init__(self, key: metric_types.PlotKey, eval_config: config.EvalConfig,
               thresholds: List[float]):
    self._key = key
    self._eval_config = eval_config
    self._thresholds = thresholds if thresholds else [0.0]

  def create_accumulator(self) -> _Matrices:
    return {}

  def add_input(self, accumulator: _Matrices,
                element: metric_types.StandardMetricInputs) -> _Matrices:
    label, predictions, example_weight = (
        metric_util.to_label_prediction_example_weight(
            element,
            eval_config=self._eval_config,
            output_name=self._key.output_name))
    if not label.shape:
      raise ValueError(
          'Label missing from example: StandardMetricInputs={}'.format(element))
    if predictions.shape in ((), (1,)):
      raise ValueError(
          'Predictions shape must be > 1 for multi-class confusion matrix: '
          'shape={}, StandardMetricInputs={}'.format(predictions.shape,
                                                     element))
    if len(label.shape) > 1:
      actual_class_id = np.argmax(label)
    else:
      actual_class_id = int(label)
    predicted_class_id = np.argmax(predictions)
    example_weight = float(example_weight)
    for threshold in self._thresholds:
      if threshold not in accumulator:
        accumulator[threshold] = {}
      if predictions[predicted_class_id] < threshold:
        predicted_class_id = NO_PREDICTED_CLASS_ID
      matrix_key = _MatrixEntryKey(actual_class_id, predicted_class_id)
      if matrix_key in accumulator[threshold]:
        accumulator[threshold][matrix_key] += example_weight
      else:
        accumulator[threshold][matrix_key] = example_weight
    return accumulator

  def merge_accumulators(self, accumulators: List[_Matrices]) -> _Matrices:
    result = {}
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
      self, accumulator: _Matrices
  ) -> Dict[metric_types.PlotKey,
            metrics_for_slice_pb2.MultiClassConfusionMatrixAtThresholds]:
    pb = metrics_for_slice_pb2.MultiClassConfusionMatrixAtThresholds()
    for threshold in sorted(accumulator.keys()):
      matrix = pb.matrices.add(threshold=threshold)
      for k in sorted(accumulator[threshold].keys()):
        matrix.entries.add(
            actual_class_id=k.actual_class_id,
            predicted_class_id=k.predicted_class_id,
            num_weighted_examples=accumulator[threshold][k])
    return {self._key: pb}
