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
"""Multi-label confusion matrix at thresholds."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import apache_beam as beam
from tensorflow_model_analysis import config
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.proto import metrics_for_slice_pb2
from typing import Dict, List, Optional, NamedTuple, Text

MULTI_LABEL_CONFUSION_MATRIX_AT_THRESHOLDS_NAME = (
    'multi_label_confusion_matrix_at_thresholds')


class MultiLabelConfusionMatrixAtThresholds(metric_types.Metric):
  """Multi-label confusion matrix.

  For each actual class (positive label) a confusion matrix is computed for each
  class based on the associated predicted values such that:

    TP = positive_prediction_class_label & positive_prediction
    TN = negative_prediction_class_label & negative_prediction
    FP = negative_prediction_class_label & positive_prediction
    FN = positive_prediction_class_label & negative_prediction

  For example, given classes 0, 1 and a given threshold, the following matrices
  will be computed:

    Actual: class_0
    Predicted: class_0
        TP = is_class_0 & is_class_0 & predict_class_0
        TN = is_class_0 & not_class_0 & predict_not_class_0
        FN = is_class_0 & is_class_0 & predict_not_class_0
        FP = is_class_0 & not_class_0 & predict_class_0
    Actual: class_0
    Predicted: class_1
        TP = is_class_0 & is_class_1 & predict_class_1
        TN = is_class_0 & not_class_1 & predict_not_class_1
        FN = is_class_0 & is_class_1 & predict_not_class_1
        FP = is_class_0 & not_class_1 & predict_class_1
    Actual: class_1
    Predicted: class_0
        TP = is_class_1 & is_class_0 & predict_class_0
        TN = is_class_1 & not_class_0 & predict_not_class_0
        FN = is_class_1 & is_class_0 & predict_not_class_0
        FP = is_class_1 & not_class_0 & predict_class_0
    Actual: class_1
    Predicted: class_1
        TP = is_class_1 & is_class_1 & predict_class_1
        TN = is_class_1 & not_class_1 & predict_not_class_1
        FN = is_class_1 & is_class_1 & predict_not_class_1
        FP = is_class_1 & not_class_1 & predict_class_1

  Note that unlike the multi-class confusion matrix, the inputs are assumed to
  be multi-label whereby the predictions may not necessarily sum to 1.0 and
  multiple classes can be true as the same time.
  """

  def __init__(self,
               thresholds: Optional[float] = None,
               name: Text = MULTI_LABEL_CONFUSION_MATRIX_AT_THRESHOLDS_NAME):
    """Initializes multi-label confusion matrix.

    Args:
      thresholds: Optional thresholds. Defaults to [0.5].
      name: Metric name.
    """
    super(MultiLabelConfusionMatrixAtThresholds, self).__init__(
        metric_util.merge_per_key_computations(
            _multi_label_confusion_matrix_at_thresholds),
        thresholds=thresholds,
        name=name)


metric_types.register_metric(MultiLabelConfusionMatrixAtThresholds)


def _multi_label_confusion_matrix_at_thresholds(
    thresholds: Optional[List[float]] = None,
    name: Text = MULTI_LABEL_CONFUSION_MATRIX_AT_THRESHOLDS_NAME,
    eval_config: Optional[config.EvalConfig] = None,
    model_name: Text = '',
    output_name: Text = '',
) -> metric_types.MetricComputations:
  """Returns computations for multi-label confusion matrix at thresholds."""
  key = metric_types.PlotKey(
      name=name, model_name=model_name, output_name=output_name)
  return [
      metric_types.MetricComputation(
          keys=[key],
          preprocessor=None,
          combiner=_MultiLabelConfusionMatrixAtThresholdsCombiner(
              key=key, eval_config=eval_config, thresholds=thresholds))
  ]


_MatrixEntryKey = NamedTuple('_MatrixEntryKey', [('actual_class_id', int),
                                                 ('predicted_class_id', int)])


class _ConfusionMatrix(object):
  """Confusion matrix."""
  __slots__ = [
      'false_negatives', 'true_negatives', 'false_positives', 'true_positives'
  ]

  def __init__(self):
    self.false_negatives = 0.0
    self.true_negatives = 0.0
    self.false_positives = 0.0
    self.true_positives = 0.0


# Thresholds -> entry -> confusion matrix
_Matrices = Dict[float, Dict[_MatrixEntryKey, _ConfusionMatrix]]


class _MultiLabelConfusionMatrixAtThresholdsCombiner(beam.CombineFn):
  """Creates multi-label confusion matrix at thresholds from standard inputs."""

  def __init__(self, key: metric_types.PlotKey, eval_config: config.EvalConfig,
               thresholds: List[float]):
    self._key = key
    self._eval_config = eval_config
    self._thresholds = thresholds if thresholds else [0.5]

  def create_accumulator(self) -> _Matrices:
    return {}

  def add_input(self, accumulator: _Matrices,
                element: metric_types.StandardMetricInputs) -> _Matrices:
    labels, predictions, example_weight = (
        metric_util.to_label_prediction_example_weight(
            element,
            eval_config=self._eval_config,
            output_name=self._key.output_name))
    if not labels.shape:
      raise ValueError(
          'Labels missing from example: StandardMetricInputs={}'.format(
              element))
    if predictions.shape in ((), (1,)):
      raise ValueError(
          'Predictions shape must be > 1 for multi-label confusion matrix: '
          'shape={}, StandardMetricInputs={}'.format(predictions.shape,
                                                     element))
    example_weight = float(example_weight)
    for threshold in self._thresholds:
      if threshold not in accumulator:
        accumulator[threshold] = {}
      for actual_class_id, label in enumerate(labels):
        if not label:
          continue
        for class_id, prediction in enumerate(predictions):
          matrix_key = _MatrixEntryKey(actual_class_id, class_id)
          fn = (labels[class_id] and prediction <= threshold) * example_weight
          fp = (not labels[class_id] and
                prediction > threshold) * example_weight
          tn = ((not labels[class_id] and prediction <= threshold) *
                example_weight)
          tp = (labels[class_id] and prediction > threshold) * example_weight
          if matrix_key in accumulator[threshold]:
            accumulator[threshold][matrix_key].false_negatives += fn
            accumulator[threshold][matrix_key].true_negatives += tn
            accumulator[threshold][matrix_key].false_positives += fp
            accumulator[threshold][matrix_key].true_positives += tp
          else:
            matrix = _ConfusionMatrix()
            matrix.false_negatives = fn
            matrix.true_negatives = tn
            matrix.false_positives = fp
            matrix.true_positives = tp
            accumulator[threshold][matrix_key] = matrix
    return accumulator

  def merge_accumulators(self, accumulators: List[_Matrices]) -> _Matrices:
    result = {}
    for accumulator in accumulators:
      for threshold, matrix in accumulator.items():
        if threshold not in result:
          result[threshold] = {}
        for k, v in matrix.items():
          if k in result[threshold]:
            result[threshold][k].false_negatives += v.false_negatives
            result[threshold][k].true_negatives += v.true_negatives
            result[threshold][k].false_positives += v.false_positives
            result[threshold][k].true_positives += v.true_positives
          else:
            result[threshold][k] = v
    return result

  def extract_output(
      self, accumulator: _Matrices
  ) -> Dict[metric_types.PlotKey,
            metrics_for_slice_pb2.MultiLabelConfusionMatrixAtThresholds]:
    pb = metrics_for_slice_pb2.MultiLabelConfusionMatrixAtThresholds()
    for threshold in sorted(accumulator.keys()):
      matrix = pb.matrices.add(threshold=threshold)
      for k in sorted(accumulator[threshold].keys()):
        matrix.entries.add(
            actual_class_id=k.actual_class_id,
            predicted_class_id=k.predicted_class_id,
            false_negatives=accumulator[threshold][k].false_negatives,
            true_negatives=accumulator[threshold][k].true_negatives,
            false_positives=accumulator[threshold][k].false_positives,
            true_positives=accumulator[threshold][k].true_positives)
    return {self._key: pb}
