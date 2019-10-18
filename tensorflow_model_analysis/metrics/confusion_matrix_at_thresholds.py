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
"""Confusion matrix at threshold."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

from tensorflow_model_analysis import config
from tensorflow_model_analysis.metrics import binary_confusion_matrices
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.proto import metrics_for_slice_pb2
from typing import Any, Dict, List, Optional, Text

CONFUSION_MATRIX_AT_THRESHOLDS_NAME = 'confusion_matrix_at_thresholds'


class ConfusionMatrixAtThresholds(metric_types.Metric):
  """Confusion matrix at thresholds."""

  def __init__(self,
               thresholds: List[float],
               name: Text = CONFUSION_MATRIX_AT_THRESHOLDS_NAME):
    """Initializes confusion matrix at thresholds.

    Args:
      thresholds: Thresholds to use for confusion matrix.
      name: Metric name.
    """
    super(ConfusionMatrixAtThresholds, self).__init__(
        metric_util.merge_per_key_computations(_confusion_matrix_at_thresholds),
        thresholds=thresholds,
        name=name)


metric_types.register_metric(ConfusionMatrixAtThresholds)


def _confusion_matrix_at_thresholds(
    thresholds: List[float],
    name: Text = CONFUSION_MATRIX_AT_THRESHOLDS_NAME,
    eval_config: Optional[config.EvalConfig] = None,
    model_name: Text = '',
    output_name: Text = '',
    sub_key: Optional[metric_types.SubKey] = None
) -> metric_types.MetricComputations:
  """Returns metric computations for confusion matrix at thresholds."""
  key = metric_types.MetricKey(
      name=name,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key)

  # Make sure matrices are calculated.
  matrices_computations = binary_confusion_matrices.binary_confusion_matrices(
      eval_config=eval_config,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key,
      thresholds=thresholds)
  matrices_key = matrices_computations[-1].keys[-1]

  def result(
      metrics: Dict[metric_types.MetricKey,
                    metrics_for_slice_pb2.ConfusionMatrixAtThresholds]
  ) -> Dict[metric_types.MetricKey, Any]:
    return {key: to_proto(thresholds, metrics[matrices_key])}

  derived_computation = metric_types.DerivedMetricComputation(
      keys=[key], result=result)
  computations = matrices_computations
  computations.append(derived_computation)
  return computations


def to_proto(
    thresholds: List[float], matrices: binary_confusion_matrices.Matrices
) -> metrics_for_slice_pb2.ConfusionMatrixAtThresholds:
  """Converts matrices into ConfusionMatrixAtThresholds proto.

  If precision or recall are undefined then 1.0 and 0.0 will be used.

  Args:
    thresholds: Thresholds.
    matrices: Confusion matrices.

  Returns:
    Matrices in ConfusionMatrixAtThresholds proto format.
  """
  pb = metrics_for_slice_pb2.ConfusionMatrixAtThresholds()
  for i, threshold in enumerate(thresholds):
    precision = 1.0
    if matrices.tp[i] + matrices.fp[i] > 0:
      precision = matrices.tp[i] / (matrices.tp[i] + matrices.fp[i])
    recall = 0.0
    if matrices.tp[i] + matrices.fn[i] > 0:
      recall = matrices.tp[i] / (matrices.tp[i] + matrices.fn[i])
    pb.matrices.add(
        threshold=round(threshold, 6),
        true_positives=matrices.tp[i],
        false_positives=matrices.fp[i],
        true_negatives=matrices.tn[i],
        false_negatives=matrices.fn[i],
        precision=precision,
        recall=recall)
  return pb
