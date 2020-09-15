# Lint as: python3
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
"""Multi-class confusion matrix plot at thresholds."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

from typing import Dict, List, Optional, Text

from tensorflow_model_analysis import config
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.metrics import multi_class_confusion_matrix_metrics
from tensorflow_model_analysis.proto import metrics_for_slice_pb2

MULTI_CLASS_CONFUSION_MATRIX_PLOT_NAME = ('multi_class_confusion_matrix_plot')


class MultiClassConfusionMatrixPlot(metric_types.Metric):
  """Multi-class confusion matrix plot.

  Computes weighted example counts for all combinations of actual / (top)
  predicted classes.

  The inputs are assumed to contain a single positive label per example (i.e.
  only one class can be true at a time) while the predictions are assumed to sum
  to 1.0.
  """

  def __init__(self,
               thresholds: Optional[List[float]] = None,
               num_thresholds: Optional[int] = None,
               name: Text = MULTI_CLASS_CONFUSION_MATRIX_PLOT_NAME):
    """Initializes multi-class confusion matrix.

    Args:
      thresholds: Optional thresholds. If the top prediction is less than a
        threshold then the associated example will be assumed to have no
        prediction associated with it (the predicted_class_id will be set to
        tfma.metrics.NO_PREDICTED_CLASS_ID). Only one of
        either thresholds or num_thresholds should be used. If both are unset,
        then [0.0] will be assumed.
      num_thresholds: Number of thresholds to use. The thresholds will be evenly
        spaced between 0.0 and 1.0 and inclusive of the boundaries (i.e. to
        configure the thresholds to [0.0, 0.25, 0.5, 0.75, 1.0], the parameter
        should be set to 5). Only one of either thresholds or num_thresholds
        should be used.
      name: Metric name.
    """
    super(MultiClassConfusionMatrixPlot, self).__init__(
        metric_util.merge_per_key_computations(
            _multi_class_confusion_matrix_plot),
        thresholds=thresholds,
        num_thresholds=num_thresholds,
        name=name)


metric_types.register_metric(MultiClassConfusionMatrixPlot)


def _multi_class_confusion_matrix_plot(
    thresholds: Optional[List[float]] = None,
    num_thresholds: Optional[int] = None,
    name: Text = MULTI_CLASS_CONFUSION_MATRIX_PLOT_NAME,
    eval_config: Optional[config.EvalConfig] = None,
    model_name: Text = '',
    output_name: Text = '',
) -> metric_types.MetricComputations:
  """Returns computations for multi-class confusion matrix plot."""
  if num_thresholds is None and thresholds is None:
    thresholds = [0.0]

  key = metric_types.PlotKey(
      name=name, model_name=model_name, output_name=output_name)

  # Make sure matrices are calculated.
  matrices_computations = (
      multi_class_confusion_matrix_metrics.multi_class_confusion_matrices(
          thresholds=thresholds,
          num_thresholds=num_thresholds,
          eval_config=eval_config,
          model_name=model_name,
          output_name=output_name))
  matrices_key = matrices_computations[-1].keys[-1]

  def result(
      metrics: Dict[metric_types.MetricKey,
                    metrics_for_slice_pb2.MultiClassConfusionMatrixAtThresholds]
  ) -> Dict[metric_types.PlotKey,
            metrics_for_slice_pb2.MultiClassConfusionMatrixAtThresholds]:
    return {key: metrics[matrices_key]}

  derived_computation = metric_types.DerivedMetricComputation(
      keys=[key], result=result)
  computations = matrices_computations
  computations.append(derived_computation)
  return computations
