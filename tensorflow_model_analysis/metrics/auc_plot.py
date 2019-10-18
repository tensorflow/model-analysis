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
"""AUC Plot."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

from tensorflow_model_analysis import config
from tensorflow_model_analysis.metrics import binary_confusion_matrices
from tensorflow_model_analysis.metrics import confusion_matrix_at_thresholds
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.proto import metrics_for_slice_pb2
from typing import Any, Dict, Optional, Text

DEFAULT_NUM_THRESHOLDS = 1000

AUC_PLOT_NAME = 'auc_plot'


class AUCPlot(metric_types.Metric):
  """AUC plot."""

  def __init__(self,
               num_thresholds: int = DEFAULT_NUM_THRESHOLDS,
               name: Text = AUC_PLOT_NAME):
    """Initializes AUC plot.

    Args:
      num_thresholds: Number of thresholds to use when discretizing the curve.
        Values must be > 1. Defaults to 1000.
      name: Metric name.
    """
    super(AUCPlot, self).__init__(
        metric_util.merge_per_key_computations(_auc_plot),
        num_thresholds=num_thresholds,
        name=name)


metric_types.register_metric(AUCPlot)


def _auc_plot(
    num_thresholds: int = DEFAULT_NUM_THRESHOLDS,
    name: Text = AUC_PLOT_NAME,
    eval_config: Optional[config.EvalConfig] = None,
    model_name: Text = '',
    output_name: Text = '',
    sub_key: Optional[metric_types.SubKey] = None
) -> metric_types.MetricComputations:
  """Returns metric computations for AUC plots."""
  key = metric_types.PlotKey(
      name=name,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key)

  # The interoploation stragety used here matches how the legacy post export
  # metrics calculated its plots.
  thresholds = [i * 1.0 / num_thresholds for i in range(0, num_thresholds + 1)]
  thresholds = [-1e-6] + thresholds

  # Make sure matrices are calculated.
  matrices_computations = binary_confusion_matrices.binary_confusion_matrices(
      eval_config=eval_config,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key,
      thresholds=thresholds)
  matrices_key = matrices_computations[-1].keys[-1]

  def result(
      metrics: Dict[metric_types.MetricKey, Any]
  ) -> Dict[metric_types.MetricKey,
            metrics_for_slice_pb2.ConfusionMatrixAtThresholds]:
    return {
        key:
            confusion_matrix_at_thresholds.to_proto(thresholds,
                                                    metrics[matrices_key])
    }

  derived_computation = metric_types.DerivedMetricComputation(
      keys=[key], result=result)
  computations = matrices_computations
  computations.append(derived_computation)
  return computations
