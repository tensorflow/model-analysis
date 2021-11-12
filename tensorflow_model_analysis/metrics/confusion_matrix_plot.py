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
"""Confusion matrix Plot."""

from typing import Any, Dict, Optional

from tensorflow_model_analysis.metrics import binary_confusion_matrices
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.proto import config_pb2

DEFAULT_NUM_THRESHOLDS = 1000

CONFUSION_MATRIX_PLOT_NAME = 'confusion_matrix_plot'


class ConfusionMatrixPlot(metric_types.Metric):
  """Confusion matrix plot."""

  def __init__(self,
               num_thresholds: int = DEFAULT_NUM_THRESHOLDS,
               name: str = CONFUSION_MATRIX_PLOT_NAME):
    """Initializes confusion matrix plot.

    Args:
      num_thresholds: Number of thresholds to use when discretizing the curve.
        Values must be > 1. Defaults to 1000.
      name: Metric name.
    """
    super().__init__(
        metric_util.merge_per_key_computations(_confusion_matrix_plot),
        num_thresholds=num_thresholds,
        name=name)


metric_types.register_metric(ConfusionMatrixPlot)


def _confusion_matrix_plot(
    num_thresholds: int = DEFAULT_NUM_THRESHOLDS,
    name: str = CONFUSION_MATRIX_PLOT_NAME,
    eval_config: Optional[config_pb2.EvalConfig] = None,
    model_name: str = '',
    output_name: str = '',
    sub_key: Optional[metric_types.SubKey] = None,
    aggregation_type: Optional[metric_types.AggregationType] = None,
    class_weights: Optional[Dict[int, float]] = None,
    example_weighted: bool = False) -> metric_types.MetricComputations:
  """Returns metric computations for confusion matrix plots."""
  key = metric_types.PlotKey(
      name=name,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key,
      example_weighted=example_weighted)

  # The interoploation strategy used here matches how the legacy post export
  # metrics calculated its plots.
  thresholds = [i * 1.0 / num_thresholds for i in range(0, num_thresholds + 1)]
  thresholds = [-1e-6] + thresholds

  # Make sure matrices are calculated.
  matrices_computations = binary_confusion_matrices.binary_confusion_matrices(
      # Use a custom name since we have a custom interpolation strategy which
      # will cause the default naming used by the binary confusion matrix to be
      # very long.
      name=(binary_confusion_matrices.BINARY_CONFUSION_MATRICES_NAME + '_' +
            name),
      eval_config=eval_config,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key,
      aggregation_type=aggregation_type,
      class_weights=class_weights,
      example_weighted=example_weighted,
      thresholds=thresholds,
      use_histogram=True)
  matrices_key = matrices_computations[-1].keys[-1]

  def result(
      metrics: Dict[metric_types.MetricKey, Any]
  ) -> Dict[metric_types.MetricKey, binary_confusion_matrices.Matrices]:
    return {
        key: metrics[matrices_key].to_proto().confusion_matrix_at_thresholds
    }

  derived_computation = metric_types.DerivedMetricComputation(
      keys=[key], result=result)
  computations = matrices_computations
  computations.append(derived_computation)
  return computations
