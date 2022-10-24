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
"""Score distribution Plot."""

import copy

from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.metrics import binary_confusion_matrices
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.proto import config_pb2

DEFAULT_NUM_THRESHOLDS = 1000

SCORE_DISTRIBUTION_PLOT_NAME = 'score_distribution_plot'


class ScoreDistributionPlot(metric_types.Metric):
  """Score distribution plot."""

  def __init__(self,
               num_thresholds: int = DEFAULT_NUM_THRESHOLDS,
               name: str = SCORE_DISTRIBUTION_PLOT_NAME):
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


metric_types.register_metric(ScoreDistributionPlot)


def _extract_prediction_and_weight(
    inputs: metric_types.StandardMetricInputs,
    **kwargs) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
  if 'predictions' in inputs:
    modified_inputs = copy.deepcopy(inputs)
    modified_inputs['labels'] = modified_inputs['predictions']
  else:
    modified_inputs = inputs
  return metric_util.to_label_prediction_example_weight(modified_inputs,
                                                        **kwargs)


def _confusion_matrix_plot(
    num_thresholds: int = DEFAULT_NUM_THRESHOLDS,
    name: str = SCORE_DISTRIBUTION_PLOT_NAME,
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

  modified_eval_config = None
  if eval_config:
    modified_eval_config = copy.deepcopy(eval_config)
    # We want to completely ignore the labels, and in particular not fail if the
    # label_key is not specified.
    for model_spec in modified_eval_config.model_specs:
      model_spec.label_key = model_spec.prediction_key
  else:
    modified_eval_config = config_pb2.EvalConfig()
    spec = config_pb2.ModelSpec()
    spec.prediction_key = constants.PREDICTIONS_KEY
    # Pass the prediction key as label key to avoid failing if labels are not
    # present.
    spec.label_key = constants.PREDICTIONS_KEY
    spec.name = model_name
    modified_eval_config.model_specs.append(spec)

  # Make sure matrices are calculated.
  matrices_computations = binary_confusion_matrices.binary_confusion_matrices(
      # Use a custom name since we have a custom interpolation strategy which
      # will cause the default naming used by the binary confusion matrix to be
      # very long.
      name=(binary_confusion_matrices.BINARY_CONFUSION_MATRICES_NAME + '_' +
            name),
      eval_config=modified_eval_config,
      extract_label_prediction_and_weight=_extract_prediction_and_weight,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key,
      aggregation_type=aggregation_type,
      class_weights=class_weights,
      example_weighted=example_weighted,
      thresholds=thresholds,
      use_histogram=False)
  matrices_key = matrices_computations[-1].keys[-1]

  def result(
      metrics: Dict[metric_types.MetricKey, Any]
  ) -> Dict[metric_types.MetricKey, binary_confusion_matrices.Matrices]:
    confusion_matrix = metrics[matrices_key].to_proto(
    ).confusion_matrix_at_thresholds
    for value in confusion_matrix.matrices:
      value.true_negatives += value.false_negatives
      value.false_negatives = 0
      value.true_positives += value.false_positives
      value.false_positives = 0
      value.precision = 0
      value.recall = 0
    return {key: confusion_matrix}

  derived_computation = metric_types.DerivedMetricComputation(
      keys=[key], result=result)
  computations = matrices_computations
  computations.append(derived_computation)
  return computations
