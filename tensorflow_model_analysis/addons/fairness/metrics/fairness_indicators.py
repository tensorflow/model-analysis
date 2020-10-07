# Lint as: python3
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
"""Fairness Indicators Metrics."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import collections
from typing import Any, Dict, List, Optional, Text

from tensorflow_model_analysis import config
from tensorflow_model_analysis.metrics import binary_confusion_matrices
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util

FAIRNESS_INDICATORS_METRICS_NAME = 'fairness_indicators_metrics'
FAIRNESS_INDICATORS_SUB_METRICS = ('false_positive_rate', 'false_negative_rate',
                                   'true_positive_rate', 'true_negative_rate',
                                   'positive_rate', 'negative_rate',
                                   'false_discovery_rate',
                                   'false_omission_rate')

DEFAULT_THERSHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class FairnessIndicators(metric_types.Metric):
  """Fairness indicators metrics."""

  def __init__(self,
               thresholds: List[float] = DEFAULT_THERSHOLDS,
               name: Text = FAIRNESS_INDICATORS_METRICS_NAME):
    """Initializes fairness indicators metrics.

    Args:
      thresholds: Thresholds to use for fairness metrics.
      name: Metric name.
    """
    super(FairnessIndicators, self).__init__(
        metric_util.merge_per_key_computations(
            _fairness_indicators_metrics_at_thresholds),
        thresholds=thresholds,
        name=name)


def calculate_digits(thresholds):
  digits = [len(str(t)) - 2 for t in thresholds]
  return max(max(digits), 1)


def _fairness_indicators_metrics_at_thresholds(
    thresholds: List[float],
    name: Text = FAIRNESS_INDICATORS_METRICS_NAME,
    eval_config: Optional[config.EvalConfig] = None,
    model_name: Text = '',
    output_name: Text = '',
    aggregation_type: Optional[metric_types.AggregationType] = None,
    sub_key: Optional[metric_types.SubKey] = None,
    class_weights: Optional[Dict[int, float]] = None
) -> metric_types.MetricComputations:
  """Returns computations for fairness metrics at thresholds."""
  metric_key_by_name_by_threshold = collections.defaultdict(dict)
  keys = []
  digits_num = calculate_digits(thresholds)
  for t in thresholds:
    for m in FAIRNESS_INDICATORS_SUB_METRICS:
      key = metric_types.MetricKey(
          name='%s/%s@%.*f' %
          (name, m, digits_num,
           t),  # e.g. "fairness_indicators_metrics/positive_rate@0.5"
          model_name=model_name,
          output_name=output_name,
          sub_key=sub_key)
      keys.append(key)
      metric_key_by_name_by_threshold[t][m] = key

  # Make sure matrices are calculated.
  computations = binary_confusion_matrices.binary_confusion_matrices(
      eval_config=eval_config,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key,
      aggregation_type=aggregation_type,
      class_weights=class_weights,
      thresholds=thresholds)
  confusion_matrices_key = computations[-1].keys[-1]

  def result(
      metrics: Dict[metric_types.MetricKey, Any]
  ) -> Dict[metric_types.MetricKey, Any]:
    """Returns fairness metrics values."""
    metric = metrics[confusion_matrices_key]
    output = {}

    for i, threshold in enumerate(thresholds):
      num_positives = metric.tp[i] + metric.fn[i]
      num_negatives = metric.tn[i] + metric.fp[i]

      tpr = metric.tp[i] / (num_positives or float('nan'))
      tnr = metric.tn[i] / (num_negatives or float('nan'))
      fpr = metric.fp[i] / (num_negatives or float('nan'))
      fnr = metric.fn[i] / (num_positives or float('nan'))
      pr = (metric.tp[i] + metric.fp[i]) / (
          (num_positives + num_negatives) or float('nan'))
      nr = (metric.tn[i] + metric.fn[i]) / (
          (num_positives + num_negatives) or float('nan'))

      fdr = metric.fp[i] / ((metric.fp[i] + metric.tp[i]) or float('nan'))
      fomr = metric.fn[i] / ((metric.fn[i] + metric.tn[i]) or float('nan'))

      output[metric_key_by_name_by_threshold[threshold]
             ['false_positive_rate']] = fpr
      output[metric_key_by_name_by_threshold[threshold]
             ['false_negative_rate']] = fnr
      output[metric_key_by_name_by_threshold[threshold]
             ['true_positive_rate']] = tpr
      output[metric_key_by_name_by_threshold[threshold]
             ['true_negative_rate']] = tnr
      output[metric_key_by_name_by_threshold[threshold]['positive_rate']] = pr
      output[metric_key_by_name_by_threshold[threshold]['negative_rate']] = nr
      output[metric_key_by_name_by_threshold[threshold]
             ['false_discovery_rate']] = fdr
      output[metric_key_by_name_by_threshold[threshold]
             ['false_omission_rate']] = fomr

    return output

  derived_computation = metric_types.DerivedMetricComputation(
      keys=keys, result=result)

  computations.append(derived_computation)
  return computations


metric_types.register_metric(FairnessIndicators)
