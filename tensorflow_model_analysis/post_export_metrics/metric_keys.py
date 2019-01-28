# Copyright 2018 Google LLC
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
"""Library defining metric keys for post export evaluation.

These keys defines the name of the post export metrics that are supported by
TFMA.
"""



from tensorflow_model_analysis.types_compat import Optional, Text

# Prefix for post export metrics keys in metric_ops.
NAME_PREFIX = 'post_export_metrics'


def add_metric_prefix(name, prefix = NAME_PREFIX):
  return '%s/%s' % (prefix, name)  # pytype: disable=bad-return-type


def upper_bound(name):
  return name + '/upper_bound'


def lower_bound(name):
  return name + '/lower_bound'


EXAMPLE_WEIGHT_BASE = 'example_weight'
EXAMPLE_WEIGHT = add_metric_prefix(EXAMPLE_WEIGHT_BASE)
EXAMPLE_COUNT_BASE = 'example_count'
EXAMPLE_COUNT = add_metric_prefix(EXAMPLE_COUNT_BASE)
CALIBRATION_PLOT_MATRICES = 'calibration_plot/matrices'
CALIBRATION_PLOT_BOUNDARIES = 'calibration_plot/boundaries'
CONFUSION_MATRIX_AT_THRESHOLDS_MATRICES = (
    'confusion_matrix_at_thresholds/matrices')
CONFUSION_MATRIX_AT_THRESHOLDS_THRESHOLDS = (
    'confusion_matrix_at_thresholds/thresholds')
CONFUSION_MATRIX_AT_THRESHOLDS = (
    'confusion_matrix_at_thresholds')  # Output-only
FAIRNESS_CONFUSION_MATRIX_MATRICES = (
    'fairness/confusion_matrix_at_thresholds/matrices')
FAIRNESS_CONFUSION_MATRIX_THESHOLDS = (
    'fairness/confusion_matrix_at_thresholds/thresholds')
FAIRNESS_CONFUSION_MATRIX = (
    'fairness/confusion_matrix_at_thresholds')  # Output-only
AUC_PLOTS_MATRICES = 'auc_plots/matrices'
AUC_PLOTS_THRESHOLDS = 'auc_plots/thresholds'
AUC = 'auc'
AUPRC = 'auprc'
PRECISION_RECALL_AT_K = 'precision_recall_at_k'
PRECISION_AT_K = 'precision_at_k'  # Output-only
RECALL_AT_K = 'recall_at_k'  # Output-only

# keys where the corresponding values are results for plots
PLOT_KEYS = [
    CALIBRATION_PLOT_MATRICES, CALIBRATION_PLOT_BOUNDARIES, AUC_PLOTS_MATRICES,
    AUC_PLOTS_THRESHOLDS
]
