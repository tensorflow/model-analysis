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



# Prefix for post export metrics keys in metric_ops.
_NAME_PREFIX = 'post_export_metrics'


def _add_metric_prefix(name):
  return '%s/%s' % (_NAME_PREFIX, name)


def upper_bound(name):
  return name + '/upper_bound'


def lower_bound(name):
  return name + '/lower_bound'


EXAMPLE_WEIGHT = _add_metric_prefix('example_weight')
EXAMPLE_COUNT = _add_metric_prefix('example_count')
CALIBRATION_PLOT_MATRICES = _add_metric_prefix('calibration_plot/matrices')
CALIBRATION_PLOT_BOUNDARIES = _add_metric_prefix('calibration_plot/boundaries')
CONFUSION_MATRIX_AT_THRESHOLDS_MATRICES = _add_metric_prefix(
    'confusion_matrix_at_thresholds/matrices')
CONFUSION_MATRIX_AT_THRESHOLDS_THRESHOLDS = _add_metric_prefix(
    'confusion_matrix_at_thresholds/thresholds')
CONFUSION_MATRIX_AT_THRESHOLDS = _add_metric_prefix(
    'confusion_matrix_at_thresholds')  # Output-only
AUC_PLOTS_MATRICES = _add_metric_prefix('auc_plots/matrices')
AUC_PLOTS_THRESHOLDS = _add_metric_prefix('auc_plots/thresholds')
AUC = _add_metric_prefix('auc')
AUPRC = _add_metric_prefix('auprc')
PRECISION_RECALL_AT_K = _add_metric_prefix('precision_recall_at_k')
PRECISION_AT_K = _add_metric_prefix('precision_at_k')  # Output-only
RECALL_AT_K = _add_metric_prefix('recall_at_k')  # Output-only

# keys where the corresponding values are results for plots
PLOT_KEYS = [
    CALIBRATION_PLOT_MATRICES, CALIBRATION_PLOT_BOUNDARIES, AUC_PLOTS_MATRICES,
    AUC_PLOTS_THRESHOLDS
]
