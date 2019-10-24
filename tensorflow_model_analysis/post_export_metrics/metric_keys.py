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

# Standard __future__ imports

from typing import Optional, Text

# Prefix for post export metrics keys in metric_ops.
DEFAULT_PREFIX = 'post_export_metrics'


def base_key(suffix: Text, prefix: Optional[Text] = DEFAULT_PREFIX) -> Text:
  """Creates a base key from a prefix and a suffix."""
  return '%s/%s' % (prefix, suffix)


def tagged_key(key: Text, tag: Text) -> Text:
  """Returns a base key tagged with a user defined tag.

  The tag is inserted after the base key's initial prefix.

  Example: tagged_key('a/c', 'b') -> 'a/b/c'
  Example: tagged_key('a', 'b') -> 'a/b'  # Use case for plots keys.

  Args:
    key: Base key.
    tag: Tag to add to base key.
  """
  parts = key.split('/')
  if len(parts) > 1:
    return '%s/%s/%s' % (parts[0], tag, '/'.join(parts[1:]))
  return '%s/%s' % (key, tag)


def upper_bound_key(key: Text) -> Text:
  """Creates an upper_bound key from a child key."""
  return key + '/upper_bound'


def lower_bound_key(key: Text) -> Text:
  """Create a lower_bound key from a child key."""
  return key + '/lower_bound'


# Not actually for any metric, just used for communicating errors.
ERROR_METRIC = '__ERROR__'

EXAMPLE_WEIGHT = base_key('example_weight')
EXAMPLE_COUNT = base_key('example_count')
SQUARED_PEARSON_CORRELATION = base_key('squared_pearson_correlation')
CALIBRATION = base_key('calibration')
_CALIBRATION_PLOT_MATRICES_SUFFIX = 'calibration_plot/matrices'
CALIBRATION_PLOT_MATRICES = base_key(_CALIBRATION_PLOT_MATRICES_SUFFIX)
_CALIBRATION_PLOT_BOUNDARIES_SUFFIX = 'calibration_plot/boundaries'
CALIBRATION_PLOT_BOUNDARIES = base_key(_CALIBRATION_PLOT_BOUNDARIES_SUFFIX)
CONFUSION_MATRIX_AT_THRESHOLDS_MATRICES = base_key(
    'confusion_matrix_at_thresholds/matrices')
CONFUSION_MATRIX_AT_THRESHOLDS_THRESHOLDS = base_key(
    'confusion_matrix_at_thresholds/thresholds')
CONFUSION_MATRIX_AT_THRESHOLDS = base_key(
    'confusion_matrix_at_thresholds')  # Output-only
FAIRNESS_CONFUSION_MATRIX_MATRICES = base_key(
    'fairness/confusion_matrix_at_thresholds/matrices')
FAIRNESS_CONFUSION_MATRIX_THESHOLDS = base_key(
    'fairness/confusion_matrix_at_thresholds/thresholds')
FAIRNESS_CONFUSION_MATRIX = base_key(
    'fairness/confusion_matrix_at_thresholds')  # Output-only
FAIRNESS_AUC = base_key('fairness/auc')
_AUC_PLOTS_MATRICES_SUFFIX = 'auc_plots/matrices'
AUC_PLOTS_MATRICES = base_key(_AUC_PLOTS_MATRICES_SUFFIX)
_AUC_PLOTS_THRESHOLDS_SUFFIX = 'auc_plots/thresholds'
AUC_PLOTS_THRESHOLDS = base_key(_AUC_PLOTS_THRESHOLDS_SUFFIX)
AUC = base_key('auc')
AUPRC = base_key('auprc')
PRECISION_AT_K = base_key('precision_at_k')
RECALL_AT_K = base_key('recall_at_k')
MEAN_ABSOLUTE_ERROR = base_key('mean_absolute_error')
MEAN_SQUARED_ERROR = base_key('mean_squared_error')
ROOT_MEAN_SQUARED_ERROR = base_key('root_mean_squared_error')

# Suffixes of keys where the corresponding values are results for plots
_PLOT_SUFFIXES = [
    _CALIBRATION_PLOT_MATRICES_SUFFIX, _CALIBRATION_PLOT_BOUNDARIES_SUFFIX,
    _AUC_PLOTS_MATRICES_SUFFIX, _AUC_PLOTS_THRESHOLDS_SUFFIX
]


def is_plot_key(key: Text) -> bool:
  """Returns true if key is a plot key."""
  # We need to check for suffixes here because metrics may have prefixes based
  # on multiple labels and/or heads.
  for suffix in _PLOT_SUFFIXES:
    if key.endswith(suffix):
      return True
  return False
