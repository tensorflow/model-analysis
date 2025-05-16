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
"""Init module for TensorFlow Model Analysis post export metrics.

WARNING: This is a legacy API that is no longer available in OSS and no longer
under active development.
"""

from tensorflow_model_analysis.post_export_metrics import metric_keys
from tensorflow_model_analysis.post_export_metrics import *

__all__ = [
  "auc",
  "auc_plots",
  "calibration",
  "calibration_plot_and_prediction_histogram",
  "confusion_matrix_at_thresholds",
  "DEFAULT_KEY_PREFERENCE",
  "example_count",
  "example_weight",
  "fairness_auc",
  "fairness_indicators",
  "mean_absolute_error",
  "mean_squared_error",
  "precision_at_k",
  "recall_at_k",
  "root_mean_squared_error",
  "squared_pearson_correlation",
]
