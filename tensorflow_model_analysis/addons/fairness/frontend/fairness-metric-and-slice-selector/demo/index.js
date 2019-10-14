/**
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

(() => {
  const element =
      document.getElementsByTagName('fairness-metric-and-slice-selector')[0];
  element.availableMetrics =
      ['post_export_metrics/false_negative_rate',
       'post_export_metrics/false_positive_rate',
       'post_export_metrics/negative_rate', 'post_export_metrics/positive_rate',
       'post_export_metrics/true_negative_rate',
       'post_export_metrics/true_positive_rate', 'accuracy',
       'accuracy_baseline', 'auc', 'auc_precision_recall', 'average_loss',
       'label/mean', 'post_export_metrics/example_count', 'precision',
       'prediction/mean', 'recall', 'totalWeightedExamples'];
})();
