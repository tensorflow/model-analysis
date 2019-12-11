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
  const createTestData = () => {
    const makeData = () => {
      return {
        'post_export_metrics/false_negative_rate@0.25': NaN,
        'post_export_metrics/false_negative_rate@0.15': Math.random(),
        'post_export_metrics/example_count': Math.floor(Math.random() * 100),
        'accuracy': Math.random(),
        'weight': 2000 * (Math.random() + 0.8),
      };
    };

    const slices = [
      'Overall', 'sex:male', 'group:A', 'sex:female', 'group:B'
    ];
    const data = slices.map(slice => {
      return {
        'slice': slice,
        'sliceValue': slice.split(':')[1] || 'Overall',
        'metrics': makeData(),
      };
    });

    // Add a slice omitted because of privacy concerns.
    data.push({
      'slice': 'slice:omitted',
      'sliceValue': 'omitted',
      'metrics': {'__ERROR__': 'error message'},
    });

    return data;
  };
  const element = document.getElementsByTagName('fairness-metrics-board')[0];
  element.data = createTestData();
  element.metrics = ['post_export_metrics/false_negative_rate', 'accuracy'];
  element.thresholdedMetrics =
      new Set(['post_export_metrics/false_negative_rate']);
  element.thresholds = ['0.25', '0.15'];
  element.weightColumn = 'weight';
})();
