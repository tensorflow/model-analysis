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
  const createSliceMetrics = () => {
    return {
      'accuracy': {
        'boundedValue': {
          'lowerBound': Math.random() * 0.3,
          'upperBound': Math.random() * 0.3 + 0.6,
          'value': Math.random() * 0.3 + 0.3,
          'methodology': 'POISSON_BOOTSTRAP'
        }
      },
      'post_export_metrics/positive_rate@0.50': {
        'doubleValue': Math.random(),
      },
      'post_export_metrics/true_positive_rate@0.50': {
        'doubleValue': Math.random(),
      },
      'post_export_metrics/false_positive_rate@0.50': {
        'doubleValue': Math.random(),
      },
      'post_export_metrics/positive_rate@0.60': {
        'doubleValue': Math.random(),
      },
      'post_export_metrics/true_positive_rate@0.60': {
        'doubleValue': Math.random(),
      },
      'post_export_metrics/false_positive_rate@0.60': {
        'doubleValue': Math.random(),
      },
      'post_export_metrics/example_count': {
        'doubleValue': Math.floor(Math.random() * 100)
      },
      'totalWeightedExamples': {'doubleValue': 2000 * (Math.random() + 0.8)}
    };
  };

  const SLICES = [
    'Overall', 'Slice:1', 'Slice:2', 'Slice:3', 'Slice:4', 'Slice:5', 'Slice:6',
    'Slice:7', 'Slice:8', 'Slice:9', 'Slice:10', 'Slice:11', 'Slice:12',
    'Slice:13', 'Slice:14', 'Slice:15', 'Slice:16'
  ];
  const INPUT = SLICES.reduce((acc, slice) => {
    acc.push({
      'slice': slice,
      'sliceValue': slice.split(':')[1] || 'Overall',
      'metrics': createSliceMetrics(),
    });
    return acc;
  }, []);
  const element =
      document.getElementsByTagName('fairness-tensorboard-container')[0];
  element.evaluationRuns_ = ['1', '2', '3'];
  element.slicingMetrics_ = INPUT;
  setTimeout(() => {
    element.slicingMetrics_ = undefined;
  }, 5000);
})();
