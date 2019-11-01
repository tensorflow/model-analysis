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
  const SLICES = [
    'Overall', 'Slice:1', 'Slice:2', 'Slice:3', 'Slice:4', 'Slice:5', 'Slice:6',
    'Slice:7', 'Slice:8', 'Slice:9', 'Slice:10', 'Slice:11', 'Slice:12',
    'Slice:13', 'Slice:14', 'Slice:15', 'Slice:16'
  ];
  const EXAMPLE_COUNTS = {};
  SLICES.forEach(function(slice) {
    EXAMPLE_COUNTS[slice] = Math.floor(Math.random() * 100);
  });

  const BOUNDED_VALUE_DATA = SLICES.map((slice) => {
    return {
      'slice': slice,
      'sliceValue': slice.split(':')[1] || 'Overall',
      'metrics': {
        'accuracy': {
          'lowerBound': Math.random() * 0.3,
          'upperBound': Math.random() * 0.3 + 0.6,
          'value': Math.random() * 0.3 + 0.3,
          'methodology': 'POISSON_BOOTSTRAP'
        },
        'post_export_metrics/false_negative_rate@0.30': {
          'lowerBound': Math.random() * 0.3,
          'upperBound': Math.random() * 0.3 + 0.6,
          'value': Math.random() * 0.3 + 0.3,
          'methodology': 'POISSON_BOOTSTRAP'
        },
        'post_export_metrics/false_negative_rate@0.50': {
          'lowerBound': Math.random() * 0.3,
          'upperBound': Math.random() * 0.3 + 0.6,
          'value': Math.random() * 0.3 + 0.3,
          'methodology': 'POISSON_BOOTSTRAP'
        },
        'post_export_metrics/false_negative_rate@0.70': {
          'lowerBound': NaN,
          'upperBound': NaN,
          'value': NaN,
          'methodology': 'POISSON_BOOTSTRAP'
        },
        'post_export_metrics/example_count': EXAMPLE_COUNTS[slice],
      }
    };
  });

  const DOUBLE_VALUE_DATA = SLICES.map((slice) => {
    return {
      'slice': slice,
      'sliceValue': slice.split(':')[1] || 'Overall',
      'metrics': {
        'accuracy': Math.random(),
        'post_export_metrics/false_negative_rate@0.30': Math.random(),
        'post_export_metrics/false_negative_rate@0.50': Math.random(),
        'post_export_metrics/false_negative_rate@0.70': Math.random(),
        'post_export_metrics/example_count': EXAMPLE_COUNTS[slice],
      }
    };
  });

  let element = document.getElementById('bounded-value');
  element.slices = SLICES;
  element.data = BOUNDED_VALUE_DATA;
  element.metric = 'post_export_metrics/false_negative_rate';
  element.thresholds = ['0.30', '0.50', '0.70'];
  element.baseline = 'Overall';

  element = document.getElementById('double-value');
  element.slices = SLICES;
  element.data = DOUBLE_VALUE_DATA;
  element.metric = 'accuracy';
  element.thresholds = ['0.30', '0.50', '0.70'];
  element.baseline = 'Overall';
})();
