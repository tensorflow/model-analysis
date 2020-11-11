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
    'Overall', 'slice:1', 'slice:2', 'slice:3', 'slice:4', 'slice:5', 'slice:6',
    'slice:7', 'slice:8', 'slice:9', 'slice:10', 'slice:11', 'slice:12',
    'slice:13', 'slice:14', 'slice:15', 'slice:16'
  ];

  // use smaller set of slices for eval comparison, for easier visualization
  const SLICES_COMPARE = ['Overall', 'slice:1', 'slice:2'];

  const METRICS = [
    'post_export_metrics/false_negative_rate@0.30',
    'post_export_metrics/false_negative_rate@0.50',
    'post_export_metrics/false_negative_rate@0.70',
  ];
  const MORE_METRICS = [
    'post_export_metrics/false_negative_rate@0.10',
    'post_export_metrics/false_negative_rate@0.30',
    'post_export_metrics/false_negative_rate@0.50',
    'post_export_metrics/false_negative_rate@0.70',
    'post_export_metrics/false_negative_rate@0.90',
  ];

  const generateBoundedValueData = (slices) => slices.map(slice => {
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
          'lowerBound': Math.random() * 0.2,
          'upperBound': Math.random() * 0.2 + 0.4,
          'value': Math.random() * 0.2 + 0.2,
          'methodology': 'POISSON_BOOTSTRAP'
        },
        'post_export_metrics/false_negative_rate@0.50': {
          'lowerBound': Math.random() * 0.3,
          'upperBound': Math.random() * 0.3 + 0.6,
          'value': Math.random() * 0.3 + 0.3,
          'methodology': 'POISSON_BOOTSTRAP'
        },
        'post_export_metrics/false_negative_rate@0.70': {
          'lowerBound': Math.random() * 0.4,
          'upperBound': Math.random() * 0.4 + 0.6,
          'value': Math.random() * 0.4 + 0.4,
          'methodology': 'POISSON_BOOTSTRAP'
        },
        'post_export_metrics/example_count': Math.floor(Math.random() * 100),
      }
    };
  });
  const generateBoundedValueDataMoreThresholds = (slices) => slices.map(slice => {
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
        'post_export_metrics/false_negative_rate@0.10': {
          'lowerBound': Math.random() * 0.2,
          'upperBound': Math.random() * 0.2 + 0.4,
          'value': Math.random() * 0.2 + 0.2,
          'methodology': 'POISSON_BOOTSTRAP'
        },
        'post_export_metrics/false_negative_rate@0.30': {
          'lowerBound': Math.random() * 0.2,
          'upperBound': Math.random() * 0.2 + 0.4,
          'value': Math.random() * 0.2 + 0.2,
          'methodology': 'POISSON_BOOTSTRAP'
        },
        'post_export_metrics/false_negative_rate@0.50': {
          'lowerBound': Math.random() * 0.3,
          'upperBound': Math.random() * 0.3 + 0.6,
          'value': Math.random() * 0.3 + 0.3,
          'methodology': 'POISSON_BOOTSTRAP'
        },
        'post_export_metrics/false_negative_rate@0.70': {
          'lowerBound': Math.random() * 0.4,
          'upperBound': Math.random() * 0.4 + 0.6,
          'value': Math.random() * 0.4 + 0.4,
          'methodology': 'POISSON_BOOTSTRAP'
        },
        'post_export_metrics/false_negative_rate@0.90': {
          'lowerBound': Math.random() * 0.4,
          'upperBound': Math.random() * 0.4 + 0.6,
          'value': Math.random() * 0.4 + 0.4,
          'methodology': 'POISSON_BOOTSTRAP'
        },
        'post_export_metrics/example_count': Math.floor(Math.random() * 100),
      }
    };
  });
  const generateDoubleValueData = (slices) => slices.map(slice => {
    return {
      'slice': slice,
      'sliceValue': slice.split(':')[1] || 'Overall',
      'metrics': {
        'accuracy': Math.random(),
        'post_export_metrics/false_negative_rate@0.30': NaN,
        'post_export_metrics/false_negative_rate@0.50': Math.random() * 0.6,
        'post_export_metrics/false_negative_rate@0.70': Math.random() * 0.8,
      }
    };
  });

  let boundedValueBarChart = document.getElementById('bounded-value');
  boundedValueBarChart.data = generateBoundedValueData(SLICES);
  boundedValueBarChart.metrics = METRICS;
  boundedValueBarChart.baseline = 'Overall';
  boundedValueBarChart.slices = SLICES.slice(1, 16);

  let doubleValueBarChart = document.getElementById('double-value');
  doubleValueBarChart.data = generateDoubleValueData(SLICES);
  doubleValueBarChart.metrics = METRICS;
  doubleValueBarChart.baseline = 'Overall';
  doubleValueBarChart.slices = SLICES.slice(1, 16);

  let boundedCompareBarChart = document.getElementById('bounded-compare');
  boundedCompareBarChart.data = generateBoundedValueDataMoreThresholds(SLICES_COMPARE);
  boundedCompareBarChart.dataCompare = generateBoundedValueDataMoreThresholds(SLICES_COMPARE);
  boundedCompareBarChart.metrics = MORE_METRICS;
  boundedCompareBarChart.baseline = 'Overall';
  boundedCompareBarChart.slices = SLICES_COMPARE.slice(1, 3);
  boundedCompareBarChart.evalName = 'EvalA';
  boundedCompareBarChart.evalNameCompare = 'EvalB';
  boundedCompareBarChart.sort = 'Slice';

  let doubleCompareBarChart = document.getElementById('double-compare');
  doubleCompareBarChart.data = generateDoubleValueData(SLICES_COMPARE);
  doubleCompareBarChart.dataCompare = generateDoubleValueData(SLICES_COMPARE);
  doubleCompareBarChart.metrics = METRICS;
  doubleCompareBarChart.baseline = 'Overall';
  doubleCompareBarChart.slices = SLICES_COMPARE.slice(1, 3);
  doubleCompareBarChart.evalName = 'EvalA';
  doubleCompareBarChart.evalNameCompare = 'EvalB';
  doubleCompareBarChart.sort = 'Eval';
})();
