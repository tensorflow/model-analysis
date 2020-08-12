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
    let result = {
      'post_export_metrics/example_count':
          {'doubleValue': Math.floor(Math.random() * 200 + 1000)},
      'totalWeightedExamples':
          {'doubleValue': Math.floor(Math.random() * 200 + 1000)},
    };
    const metrics = [
      'accuracy',
      'post_export_metrics/false_positive_rate@0.30',
      'post_export_metrics/false_negative_rate@0.30',
      'post_export_metrics/false_positive_rate@0.50',
      'post_export_metrics/false_negative_rate@0.50',
      'post_export_metrics/false_positive_rate@0.70',
      'post_export_metrics/false_negative_rate@0.70',
    ];
    metrics.forEach(metric => {
      let value = Math.random() * 0.7 + 0.15;
      result[metric] = {
        'boundedValue': {
          'lowerBound': value - 0.1,
          'upperBound': value + 0.1,
          'value': value,
          'methodology': 'POISSON_BOOTSTRAP'
        }
      };
    });
    return result;
  };

  const SLICES = [
    'Overall',
    'Sex:Male',
    'Sex:Female',
    'Sex:Transgender',
    'Race:asian',
    'Race:latino',
    'Race:black',
    'Race:white',
    'Religion:atheist',
    'Religion:buddhist',
    'Religion:christian',
    'Religion:hindu',
    'Religion:jewish',
  ];

  const input = SLICES.map((slice) => {
    return {
      'slice': slice,
      'sliceValue': slice.split(':')[1] || 'Overall',
      'metrics': createSliceMetrics(),
    };
  });

  const element = document.getElementsByTagName('fairness-nb-container')[0];
  element.slicingMetrics = input;
})();
