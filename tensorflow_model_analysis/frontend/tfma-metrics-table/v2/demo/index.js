/**
 * Copyright 2018 Google LLC
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
  const metrics = [
    'count',
    'logisticLoss',
    'boundedAuc',
    'calibration',
    'multiClassConfusionMatrix',
  ];
  const table = document.getElementById('table');
  table.metrics = metrics;
  table.metricFormats = {
    'count': {'type': tfma.MetricValueFormat.INT64},
  };
  table.data = tfma.Data.build(metrics, [
    {
      'slice': 'col:1',
      'metrics': {
        'logisticLoss': 0.7,
        'averageRefinedPrediction': 0.6,
        'averageLabel': 0.5,
        'count': '1000000',
        'totalWeightedExamples': 2000000,
        'auprc': 0.7,
        'boundedAuc': {'value': 0.61, 'lowerBound': 0.60, 'upperBound': 0.62},
        'multiClassConfusionMatrix': {
          'entries': [
            {'actualClass': 'A', 'predictedClass': 'A', 'weight': 100},
            {'actualClass': 'A', 'predictedClass': 'B', 'weight': 10},
            {'actualClass': 'B', 'predictedClass': 'A', 'weight': 50},
            {'actualClass': 'B', 'predictedClass': 'B', 'weight': 30}
          ],
        },
      }
    },
    {
      'slice': 'col:2',
      'metrics': {
        'logisticLoss': 0.72,
        'averageRefinedPrediction': 0.62,
        'averageLabel': 0.52,
        'count': '1000002',
        'totalWeightedExamples': 2000002,
        'auprc': 0.72,
        'boundedAuc':
            {'value': 0.612, 'lowerBound': 0.602, 'upperBound': 0.622},
      }
    },
    {
      'slice': 'col:3',
      'metrics': {
        'logisticLoss': 0.73,
        'averageRefinedPrediction': 0.63,
        'averageLabel': 0.53,
        'count': '2000003',
        'totalWeightedExamples': 2000003,
        'auprc': 0.73,
        'boundedAuc':
            {'value': 0.613, 'lowerBound': 0.603, 'upperBound': 0.623},
      },
    }
  ]);
})();
