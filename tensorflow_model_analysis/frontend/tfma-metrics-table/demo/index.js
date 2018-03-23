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
    'precisionAtK',
    'binaryConfusionMatrixFromRegression',
    'plot',
    'multiValuesPrecisionAtK',
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
        'precisionAtK': [{'k': 2, 'value': 0.25, 'totalPositives': 100}],
        'multiValuesPrecisionAtK': [{
          'k': 2,
          'macroValue': 0.25,
          'microValue': 0.3,
          'weightedValue': 0.4
        }],
        'multiValuesThresholdBasedBinaryClassificationMetrics': [{
          'binaryClassificationThreshold': {'predictionThreshold': 0.5},
          'macroPrecision': 0.91,
          'macroRecall': 0.81,
          'macroF1Score': 0.71,
          'macroAccuracy': 0.61,
          'microPrecision': 0.92,
          'microRecall': 0.82,
          'microF1Score': 0.72,
          'microAccuracy': 0.62,
        }],
        'multiClassConfusionMatrix': {
          'entries': [
            {'actualClass': 'A', 'predictedClass': 'A', 'weight': 100},
            {'actualClass': 'A', 'predictedClass': 'B', 'weight': 10},
            {'actualClass': 'B', 'predictedClass': 'A', 'weight': 50},
            {'actualClass': 'B', 'predictedClass': 'B', 'weight': 30}
          ],
        },
        'plot': {
          'types': [tfma.PlotTypes.PRECISION_RECALL_CURVE],
          'slice': 'col:1'
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
        'precisionAtK': [{'k': 2, 'value': 0.252, 'totalPositives': 100}],
        'multiValuesPrecisionAtK': [
          {'k': 2, 'macroValue': 0, 'microValue': 0.32, 'weightedValue': 0.42}
        ],
        'binaryConfusionMatrixFromRegression': [{
          'binaryClassificationThreshold': {'predictionThreshold': 0.5},
          'matrix': {
            'f1Score': 0.8,
            'accuracy': 0.7,
            'precision': 0.6,
            'recall': 0.9,
          }
        }],
        'plot': {
          'types': [
            tfma.PlotTypes.PREDICTION_DISTRIBUTION,
            tfma.PlotTypes.CALIBRATION_PLOT
          ],
          'slice': 'col:2'
        }
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
        'precisionAtK': [{'k': 2, 'value': 0.253, 'totalPositives': 100}],
        'multiValuesPrecisionAtK': [
          {'k': 2, 'macroValue': 0.253, 'microValue': 0.33}, {
            'k': 1,
            'macroValue': 0.252,
            'microValue': 0.32,
            'weightedValue': 0.42
          },
          {'k': 0, 'macroValue': 0.251, 'microValue': 0.31}
        ],
      }
    }
  ]);
})();
