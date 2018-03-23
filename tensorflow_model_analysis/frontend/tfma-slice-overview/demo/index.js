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
  const overview = document.getElementById('overview');
  overview.metricToShow = 'weightedExamples';
  overview.slices = tfma.Data.build(
      [
        'logisticLoss', 'boundedAuc', 'calibration', 'precisionAtK',
        'binaryConfusionMatrixFromRegression', 'plot', 'weightedExamples'
      ],
      [
        {
          'slice': 'col:1',
          'metrics': {
            'logisticLoss': 0.7,
            'averageRefinedPrediction': 0.6,
            'averageLabel': 0.5,
            'totalWeightedExamples': 2000000,
            'auprc': 0.7,
            'boundedAuc':
                {'value': 0.61, 'lowerBound': 0.60, 'upperBound': 0.62},
            'precisionAtK': [{'k': 2, 'value': 0.25, 'totalPositives': 100}],
            'weightedExamples': 123
          }
        },
        {
          'slice': 'col:2',
          'metrics': {
            'logisticLoss': 0.72,
            'averageRefinedPrediction': 0.62,
            'averageLabel': 0.52,
            'totalWeightedExamples': 2000002,
            'auprc': 0.72,
            'boundedAuc':
                {'value': 0.612, 'lowerBound': 0.602, 'upperBound': 0.622},
            'precisionAtK': [{'k': 2, 'value': 0.252, 'totalPositives': 100}],
            'binaryConfusionMatrixFromRegression': [{
              'binaryClassificationThreshold': {'predictionThreshold': 0.5},
              'matrix': {
                'f1Score': 0.8,
                'accuracy': 0.7,
                'precision': 0.6,
                'recall': 0.9,
              }
            }],
            'weightedExamples': 123
          }
        },
        {
          'slice': 'col:3',
          'metrics': {
            'logisticLoss': 0.73,
            'averageRefinedPrediction': 0.63,
            'averageLabel': 0.53,
            'totalWeightedExamples': 2000003,
            'auprc': 0.73,
            'boundedAuc':
                {'value': 0.613, 'lowerBound': 0.603, 'upperBound': 0.623},
            'precisionAtK': [{'k': 2, 'value': 0.253, 'totalPositives': 100}],
            'weightedExamples': 123
          }
        }
      ]);
})();
