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
  const browser = document.getElementById('browser');
  const metrics = [
    'logisticLoss', 'boundedAuc', 'calibration', 'precisionAtK',
    'binaryConfusionMatrixFromRegression', 'plot', 'weightedExamples',
  ];
  browser.metrics = metrics;
  browser.weightedExamplesColumn = 'weightedExamples';
  browser.data = [
    {
      'slice': 'col:1',
      'metrics': {
        'logisticLoss': 0.7,
        'averageRefinedPrediction': 0.6,
        'averageLabel': 0.5,
        'totalWeightedExamples': 2000000,
        'auprc': 0.7,
        'boundedAuc': {'value': 0.61, 'lowerBound': 0.60, 'upperBound': 0.62},
        'precisionAtK': { 'values': [{'cutoff': 2, 'value': 0.25}]},
        'binaryConfusionMatrixFromRegression': {
          'matrices': [{
            'threshold': 0.5,
            'boundedFalseNegatives': {'value': 0.61, 'lowerBound': 0.602, 'upperBound': 0.611},
            'boundedTrueNegatives': {'value': 0.62, 'lowerBound': 0.602, 'upperBound': 0.622},
            'boundedFalsePositives': {'value': 0.63, 'lowerBound': 0.602, 'upperBound': 0.633},
            'boundedTruePositives': {'value': 0.64, 'lowerBound': 0.602, 'upperBound': 0.642},
            'boundedPrecision': {'value': 0.6, 'lowerBound': 0.602, 'upperBound': 0.622},
            'boundedRecall': {'value': 0.9, 'lowerBound': 0.602, 'upperBound': 0.622},
          }],
        },
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
        'precisionAtK': {'values': [{'cutoff': 2, 'value': 0.252}]},
        'binaryConfusionMatrixFromRegression': {
          'matrices': [{
            'threshold': 0.5,
            'falseNegatives': 10,
            'trueNegatives': 20,
            'falsePositives': 30,
            'truePositives': 40,
            'precision': 0.6,
            'recall': 0.9,
          }],
        },
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
        'precisionAtK': {'values': [{'cutoff': 2, 'value': 0.253}]},
        'weightedExamples': 123,
      }
    },
  ];
})();
