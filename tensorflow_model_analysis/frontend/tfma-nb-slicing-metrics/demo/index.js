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
  const view = document.getElementById('slicing-metrics');
  view.config = {'weightedExamplesColumn': 'weightedExamples'};
  view.data = [
    {
      'slice': 'col:1',
      'metrics': {
        'logisticLoss': {
          'doubleValue': 0.7,
        },
        'averageRefinedPrediction': {
          'doubleValue': 0.6,
        },
        'averageLabel': {
          'doubleValue': 0.5,
        },
        'totalWeightedExamples': {
          'doubleValue': 2000000,
        },
        'auprc': {
          'doubleValue': 0.7,
        },
        'boundedAuc': {
          'boundedValue':
              {'value': 0.61, 'lowerBound': 0.60, 'upperBound': 0.62},
        },
        'precisionAtK': {
          'value_at_cutoffs': {
            'values': [{'cutoff': 2, 'value': 0.25}],
          },
        },
        'weightedExamples': {'doubleValue': 123},
      }
    },
    {
      'slice': 'col:2',
      'metrics': {
        'logisticLoss': {
          'doubleValue': 0.72,
        },
        'averageRefinedPrediction': {
          'doubleValue': 0.62,
        },
        'averageLabel': {
          'doubleValue': 0.52,
        },
        'totalWeightedExamples': {
          'doubleValue': 2000002,
        },
        'auprc': {
          'doubleValue': 0.72,
        },
        'boundedAuc': {
          'boundedValue':
              {'value': 0.612, 'lowerBound': 0.602, 'upperBound': 0.622},
        },
        'precisionAtK': {
          'value_at_cutoffs': {
            'values': [{'cutoff': 2, 'value': 0.252}],
          },
        },
        'weightedExamples': {'doubleValue': 123},
      }
    },
    {
      'slice': 'col:3',
      'metrics': {
        'logisticLoss': {
          'doubleValue': 0.73,
        },
        'averageRefinedPrediction': {
          'doubleValue': 0.63,
        },
        'averageLabel': {
          'doubleValue': 0.53,
        },
        'totalWeightedExamples': {
          'doubleValue': 2000003,
        },
        'auprc': {
          'doubleValue': 0.73,
        },
        'boundedAuc': {
          'boundedValue':
              {'value': 0.613, 'lowerBound': 0.603, 'upperBound': 0.623},
        },
        'precisionAtK': {
          'value_at_cutoffs': {
            'values': [{'cutoff': 2, 'value': 0.253}],
          },
        },
        'weightedExamples': {'doubleValue': 123},
      }
    }
  ];
})();
