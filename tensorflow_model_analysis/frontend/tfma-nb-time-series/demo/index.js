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
  const view = document.getElementById('time-series');
  view.config = {'isModelCentric': true};
  view.data = [
    {
      'config': {
        'modelIdentifier': 'a.model',
        'dataIdentifier': 'a.data',
      },
      'metrics': {
        'output1': {
          '': {
            'logisticLoss': {'doubleValue': 0.7},
            'averageRefinedPrediction': {'doubleValue': 0.6},
            'averageLabel': {'doubleValue': 0.5},
            'totalWeightedExamples': {'doubleValue': 2000000},
            'auprc': {'doubleValue': 0.7},
            'boundedAuc': {
              'boundedValue':
                  {'value': 0.61, 'lowerBound': 0.60, 'upperBound': 0.62}
            },
            'precisionAtK': {
              'valueAtCutoffs': {
                'values': [{'cutoff': 2, 'value': 0.25}],
              }
            },
            'weightedExamples': {'doubleValue': 123},
          },
          'classId:0': {
            'logisticLoss': {'doubleValue': 0.71},
            'averageRefinedPrediction': {'doubleValue': 0.61},
            'averageLabel': {'doubleValue': 0.51},
            'totalWeightedExamples': {'doubleValue': 2000001},
            'auprc': {'doubleValue': 0.71},
            'boundedAuc': {
              'boundedValue':
                  {'value': 0.60, 'lowerBound': 0.61, 'upperBound': 0.62}
            },
            'precisionAtK': {
              'valueAtCutoffs': {
                'values': [{'cutoff': 2, 'value': 0.251}],
              }
            },
            'weightedExamples': {'doubleValue': 122},
          },
          'classId:1': {
            'logisticLoss': {'doubleValue': 0.72},
            'averageRefinedPrediction': {'doubleValue': 0.62},
            'averageLabel': {'doubleValue': 0.52},
            'totalWeightedExamples': {'doubleValue': 2000002},
            'auprc': {'doubleValue': 0.72},
            'boundedAuc': {
              'boundedValue':
                  {'value': 0.602, 'lowerBound': 0.612, 'upperBound': 0.622}
            },
            'precisionAtK': {
              'valueAtCutoffs': {
                'values': [{'cutoff': 2, 'value': 0.252}],
              }
            },
            'weightedExamples': {'doubleValue': 1},
          },
        },
      },
    },
    {
      'config': {
        'modelIdentifier': 'b.model',
        'dataIdentifier': 'b.data',
      },
      'metrics': {
        'output1': {
          '': {
            'logisticLoss': {'doubleValue': 0.72},
            'averageRefinedPrediction': {'doubleValue': 0.62},
            'averageLabel': {'doubleValue': 0.52},
            'totalWeightedExamples': {'doubleValue': 2000002},
            'auprc': {'doubleValue': 0.72},
            'boundedAuc': {
              'boundedValue':
                  {'value': 0.612, 'lowerBound': 0.602, 'upperBound': 0.622},
            },
            'precisionAtK': {
              'valueAtCutoffs': {
                'values': [{'cutoff': 2, 'value': 0.252}],
              }
            },
            'confusionMatrix': {
              'confusionMatrixAtThresholds': {
                'matrices': [{
                  'threshold': 0.5,
                  'precision': 0.6,
                  'recall': 0.7,
                  'truePositives': 0.8,
                  'trueNegatives': 0.9,
                  'falsePositives': 0.81,
                  'falseNegatives': 0.91,
                }],
              }
            },
            'weightedExamples': {'doubleValue': 123},
          },
          'classId:0': {
            'logisticLoss': {'doubleValue': 0.721},
            'averageRefinedPrediction': {'doubleValue': 0.621},
            'averageLabel': {'doubleValue': 0.521},
            'totalWeightedExamples': {'doubleValue': 2000001},
            'auprc': {'doubleValue': 0.721},
            'boundedAuc': {
              'boundedValue':
                  {'value': 0.612, 'lowerBound': 0.6021, 'upperBound': 0.622},
            },
            'precisionAtK': {
              'valueAtCutoffs': {
                'values': [{'cutoff': 2, 'value': 0.251}],
              }
            },
            'confusionMatrix': {
              'confusionMatrixAtThresholds': {
                'matrices': [{
                  'threshold': 0.51,
                  'precision': 0.61,
                  'recall': 0.71,
                  'truePositives': 0.81,
                  'trueNegatives': 0.91,
                  'falsePositives': 0.811,
                  'falseNegatives': 0.911,
                }],
              }
            },
            'weightedExamples': {'doubleValue': 123},
          },
        },
      },
    },
    {
      'config': {
        'modelIdentifier': 'c.model',
        'dataIdentifier': 'c.data',
      },
      'metrics': {
        'output1': {
          '': {
            'logisticLoss': {'doubleValue': 0.73},
            'averageRefinedPrediction': {'doubleValue': 0.63},
            'averageLabel': {'doubleValue': 0.53},
            'totalWeightedExamples': {'doubleValue': 2000003},
            'auprc': {'doubleValue': 0.73},
            'boundedAuc': {
              'doubleValue':
                  {'value': 0.613, 'lowerBound': 0.603, 'upperBound': 0.623}
            },
            'precisionAtK': {
              'valueAtCutoffs': {
                'values': [{'cutoff': 2, 'value': 0.253}],
              }
            },
            'weightedExamples': {'doubleValue': 123},
          },
          'classId:1': {
            'logisticLoss': {'doubleValue': 0.732},
            'averageRefinedPrediction': {'doubleValue': 0.632},
            'averageLabel': {'doubleValue': 0.532},
            'totalWeightedExamples': {'doubleValue': 2000002},
            'auprc': {'doubleValue': 0.722},
            'boundedAuc': {
              'doubleValue':
                  {'value': 0.613, 'lowerBound': 0.6032, 'upperBound': 0.623}
            },
            'precisionAtK': {
              'valueAtCutoffs': {
                'values': [{'cutoff': 2, 'value': 0.2532}],
              }
            },
            'weightedExamples': {'doubleValue': 123},
          },
        },
        'output2': {
          'classId:0': {
            'logisticLoss': {'doubleValue': 0.73},
            'averageRefinedPrediction': {'doubleValue': 0.63},
            'averageLabel': {'doubleValue': 0.53},
            'totalWeightedExamples': {'doubleValue': 2000003},
            'auprc': {'doubleValue': 0.73},
            'boundedAuc': {
              'doubleValue':
                  {'value': 0.613, 'lowerBound': 0.603, 'upperBound': 0.623}
            },
            'precisionAtK': {
              'valueAtCutoffs': {
                'values': [{'cutoff': 2, 'value': 0.253}],
              }
            },
            'weightedExamples': {'doubleValue': 123},
          },
          'classId:1': {
            'logisticLoss': {'doubleValue': 0.732},
            'averageRefinedPrediction': {'doubleValue': 0.632},
            'averageLabel': {'doubleValue': 0.532},
            'totalWeightedExamples': {'doubleValue': 2000002},
            'auprc': {'doubleValue': 0.722},
            'boundedAuc': {
              'doubleValue':
                  {'value': 0.613, 'lowerBound': 0.6032, 'upperBound': 0.623}
            },
            'precisionAtK': {
              'valueAtCutoffs': {
                'values': [{'cutoff': 2, 'value': 0.2532}],
              }
            },
            'weightedExamples': {'doubleValue': 123},
          },
        },
      },
    },
  ];
})();
