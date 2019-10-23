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
suite('tests', () => {
  const makeMultiClassData = () => {
    return {
      'matrices': [
        {
          'threshold': 0.2,
          'entries': [
            {'numWeightedExamples': 45},
            {'predictedClassId': 1, 'numWeightedExamples': 5}, {
              'actualClassId': 1,
            },
            {
              'actualClassId': 1,
              'predictedClassId': 1,
              'numWeightedExamples': 55
            },
            {'predictedClassId': -1, 'numWeightedExamples': 5}, {
              'actualClassId': 1,
              'predictedClassId': -1,
              'numWeightedExamples': 10
            }
          ],
        },
        {
          'threshold': 0.5,
          'entries': [
            {'numWeightedExamples': 45},
            {'predictedClassId': 1, 'numWeightedExamples': 10},
            {'actualClassId': 1}, {
              'actualClassId': 1,
              'predictedClassId': 1,
              'numWeightedExamples': 50
            },
            {'predictedClassId': -1, 'numWeightedExamples': 5}, {
              'actualClassId': 1,
              'predictedClassId': -1,
              'numWeightedExamples': 10
            }
          ],
        },
        {
          'threshold': 0.8,
          'entries': [{
            'actualClassId': 1,
            'predictedClassId': -1,
            'numWeightedExamples': 120
          }],
        }
      ]
    };
  };
  let element;

  const makeMultiLabelData = () => {
    return {
      'matrices': [{
        'threshold': 0.5,
        'entries': [
          {
            'truePositives': 45,
            'trueNegatives': 15,
          },
          {
            'predictedClassId': 1,
            'truePositives': 10,
            'trueNegatives': 30,
            'falsePositives': 12,
            'falseNegatives': 8.
          },
          {
            'actualClassId': 1,
            'truePositives': 3,
            'trueNegatives': 15,
            'falsePositives': 25,
            'falseNegatives': 13,
          },
          {
            'actualClassId': 1,
            'predictedClassId': 1,
            'truePositives': 22,
            'trueNegatives': 16,
            'falsePositives': 12,
            'falseNegatives': 10,
          },
        ],
      }],
    };
  };

  const run = (multiLabel, cb) => {
    element = fixture('matrix');
    element.multiLabel = multiLabel;
    element.data = multiLabel ? makeMultiLabelData() : makeMultiClassData();
    setTimeout(cb, 0);
  };

  test('ParsesMultiClassSingleLabelData', done => {
    const dataLoadedProperly = () => {
      assert.equal(element.selectedThreshold_, 0.5);
      const matrix = element.selectedMatrix_;
      assert.deepEqual(matrix[0].entries, {
        '0': {
          positives: 45,
          truePositives: 45,
          falsePositives: 0,
          falseNegatives: 0
        },
        '1': {
          positives: 10,
          truePositives: 0,
          falsePositives: 10,
          falseNegatives: 0
        },
        '-1': {
          positives: 5,
          truePositives: 0,
          falsePositives: 5,
          falseNegatives: 0
        }
      });
      assert.deepEqual(matrix[1].entries, {
        '0': {
          positives: 0,
          truePositives: 0,
          falsePositives: 0,
          falseNegatives: 0
        },
        '1': {
          positives: 50,
          truePositives: 50,
          falsePositives: 0,
          falseNegatives: 0
        },
        '-1': {
          positives: 10,
          truePositives: 0,
          falsePositives: 10,
          falseNegatives: 0
        }
      });
      done();
    };
    run(false, dataLoadedProperly);
  });
  test('ParseMultiLabelData', done => {
    const dataLoadedProperly = () => {
      assert.equal(element.selectedThreshold_, 0.5);
      const matrix = element.selectedMatrix_;
      assert.deepEqual(matrix[0].entries, {
        '0': {
          positives: 45,
          truePositives: 45,
          falsePositives: 0,
          falseNegatives: 0
        },
        '1': {
          positives: 22,
          truePositives: 10,
          falsePositives: 12,
          falseNegatives: 8
        }
      });
      assert.deepEqual(matrix[1].entries, {
        '0': {
          positives: 28,
          truePositives: 3,
          falsePositives: 25,
          falseNegatives: 13
        },
        '1': {
          positives: 34,
          truePositives: 22,
          falsePositives: 12,
          falseNegatives: 10
        }
      });
      done();
    };
    run(true, dataLoadedProperly);
  });
});
