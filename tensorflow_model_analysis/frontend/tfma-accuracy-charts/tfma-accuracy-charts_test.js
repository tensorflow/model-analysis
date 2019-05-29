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
suite('tests', () => {
  test('ExtractsData', () => {
    const element = fixture('charts');
    element.data = [
      {
        'precision': 0.75,
        'recall': 0.125,
        'accuracy': 0.7,
        'threshold': Infinity
      },
      {'precision': 0.625, 'recall': 0.25, 'accuracy': 0.6, 'threshold': 0.5}, {
        'precision': 0.5,
        'recall': 0.375,
        'accuracy': 0.5,
        'threshold': -Infinity
      }
    ];
    const chartData = element.shadowRoot.querySelector('google-chart').data;
    assert.equal(chartData.length, 4);
    assert.deepEqual(chartData[0], [
      'Threshold', 'Accuracy', {'type': 'string', 'role': 'tooltip'},
      'Precision', {'type': 'string', 'role': 'tooltip'}, 'Recall',
      {'type': 'string', 'role': 'tooltip'}, 'F1',
      {'type': 'string', 'role': 'tooltip'}
    ]);

    assert.deepEqual(chartData[1], [
      1, 0.7, 'Accuracy: 0.70000, threshold: 1.00000', 0.75,
      'Precision: 0.75000, threshold: 1.00000', 0.125,
      'Recall: 0.12500, threshold: 1.00000', 0.21428571428571427,
      'F1 Score: 0.21429, threshold: 1.00000'
    ]);
    assert.deepEqual(chartData[2], [
      0.5, 0.6, 'Accuracy: 0.60000, threshold: 0.50000', 0.625,
      'Precision: 0.62500, threshold: 0.50000', 0.25,
      'Recall: 0.25000, threshold: 0.50000', 0.35714285714285715,
      'F1 Score: 0.35714, threshold: 0.50000'
    ]);
    assert.deepEqual(chartData[3], [
      0, 0.5, 'Accuracy: 0.50000, threshold: 0.00000', 0.5,
      'Precision: 0.50000, threshold: 0.00000', 0.375,
      'Recall: 0.37500, threshold: 0.00000', 0.42857142857142855,
      'F1 Score: 0.42857, threshold: 0.00000'
    ]);
  });
  test('handleNaNAndInfinity', () => {
    const element = fixture('charts');
    element.data = [
      {'precision': 'NaN', 'recall': 'Infinity', 'threshold': 1},
    ];
    const chartData = element.shadowRoot.querySelector('google-chart').data;
    assert.equal(chartData.length, 2);
    assert.deepEqual(chartData[1], [
      1, 0, 'Accuracy: 0.00000, threshold: 1.00000', NaN,
      'Precision: NaN, threshold: 1.00000', Infinity,
      'Recall: Infinity, threshold: 1.00000', NaN,
      'F1 Score: NaN, threshold: 1.00000'
    ]);
  });
});
