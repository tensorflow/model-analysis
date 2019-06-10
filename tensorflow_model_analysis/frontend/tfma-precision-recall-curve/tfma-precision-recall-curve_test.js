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
    const element = fixture('pr-curve-fixture');
    element.data = [
      {'precision': 0.75, 'recall': 0.125, 'threshold': Infinity},
      {'precision': 0.625, 'recall': 0.25, 'threshold': 0.5},
      {'precision': 0.5, 'recall': 0.375, 'threshold': -Infinity}
    ];
    const chartData =
        element.shadowRoot.querySelector('tfma-google-chart-wrapper').data;
    assert.equal(chartData.length, 4);
    assert.deepEqual(
        chartData[0],
        ['Recall', 'Precision', {'type': 'string', 'role': 'tooltip'}]);
    assert.deepEqual(chartData[1], [
      0.125, 0.75,
      'Prediction threshold: 1.00000\nRecall: 0.12500\nPrecision: 0.75000'
    ]);
    assert.deepEqual(chartData[2], [
      0.25, 0.625,
      'Prediction threshold: 0.50000\nRecall: 0.25000\nPrecision: 0.62500'
    ]);
    assert.deepEqual(chartData[3], [
      0.375, 0.5,
      'Prediction threshold: 0.00000\nRecall: 0.37500\nPrecision: 0.50000'
    ]);
  });
  test('handleNaNAndInfinity', () => {
    const element = fixture('pr-curve-fixture');
    element.data = [
      {'precision': 'NaN', 'recall': 'Infinity', 'threshold': 1},
    ];
    const chartData =
        element.shadowRoot.querySelector('tfma-google-chart-wrapper').data;
    assert.equal(chartData.length, 2);
    assert.deepEqual(chartData[1], [
      Infinity, NaN,
      'Prediction threshold: 1.00000\nRecall: Infinity\nPrecision: NaN'
    ]);
  });
});
