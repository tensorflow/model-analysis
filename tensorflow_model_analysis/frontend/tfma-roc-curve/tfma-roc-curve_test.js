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
    const element = fixture('roc-curve-fixture');
    element.data = [
      {
        'truePositives': 0,
        'falseNegatives': 64,
        'falsePositives': 0,
        'trueNegatives': 128,
        'threshold': Infinity
      },
      {
        'truePositives': 75,
        'falseNegatives': 25,
        'falsePositives': 128,
        'trueNegatives': 128,
        'threshold': 0.5
      },
      {
        'truePositives': 112,
        'falseNegatives': 16,
        'falsePositives': 224,
        'trueNegatives': 32,
        'threshold': -Infinity
      }
    ];
    const chartData =
        element.shadowRoot.querySelector('tfma-google-chart-wrapper').data;
    assert.equal(4, chartData.length);
    assert.deepEqual(chartData[0], [
      'FPR', '', {'type': 'string', 'role': 'tooltip'}, 'TPR',
      {'type': 'string', 'role': 'tooltip'}
    ]);
    assert.deepEqual(chartData[1], [
      0, 0, 'Random', 0,
      'Prediction threshold: 1.00000\nFPR: 0.00000\nTPR: 0.00000'
    ]);
    assert.deepEqual(chartData[2], [
      0.5, 0.5, 'Random', 0.75,
      'Prediction threshold: 0.50000\nFPR: 0.50000\nTPR: 0.75000'
    ]);
    assert.deepEqual(chartData[3], [
      0.875, 0.875, 'Random', 0.875,
      'Prediction threshold: 0.00000\nFPR: 0.87500\nTPR: 0.87500'
    ]);
  });
});
