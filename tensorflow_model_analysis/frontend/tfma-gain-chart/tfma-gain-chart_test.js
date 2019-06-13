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
        'truePositives': 0,
        'falsePositives': 0,
        'trueNegatives': 40,
        'falseNegatives': 240,
        'threshold': 1,
      },
      {
        'truePositives': 33,
        'falsePositives': 5,
        'trueNegatives': 35,
        'falseNegatives': 207,
        'threshold': 0.875,
      },
      {
        // 25%
        'truePositives': 63,
        'falsePositives': 7,
        'trueNegatives': 33,
        'falseNegatives': 177,
        'threshold': 0.75,
      },
      {
        'truePositives': 83,
        'falsePositives': 10,
        'trueNegatives': 30,
        'falseNegatives': 157,
        'threshold': 0.625,
      },
      {
        'truePositives': 100,
        'falsePositives': 28,
        'trueNegatives': 12,
        'falseNegatives': 140,
        'threshold': 0.5,
      },
      {
        // 50%
        'truePositives': 126,
        'falsePositives': 14,
        'trueNegatives': 26,
        'falseNegatives': 114,
        'threshold': 0.375,
      },
      {
        // 75%.
        'truePositives': 189,
        'falsePositives': 21,
        'trueNegatives': 19,
        'falseNegatives': 51,
        'threshold': 0.25,
      },
      {
        // 100%
        'truePositives': 240,
        'falsePositives': 40,
        'trueNegatives': 0,
        'falseNegatives': 0,
        'threshold': 0.125,
      },
      {
        'truePositives': 240,
        'falsePositives': 40,
        'trueNegatives': 0,
        'falseNegatives': 0,
        'threshold': 0,
      }
    ];
    element.steps = 4;
    const chartData =
        element.shadowRoot.querySelector('tfma-google-chart-wrapper').data;
    assert.equal(chartData.length, 6);

    assert.deepEqual(
        chartData[0],
        ['Percentile', '', {'type': 'string', 'role': 'tooltip'}, 'Gain', {'type': 'string', 'role': 'tooltip'}]);

    assert.deepEqual(chartData[1], [0, 0, 'Random', 0, 'Origin']);
    assert.deepEqual(chartData[2], [
      25, 25, 'Random', 26.25,
      'True Positives: 63\nPredicted Positives: 70\nThreshold: 0.75000\nPercentile: 25'
    ]);
    assert.deepEqual(chartData[3], [
      50, 50, 'Random', 52.5,
      'True Positives: 126\nPredicted Positives: 140\nThreshold: 0.37500\nPercentile: 50'
    ]);
    assert.deepEqual(chartData[4], [
      75, 75, 'Random', 78.75,
      'True Positives: 189\nPredicted Positives: 210\nThreshold: 0.25000\nPercentile: 75'
    ]);
    assert.deepEqual(chartData[5], [
      100, 100, 'Random', 100,
      'True Positives: 240\nPredicted Positives: 280\nThreshold: 0.12500\nPercentile: 100'
    ]);
  });
});
