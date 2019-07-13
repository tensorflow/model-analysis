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
    const element = fixture('residual-plot-fixture');
    element.data = [
      {
        'numWeightedExamples': 5,
        'upperThresholdExclusive': 0.25,
        'lowerThresholdInclusive': 0,
        'totalWeightedRefinedPrediction': 0.4,
      },
      {
        'numWeightedExamples': 10,
        'upperThresholdExclusive': 0.5,
        'lowerThresholdInclusive': 0.25,
        'totalWeightedRefinedPrediction': 4,
      },
      {
        'numWeightedExamples': 20,
        'upperThresholdExclusive': 0.75,
        'lowerThresholdInclusive': 0.5,
        'totalWeightedRefinedPrediction': 10,
      },
      {
        'numWeightedExamples': 1,
        'upperThresholdExclusive': 1,
        'lowerThresholdInclusive': 0.75,
        'totalWeightedRefinedPrediction': 0.875,
      },
    ];
    const chartData =
        element.shadowRoot.querySelector('tfma-google-chart-wrapper').data;
    assert.deepEqual(chartData[0], [
      'Label',
      'Residual',
      {'type': 'string', 'role': 'tooltip'},
      '',
      {'type': 'string', 'role': 'tooltip'},
      'Count',
      {'type': 'string', 'role': 'tooltip'},
    ]);
    assert.deepEqual(chartData[1], [
      0.125, 0.045, 'Residual is 0.04500 for label in [0.00000, 0.25000)', 0,
      'Prediciton range is [0.00000, 0.25000)', 5,
      'There are 5 example(s) for label in [0.00000, 0.25000)'
    ]);
    assert.deepEqual(chartData[2], [
      0.375, -0.025000000000000022,
      'Residual is -0.02500 for label in [0.25000, 0.50000)', 0,
      'Prediciton range is [0.25000, 0.50000)', 10,
      'There are 10 example(s) for label in [0.25000, 0.50000)'
    ]);
    assert.deepEqual(chartData[3], [
      0.625, 0.125, 'Residual is 0.12500 for label in [0.50000, 0.75000)', 0,
      'Prediciton range is [0.50000, 0.75000)', 20,
      'There are 20 example(s) for label in [0.50000, 0.75000)'
    ]);
    assert.deepEqual(chartData[4], [
      0.875, 0, 'Residual is 0.00000 for label in [0.75000, 1.00000)', 0,
      'Prediciton range is [0.75000, 1.00000)', 1,
      'There are 1 example(s) for label in [0.75000, 1.00000)'
    ]);
  });
});
