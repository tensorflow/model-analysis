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
        'totalWeightedLabel': 10,
        'totalWeightedRefinedPrediction': 0.4,
      },
      {
        'numWeightedExamples': 10,
        'upperThresholdExclusive': 0.5,
        'lowerThresholdInclusive': 0.25,
        'totalWeightedLabel': 0,
        'totalWeightedRefinedPrediction': 4,
      },
      {
        'numWeightedExamples': 20,
        'upperThresholdExclusive': 0.75,
        'lowerThresholdInclusive': 0.5,
        'totalWeightedLabel': 10,
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
    assert.equal(chartData.length, 5);
    assert.deepEqual(chartData[0], [
      'Prediction',
      'Residual',
      {'type': 'string', 'role': 'tooltip'},
      '',
      {'type': 'string', 'role': 'tooltip'},
      'Count',
      {'type': 'string', 'role': 'tooltip'},
    ]);
    assert.deepEqual(chartData[1], [
      0.125, 1.92, 'Residual is 1.92000 for predictions in [0.00000, 0.25000)',
      0, 'Prediction range is [0.00000, 0.25000)', 5,
      'There are 5 predictions in [0.00000, 0.25000)'
    ]);
    assert.deepEqual(chartData[2], [
      0.375, -0.4, 'Residual is -0.40000 for predictions in [0.25000, 0.50000)',
      0, 'Prediction range is [0.25000, 0.50000)', 10,
      'There are 10 predictions in [0.25000, 0.50000)'
    ]);
    assert.deepEqual(chartData[3], [
      0.625, 0, 'Residual is 0.00000 for predictions in [0.50000, 0.75000)', 0,
      'Prediction range is [0.50000, 0.75000)', 20,
      'There are 20 predictions in [0.50000, 0.75000)'
    ]);
    assert.deepEqual(chartData[4], [
      0.875, -.875,
      'Residual is -0.87500 for predictions in [0.75000, 1.00000)', 0,
      'Prediction range is [0.75000, 1.00000)', 1,
      'There are 1 predictions in [0.75000, 1.00000)'
    ]);
  });
  test('HandleInfinity', () => {
    const element = fixture('residual-plot-fixture');
    element.data = [
      {
        'numWeightedExamples': 5,
        'upperThresholdExclusive': 0,
        'lowerThresholdInclusive': '-Infinity',
        'totalWeightedLabel': 5,
        'totalWeightedRefinedPrediction': 0.4,
      },
      {
        'numWeightedExamples': 10,
        'upperThresholdExclusive': 0.75,
        'lowerThresholdInclusive': 0.25,
        'totalWeightedLabel': 2,
        'totalWeightedRefinedPrediction': 4,
      },
      {
        'numWeightedExamples': 1,
        'upperThresholdExclusive': 'Infinity',
        'lowerThresholdInclusive': 1,
        'totalWeightedRefinedPrediction': 0.875,
      },
    ];
    const chartData =
        element.shadowRoot.querySelector('tfma-google-chart-wrapper').data;
    assert.equal(chartData.length, 4);
    assert.deepEqual(chartData[0], [
      'Prediction',
      'Residual',
      {'type': 'string', 'role': 'tooltip'},
      '',
      {'type': 'string', 'role': 'tooltip'},
      'Count',
      {'type': 'string', 'role': 'tooltip'},
    ]);
    assert.deepEqual(chartData[1], [
      0, 0.92, 'Residual is 0.92000 for predictions in [-Infinity, 0.00000)', 0,
      'Prediction range is [-Infinity, 0.00000)', 5,
      'There are 5 predictions in [-Infinity, 0.00000)'
    ]);
    assert.deepEqual(chartData[2], [
      0.5, -0.2, 'Residual is -0.20000 for predictions in [0.25000, 0.75000)',
      0, 'Prediction range is [0.25000, 0.75000)', 10,
      'There are 10 predictions in [0.25000, 0.75000)'
    ]);
    assert.deepEqual(chartData[3], [
      1, -0.875, 'Residual is -0.87500 for predictions in [1.00000, Infinity)',
      0, 'Prediction range is [1.00000, Infinity)', 1,
      'There are 1 predictions in [1.00000, Infinity)'
    ]);
  });
});
