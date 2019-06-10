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
  let element;

  test('PutDataInBuckets', () => {
    element = fixture('pd-fixture');
    element.bucketSize = 0.25;
    element.data = [
      {'numWeightedExamples': 2, 'totalWeightedRefinedPrediction': 0.1 * 2},
      {'numWeightedExamples': 4, 'totalWeightedRefinedPrediction': 0.2 * 4},
      {'numWeightedExamples': 7, 'totalWeightedRefinedPrediction': 0.8 * 7}
    ];
    const chartData =
        element.shadowRoot.querySelector('tfma-google-chart-wrapper').data;
    assert.equal(chartData.length, 5);
    assert.deepEqual(
        chartData[0],
        ['Prediction', 'Count', {'type': 'string', 'role': 'tooltip'}]);
    assert.deepEqual(
        chartData[1],
        [0.125, 6, '6 weighted example(s) between 0.0000 and 0.2500']);
    assert.deepEqual(
        chartData[2],
        [0.375, 0, '0 weighted example(s) between 0.2500 and 0.5000']);
    assert.deepEqual(
        chartData[3],
        [0.625, 0, '0 weighted example(s) between 0.5000 and 0.7500']);
    assert.deepEqual(
        chartData[4],
        [0.875, 7, '7 weighted example(s) between 0.7500 and 1.0000']);
  });

  test('BoundaryValues', () => {
    element.bucketSize = 0.5;
    element.data = [
      {'numWeightedExamples': 2, 'totalWeightedRefinedPrediction': 0},
      {'numWeightedExamples': 1, 'totalWeightedRefinedPrediction': 1}
    ];
    const chartData =
        element.shadowRoot.querySelector('tfma-google-chart-wrapper').data;
    assert.equal(chartData.length, 3);
    assert.deepEqual(
        chartData[0],
        ['Prediction', 'Count', {'type': 'string', 'role': 'tooltip'}]);
    assert.deepEqual(
        chartData[1],
        [0.25, 2, '2 weighted example(s) between 0.0000 and 0.5000']);
    assert.deepEqual(
        chartData[2],
        [0.75, 1, '1 weighted example(s) between 0.5000 and 1.0000']);
  });
});
