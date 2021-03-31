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

goog.module('tensorflow_model_analysis.addons.fairness.frontend.fairness_bounded_value_bar_chart.D3DataObjectTest');
goog.setTestOnly('tensorflow_model_analysis.addons.fairness.frontend.fairness_bounded_value_bar_chart.D3DataObjectTest');

const D3DataObject = goog.require('tensorflow_model_analysis.addons.fairness.frontend.fairness_bounded_value_bar_chart.D3DataObject');
const testSuite = goog.require('goog.testing.testSuite');

/**
 * Create a basic metricsData object, used for instantiating D3DataObjects
 * @param {string} fullSliceName
 * @param {string} evalName to distinguish evals.
 * @return {!Array<!Object>}
 * @private
 */
const metricsData = (fullSliceName, evalName) => {
  const obj = new Object();
  obj.fullSliceName = fullSliceName;
  obj.evalName = evalName;
  return [obj];
};

const D3_DATA = [
  D3DataObject.create(metricsData('Overall', 'eval1'), {}),
  D3DataObject.create(metricsData('slice:A', 'eval1'), {}),
  D3DataObject.create(metricsData('slice:B', 'eval1'), {}),
  D3DataObject.create(metricsData('Overall', 'eval2'), {}),
  D3DataObject.create(metricsData('slice:A', 'eval2'), {}),
  D3DataObject.create(metricsData('slice:B', 'eval2'), {})
];

const ALTERNATING_EVALS =
    ['eval1', 'eval2', 'eval1', 'eval2', 'eval1', 'eval2'];
const ALTERNATING_SLICES =
    ['Overall', 'slice:A', 'slice:B', 'Overall', 'slice:A', 'slice:B'];

const SLICE_SORT = 'Slice';
const EVAL_SORT = 'Eval';

testSuite({
  testSortBySlice_BaselineOverall() {
    const sortedD3data =
        D3_DATA.sort(D3DataObject.sortFunction('Overall', SLICE_SORT, true));
    const actualSlices = sortedD3data.map(d => d.fullSliceName);
    const actualEvals = sortedD3data.map(d => d.evalName);

    const expectedSlicesOverall =
        ['Overall', 'Overall', 'slice:A', 'slice:A', 'slice:B', 'slice:B'];

    assertArrayEquals(actualSlices, expectedSlicesOverall);
    assertArrayEquals(actualEvals, ALTERNATING_EVALS);
  },

  testSortBySlice_BaselineA() {
    const sortedD3data =
        D3_DATA.sort(D3DataObject.sortFunction('slice:A', SLICE_SORT, true));
    const actualSlices = sortedD3data.map(d => d.fullSliceName);
    const actualEvals = sortedD3data.map(d => d.evalName);

    const expectedSlicesA = [
      'slice:A',
      'slice:A',
      'Overall',
      'Overall',
      'slice:B',
      'slice:B',
    ];

    assertArrayEquals(actualSlices, expectedSlicesA);
    assertArrayEquals(actualEvals, ALTERNATING_EVALS);
  },

  testSortBySlice_BaselineB() {
    const sortedD3data =
        D3_DATA.sort(D3DataObject.sortFunction('slice:B', SLICE_SORT, true));
    const actualSlices = sortedD3data.map(d => d.fullSliceName);
    const actualEvals = sortedD3data.map(d => d.evalName);

    const expectedSlicesB = [
      'slice:B',
      'slice:B',
      'Overall',
      'Overall',
      'slice:A',
      'slice:A',
    ];

    assertArrayEquals(actualSlices, expectedSlicesB);
    assertArrayEquals(actualEvals, ALTERNATING_EVALS);
  },

  testSortByEval() {
    const sortedD3data =
        D3_DATA.sort(D3DataObject.sortFunction('Overall', EVAL_SORT, true));
    const actualSlices = sortedD3data.map(d => d.fullSliceName);
    const actualEvals = sortedD3data.map(d => d.evalName);

    const expectedEvals =
        ['eval1', 'eval1', 'eval1', 'eval2', 'eval2', 'eval2'];

    assertArrayEquals(actualSlices, ALTERNATING_SLICES);
    assertArrayEquals(actualEvals, expectedEvals);
  }
});
