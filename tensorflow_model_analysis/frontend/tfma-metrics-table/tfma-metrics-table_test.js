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
goog.module('tfma.MetricsHistogramTest');

const TestUtil = goog.require('tfma.TestUtil');

suite('tests', () => {
  /** @const {number} */
  const INITIALIZATION_TIMEOUT_MS = 2000;

  /** @const {number} */
  const TIMEOUT_MS = 200;

  /**
   * Test component element.
   * @private {Element}
   */
  let element;

  /**
   * The GViz table under the test component.
   * @private {!Element}
   */
  let table;

  function next(step) {
    setTimeout(step, TIMEOUT_MS);
  }

  /**
   * Sets up the test fixture and run the test.
   * @param {!tfma.Data} testData
   * @param {function()} cb
   */
  function run(testData, cb) {
    element = fixture('test-table');
    element.metrics = testData.getMetrics();
    element.data = testData;
    table = element.$.table;
    table.addEventListener('google-chart-ready', () => {
      setTimeout(cb, INITIALIZATION_TIMEOUT_MS);
    }, {once: true});
  }

  test('ComponentSetup', done => {
    run(TestUtil.createDefaultTestData(), () => {
      assert.equal(
          TestUtil.getElementsWithText(table, 'th', 'feature').length, 1);
      assert.equal(
          TestUtil.getElementsWithText(table, 'td', 'col:1').length, 1);
      assert.equal(
          TestUtil.getElementsWithText(table, 'td', '4.00000').length, 1);
      assert.equal(
          TestUtil.getElementsWithText(table, 'td', '2.0000e-3').length, 1);
      assert.equal(
          TestUtil.getElementsWithText(table, 'td', '0.33200').length, 1);
      done();
    });
  });

  test('TableSelection', done => {
    const highlightFeature1 = () => {
      element.highlight({'feature': 1});
      next(clearSelection);
    };

    const clearSelection = () => {
      const row = table.shadowRoot.querySelector(
          'tr.google-visualization-table-tr-sel');
      assert.isNotNull(row);
      assert.equal(TestUtil.getElementsWithText(row, 'td', 'col:1').length, 1);
      const selected = table.selection;
      assert.equal(selected.length, 1);
      assert.deepEqual(selected, [{'row': 0}]);
      element.highlight(null);
      next(selectionCleared);
    };

    const selectionCleared = () => {
      assert.isNull(table.shadowRoot.querySelector(
          'tr.google-visualization-table-tr-sel'));
      assert.deepEqual(table.selection, []);
      done();
    };
    run(TestUtil.createDefaultTestData(), highlightFeature1);
  });

  test('SortChangesVisibleRows', done => {
    const resizeTable = () => {
      element.pageSize = 3;
      next(checkVisibleRows);
    };
    const checkVisibleRows = () => {
      assert.deepEqual(element.visibleRows, [0, 1, 2]);
      next(sortByWeightedExamples);
    };

    const sortByWeightedExamples = () => {
      const weightedEexampleHeader =
          table.shadowRoot.querySelector('thead tr th:nth-child(2)');
      weightedEexampleHeader.click();
      next(checkVisibleRowsAfterSort);
    };

    const checkVisibleRowsAfterSort = () => {
      assert.deepEqual(element.visibleRows, [2, 0, 1]);
      done();
    };

    run(TestUtil.createDefaultTestData(), resizeTable);
  });

  test('CheckInjectedStyle', done => {
    const resizeTable = () => {
      // Ensure that code injecting CSS to the google-chart component works by
      // checking some of the expected outcome.
      const chart = element.$.table.$['chartdiv'];

      const tr = chart.querySelector('tr.google-visualization-table-tr-head');
      const trStyles = getComputedStyle(tr);
      // Default background color is rgb(228, 233, 244).
      assert.equal(trStyles['backgroundColor'], 'rgba(0, 0, 0, 0)');

      const th = chart.querySelector('.google-visualization-table-th');
      const thStyles = getComputedStyle(th);
      // Default background is a gradient image.
      assert.equal(thStyles['backgroundColor'], 'rgba(0, 0, 0, 0.05)');

      done();
    };

    run(TestUtil.createDefaultTestData(), resizeTable);
  });
});
