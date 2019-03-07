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
  /**
   * Test component element.
   * @type {!Element}
   */
  let element;

  /**
   * The underlying google-chart component.
   * @type {!Element}
   */
  let chart;

  test('mouseOverTriggersSelect', done => {
    let selectFired = false;
    const onSelect = () => {
      selectFired = true;
    };

    const addListener = () => {
      element.addEventListener('select', onSelect);
      setTimeout(mouseOver, 0);
    };

    const mouseOver = () => {
      chart.dispatchEvent(new CustomEvent(
          'google-chart-onmouseover', {detail: {data: {row: 0, column: 0}}}));
      setTimeout(checkSelectFired, 0);
    };

    const checkSelectFired = () => {
      assert.isTrue(selectFired);
      done();
    };

    run(addListener);
  });

  test('mouseOverSetsChartSelection', done => {
    const selectedPoint = {row: 0, column: 0};

    const mouseOver = () => {
      chart.dispatchEvent(new CustomEvent(
          'google-chart-onmouseover', {detail: {data: selectedPoint}}));
      setTimeout(checkSelection, 0);
    };

    const checkSelection = () => {
      assert.deepEqual(chart.selection, [selectedPoint]);
      done();
    };

    run(mouseOver);
  });

  test('mouseOutTriggersClearSelection', done => {
    let clearSelectionFired = false;
    const onClearSelection = () => {
      clearSelectionFired = true;
    };

    const addListener = () => {
      element.addEventListener('clear-selection', onClearSelection);
      setTimeout(mouseOut, 0);
    };

    const mouseOut = () => {
      chart.dispatchEvent(new CustomEvent('google-chart-onmouseout'));
      setTimeout(checkClearSelectionFired, 0);
    };

    const checkClearSelectionFired = () => {
      assert.isTrue(clearSelectionFired);
      done();
    };

    run(addListener);
  });

  test('mouseOutClearChartSelection', done => {
    const selectedPoint = {row: 0, column: 0};

    const mouseOver = () => {
      chart.dispatchEvent(new CustomEvent(
          'google-chart-onmouseover', {detail: {data: selectedPoint}}));
      setTimeout(checkSelection, 0);
    };

    const checkSelection = () => {
      assert.deepEqual(chart.selection, [selectedPoint]);
      setTimeout(mouseOut, 0);
    };

    const mouseOut = () => {
      chart.dispatchEvent(new CustomEvent('google-chart-onmouseout'));
      setTimeout(checkSelectionCleared, 0);
    };

    const checkSelectionCleared = () => {
      assert.deepEqual(chart.selection, []);
      done();
    };

    run(mouseOver);
  });

  test('selectAndClearSelection', done => {
    const selectedDataPoint = {row: 0, column: 0};
    const select = () => {
      element.select(selectedDataPoint);
      setTimeout(checkSelection, 0);
    };

    const checkSelection = () => {
      assert.deepEqual(chart.selection, [selectedDataPoint]);
      setTimeout(clearSelection, 0);
    };

    const clearSelection = () => {
      element.clearSelection();
      setTimeout(checkSelectionAgain, 0);
    };

    const checkSelectionAgain = () => {
      assert.deepEqual(chart.selection, []);
      done();
    };
    run(select);
  });

  const run = (cb) => {
    element = fixture('chart-fixture');
    element.data = [
      [{'label': 'x', 'type': 'number'}, {'label': 'y', 'type': 'number'}],
      [0, 1], [1, 2], [2, 4], [3, 8], [4, 16]
    ];
    chart = element.$['chart'];
    setTimeout(cb, 0);
  };
});
