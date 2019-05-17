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
  /** @private @const {number} */
  const INITIAL_SETUP_MS = 1000;

  /** @private @const {number} */
  const TIMEOUT_MS = 200;

  const METRIC1 = 'metric1';
  const METRIC2 = 'metric2';
  const METRIC3 = 'metric3';

  let element;

  function next(step) {
    setTimeout(step, TIMEOUT_MS);
  }

  test('CanShowAndSortByNumericalMetrics', done => {
    const checkComponent = () => {
      const metricsToShow = TestUtil.selectAll(
          element, 'paper-dropdown-menu[label="Show"] paper-item');

      assert.equal(metricsToShow.length, 3);

      assert.equal(metricsToShow[0].textContent.trim(), METRIC1);
      assert.isFalse(metricsToShow[0].disabled);

      assert.equal(metricsToShow[1].textContent.trim(), METRIC2);
      assert.isFalse(metricsToShow[1].disabled);

      assert.equal(metricsToShow[2].textContent.trim(), METRIC3);
      assert.isFalse(metricsToShow[2].disabled);

      let metricsToSortBy = TestUtil.selectAll(
          element, 'paper-dropdown-menu[label="Sort by"] paper-item');

      assert.equal(metricsToSortBy.length, 4);

      assert.equal(metricsToSortBy[0].textContent.trim(), 'Slice');

      assert.equal(metricsToSortBy[1].textContent.trim(), METRIC1);
      assert.isFalse(metricsToSortBy[1].disabled);

      assert.equal(metricsToSortBy[2].textContent.trim(), METRIC2);
      assert.isFalse(metricsToSortBy[2].disabled);

      assert.equal(metricsToSortBy[3].textContent.trim(), METRIC3);
      assert.isFalse(metricsToSortBy[3].disabled);
      done();
    };
    runWithDefaultData(checkComponent);
  });

  test('ShowDefaultMetric', done => {
    const data =
        tfma.Data.build([METRIC1, METRIC2, METRIC3], [{
                          'slice': 'col:1',
                          'metrics': {'metric1': 4, 'metric2': 1, 'metric3': 2},
                        }]);

    const defaultShown = () => {
      const showMenu =
          element.shadowRoot.querySelector('paper-dropdown-menu[label="Show"]');
      const item3 = showMenu.querySelector('paper-item:nth-child(3)');
      assert.isNotNull(item3);
      assert.equal(showMenu.selectedItem, item3);
      done();
    };

    run(METRIC3, data, defaultShown);
  });

  test('SelectDefaultMetric', done => {
    const data = tfma.Data.build([METRIC1, METRIC2, METRIC3], [
      {'slice': 'col:1', 'metrics': {'metric1': 4, 'metric2': 1, 'metric3': 2}}
    ]);

    const defaultSelected = () => {
      const showMenu =
          element.shadowRoot.querySelector('paper-dropdown-menu[label="Show"]');
      const item3 = showMenu.querySelector('paper-item:nth-child(3)');
      assert.isNotNull(item3);
      assert.equal(showMenu.selectedItem, item3);
      done();
    };

    run(METRIC3, data, defaultSelected);
  });

  test('RenderExpectedData', done => {
    const checkRenderResults = () => {
      assert.equal(element.dataView_.getNumberOfColumns(), 2);
      assert.equal(element.dataView_.getNumberOfRows(), 3);

      assert.equal(element.dataView_.getValue(0, 0), 'col:1');
      assert.equal(element.dataView_.getValue(1, 0), 'col:2');
      assert.equal(element.dataView_.getValue(2, 0), 'col:3');

      assert.equal(element.dataView_.getFormattedValue(0, 0), 'col:1');
      assert.equal(element.dataView_.getFormattedValue(1, 0), 'col:2');
      assert.equal(element.dataView_.getFormattedValue(2, 0), 'col:3');

      assert.equal(element.dataView_.getValue(0, 1), 31);
      assert.equal(element.dataView_.getValue(1, 1), 22);
      assert.equal(element.dataView_.getValue(2, 1), 13);
      done();
    };

    runWithDefaultData(checkRenderResults);
  });

  test('SortDataBySelectedMetric', done => {
    let sortByListbox;

    const selectMetric2 = () => {
      sortByListbox = element.shadowRoot.querySelector(
          'paper-dropdown-menu[label="Sort by"] paper-listbox');
      sortByListbox.select(METRIC2);
      next(checkMetric2Selected);
    };

    const checkMetric2Selected = () => {
      assert.equal(element.dataView_.getValue(0, 0), 'col:3');
      assert.equal(element.dataView_.getValue(1, 0), 'col:2');
      assert.equal(element.dataView_.getValue(2, 0), 'col:1');

      assert.equal(element.dataView_.getValue(0, 1), 13);
      assert.equal(element.dataView_.getValue(1, 1), 22);
      assert.equal(element.dataView_.getValue(2, 1), 31);

      next(selecctMetric3);
    };

    const selecctMetric3 = () => {
      sortByListbox.select(METRIC3);
      next(checkMetric3Selected);
    };

    const checkMetric3Selected = () => {
      assert.equal(element.dataView_.getValue(0, 0), 'col:3');
      assert.equal(element.dataView_.getValue(1, 0), 'col:1');
      assert.equal(element.dataView_.getValue(2, 0), 'col:2');

      assert.equal(element.dataView_.getValue(0, 1), 13);
      assert.equal(element.dataView_.getValue(1, 1), 31);
      assert.equal(element.dataView_.getValue(2, 1), 22);
      next(selecctMetric1);
    };

    const selecctMetric1 = () => {
      sortByListbox.select(METRIC1);
      next(checkMetric1Selected);
    };

    const checkMetric1Selected = () => {
      assert.equal(element.dataView_.getValue(0, 0), 'col:3');
      assert.equal(element.dataView_.getValue(1, 0), 'col:2');
      assert.equal(element.dataView_.getValue(2, 0), 'col:1');

      assert.equal(element.dataView_.getValue(0, 1), 13);
      assert.equal(element.dataView_.getValue(1, 1), 22);
      assert.equal(element.dataView_.getValue(2, 1), 31);
      done();
    };

    runWithDefaultData(selectMetric2);
  });

  test('SortByLexicalOrderOfSliceIfNonParseable', done => {
    const data = tfma.Data.build([METRIC1, METRIC2, METRIC3], [
      {
        'slice': 'col:c',
        'metrics': {'metric1': 2, 'metric2': 5, 'metric3': 9},
      },
      {
        'slice': 'col:b',
        'metrics': {'metric1': 3, 'metric2': 4, 'metric3': 7},
      },
      {
        'slice': 'col:a',
        'metrics': {'metric1': 1, 'metric2': 6, 'metric3': 8},
      },
    ]);

    const checkRenderResults = () => {
      assert.equal(element.dataView_.getNumberOfColumns(), 2);
      assert.equal(element.dataView_.getNumberOfRows(), 3);

      assert.equal(element.dataView_.getValue(0, 0), 'col:a');
      assert.equal(element.dataView_.getValue(1, 0), 'col:b');
      assert.equal(element.dataView_.getValue(2, 0), 'col:c');

      assert.equal(element.dataView_.getFormattedValue(0, 0), 'col:a');
      assert.equal(element.dataView_.getFormattedValue(1, 0), 'col:b');
      assert.equal(element.dataView_.getFormattedValue(2, 0), 'col:c');

      assert.equal(element.dataView_.getValue(0, 1), 1);
      assert.equal(element.dataView_.getValue(1, 1), 3);
      assert.equal(element.dataView_.getValue(2, 1), 2);
      done();
    };


    run(METRIC1, data, checkRenderResults);
  });

  test('ShowSelectedMetric', done => {
    let showMetricListbox;
    const selectMetric2 = () => {
      showMetricListbox = element.shadowRoot.querySelector(
          'paper-dropdown-menu[label="Show"] paper-listbox');
      showMetricListbox.select(METRIC2);
      next(checkMetric2Selected);
    };

    const checkMetric2Selected = () => {
      assert.equal(element.dataView_.getValue(0, 0), 'col:1');
      assert.equal(element.dataView_.getValue(1, 0), 'col:2');
      assert.equal(element.dataView_.getValue(2, 0), 'col:3');

      assert.equal(element.dataView_.getValue(0, 1), 6);
      assert.equal(element.dataView_.getValue(1, 1), 5);
      assert.equal(element.dataView_.getValue(2, 1), 4);

      next(selecctMetric3);
    };

    const selecctMetric3 = () => {
      showMetricListbox.select(METRIC3);
      next(checkMetric3Selected);
    };
    const checkMetric3Selected = () => {
      assert.equal(element.dataView_.getValue(0, 1), 8);
      assert.equal(element.dataView_.getValue(1, 1), 9);
      assert.equal(element.dataView_.getValue(2, 1), 7);
      done();
    };

    runWithDefaultData(selectMetric2);
  });

  test('ShowAndSortMetric', done => {
    const selectAndSort = () => {
      const showMetricListbox = element.shadowRoot.querySelector(
          'paper-dropdown-menu[label="Show"] paper-listbox');
      const sortByMetricListbox = element.shadowRoot.querySelector(
          'paper-dropdown-menu[label="Sort by"] paper-listbox');
      showMetricListbox.select(METRIC2);
      sortByMetricListbox.select(METRIC3);
      next(checkResults);
    };

    const checkResults = () => {
      assert.equal(element.dataView_.getValue(0, 0), 'col:3');
      assert.equal(element.dataView_.getValue(1, 0), 'col:1');
      assert.equal(element.dataView_.getValue(2, 0), 'col:2');

      assert.equal(element.dataView_.getValue(0, 1), 4);
      assert.equal(element.dataView_.getValue(1, 1), 6);
      assert.equal(element.dataView_.getValue(2, 1), 5);

      done();
    };

    runWithDefaultData(selectAndSort);
  });

  test('DisplaySortByMetricIfDifferntFromShowMetric', done => {
    const selectAndSort = () => {
      const showMetricListbox = element.shadowRoot.querySelector(
          'paper-dropdown-menu[label="Show"] paper-listbox');
      const sortByMetricListbox = element.shadowRoot.querySelector(
          'paper-dropdown-menu[label="Sort by"] paper-listbox');
      showMetricListbox.select(METRIC1);
      sortByMetricListbox.select(METRIC3);
      next(checkResults);
    };
    const checkResults = () => {
      assert.equal(
          element.dataView_.getFormattedValue(0, 0), 'col:3, metric3:7.00000');
      assert.equal(
          element.dataView_.getFormattedValue(1, 0), 'col:1, metric3:8.00000');
      assert.equal(
          element.dataView_.getFormattedValue(2, 0), 'col:2, metric3:9.00000');
      done();
    };

    runWithDefaultData(selectAndSort);
  });

  test('DisableNonNumericalMetrics', done => {
    const defaultMetric = 'weights';
    const metrics = [defaultMetric, 'stringMetric', 'objectMetric'];
    const data = tfma.Data.build(
        metrics, [{
          'slice': 'col:1',
          'metrics': {weights: 4, stringMetric: 'abc', objectMetric: {foo: 1}}
        }]);

    const checkState = () => {
      const metricsToShow = TestUtil.selectAll(
          element, 'paper-dropdown-menu[label="Show"] paper-item');
      let itemsCount = metricsToShow.length;
      assert.equal(itemsCount, 3);
      for (let i = 0; i < itemsCount; i++) {
        let metricToShow = metricsToShow[i];
        assert.equal(metricToShow.textContent.trim(), metrics[i]);
        assert.isFalse(metricToShow.disabled);
      }

      const metricsToSortBy = TestUtil.selectAll(
          element, 'paper-dropdown-menu[label="Sort by"] paper-item');
      itemsCount = metricsToSortBy.length;
      assert.equal(itemsCount, 4);
      assert.equal(metricsToSortBy[0].textContent.trim(), 'Slice');
      for (let i = 1; i < itemsCount; i++) {
        let metricToSortBy = metricsToSortBy[i];
        assert.equal(metrics[i - 1], metricToSortBy.textContent.trim());
        assert.isFalse(metricToSortBy.disabled);
      }
      done();
    };

    run(defaultMetric, data, checkState);
  });

  test('UseNaNForMissingValue', done => {
    const data = tfma.Data.build([METRIC1, METRIC2], [
      {'slice': 'col:1', 'metrics': {'metric1': 4}},
      {'slice': 'col:2', 'metrics': {'metric1': 5, 'metric2': 1}}
    ]);

    const checkNanForMissing = () => {
      const metricsToShow = TestUtil.selectAll(
          element, 'paper-dropdown-menu[label="Show"] paper-item');
      assert.isFalse(metricsToShow[0].disabled);
      assert.isFalse(metricsToShow[1].disabled);

      let metricsToSortBy = TestUtil.selectAll(
          element, 'paper-dropdown-menu[label="Sort by"] paper-item');
      assert.isFalse(metricsToSortBy[1].disabled);
      assert.isFalse(metricsToSortBy[2].disabled);

      assert.isNaN(element.dataView_.getValue(0, 1));
      assert.equal(1, element.dataView_.getValue(1, 1));
      done();
    };

    run(METRIC2, data, checkNanForMissing);
  });

  test('FireSelectEventWhenSelectionChanged', done => {
    const sliceToSelect = 'col:2';
    let selectedSlice;
    const setUpListener = () => {
      element.addEventListener(tfma.Event.SELECT, (event) => {
        selectedSlice = event.detail;
      });

      setTimeout(select, 0);
    };
    const select = () => {
      element.selectedSlice_ = sliceToSelect;
      element.$['loader'].dispatchEvent(new Event('google-chart-select'));
      setTimeout(verify, 0);
    };
    const verify = () => {
      assert.equal(selectedSlice, sliceToSelect);
      done();
    };

    runWithDefaultData(setUpListener);
  });

  /**
   * @param {function()} cb
   */
  function runWithDefaultData(cb) {
    const data = tfma.Data.build([METRIC1, METRIC2, METRIC3], [
      {
        'slice': 'col:2',
        'metrics': {'metric1': 22, 'metric2': 5, 'metric3': 9},
      },
      {
        'slice': 'col:3',
        'metrics': {'metric1': 13, 'metric2': 4, 'metric3': 7},
      },
      {
        'slice': 'col:1',
        'metrics': {'metric1': 31, 'metric2': 6, 'metric3': 8},
      },
    ]);

    run(METRIC1, data, cb);
  }


  /**
   * Sets up the slice-overview element.
   * @param {string} metricToShow
   * @param {!tfma.Data} data
   * @param {function()} cb
   * @private
   */
  function run(metricToShow, data, cb) {
    element = fixture('test-overview');
    element.metricToShow = metricToShow;
    element.slices = data;
    element.displayed = true;
    setTimeout(cb, INITIAL_SETUP_MS);
  }
});
