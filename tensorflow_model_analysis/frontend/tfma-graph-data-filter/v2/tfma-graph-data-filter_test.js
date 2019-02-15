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
goog.module('tfma.GraphDataFilterTest');

const TestUtil = goog.require('tfma.TestUtil');

suite('tests', () => {
  /** @private @const {number} */
  const INITIAL_SETUP_MS = 1000;

  /** @const {number} */
  const TIMEOUT_MS = 200;

  /**
   * Test element.
   * @private {!Element}
   */
  let element;

  /** @enum {string} */
  const ElementId = {
    CHART_TYPE: 'chart-type',
    WEIGHTED_EXAMPLES_THRESHOLD: 'weighted-examples-threshold'
  };

  /** @enum {number} */
  const ChartType = {
    SLICE_OVERVIEW: 0,
    METRICS_HISTOGRAM: 1,
  };

  /** @const {number} */
  const SLICE_COUNT_THRESHOLD_PLUS_ONE = 51;

  /**
   * Runs the next step in a setTimeout.
   * @param {function()} step
   */
  function next(step) {
    setTimeout(step, TIMEOUT_MS);
  }

  /**
   * Sets up the test fixture and runs the test by calling the provided
   * callback. This method creates an instance of {!tfma.SingleSeriesGraphData}
   * to perform the data operations.
   * @param {!Array<string>} metricNames
   * @param {!Array<!Object>} metricResults
   * @param {function()} cb
   */
  function run(metricNames, metricResults, cb) {
    element = fixture('test-filter');
    element.data = new tfma.SingleSeriesGraphData(metricNames, metricResults);
    element.weightedExamplesColumn = 'weightedExamples';
    setTimeout(cb, INITIAL_SETUP_MS);
  }

  /**
   * Sets up the test fixture with default data and runs the test by calling the
   * provided callback.
   * @param {function()} cb
   */
  function runWithDefaultData(cb) {
    run(TestUtil.createDefaultMetricsList(),
        TestUtil.createDefaultMetricsResult(), cb);
  }

  function getElementsWithTextInChart(selector, text) {
    return TestUtil.getElementsWithText(
        element.$.table.shadowRoot.querySelector('google-chart'), selector,
        text);
  }

  /**
   * Selects the given chart type in the chart type UI selector.
   * @param {number} chartType a value from the ChartType enum.
   */
  function selectChartType(chartType) {
    const chartTypeSelector = element.shadowRoot.querySelector(
        '#' + ElementId.CHART_TYPE + ' paper-listbox');
    chartTypeSelector.select(chartType);
  }

  test('ComponentSetup', done => {
    const checkComponents = () => {
      assert.isNotNull(element.$[ElementId.CHART_TYPE]);
      assert.isNotNull(element.$[ElementId.WEIGHTED_EXAMPLES_THRESHOLD]);
      assert.isNotNull(
          element.shadowRoot.querySelector('tfma-metrics-histogram'));
      assert.isNotNull(element.shadowRoot.querySelector('tfma-slice-overview'));
      done();
    };
    runWithDefaultData(checkComponents);
  });

  test('WeightedExampleThresholdUpdatesFilteredData', done => {
    const setThresholdValue = (value) => {
      const sliceWeightFilter = element.shadowRoot.querySelector(
          '#' + ElementId.WEIGHTED_EXAMPLES_THRESHOLD + ' input');
      sliceWeightFilter.value = value;
      sliceWeightFilter.dispatchEvent(new CustomEvent('change'));
    };
    const changeThreshold = () => {
      setThresholdValue('2');
      next(checkandChangeThresholdAgain);
    };
    const checkandChangeThresholdAgain = () => {
      assert.equal(element.weightedExamplesThreshold_, 2);
      const filteredData = element.filteredData_;
      assert.isNotNull(filteredData);
      const table = filteredData.getDataTable('');
      assert.equal(3, table.length);
      assert.equal('col:1', table[0][0]);
      assert.equal('col:2', table[1][0]);
      assert.equal('col:4', table[2][0]);
      setThresholdValue('10');
      next(checkNewThreshold);
    };

    const checkNewThreshold = () => {
      assert.equal(element.weightedExamplesThreshold_, 10);
      const filteredData = element.filteredData_;
      assert.isNotNull(filteredData);
      const table = filteredData.getDataTable('');
      assert.equal(1, table.length);
      assert.equal('col:4', table[0][0]);
      done();
    };
    runWithDefaultData(changeThreshold);
  });

  test('SelectSliceOverviewChartType', done => {
    const changeChartType = () => {
      element.chartType = ChartType.METRICS_HISTOGRAM;

      assert.isFalse(
          element.shadowRoot.querySelector('tfma-slice-overview').displayed);
      selectChartType(ChartType.SLICE_OVERVIEW);
      next(sliceOverviewSelected);
    };

    const sliceOverviewSelected = () => {
      assert.isTrue(
          element.shadowRoot.querySelector('tfma-slice-overview').displayed);
      assert.equal(element.chartType, ChartType.SLICE_OVERVIEW);
      done();
    };

    run(TestUtil.createDefaultMetricsList(),
        TestUtil.createDefaultMetricsResult(), changeChartType);
  });

  test('StartWithMetricsHistogramIfSliceCountGreaterThanCutoff', done => {
    const sliceOverviewHidden = () => {
      assert.equal(element.chartType, ChartType.METRICS_HISTOGRAM);
      assert.isFalse(
          element.shadowRoot.querySelector('tfma-slice-overview').displayed);
      selectChartType(ChartType.SLICE_OVERVIEW);
      next(sliceOverviewShown);
    };

    const sliceOverviewShown = () => {
      assert.isTrue(
          element.shadowRoot.querySelector('tfma-slice-overview').displayed);
      assert.equal(element.chartType, ChartType.SLICE_OVERVIEW);
      done();
    };

    const metrics = [];
    for (let i = 0; i < SLICE_COUNT_THRESHOLD_PLUS_ONE; i++) {
      metrics.push({
        'slice': 'col:' + i,
        'metrics': {'weightedExamples': 10, 'metricA': i, 'metricB': i * i}
      });
    }
    run(TestUtil.createDefaultMetricsList(), metrics, sliceOverviewHidden);
  });

  test('StartWithSliceOverviewIfSliceCountLessThanCutoff', done => {
    const sliceOverviewShown = () => {
      assert.isTrue(
          element.shadowRoot.querySelector('tfma-slice-overview').displayed);
      assert.equal(element.chartType, ChartType.SLICE_OVERVIEW);
      selectChartType(ChartType.METRICS_HISTOGRAM);
      next(sliceOverviewHidden);
    };

    const sliceOverviewHidden = () => {
      assert.equal(element.chartType, ChartType.METRICS_HISTOGRAM);
      done();
    };

    const metrics = [];
    const SLICE_COUNT_THRESHOLD_MINUS_ONE = SLICE_COUNT_THRESHOLD_PLUS_ONE - 2;
    for (let i = 0; i < SLICE_COUNT_THRESHOLD_MINUS_ONE; i++) {
      metrics.push({
        'slice': 'col:' + i,
        'metrics': {'weightedExamples': 10, 'metricA': i, 'metricB': i * i}
      });
    }
    run(TestUtil.createDefaultMetricsList(), metrics, sliceOverviewShown);
  });

  test('TableChangesDataSourceBasedOnChartType', done => {
    const checkDataShownAndSwitchToHistorram = () => {
      assert.equal(
          element.tableData.getFeatures().length,
          SLICE_COUNT_THRESHOLD_PLUS_ONE);
      selectChartType(ChartType.METRICS_HISTOGRAM);
      next(filterHistogram);
    };

    const filterHistogram = () => {
      element.$.histogram.updateFocusRange(0, 0.5);
      next(checkTableFiltered);
    };

    const checkTableFiltered = () => {
      // Ensure that filtering in histogram is applied to metrics table.
      assert.equal(element.tableData.getFeatures().length, 26);
      done();
    };

    const metrics = [];
    for (let i = 0; i < SLICE_COUNT_THRESHOLD_PLUS_ONE; i++) {
      metrics.push({
        'slice': 'col:' + i,
        'metrics': {'weightedExamples': 10, 'metricA': i, 'metricB': i * i}
      });
    }
    run(TestUtil.createDefaultMetricsList(), metrics,
        checkDataShownAndSwitchToHistorram);
  });
});
