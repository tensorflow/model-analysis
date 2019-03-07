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
goog.module('tfma.SlicingMetricsBrowserTest');

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

  /**
   * Runs the next step in a setTimeout.
   * @param {function()} step
   */
  function next(step) {
    setTimeout(step, TIMEOUT_MS);
  }

  /**
   * Sets up the test fixture and runs the test by calling the provided
   * callback.
   * @param {function()} cb
   */
  function run(cb) {
    element = fixture('test-browser');
    element.metrics = TestUtil.createDefaultMetricsList();
    element.data = TestUtil.createDefaultMetricsResult();
    element.weightedExamplesColumn = 'weightedExamples';
    setTimeout(cb, INITIAL_SETUP_MS);
  }

  function getElementsWithTextInChart(selector, text) {
    return TestUtil.getElementsWithText(
        element.$.table.shadowRoot.querySelector('google-chart'), selector,
        text);
  }

  test('ComponentSetup', done => {
    const checkComponents = () => {
      assert.isNotNull(element.$[ElementId.CHART_TYPE]);
      assert.isNotNull(element.$[ElementId.WEIGHTED_EXAMPLES_THRESHOLD]);
      assert.isNotNull(
          element.shadowRoot.querySelector('tfma-graph-data-filter'));
      assert.isNotNull(element.shadowRoot.querySelector('tfma-metrics-table'));
      done();
    };
    run(checkComponents);
  });

  test('DefaultFormatOverride', done => {
    const defaultOverrideSetOnMetricsTable = () => {
      const metricsTable = element.$.table;
      let formats = metricsTable['metricFormats'];
      assert.equal(2, Object.keys(formats).length);
      assert.equal(
          tfma.MetricValueFormat.INT64, formats['totalExampleCount']['type']);
      assert.equal(
          tfma.MetricValueFormat.INT, formats['weightedExamples']['type']);
      done();
    };
    run(defaultOverrideSetOnMetricsTable);
  });

  test('MergeFormatOverride', done => {
    const setFormatOverride = () => {
      // No override for averageLable initially.
      assert.isUndefined(element.$.table['metricFormats']['averageLabel']);
      element.formats = {
        'averageLabel': {'type': tfma.MetricValueFormat.FLOAT}
      };
      next(overrideSetOnMetricsTable);
    };
    const overrideSetOnMetricsTable = () => {
      // Override applied to averageLabel.
      assert.equal(
          element.$.table['metricFormats']['averageLabel']['type'],
          tfma.MetricValueFormat.FLOAT);
      done();
    };
    run(setFormatOverride);
  });
});
