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
  /**
   * Test component element.
   * @type {!Element}
   */
  let element;

  /** @enum {string} */
  const ElementId = {
    TYPE_SELECT: 'type-select',
    METRIC_SELECT: 'metric-select',
    LOGARITHM_SCALE: 'logarithm-scale',
    OPTIONS: 'options',
    OPTIONS_TOGGLE: 'options-toggle',
    NUM_BUCKETS: 'num-buckets',
    EMPTY: 'empty',
    FOCUS: 'focus',
    OVERVIEW: 'overview'
  };

  /** @enum {string} */
  const ElementClass = {WEIGHTED: 'weighted', UNWEIGHTED: 'unweighted'};

  /** @type {string} */
  const WEIGHTED_EXAMPLES = 'weightedExamples';

  /** @type {string} */
  const METRIC_A = 'metricA';

  /** @type {string} */
  const METRIC_B = 'metricB';

  /** @const {number} */
  const INITIALIZATION_TIMEOUT_MS = 2000;

  /** @type {number} */
  const WAIT_RENDER_MS = 200;

  /**
   * Runs the test by setting up the histogram element in the fixture and call
   * the callback asynchronously.
   * @param {!tfma.Data} testData
   * @param {function()} cb
   * @private
   */
  function run(testData, cb) {
    element = fixture('test-histogram');
    element.data = testData;
    element.weightedExamplesColumn = WEIGHTED_EXAMPLES;
    setTimeout(cb, INITIALIZATION_TIMEOUT_MS);
  }

  function next(step) {
    setTimeout(step, WAIT_RENDER_MS);
  }

  test('ComponentSetup', done => {
    const checkInitialSetUp = () => {
      assert.equal(element.$[ElementId.TYPE_SELECT].style.display, '');
      assert.equal(element.$[ElementId.METRIC_SELECT].style.display, '');
      assert.equal(element.$[ElementId.NUM_BUCKETS].style.display, '');
      assert.equal(element.$[ElementId.OPTIONS_TOGGLE].style.display, '');
      assert.equal(element.$[ElementId.OPTIONS].style.display, 'none');
      next(checkSvgRendered);
    };

    const checkSvgRendered = () => {
      assert.equal(
          element.shadowRoot.querySelector('#' + ElementId.FOCUS).style.display,
          '');
      assert.equal(element.$[ElementId.EMPTY].style.display, 'none');
      done();
    };

    run(TestUtil.createDefaultTestData(), checkInitialSetUp);
  });

  test('Highlighting', done => {
    const highlight = () => {
      assert.isNull(element.shadowRoot.querySelector('rect.highlighted'));
      assert.isNull(element.shadowRoot.querySelector('g.highlighted'));

      element.selectedFeatures = [1];
      next(unhighlight);
    };

    const unhighlight = () => {
      assert.isNotNull(element.shadowRoot.querySelector('rect.highlighted'));
      assert.isNotNull(element.shadowRoot.querySelector('g.highlighted'));

      element.selectedFeatures = [];
      next(select);
    };

    const select = () => {
      assert.isNull(element.shadowRoot.querySelector('rect.highlighted'));
      assert.isNull(element.shadowRoot.querySelector('g.highlighted'));

      // Switch to histogram type 'both' and choose 'col:1'.
      const paperItem = element.shadowRoot.querySelector(
          '#' + ElementId.TYPE_SELECT + ' paper-item[value=both]');
      paperItem.click();
      element.selectedFeatures = [1];

      next(setFocusRange);
    };

    const setFocusRange = () => {
      // Two bars and texts should be highlighted in 'both' mode.
      assert.equal(TestUtil.selectAll(element, 'rect.highlighted').length, 2);
      assert.equal(TestUtil.selectAll(element, 'g.highlighted').length, 2);

      element.updateFocusRange(.1, .5);
      next(changeFocusRange);
    };

    const changeFocusRange = () => {
      // Slices are still highlighted after histogram zoom in.
      assert.equal(TestUtil.selectAll(element, 'rect.highlighted').length, 2);
      assert.equal(TestUtil.selectAll(element, 'g.highlighted').length, 2);

      element.updateFocusRange(.5, 1.0);
      next(highlighted);
    };

    const highlighted = () => {
      // Highlighted slices out of histogram focus range.
      assert.isNull(element.shadowRoot.querySelector('rect.highlighted'));
      assert.isNull(element.shadowRoot.querySelector('g.highlighted'));
      done();
    };

    run(TestUtil.createDefaultTestData(), highlight);
  });

  test('NumberOfBuckets', done => {
    const checkInitialState = () => {
      assert.equal(
          TestUtil.getElementsWithText(element, 'text', '0.18000').length, 1);
      assert.equal(
          TestUtil.getElementsWithText(element, 'text', '0.27600').length, 1);
      assert.equal(
          TestUtil.getElementsWithText(element, 'text', '1.14000').length, 1);
      next(updateNumberOfBuckets);
    };
    const updateNumberOfBuckets = () => {
      const input = element.$[ElementId.NUM_BUCKETS].getElementsByTagName(
          'paper-input')[0];
      input.updateValueAndPreserveCaret('5');
      assert.equal(
          TestUtil.getElementsWithText(element, 'text', '0.37200').length, 1);
      assert.equal(
          TestUtil.getElementsWithText(element, 'text', '0.27600').length, 0);

      const slider = element.shadowRoot.querySelector(
          '#' + ElementId.NUM_BUCKETS + ' paper-slider');
      slider.value = 10;
      next(checkNewState);
    };
    const checkNewState = () => {
      assert.equal(
          TestUtil.getElementsWithText(element, 'text', '0.18000').length, 1);
      assert.equal(
          TestUtil.getElementsWithText(element, 'text', '0.27600').length, 1);
      assert.equal(
          TestUtil.getElementsWithText(element, 'text', '1.14000').length, 1);
      done();
    };
    run(TestUtil.createDefaultTestData(), checkInitialState);
  });

  test('MetricSelect', () => {
    const selectMetricB = () => {
      var metricSelect = element.$[ElementId.METRIC_SELECT];
      var paperItem = TestUtil.getElementsWithText(
          metricSelect, 'paper-item', 'metricB')[0];
      paperItem.click();
      next(checkState);
    };
    const checkState = () => {
      assert.equal(
          TestUtil.getElementsWithText(element, 'text', '0.00200').length, 1);
      assert.equal(
          TestUtil.getElementsWithText(element, 'text', '0.75300').length, 1);
    };
    run(TestUtil.createDefaultTestData(), selectMetricB);
  });

  test('TypeSelect', done => {
    /**
     * @param {boolean} hasUnweighted If there should be an unweighted overview.
     * @param {boolean} hasWeighted If there should be a weighted overview.
     */
    const checkOverview = function(hasUnweighted, hasWeighted) {
      const pathInUnweightedOverview = element.shadowRoot.querySelector(
          'g.' + ElementClass.UNWEIGHTED + ' path');
      const pathInWeightedOverview = element.shadowRoot.querySelector(
          'g.' + ElementClass.WEIGHTED + ' path');

      if (hasUnweighted) {
        assert.isNotNull(pathInUnweightedOverview);
      } else {
        assert.isNull(pathInUnweightedOverview);
      }
      if (hasWeighted) {
        assert.isNotNull(pathInWeightedOverview);
      } else {
        assert.isNull(pathInWeightedOverview);
      }
    };

    const selectWeighted = () => {
      checkOverview(true, false);
      // Check there are bars for unweighted histogram.
      assert.equal(
          TestUtil.getElementsWithText(element, 'text', '1').length, 4);

      const histogramType = element.$[ElementId.TYPE_SELECT];
      const paperItem = TestUtil.getElementsWithText(
          histogramType, 'paper-item', 'Example Counts')[0];
      paperItem.click();
      next(selectBoth);
    };

    const selectBoth = () => {
      // Check there are bars for weighted histogram.
      assert.equal(
          TestUtil.getElementsWithText(element, 'text', '12').length, 1);
      assert.equal(
          TestUtil.getElementsWithText(element, 'text', '4').length, 1);
      assert.equal(
          TestUtil.getElementsWithText(element, 'text', '9').length, 1);
      assert.equal(
          TestUtil.getElementsWithText(element, 'text:not([aria-hidden])', '1')
              .length,
          1);
      // Check there is only the weighted histogram overview.
      checkOverview(false, true);

      var histogramType = element.$[ElementId.TYPE_SELECT];
      var paperItem =
          TestUtil.getElementsWithText(histogramType, 'paper-item', 'Both')[0];
      paperItem.click();
      next(selectSliceCount);
    };

    const selectSliceCount = () => {
      // Check there are bars for both histogram types.
      assert.equal(
          TestUtil.getElementsWithText(element, 'text', '12').length, 1);
      assert.equal(
          TestUtil.getElementsWithText(element, 'text', '4').length, 1);
      assert.equal(
          TestUtil.getElementsWithText(element, 'text', '9').length, 1);
      assert.equal(
          TestUtil.getElementsWithText(element, 'text:not([aria-hidden])', '1')
              .length,
          5);
      // Check there are overviews for both histogram types.
      checkOverview(true, true);

      var histogramType = element.$[ElementId.TYPE_SELECT];
      var paperItem = TestUtil.getElementsWithText(
          histogramType, 'paper-item', 'Slice Counts')[0];
      paperItem.click();
      next(checkState);
    };

    const checkState = () => {
      // Check unweighted histogram can be rendered properly after histogram
      // type is switched back from both to unweighted.
      checkOverview(true, false);
      assert.equal(
          TestUtil.getElementsWithText(element, 'text', '1').length, 4);
      done();
    };

    run(TestUtil.createDefaultTestData(), selectWeighted);
  });

  test('LogarithmScale', done => {
    const switchToLogScale = () => {
      assert.equal(
          TestUtil.getElementsWithText(element, 'text', '1').length, 4);
      element.$[ElementId.LOGARITHM_SCALE].click();
      next(checkLogScale);
    };
    const checkLogScale = () => {
      assert.equal(
          TestUtil.getElementsWithText(element, 'text', '0.693').length, 4);
      done();
    };
    run(TestUtil.createDefaultTestData(), switchToLogScale);
  });

  test('FocusRange', done => {
    const setFocusRange = () => {
      element.updateFocusRange(.1, .5);
      next(resetFocusRange);
    };
    const resetFocusRange = () => {
      assert.equal(
          TestUtil.getElementsWithText(element, 'text', '1').length, 1);
      const histogramOverview =
          element.shadowRoot.querySelector('svg#' + ElementId.OVERVIEW);
      histogramOverview.dispatchEvent(new CustomEvent('dblclick'));
      next(checkStateRestored);
    };
    const checkStateRestored = () => {
      assert.equal(
          TestUtil.getElementsWithText(element, 'text', '1').length, 4);
      done();
    };
    run(TestUtil.createDefaultTestData(), setFocusRange);
  });

  test('SkipUndefinedMetric', done => {
    const data = tfma.Data.build([WEIGHTED_EXAMPLES, METRIC_A, METRIC_B], [
      {
        'slice': 'col:1',
        'metrics': {'weightedExamples': 4, 'metricA': 0.333, 'metricB': 0.25}
      },
      {'slice': 'col:2', 'metrics': {'weightedExamples': 10, 'metricB': 0.75}},
      {
        'slice': 'col:3',
        'metrics': {'weightedExamples': 6, 'metricA': 0.5, 'metricB': 0.5}
      }
    ]);

    const metricAFilteredOutforCol2 = () => {
      // col:2 is filtered out side its metricA is undefined.
      assert.equal(
          TestUtil.getElementsWithText(element, 'text', '1').length, 2);
      var metricSelect = element.$[ElementId.METRIC_SELECT];
      var paperItem = TestUtil.getElementsWithText(
          metricSelect, 'paper-item', 'metricB')[0];
      paperItem.click();
      next(metricBAvailableForAll);
    };
    const metricBAvailableForAll = () => {
      // All three slices should show up.
      assert.equal(
          TestUtil.getElementsWithText(element, 'text', '1').length, 3);
      done();
    };
    run(data, metricAFilteredOutforCol2);
  });
});
