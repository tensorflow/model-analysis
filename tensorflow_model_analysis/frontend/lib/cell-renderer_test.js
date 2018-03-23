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
goog.module('tfma.tests.CellRendererTest');

goog.setTestOnly();

const CellRenderer = goog.require('tfma.CellRenderer');
const Constants = goog.require('tfma.Constants');
const googString = goog.require('goog.string');
const testSuite = goog.require('goog.testing.testSuite');
goog.require('goog.testing.jsunit');

const dummyTableProvider = {
  'applyOverride': function(value, override) {
    return value;
  }
};

testSuite({

  testRenderValueWithFloat: function() {
    const float = 0.12345;
    const cell = CellRenderer.renderValue(float);
    assertEquals(float, cell.v);
    assertEquals(float.toString(), cell.f);
  },

  testRenderValueWithBoundedValue: function() {
    const value = 1;
    const boundedValue = makeBoundedValueObject(value, 2, 3);
    const cell = CellRenderer.renderValue(boundedValue);
    assertEquals(value, cell.v);
    assertEquals(
        '<tfma-bounded-value value=1.00000 lower-bound=2.00000 ' +
            'upper-bound=3.00000>' +
            '</tfma-bounded-value>',
        cell.f);
  },

  testRenderValueWithPrecisionAtK: function() {
    const value = 0.95;
    const precisionAtK = makePrecisionAtKData(3, 100, value);
    const cell = CellRenderer.renderValue(precisionAtK);
    assertEquals(value, cell.v);
    assertEquals(
        '<tfma-metric-at-k data="' +
            googString.htmlEscape(JSON.stringify(precisionAtK)) +
            '"></tfma-metric-at-k>',
        cell.f);
  },

  testSortPrecisionAtKDataByK: function() {
    const unsorted = [
      {k: 2, value: 0.5, totalPositives: 10},
      {k: 0, value: 0.2, totalPositives: 5},
      {k: 12, value: 0.7, totalPositives: 20},
    ];
    const sorted = [
      {k: 0, value: 0.2, totalPositives: 5},
      {k: 2, value: 0.5, totalPositives: 10},
      {k: 12, value: 0.7, totalPositives: 20},
    ];
    const cell = CellRenderer.renderValue(unsorted);
    assertEquals(sorted[0].value, cell.v);
    assertEquals(
        '<tfma-metric-at-k data="' +
            googString.htmlEscape(JSON.stringify(sorted)) +
            '"></tfma-metric-at-k>',
        cell.f);
  },

  testRenderValueWithBinaryConfusionMatricesFromRegression: function() {
    const f1Score = 0.8;
    const data = makeBinaryConfusionMatricesFromRegressionData(
        1, 0.75, f1Score, 0.7, 0.5, 0.6);
    const cell = CellRenderer.renderValue(data);
    assertEquals(f1Score, cell.v);
    assertEquals(
        '<tfma-binary-confusion-matrices-from-regression data="' +
            googString.htmlEscape(JSON.stringify(data)) +
            '"></tfma-binary-confusion-matrices-from-regression>',
        cell.f);
  },

  testRenderValueWithPlotTrigger: function() {
    const value = makePlotTriggerData(null, 3);

    const cell = CellRenderer.renderValue(value);
    assertEquals(0, cell.v);
    assertEquals(
        '<tfma-plot-trigger data="' +
            googString.htmlEscape(JSON.stringify(value)) +
            '"></tfma-plot-trigger>',
        cell.f);
  },

  testRenderValueWithRecallAtK: function() {
    const value = 0.95;
    const recallAtK = makeRecallAtKData(3, 100, value);
    const cell = CellRenderer.renderValue(recallAtK);
    assertEquals(value, cell.v);
    assertEquals(
        '<tfma-metric-at-k data="' +
            googString.htmlEscape(JSON.stringify(recallAtK)) +
            '"></tfma-metric-at-k>',
        cell.f);
  },

  testSortRecallAtKDataByK: function() {
    const unsorted = [
      {k: 2, value: 0.5, totalActualPositives: 10},
      {k: 0, value: 0., totalActualPositives: 5},
      {k: 21, value: 0.7, totalActualPositives: 20},
    ];
    const sorted = [
      {k: 0, value: 0., totalActualPositives: 5},
      {k: 2, value: 0.5, totalActualPositives: 10},
      {k: 21, value: 0.7, totalActualPositives: 20},
    ];
    const cell = CellRenderer.renderValue(unsorted);
    assertEquals(sorted[0].value, cell.v);
    assertEquals(
        '<tfma-metric-at-k data="' +
            googString.htmlEscape(JSON.stringify(sorted)) +
            '"></tfma-metric-at-k>',
        cell.f);
  },

  testRenderValueWithMultiValuesMetricAtK: function() {
    const value = 0.95;
    const multiValuesMetricAtK = makeMultiValuesMetricAtKData(3, value);
    const cell = CellRenderer.renderValue(multiValuesMetricAtK);
    assertEquals(value, cell.v);
    assertEquals(
        '<tfma-multi-values-metric-at-k data="' +
            googString.htmlEscape(JSON.stringify(multiValuesMetricAtK)) +
            '"></tfma-multi-values-metric-at-k>',
        cell.f);
  },

  testRenderValueWithMultiValuesThresholdBasedBinaryClassificationMetrics:
      function() {
        const macroF1Score = 0.999;
        const data = [{
          'binaryClassificationThreshold': {'predictionThreshold': 0.5},
          'macroPrecision': 0.91,
          'macroRecall': 0.81,
          'macroF1Score': macroF1Score,
          'macroAccuracy': 0.61,
        }];
        const cell = CellRenderer.renderValue(data);
        assertEquals(macroF1Score, cell.v);
        assertEquals(
            '<tfma-multi-values-threshold-based-binary-classification-metrics' +
            ' data="' + googString.htmlEscape(JSON.stringify(data)) + '"></' +
            'tfma-multi-values-threshold-based-binary-classification-metrics>',
            cell.f);
      },

  testSortMultiValuesMetricAtKDataByK: function() {
    const unsorted = [
      {k: 2, microValue: 0},
      {k: 0, macroValue: 0.3, microValue: 0.5, weightedValue: 10},
      {k: 21, macroValue: 0.7, microValue: 0.8},
    ];
    const sorted = [
      {k: 0, macroValue: 0.3, microValue: 0.5, weightedValue: 10},
      {k: 2, microValue: 0},
      {k: 21, macroValue: 0.7, microValue: 0.8},
    ];
    const cell = CellRenderer.renderValue(unsorted);
    assertEquals(sorted[0].macroValue, cell.v);
    assertEquals(
        '<tfma-multi-values-metric-at-k data="' +
            googString.htmlEscape(JSON.stringify(sorted)) +
            '"></tfma-multi-values-metric-at-k>',
        cell.f);
  },

  testRenderValueWithMultiClassConfusionMatrix: function() {
    const matrix = {
      'entries': [
        {'actualClass': 'A', 'predictedClass': 'A', 'weight': 100},
        {'actualClass': 'A', 'predictedClass': 'B', 'weight': 10},
        {'actualClass': 'B', 'predictedClass': 'A', 'weight': 50},
        {'actualClass': 'B', 'predictedClass': 'B', 'weight': 30}
      ],
    };
    const cell = CellRenderer.renderValue(matrix);
    assertEquals(0, cell.v);
    assertEquals(
        '<tfma-multi-class-confusion-matrix data="' +
            googString.htmlEscape(JSON.stringify(matrix)) +
            '"></tfma-multi-class-confusion-matrix>',
        cell.f);
  },

  testRenderValueWithUnsupported: function() {
    const unsupported = {foo: 1, bar: 'baz'};
    const cell = CellRenderer.renderValue(unsupported);
    assertEquals(0, cell.v);
    assertEquals('Unsupported: {"foo":1,"bar":"baz"}', cell.f);
  },

  testRenderValueWithRowIdOverride: function() {
    const id = 123;
    const rowId = 'my_id:' + id;
    const override = {type: Constants.MetricValueFormat.ROW_ID};
    const cell = CellRenderer.renderValueWithFormatOverride(
        rowId, dummyTableProvider, override);
    assertEquals(id, cell.v);
    assertEquals(rowId, cell.f);
  },

  testRenderValueWithIntegerOverride: function() {
    const int = 123;
    const override = {type: Constants.MetricValueFormat.INT};
    const cell = CellRenderer.renderValueWithFormatOverride(
        int, dummyTableProvider, override);
    assertEquals(int, cell.v);
    assertEquals(int.toString(), cell.f);
  },

  testRenderValueWithFloatOverride: function() {
    const float = 0.1234567;
    const override = {type: Constants.MetricValueFormat.FLOAT};
    const cell = CellRenderer.renderValueWithFormatOverride(
        float, dummyTableProvider, override);
    assertEquals(float, cell.v);
    assertEquals('0.12346', cell.f);
  },

  testRenderValueWithHtmlOverride: function() {
    const html = '<a href=#>link</a>';
    const override = {type: Constants.MetricValueFormat.HTML};
    const cell = CellRenderer.renderValueWithFormatOverride(
        html, dummyTableProvider, override);
    assertEquals(html, cell.v);
    assertEquals(html, cell.f);
  },

  testRenderValueWithStringOverride: function() {
    const string = 'abc';
    const override = {type: Constants.MetricValueFormat.STRING};
    const cell = CellRenderer.renderValueWithFormatOverride(
        string, dummyTableProvider, override);
    assertEquals(string, cell.v);
    assertEquals(string, cell.f);
  },

  testRenderValueWithOverrideAppliesOverrideThroughTableProvider: function() {
    const value = 'abc';
    const overriddenValue = 0.54321;
    const override = {type: Constants.MetricValueFormat.FLOAT};
    const tableProvider = {
      applyOverride: function(value, override) {
        return overriddenValue;
      }
    };
    const cell = CellRenderer.renderValueWithFormatOverride(
        value, tableProvider, override);
    assertEquals(overriddenValue, cell.v);
    assertEquals(overriddenValue.toString(), cell.f);
  },

  testRenderValueWithOverrideWithNullValue: function() {
    const override = {type: Constants.MetricValueFormat.FLOAT};
    const tableProvider = {
      applyOverride: function(value, override) {
        return 12345;
      }
    };
    const cell = CellRenderer.renderValueWithFormatOverride(
        null, tableProvider, override);
    assertEquals('n/a', cell.f);
  },

  testRenderValueWithUndefinedOverride: function() {
    const value = 123;
    const cell = CellRenderer.renderValueWithFormatOverride(
        value, dummyTableProvider, undefined);
    assertEquals(value, cell.v);
    assertEquals('123.00000', cell.f);
  },

  testRenderValueWithOverrideLeadingToJsError: function() {
    const override = {type: Constants.MetricValueFormat.INT64};
    const value = {foo: 'bar'};
    const cell = CellRenderer.renderValueWithFormatOverride(
        value, dummyTableProvider, override);
    assertEquals(0, cell.v);
    assertEquals(
        'Unsupported: {"override":' + JSON.stringify(override) +
            ',"value":' + JSON.stringify(value) + '}',
        cell.f);
  },

  testRenderValueWithInt64Override: function() {
    const int64 = '123';
    const override = {type: Constants.MetricValueFormat.INT64};
    const cell = CellRenderer.renderValueWithFormatOverride(
        int64, dummyTableProvider, override);
    assertEquals('00000000000000000123', cell.v);
    assertEquals('<tfma-int64 data="' + int64 + '"></tfma-int64>', cell.f);
  },

  testRenderUndefined: function() {
    const override = {type: Constants.MetricValueFormat.FLOAT};
    const cell = CellRenderer.renderValueWithFormatOverride(
        undefined, dummyTableProvider, override);
    assertEquals(0, cell.v);
    assertEquals('n/a', cell.f);
  },

  testParseRowId: function() {
    assertEquals(123, CellRenderer.parseRowId('123'));
    assertEquals(123, CellRenderer.parseRowId('foo:123'));
    assertEquals(123, CellRenderer.parseRowId('foo_123'));

    assertNaN(CellRenderer.parseRowId('foo'));
    assertNaN(CellRenderer.parseRowId('foo:bar'));
    assertNaN(CellRenderer.parseRowId('foo:123|456'));
  },

  testNewRenderer: function() {
    const newType = 'foo';
    const value = {a: 'abc'};
    const renderedCell = {'f': 'abc', 'v': 123};
    const renderer = () => {
      return renderedCell;
    };
    const typeChecker = (valueToCheck) => {
      return value === valueToCheck;
    };
    CellRenderer.registerRenderer(newType, renderer, typeChecker);
    assertEquals(renderedCell, CellRenderer.renderValue(value));
  },

  testNewFormatOverrideRenderer() {
    const newType = 'bar';
    const override = {type: newType};
    const dummyProvider = {
      applyOverride: (value, override) => value,
    };
    const value = {b: '123'};
    const renderedCell = {a: 123, b: 456};
    const newRenderer = () => {
      return renderedCell;
    };
    CellRenderer.registerOverrideRenderer(newType, newRenderer);
    assertEquals(
        renderedCell,
        CellRenderer.renderValueWithFormatOverride(
            value, dummyProvider, override));
  },
});


/**
 * Creates a bound value object with the given inputs.
 * @param {number} value
 * @param {number} lowerBound
 * @param {number} upperBound
 * @return {!Object}
 */
function makeBoundedValueObject(value, lowerBound, upperBound) {
  return {value: value, lowerBound: lowerBound, upperBound: upperBound};
}


/**
 * Creates an array of binary confusion matrices from regression data. At each
 * point, the values will be scaled by 0.75.
 * @param {number} count
 * @param {number} threshold
 * @param {number} f1Score
 * @param {number} precision
 * @param {number} recall
 * @param {number} accuracy
 * @return {!Object}
 */
function makeBinaryConfusionMatricesFromRegressionData(
    count, threshold, f1Score, precision, recall, accuracy) {
  const matrices = [];
  const scale = 0.75;
  for (let i = 0; i < count; i++) {
    const classificationThreshold = {'predictionThreshold': threshold};
    const matrix = {
      'f1Score': f1Score,
      'precision': precision,
      'recall': recall,
      'accuracy': accuracy
    };
    matrices.push({
      'binaryClassificationThreshold': classificationThreshold,
      'matrix': matrix
    });
    threshold *= scale;
    f1Score *= scale;
    precision *= scale;
    recall *= scale;
    accuracy *= scale;
  }
  return matrices;
}


/**
 * Creates an array of precision at k data. At each position, the precision will
 * decrease by 25%.
 * @param {number} count
 * @param {number} positiveCount
 * @param {number} startingPrecision
 * @return {!Object}
 */
function makePrecisionAtKData(count, positiveCount, startingPrecision) {
  const data = [];
  let precision = startingPrecision;
  for (let i = 1; i <= count; i++) {
    data.push({k: i, value: precision, totalPositives: positiveCount * i});
    precision *= 0.75;
  }
  return data;
}


/**
 * Creates a plot-trigger object with given input.
 * @param {?string} slice
 * @param {?number} span
 * @return {!Object}
 */
function makePlotTriggerData(slice, span) {
  const data = {'types': [Constants.PlotTypes.CALIBRATION_PLOT]};
  if (slice !== null) {
    data['slice'] = slice;
  } else {
    data['span'] = span;
  }
  return data;
}

/**
 * Creates an array of precision at k data. At each position, the precision will
 * decrease by 25%.
 * @param {number} count
 * @param {number} positiveCount
 * @param {number} startingPrecision
 * @return {!Object}
 */
function makeRecallAtKData(count, positiveCount, startingPrecision) {
  const data = [];
  let precision = startingPrecision;
  for (let i = 1; i <= count; i++) {
    data.push(
        {k: i, value: precision, totalActualPositives: positiveCount * i});
    precision *= 0.75;
  }
  return data;
}

/**
 * Creates an array of multi values metric at k data. At each position, the
 * macro value will decrease by 25% and micro value is 90% of macro value.
 * @param {number} count
 * @param {number} startingValue
 * @return {!Object}
 */
function makeMultiValuesMetricAtKData(count, startingValue) {
  const data = [];
  let macroValue = startingValue;
  for (let i = 1; i <= count; i++) {
    data.push({k: i, macroValue: macroValue, microValue: macroValue * 0.9});
    macroValue *= 0.75;
  }
  return data;
}
