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

  testRenderValueWithSmallFloatUsesScientificNotation: function() {
    const float = -0.0012345;
    const cell = CellRenderer.renderValue(float);
    assertEquals(float, cell.v);
    assertEquals('-1.2345e-3', cell.f);
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

  testRenderValueWithValueAtCutoffs: function() {
    const value = 0.2;
    const valueAtCutoffsData = {
      'values': [
        {'cutoff': 0, 'value': value},
        {'cutoff': 2, 'value': 0.5},
        {'cutoff': 12, 'value': 0.7},
      ]
    };
    const cell = CellRenderer.renderValue(valueAtCutoffsData);
    assertEquals(value, cell.v);
    assertEquals(
        '<tfma-value-at-cutoffs data="' +
            googString.htmlEscape(JSON.stringify(valueAtCutoffsData)) +
            '"></tfma-value-at-cutoffs>',
        cell.f);
  },

  testSortValueAtCutoffsDataByCutoff: function() {
    const unsorted = {
      'values': [
        {'cutoff': 2, 'value': 0.5},
        {'cutoff': 0, 'value': 0.2},
        {'cutoff': 12, 'value': 0.7},
      ]
    };
    const sorted = {
      'values': [
        {'cutoff': 0, 'value': 0.2},
        {'cutoff': 2, 'value': 0.5},
        {'cutoff': 12, 'value': 0.7},
      ]
    };
    const cell = CellRenderer.renderValue(unsorted);
    assertEquals(sorted['values'][0].value, cell.v);
    assertEquals(
        '<tfma-value-at-cutoffs data="' +
            googString.htmlEscape(JSON.stringify(sorted)) +
            '"></tfma-value-at-cutoffs>',
        cell.f);
  },

  testRenderValueWithConfusionMatrixAtThresholds: function() {
    const precision = 0.81;
    const confusionMatrixAtThresholds = {
      'matrices': [{
        'threshold': 0.8,
        'precision': precision,
        'recall': 0.82,
        'truePositives': 0.83,
        'trueNegatives': 0.84,
        'falsePositives': 0.85,
        'falseNegatives': 0.86
      }],
    };
    const cell = CellRenderer.renderValue(confusionMatrixAtThresholds);
    assertEquals(precision, cell.v);
    assertEquals(
        '<tfma-confusion-matrix-at-thresholds data="' +
            googString.htmlEscape(JSON.stringify(confusionMatrixAtThresholds)) +
            '"></tfma-confusion-matrix-at-thresholds>',
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
