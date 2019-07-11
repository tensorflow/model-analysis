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
goog.module('tfma.tests.DataTest');

goog.setTestOnly();

const Data = goog.require('tfma.Data');
const testSuite = goog.require('goog.testing.testSuite');
goog.require('goog.testing.jsunit');

const METRICS = ['a', 'b', 'c'];
const SLICE = 'col:1';

testSuite({
  testBuilder: function() {
    const json = [{metrics: {a: 123, b: 456, c: 789}, slice: SLICE}];
    const data = Data.build(METRICS, json);
    assertEquals(METRICS, data.getMetrics());
    assertArrayEquals([SLICE], data.getFeatures());
    assertEquals(456, data.getMetricValue(SLICE, 'b'));
  },

  testCalculateCalibrationFromLabelAndPrediciton: function() {
    const metrics = ['calibration', 'averageLabel'];
    const stats = [
      {
        slice: SLICE,
        metrics: {
          averageLabel: 0.5,
          averageRefinedPrediction: 0.75,
          calibration: 0
        }
      },
    ];
    const data = Data.build(metrics, stats);
    assertEquals(1.5, data.getMetricValue(SLICE, 'calibration'));
  },

  testCalibrationNotAvailableIfPredicitonAbsent: function() {
    const metrics = ['calibration', 'averageLabel'];
    const stats = [
      {
        slice: SLICE,
        metrics: {
          averageLabel: 0.5,
        }
      },
    ];
    const data = Data.build(metrics, stats);
    assertTrue(isNaN(data.getMetricValue(SLICE, 'calibration')));
  },

  testIsEmptyData: function() {
    const json = [];
    const data = Data.build(METRICS, json);
    assertTrue(data.isEmpty());
  },

  testGetFeatureDataSeriesKeys: function() {
    const metrics = ['calibration', 'averageLabel'];
    const stats = [
      {slice: SLICE, metrics: {averageLabel: 0.5}},
    ];
    const data = Data.build(metrics, stats);
    assertEquals(1, data.getFeatureDataSeriesKeys().length);
    assertEquals('col:1', data.getFeatureDataSeriesKeys()[0]);
  },

  testIsEmptyMetrics: function() {
    const json = [{metrics: {a: 123, b: 456, c: 789}, slice: SLICE}];
    const data = Data.build([], json);
    assertTrue(data.isEmpty());
  },

  testGetDataTable: function() {
    const serializedData = createDefaultJsonData();
    serializedData.push({slice: 'col2:1', metrics: {a: 10, b: 20, c: 30}});
    const data = Data.build(METRICS, serializedData);
    let table = data.getDataTable();
    assertEquals(4, table.length);
    assertArrayEquals(['col:1', 1, 4, 7], table[0]);
    assertArrayEquals(['col:2', 2, 5, 8], table[1]);
    assertArrayEquals(['col:3', 3, 6, 9], table[2]);
    assertArrayEquals(['col2:1', 10, 20, 30], table[3]);
  },

  testFilter: function() {
    const data = createDefaultData();
    const bIndex = data.getMetricIndex('b');
    const filteredData = data.filter((series) => {
      return series.getMetricValuesList()[bIndex] >= 5;
    });
    const table = filteredData.getDataTable();
    assertEquals(2, table.length);
    assertArrayEquals(['col:2', 2, 5, 8], table[0]);
    assertArrayEquals(['col:3', 3, 6, 9], table[1]);
  },

  testApplyOverride: function() {
    let overrideApplied = false;
    const value = {abc: 123};
    const override = {
      transform: (value) => {
        overrideApplied = true;
      }
    };
    const data = createDefaultData();
    data.applyOverride(value, override);
    assertTrue(overrideApplied);
  },

  testGetColumnRangeIgnoresMissingData: function() {
    const minFoo = 1;
    const maxFoo = 2;
    const minBar = 3;
    const maxBar = 4;
    const metrics = ['foo', 'bar'];
    const jsonData = [
      {slice: 'c:1', metrics: {foo: minFoo, bar: maxBar}},
      {slice: 'c:2', metrics: {bar: minBar}},
      {slice: 'c:3', metrics: {foo: maxFoo}},
    ];
    const data = Data.build(metrics, jsonData);

    const fooRange = data.getColumnRange('foo');
    assertEquals(minFoo, fooRange.min);
    assertEquals(maxFoo, fooRange.max);
    const barRange = data.getColumnRange('bar');
    assertEquals(minBar, barRange.min);
    assertEquals(maxBar, barRange.max);
  },

  testExtractAllMetrics: function() {
    const dataArray = [
      [{test: {foo: 1, bar: 2}, other: {a: 1, b: 2}}, {test: {foo: 3, bar: 4}}],
      [
        {test: {bar: 1, baz: 2}},
        {test: {bar: 3, baz: 4}, other: {a: 2, b: 3}},
      ]
    ];
    const metrics = Data.util.getAvailableMetrics(dataArray, 'test');
    assertEquals(3, metrics.length);
    assertEquals('bar', metrics[0]);
    assertEquals('baz', metrics[1]);
    assertEquals('foo', metrics[2]);
  },

  testGetNumericalMetricValue: function() {
    const a = 0;
    const b = 1;
    const c = '234.0';
    const json = [{
      metrics: {a: a, b: {value: b, lowerBound: 0.9, upperBound: 1.1}, c: c},
      slice: SLICE
    }];
    const data = Data.build(METRICS, json);
    assertEquals(a, data.getMetricValue(SLICE, 'a'));
    assertEquals(b, data.getMetricValue(SLICE, 'b'));
    assertEquals(parseFloat(c), data.getMetricValue(SLICE, 'c'));
  },

  testFlattenMetrics: function() {
    const run1 = {
      toFlatten: {
        metric1: {
          redirectionLayer1: {subfield1: 'preserved'},
        },
        metric2: {
          redirectionLayer2:
              {subfield2: 'preserved, too', subfield3: 'also preserved'},
        },
      },
      otherInfo: 'unchanged',
    };
    const run2 = {
      toFlatten: {
        metric3: {
          redirectionLayer3: {subfield4: 'preserved'},
        },
      },
      otherInfo: 'unchanged',
    };

    const flattenedRun1 = {
      toFlatten: {
        metric1: {subfield1: 'preserved'},
        metric2: {subfield2: 'preserved, too', subfield3: 'also preserved'}
      },
      otherInfo: 'unchanged'
    };

    const flattenedRun2 = {
      toFlatten: {
        metric3: {subfield4: 'preserved'},
      },
      otherInfo: 'unchanged'
    };

    const runs = [run1, run2];
    Data.util.flattenMetrics(runs, 'toFlatten');

    assertEquals(JSON.stringify(run1), JSON.stringify(flattenedRun1));
    assertEquals(JSON.stringify(run2), JSON.stringify(flattenedRun2));
  },
});

/**
 * Creates a Data.Data object with default input.
 * @return {!Object}
 */
function createDefaultData() {
  return Data.build(METRICS, createDefaultJsonData());
}

/**
 * @return {!Array<!Object>} The default json data for testing.
 */
function createDefaultJsonData() {
  return [
    {slice: 'col:1', metrics: {a: 1, b: 4, c: 7}},
    {slice: 'col:2', metrics: {a: 2, b: 5, c: 8}},
    {slice: 'col:3', metrics: {a: 3, b: 6, c: 9}},
  ];
}
