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
goog.module('tfma.tests.SeriesDataTest');

goog.setTestOnly();

const Data = goog.require('tfma.Data');
const SeriesData = goog.require('tfma.SeriesData');
const testSuite = goog.require('goog.testing.testSuite');
goog.require('goog.testing.jsunit');

const MODEL_CENTRIC = true;
const METRICS = ['a', 'b', 'c'];

let series;

testSuite({
  setUp: function() {
    series = createDefaultSeries();
  },

  testGetLineChartData: function() {
    assertArrayEquals(
        [
          [{'v': 2, 'f': 'Model 2 at Data 1'}, 2, 'Model: 2', {'f': 6, 'v': 6}],

          [{'v': 1, 'f': 'Model 1 at Data 0'}, 1, 'Model: 1', {'f': 3, 'v': 3}],
          [{'v': 0, 'f': 'Model 0 at Data 0'}, 0, 'Model: 0', {'f': 0, 'v': 0}],
        ],
        series.getLineChartData('b'));
  },

  testGetDataTable: function() {
    assertArrayEquals(
        [
          ['2', '1', 3, 6, 1],
          ['1', '0', 2, 3, 0.5],
          ['0', '0', 1, 0, 0],
        ],
        series.getDataTable());
  },

  testGetMetrics: function() {
    assertArrayEquals(METRICS, series.getMetrics());
  },

  testGetModelIds: function() {
    assertArrayEquals([2, 1, 0], series.getModelIds());
  },

  testGetHeader: function() {
    assertArrayEquals(
        ['Model', 'Data', 'a', 'b', 'c'], series.getHeader(METRICS));
  },

  testGetFormats: function() {
    assertObjectEquals({foo: 'bar'}, series.getFormats({foo: 'bar'}));
  },

  testApplyOverride: function() {
    const expectedValue = {a: '123'};
    const overriddenValue = 1;
    const override = {
      transform: (value) => {
        assertEquals(expectedValue, value);
        return overriddenValue;
      }
    };
    assertEquals(
        overriddenValue, series.applyOverride(expectedValue, override));
  },

  testGetEvalConfig: function() {
    assertObjectEquals({model: 2, data: 1}, series.getEvalConfig(0));
  },

  testSortDataCentric: function() {
    series = new SeriesData(
        [
          {
            config: {dataIdentifier: '1', modelIdentifier: '0'},
            data: makeData(0),
          },
          {
            config: {dataIdentifier: '2', modelIdentifier: '1'},
            data: makeData(1),
          },
          {
            config: {dataIdentifier: '1', modelIdentifier: '2'},
            data: makeData(2),
          }
        ],
        !MODEL_CENTRIC);
    assertArrayEquals(
        [
          ['2', '1', 2, 3, 0.5],
          ['1', '2', 3, 6, 1],
          ['1', '0', 1, 0, 0],
        ],
        series.getDataTable());
  },

  testSetProperXCoordinateForChart: function() {
    series = new SeriesData(
        [
          {
            config: {dataIdentifier: 'a', modelIdentifier: 'b'},
            data: makeData(0),
          },
          {
            config: {dataIdentifier: 'c', modelIdentifier: 'd'},
            data: makeData(1),
          },
          {
            config: {dataIdentifier: 'e', modelIdentifier: 'f'},
            data: makeData(2),
          }
        ],
        MODEL_CENTRIC);
    assertArrayEquals(
        [
          [
            {'v': 3, 'f': 'Model f at Data e'}, 'f', 'Model: f',
            {'f': 6, 'v': 6}
          ],
          [
            {'v': 2, 'f': 'Model d at Data c'}, 'd', 'Model: d',
            {'f': 3, 'v': 3}
          ],
          [
            {'v': 1, 'f': 'Model b at Data a'}, 'b', 'Model: b',
            {'f': 0, 'v': 0}
          ],
        ],
        series.getLineChartData('b'));
  },
});


/**
 * Creates a Data.Data object with default input.
 * @return {!Object}
 */
function createDefaultSeries() {
  return new SeriesData(
      [
        {
          config: {dataIdentifier: '0', modelIdentifier: '0'},
          data: makeData(0),
        },
        {
          config: {dataIdentifier: '0', modelIdentifier: '1'},
          data: makeData(1),
        },
        {
          config: {dataIdentifier: '1', modelIdentifier: '2'},
          data: makeData(2),
        }
      ],
      MODEL_CENTRIC);
}

/**
 * Creates an evaluation run object with the given index.
 * @param {number} index
 * @return {!Object}
 */
function makeData(index) {
  return Data.build(
      METRICS, [{metrics: {a: index + 1, b: index * 3, c: index * 0.5}}]);
}
