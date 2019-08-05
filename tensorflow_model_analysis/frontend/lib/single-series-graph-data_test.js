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
goog.module('tfma.tests.SingleSeriesGraphDataTest');

goog.setTestOnly();

const SingleSeriesGraphData = goog.require('tfma.SingleSeriesGraphData');
const testSuite = goog.require('goog.testing.testSuite');
goog.require('goog.testing.jsunit');

const METRICS = ['a', 'b', 'c'];
const DEFAULT_JSON_DATA = [
  {slice: 'col:1', metrics: {a: 1, b: 4, c: 7}},
  {slice: 'col:2', metrics: {a: 2, b: 5, c: 8}},
  {slice: 'col:3', metrics: {a: 3, b: 6, c: 9}}
];

testSuite({
  testSingleSeriesThreshold: function() {
    const singleGraphData = new SingleSeriesGraphData(
        METRICS, DEFAULT_JSON_DATA);
    const filteredData = singleGraphData.applyThreshold('b', 5);
    const table = filteredData.getDataTable('');
    assertEquals(2, table.length);
    assertArrayEquals(['col:2', 2, 5, 8], table[0]);
    assertArrayEquals(['col:3', 3, 6, 9], table[1]);
  },

  testSingleSeriesThresholdShowAllWhenThresholdUndefined: function() {
    const singleGraphData =
        new SingleSeriesGraphData(METRICS, DEFAULT_JSON_DATA);
    const filteredData = singleGraphData.applyThreshold('k', 0);
    const table = filteredData.getDataTable('');
    assertEquals(3, table.length);
    assertArrayEquals(['col:1', 1, 4, 7], table[0]);
    assertArrayEquals(['col:2', 2, 5, 8], table[1]);
    assertArrayEquals(['col:3', 3, 6, 9], table[2]);
  },

  testSingleSeriesGetColumnSteppingInfo: function() {
    const minFoo = 1;
    const maxFoo = 2;
    const minBar = 3;
    const maxBar = 300;
    const metrics = ['foo', 'bar'];
    const jsonData = [
      {slice: 'c:1', metrics: {foo: minFoo, bar: maxBar}},
      {slice: 'c:2', metrics: {bar: minBar}},
      {slice: 'c:3', metrics: {foo: maxFoo}},
    ];
    const singleGraphData =
      new SingleSeriesGraphData(metrics, jsonData);

    const fooRange = singleGraphData.getColumnSteppingInfo('foo');
    assertEquals(maxFoo + 1, fooRange.max);
    assertEquals(1, fooRange.step);

    const barRange = singleGraphData.getColumnSteppingInfo('bar');
    assertEquals(3, barRange.step);
    assertEquals(303 /* (step * ceil((300 + 1) / step)) */, barRange.max);
  },

  testGetFeatures: function() {
    const singleGraphData = new SingleSeriesGraphData(
        METRICS, DEFAULT_JSON_DATA);
    const features = singleGraphData.getFeatures();
    assertEquals(3, features.length);
    assertArrayEquals(['col:1', 'col:2', 'col:3'], features);
  },

  testGetTableDataFromDataset: function() {
    const singleGraphData = new SingleSeriesGraphData(
        METRICS, DEFAULT_JSON_DATA);
    // Default dataset
    const filteredData = singleGraphData.applyThreshold('a', 0);
    assertEquals(filteredData,
        singleGraphData.getTableDataFromDataset(filteredData));
  }
});
