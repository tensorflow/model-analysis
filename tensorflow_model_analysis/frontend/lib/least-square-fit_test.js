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
goog.module('tfma.tests.LeastSquareFitTest');

goog.setTestOnly();

const LeastSquareFit = goog.require('tfma.LeastSquareFit');
const testSuite = goog.require('goog.testing.testSuite');
goog.require('goog.testing.jsunit');

testSuite({
  testGetLine: function() {
    const points = [{x: 1, y: 6}, {x: 2, y: 5}, {x: 3, y: 7}, {x: 4, y: 10}];
    const line = LeastSquareFit.getLine(points);
    assertEquals(1.4, line.slope);
    assertEquals(3.5, line.intercept);
  },

  testGetHorizontalLine: function() {
    const points = [{x: 0, y: 0.5}, {x: 1, y: 0.5}];
    const line = LeastSquareFit.getLine(points);
    assertEquals(0, line.slope);
    assertEquals(0.5, line.intercept);
  },

  testVerticalLine: function() {
    const points = [{x: 0.5, y: 0}, {x: 0.5, y: 1}];
    const line = LeastSquareFit.getLine(points);
    assertTrue(isNaN(line.slope));
    assertTrue(isNaN(line.intercept));
  },

  testDiagonalLine: function() {
    const points = [{x: 0, y: 0}, {x: 1, y: 1}];
    const line = LeastSquareFit.getLine(points);
    assertEquals(1, line.slope);
    assertEquals(0, line.intercept);
  },

  testFitKnownLine: function() {
    const points = [];
    const slope = 2.25;
    const intercept = 1.75;
    for (let i = 0; i < 1024; i++) {
      const x = i / 1024;
      points.push({x: x, y: slope * x + intercept});
    }
    const line = LeastSquareFit.getLine(points);
    assertEquals(slope, line.slope);
    assertEquals(intercept, line.intercept);
  },

  testFitZeroPoint: function() {
    const points = [];
    const line = LeastSquareFit.getLine(points);
    assertTrue(isNaN(line.slope));
    assertTrue(isNaN(line.intercept));
  },

  testFitOnePoint: function() {
    const points = [{x: 1, y: 2}];
    const line = LeastSquareFit.getLine(points);
    assertTrue(isNaN(line.slope));
    assertTrue(isNaN(line.intercept));
  },

  testFitTooManyPoint: function() {
    const points = [];
    points[10e7 + 1] = {x: 1, y: 2};
    const line = LeastSquareFit.getLine(points);
    assertTrue(isNaN(line.slope));
    assertTrue(isNaN(line.intercept));
  }
});
