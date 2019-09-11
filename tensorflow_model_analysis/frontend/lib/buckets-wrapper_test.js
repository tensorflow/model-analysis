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
goog.module('tfma.tests.BucketsWrapperTest');

goog.setTestOnly();

const BucketsWrapper = goog.require('tfma.BucketsWrapper');
const Constants = goog.require('tfma.Constants');
const testSuite = goog.require('goog.testing.testSuite');
goog.require('goog.testing.jsunit');

const X_INDEX = 1;
const Y_INDEX = 2;
const COLOR_INDEX = 3;
const SIZE_INDEX = 4;
const DEFAULT_SLOPE = 0.75;
const DEFAULT_INTERCEPT = 0.0625;
const NO_REBUCKET = 0;
const EPSILON = 0.001;

testSuite({
  testParseInput: function() {
    const length = 4;
    const buckets =
        generateStraightLine(length, DEFAULT_SLOPE, DEFAULT_INTERCEPT);
    const holder = [];
    BucketsWrapper.getCalibrationPlotData(
        buckets, Constants.PlotFit.LEAST_SQUARE, Constants.PlotScale.LINEAR,
        Constants.PlotHighlight.ERROR, Constants.PlotHighlight.WEIGHTS,
        NO_REBUCKET, holder);

    assertEquals(length, holder.length);
    assertEquals('', holder[0][0]);
    assertEquals('', holder[1][0]);
    assertEquals('', holder[2][0]);
    assertEquals('', holder[3][0]);
  },

  testLogScaleFloorApplied: function() {
    const length = 4;
    // Create data that is a horizontal line with intercept half the minimum
    // allowed value. All y scale should use the minimum log scale value.
    const buckets = generateStraightLine(
        length, 0, BucketsWrapper.MIN_ALLOWED_VALUE_ON_LOG_SCALE / 2);
    const holder = [];
    BucketsWrapper.getCalibrationPlotData(
        buckets, Constants.PlotFit.LEAST_SQUARE, Constants.PlotScale.LOG,
        Constants.PlotHighlight.ERROR, Constants.PlotHighlight.WEIGHTS,
        NO_REBUCKET, holder);

    let entry = holder[0];
    assertEquals(BucketsWrapper.MIN_ALLOWED_VALUE_ON_LOG_SCALE, entry[Y_INDEX]);

    entry = holder[1];
    assertEquals(BucketsWrapper.MIN_ALLOWED_VALUE_ON_LOG_SCALE, entry[Y_INDEX]);

    entry = holder[2];
    assertEquals(BucketsWrapper.MIN_ALLOWED_VALUE_ON_LOG_SCALE, entry[Y_INDEX]);

    entry = holder[3];
    assertEquals(BucketsWrapper.MIN_ALLOWED_VALUE_ON_LOG_SCALE, entry[Y_INDEX]);
  },

  testUseErrorForColor: function() {
    const length = 4;
    const buckets = generateStraightLine(length, 1, DEFAULT_INTERCEPT);
    const holder = [];
    BucketsWrapper.getCalibrationPlotData(
        buckets, Constants.PlotFit.PERFECT, Constants.PlotScale.LINEAR,
        Constants.PlotHighlight.ERROR, Constants.PlotHighlight.ERROR,
        NO_REBUCKET, holder);

    let entry = holder[0];
    assertRoughlyEquals(DEFAULT_INTERCEPT, entry[COLOR_INDEX], EPSILON);

    entry = holder[1];
    assertRoughlyEquals(DEFAULT_INTERCEPT, entry[COLOR_INDEX], EPSILON);

    entry = holder[2];
    assertRoughlyEquals(DEFAULT_INTERCEPT, entry[COLOR_INDEX], EPSILON);

    entry = holder[3];
    assertRoughlyEquals(DEFAULT_INTERCEPT, entry[COLOR_INDEX], EPSILON);
  },

  testUseWeightsForColor: function() {
    const length = 4;
    const buckets =
        generateStraightLine(length, DEFAULT_SLOPE, DEFAULT_INTERCEPT, 10);
    const holder = [];
    BucketsWrapper.getCalibrationPlotData(
        buckets, Constants.PlotFit.LEAST_SQUARE, Constants.PlotScale.LOG,
        Constants.PlotHighlight.WEIGHTS, Constants.PlotHighlight.WEIGHTS,
        NO_REBUCKET, holder);

    var entry = holder[0];
    assertRoughlyEquals(1, entry[COLOR_INDEX], EPSILON);

    entry = holder[1];
    assertRoughlyEquals(2, entry[COLOR_INDEX], EPSILON);

    entry = holder[2];
    assertRoughlyEquals(3, entry[COLOR_INDEX], EPSILON);

    entry = holder[3];
    assertRoughlyEquals(4, entry[COLOR_INDEX], EPSILON);
  },

  testUseErrorForSize: function() {
    const length = 4;
    const buckets = generateStraightLine(length, 1, DEFAULT_INTERCEPT);
    const holder = [];
    BucketsWrapper.getCalibrationPlotData(
        buckets, Constants.PlotFit.PERFECT, Constants.PlotScale.LINEAR,
        Constants.PlotHighlight.ERROR, Constants.PlotHighlight.ERROR,
        NO_REBUCKET, holder);

    let entry = holder[0];
    assertRoughlyEquals(DEFAULT_INTERCEPT, entry[SIZE_INDEX], EPSILON);

    entry = holder[1];
    assertRoughlyEquals(DEFAULT_INTERCEPT, entry[SIZE_INDEX], EPSILON);

    entry = holder[2];
    assertRoughlyEquals(DEFAULT_INTERCEPT, entry[SIZE_INDEX], EPSILON);

    entry = holder[3];
    assertRoughlyEquals(DEFAULT_INTERCEPT, entry[SIZE_INDEX], EPSILON);
  },

  testUseWeightsForSize: function() {
    const length = 4;
    const buckets =
        generateStraightLine(length, DEFAULT_SLOPE, DEFAULT_INTERCEPT, 10);
    const holder = [];
    BucketsWrapper.getCalibrationPlotData(
        buckets, Constants.PlotFit.LEAST_SQUARE, Constants.PlotScale.LOG,
        Constants.PlotHighlight.WEIGHTS, Constants.PlotHighlight.WEIGHTS,
        NO_REBUCKET, holder);

    var entry = holder[0];
    assertRoughlyEquals(1, entry[SIZE_INDEX], EPSILON);

    entry = holder[1];
    assertRoughlyEquals(2, entry[SIZE_INDEX], EPSILON);

    entry = holder[2];
    assertRoughlyEquals(3, entry[SIZE_INDEX], EPSILON);

    entry = holder[3];
    assertRoughlyEquals(4, entry[SIZE_INDEX], EPSILON);
  },

  testPerfectFit: function() {
    const length = 4;
    const buckets = generateStraightLine(length, 1, DEFAULT_INTERCEPT);
    const holder = [];
    BucketsWrapper.getCalibrationPlotData(
        buckets, Constants.PlotFit.PERFECT, Constants.PlotScale.LINEAR,
        Constants.PlotHighlight.ERROR, Constants.PlotHighlight.ERROR,
        NO_REBUCKET, holder);

    // Since the data is always DEFAULT_INTERCEPT from the perfect fit, the
    // error should be DEFAULT_INTERCEPT.
    const entry = holder[0];
    assertRoughlyEquals(DEFAULT_INTERCEPT, entry[COLOR_INDEX], EPSILON);
    assertRoughlyEquals(DEFAULT_INTERCEPT, entry[SIZE_INDEX], EPSILON);
  },

  testLeastSquareFit: function() {
    const length = 4;
    const buckets = generateStraightLine(length, 1, DEFAULT_INTERCEPT);
    const holder = [];
    BucketsWrapper.getCalibrationPlotData(
        buckets, Constants.PlotFit.LEAST_SQUARE, Constants.PlotScale.LINEAR,
        Constants.PlotHighlight.ERROR, Constants.PlotHighlight.ERROR,
        NO_REBUCKET, holder);

    // Since data is on a straight line, the erro should be 0.
    const entry = holder[0];
    assertRoughlyEquals(0, entry[COLOR_INDEX], EPSILON);
    assertRoughlyEquals(0, entry[SIZE_INDEX], EPSILON);
  },

  testRebucketInput: function() {
    const length = 32;
    const buckets = generateStraightLine(length, 1, DEFAULT_INTERCEPT);
    const holder = [];
    const expectedColor = Math.log(40) / Math.log(10);
    BucketsWrapper.getCalibrationPlotData(
        buckets, Constants.PlotFit.PERFECT, Constants.PlotScale.LINEAR,
        Constants.PlotHighlight.WEIGHTS, Constants.PlotHighlight.ERROR, 8,
        holder);

    assertEquals(8, holder.length);

    let entry = holder[0];
    let expectedX = (1 + 2 + 3 + 4) / 32 / 4;
    let expectedY = expectedX + DEFAULT_INTERCEPT;
    assertEquals(expectedX, entry[X_INDEX]);
    assertEquals(expectedY, entry[Y_INDEX]);
    assertEquals(expectedColor, entry[COLOR_INDEX]);
    assertEquals(DEFAULT_INTERCEPT, entry[SIZE_INDEX]);

    entry = holder[1];
    expectedX = (5 + 6 + 7 + 8) / 32 / 4;
    expectedY = expectedX + DEFAULT_INTERCEPT;
    assertEquals(expectedX, entry[X_INDEX]);
    assertEquals(expectedY, entry[Y_INDEX]);
    assertEquals(expectedColor, entry[COLOR_INDEX]);
    assertEquals(DEFAULT_INTERCEPT, entry[SIZE_INDEX]);

    entry = holder[2];
    expectedX = (9 + 10 + 11 + 12) / 32 / 4;
    expectedY = expectedX + DEFAULT_INTERCEPT;
    assertEquals(expectedX, entry[X_INDEX]);
    assertEquals(expectedY, entry[Y_INDEX]);
    assertEquals(expectedColor, entry[COLOR_INDEX]);
    assertEquals(DEFAULT_INTERCEPT, entry[SIZE_INDEX]);

    entry = holder[3];
    expectedX = (13 + 14 + 15 + 16) / 32 / 4;
    expectedY = expectedX + DEFAULT_INTERCEPT;
    assertEquals(expectedX, entry[X_INDEX]);
    assertEquals(expectedY, entry[Y_INDEX]);
    assertEquals(expectedColor, entry[COLOR_INDEX]);
    assertEquals(DEFAULT_INTERCEPT, entry[SIZE_INDEX]);

    entry = holder[4];
    expectedX = (17 + 18 + 19 + 20) / 32 / 4;
    expectedY = expectedX + DEFAULT_INTERCEPT;
    assertEquals(expectedX, entry[X_INDEX]);
    assertEquals(expectedY, entry[Y_INDEX]);
    assertEquals(expectedColor, entry[COLOR_INDEX]);
    assertEquals(DEFAULT_INTERCEPT, entry[SIZE_INDEX]);

    entry = holder[5];
    expectedX = (21 + 22 + 23 + 24) / 32 / 4;
    expectedY = expectedX + DEFAULT_INTERCEPT;
    assertEquals(expectedX, entry[X_INDEX]);
    assertEquals(expectedY, entry[Y_INDEX]);
    assertEquals(expectedColor, entry[COLOR_INDEX]);
    assertEquals(DEFAULT_INTERCEPT, entry[SIZE_INDEX]);

    entry = holder[6];
    expectedX = (25 + 26 + 27 + 28) / 32 / 4;
    expectedY = expectedX + DEFAULT_INTERCEPT;
    assertEquals(expectedX, entry[X_INDEX]);
    assertEquals(expectedY, entry[Y_INDEX]);
    assertEquals(expectedColor, entry[COLOR_INDEX]);
    assertEquals(DEFAULT_INTERCEPT, entry[SIZE_INDEX]);

    entry = holder[7];
    expectedX = (29 + 30 + 31 + 32) / 32 / 4;
    expectedY = expectedX + DEFAULT_INTERCEPT;
    assertEquals(expectedX, entry[X_INDEX]);
    assertEquals(expectedY, entry[Y_INDEX]);
    assertEquals(expectedColor, entry[COLOR_INDEX]);
    assertEquals(DEFAULT_INTERCEPT, entry[SIZE_INDEX]);
  },
});


/**
 * Generates an array of test data that falls on a straight line.
 * @param {number} count
 * @param {number} slope
 * @param {number} intercept
 * @param {number=} opt_weightMultiplier
 * @return {lantern.data.BucketsWrapper.Buckets_}
 */
function generateStraightLine(count, slope, intercept, opt_weightMultiplier) {
  const data = [];
  let weight = 10;
  data.push({
      'upperThresholdExclusive': 0,
      'lowerThresholdInclusive': -Infinity,
      'numWeightedExamples': 0,
      'totalWeightedLabel': 0,
      'totalWeightedRefinedPrediction': 0
    });
  let oldX = 0;
  for (var i = 1; i <= count; i++) {
    const x = i / count;
    const y = x * slope + intercept;
    data.push({
      'upperThresholdExclusive': x,
      'lowerThresholdInclusive': oldX,
      'numWeightedExamples': weight,
      'totalWeightedLabel': y * weight,
      'totalWeightedRefinedPrediction': x * weight
    });
    if (opt_weightMultiplier) {
      weight *= opt_weightMultiplier;
    }
    oldX = x;
  }
  data.push({
      'upperThresholdExclusive': Infinity,
      'lowerThresholdInclusive': 1,
      'numWeightedExamples': 0,
      'totalWeightedLabel': 0,
      'totalWeightedRefinedPrediction': 0
    });
  return data;
}
