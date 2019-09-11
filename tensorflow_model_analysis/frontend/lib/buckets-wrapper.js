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
goog.module('tfma.BucketsWrapper');

const Constants = goog.require('tfma.Constants');
const LeastSquareFit = goog.require('tfma.LeastSquareFit');

/**
 * @typedef {{
 *   lowerThresholdInclusive: (number|undefined),
 *   upperThresholdExclusive: (number|undefined),
 *   numWeightedExamples: (number|undefined),
 *   totalWeightedLabel: (number|undefined),
 *   totalWeightedRefinedPrediction: (number|undefined),
 * }}
 */
let Bucket;

/**
 * @typedef {!Array<!Bucket>};
 */
let Buckets;

/**
 * @typedef {{
 *   x: number,
 *   y:number,
 *   w:number
 * }};
 */
let BucketEntry;

/**
 * Field names in the JSON representation of Bucket.
 * @enum {string}
 */
const FieldNames = {
  LABEL: 'totalWeightedLabel',
  LOWER_THRESHOLD: 'lowerThresholdInclusive',
  PREDICTION: 'totalWeightedRefinedPrediction',
  UPPER_THRESHOLD: 'upperThresholdExclusive',
  WEIGHTS: 'numWeightedExamples',
};

/**
 * The best fit line for a perfectly calibrated model; specifically, y = x.
 * @private {!LeastSquareFit.Line}
 */
const PERFECT_CALIBRATION_LINE = {
  slope: 1,
  intercept: 0
};

/**
 * Minimum value on log scale.
 * @private {number}
 */
const MIN_ON_LOG_SCALE_VALUE = -5;

/**
 * The value of log(10). It is used for determining log10(v).
 * @const {number}
 */
const LOG_10 = Math.log(10);

/**
 * Minimum allowed value on log scale. We will use this value for any value
 * smaller.
 * @const {number}
 */
const MIN_ALLOWED_VALUE_ON_LOG_SCALE = Math.pow(10, MIN_ON_LOG_SCALE_VALUE);

/**
 * Generates calibration plot data from the buckets.
 * @param {!Array<!Object>} buckets
 * @param {!Constants.PlotFit} fit
 * @param {!Constants.PlotScale} scale
 * @param {!Constants.PlotHighlight} color
 * @param {!Constants.PlotHighlight} size
 * @param {number} numberOfBuckets
 * @param {!Array<!Array<string|number>>} outputArray
 */
function getCalibrationPlotData(
    buckets, fit, scale, color, size, numberOfBuckets, outputArray) {
  const entries = regroupBucketToEntries(
      buckets, numberOfBuckets, scale == Constants.PlotScale.LOG);

  const line = fit == Constants.PlotFit.PERFECT ?
      PERFECT_CALIBRATION_LINE :
      LeastSquareFit.getLine(entries);

  entries.forEach(entry => {
    outputArray.push([
      '',                // Skip label to avoid clutter.
      entry.x, entry.y,  // Coordinate
      determineCalibrationPlotValue(
          color == Constants.PlotHighlight.WEIGHTS, entry, line),  // color
      determineCalibrationPlotValue(
          size == Constants.PlotHighlight.WEIGHTS, entry, line)  // size
    ]);
  });
}

/**
 * Iterates over the given buckets and regroups them into new buckets with the
 * provided bucket size when applicable. Determine the x, y, and w that will be
 * used for plotting calibration based on the average label, average prediction
 * and total weight in each new bucket.
 * @param {!Buckets} buckets Buckets are assumed to be sorted in increasing
 *     order by its upper threshold.
 * @param {number} numberOfBuckets Note that if numberOfBuckets is 0, the
 *     threshold will remain 0 the entire time and the buckets will be
 *     transformed one to one from Bucket to BucketEntry without regrouping.
 * @param {boolean} useLogScale If using log sclae, make sure all values are at
 *     least MIN_ALLOWED_VALUE_ON_LOG_SCALE.
 * @return {!Array<!BucketEntry>}
 * @private
 */
function regroupBucketToEntries(buckets, numberOfBuckets, useLogScale) {
  let weightSum = 0;
  let labelSum = 0;
  let predictionSum = 0;
  const minValue =
      buckets && buckets[0] && buckets[0][FieldNames.UPPER_THRESHOLD] || 0;
  const maxValue = buckets && buckets[buckets.length - 1] &&
          buckets[buckets.length - 1][FieldNames.LOWER_THRESHOLD] ||
      0;
  const bucketSize =
      numberOfBuckets ? (maxValue - minValue) / numberOfBuckets : 0;
  let currentThreshold = minValue + bucketSize;
  /**
   * @type {!Array<!BucketEntry>}
   */
  const entries = [];
  buckets.forEach(bucket => {
    if (bucket[FieldNames.LABEL] != null &&
        bucket[FieldNames.PREDICTION] != null &&
        bucket[FieldNames.WEIGHTS] != null &&
        bucket[FieldNames.UPPER_THRESHOLD] != null) {
      labelSum += bucket[FieldNames.LABEL];
      predictionSum += bucket[FieldNames.PREDICTION];
      weightSum += bucket[FieldNames.WEIGHTS];
      if (bucket[FieldNames.UPPER_THRESHOLD] >= currentThreshold) {
        // Once the current threshold is reached, make a new bucket if it will
        // not be empty.
        if (weightSum > 0) {
          const x = predictionSum / weightSum;
          const y = labelSum / weightSum;
          entries.push({
            x: useLogScale ? Math.max(MIN_ALLOWED_VALUE_ON_LOG_SCALE, x) : x,
            y: useLogScale ? Math.max(MIN_ALLOWED_VALUE_ON_LOG_SCALE, y) : y,
            w: weightSum
          });
          weightSum = 0;
          labelSum = 0;
          predictionSum = 0;
        }
        currentThreshold += bucketSize;
      }
    }
  });
  return entries;
}

/**
 * Determines the value to use in the calibration plot.
 * @param {boolean} useWeight
 * @param {!BucketEntry} entry
 * @param {!LeastSquareFit.Line} fit
 * @return {number} The value to use for determining the size of the dot.
 * @private
 */
function determineCalibrationPlotValue(useWeight, entry, fit) {
  return useWeight ? logBase10WithFloor(entry.w) :
                     approximateAbsoluteError(entry, fit);
}

/**
 * @param {!LeastSquareFit.Point} point
 * @param {!LeastSquareFit.Line} line
 * @return {number} The difference in y between the point and the line.
 * @private
 */
function approximateAbsoluteError(point, line) {
  return Math.abs(point.y - line.slope * point.x - line.intercept);
}

/**
 * @param {number} value
 * @return {number} log10(value) if it is greater than some preset threshold;
 *     otherwise, the preset minimum value.
 * @private
 */
function logBase10WithFloor(value) {
  return value < MIN_ALLOWED_VALUE_ON_LOG_SCALE ? MIN_ON_LOG_SCALE_VALUE :
                                                  Math.log(value) / LOG_10;
}

goog.exportSymbol(
    'tfma.BucketsWrapper.getCalibrationPlotData', getCalibrationPlotData);

exports = {
  getCalibrationPlotData,
  MIN_ALLOWED_VALUE_ON_LOG_SCALE,
};
