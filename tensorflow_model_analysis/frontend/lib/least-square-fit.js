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
goog.module('tfma.LeastSquareFit');

/**
 * The maximum number of poitns supported.
 * @private {number}
 */
const MAX_POINTS_SUPPORTED = 10e7;

/**
 * @typedef {{
 *   x: number,
 *   y: number
 * }}
 */
let Point;

/**
 * @typedef {{
 *   slope: number,
 *   intercept: number
 * }}
 */
let Line;

/**
 * Find a line that fits the given points using least square fit method.
 * @param {!Array<!Point>} points
 * @return {!Line}
 */
function getLine(points) {
  const count = points.length;

  // If we do not have enough inputs or too many inputs, simply bail here and
  // warn the user about it.
  if (count < 2 || count > MAX_POINTS_SUPPORTED) {
    return {intercept: NaN, slope: NaN};
  }

  let sumX = 0;
  let sumY = 0;
  points.forEach(point => {
    sumX += point.x;
    sumY += point.y;
  });

  const avgX = sumX / count;
  const avgY = sumY / count;

  let sumXY = 0;
  let sumXError = 0;
  let diffX;
  points.forEach(point => {
    diffX = (point.x - avgX);
    sumXY += diffX * (point.y - avgY);
    sumXError += diffX * diffX;
  });

  var slope = sumXY / sumXError;
  return {slope: slope, intercept: avgY - slope * avgX};
}

exports = {
  getLine,
  Line,
  Point,
};
