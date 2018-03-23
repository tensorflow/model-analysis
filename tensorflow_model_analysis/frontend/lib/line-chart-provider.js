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
goog.module('tfma.LineChartProvider');

/**
 * Defines an interface that provides necessary data to render a time series
 * chart inside tfma-line-chart-grid component.
 * @interface
 */
class LineChartProvider {
  constructor() {}

  /**
   * Returns the necessary data to render the line chart for the named metric.
   * @param {string} metric
   * @return {!Array<!Array<string|number|!GVizCell>>}
   * @export
   */
  getLineChartData(metric) {}

  /**
   * @return {!Array<string|number>}
   * @export
   */
  getModelIds() {}

  /**
   * Get eval config of the run with the given index.
   * @param {number} index
   * @return {?{
   *   model: (string|number),
   *   data: (string|number)
   * }}
   * @export
   */
  getEvalConfig(index) {}

  /**
   * @return {string} The name of the model column. Can be used for matching
   *     evaluation runs.
   * @export
   */
  getModelColumnName() {}

  /**
   * @return {string} The name of the data column. Can be used for matching
   *     evaluation runs.
   * @export
   */
  getDataColumnName() {}
}

/**
 * @typedef {{
 *   v: (number|string),
 *   f: string
 * }}
 */
let GVizCell;

exports = LineChartProvider;
