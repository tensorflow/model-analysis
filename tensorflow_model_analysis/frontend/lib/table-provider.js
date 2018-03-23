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
goog.module('tfma.TableProvider');


/**
 * Defines an interface that provides necessary data to render
 * tfma-metrics-table.
 * @interface
 */
class TableProvider {
  constructor() {}

  /**
   * Generates a 2D array representing the data.
   * @return {!TableProvider.RawCellTable}
   * @export
   */
  getDataTable() {}

  /**
   * @param {!Array<string>} requiredColumns The names of the columns specified
   *     by the user.
   * @return {!Array<string>} An array containing all columns in the header.
   * @export
   */
  getHeader(requiredColumns) {}

  /**
   * @param {!Object<!tfma.MetricValueFormatSpec>} specifiedFormats
   * @return {!Object<!tfma.MetricValueFormatSpec>} A formats object based
   *     on the specified formats given.
   * @export
   */
  getFormats(specifiedFormats) {}

  /**
   * @return {boolean} True if the data is ready to be rendered.
   * @export
   */
  readyToRender() {}

  /**
   * Applies the format override specified in the MetricValueFormatSpec.
   * @param {number|string} value
   * @param {!tfma.MetricValueFormatSpec} override
   * @return {number|string}
   * @export
   */
  applyOverride(value, override) {}
}

/**
 * @typedef {string|number}
 */
TableProvider.RawCellData;

/**
 * @typedef {!Array<!Array<TableProvider.RawCellData>>}
 */
TableProvider.RawCellTable;

/**
 * @typedef {{
 *   f: (string),
 *   v: (string|number)
 * }}
 */
TableProvider.GvizCell;

exports = TableProvider;
