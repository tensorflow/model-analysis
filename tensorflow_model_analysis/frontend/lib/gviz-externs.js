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
/**
 * @externs
 */

/**
 * @type {Object}
 */
var google;

/**
 * @type {!Object}
 */
google.visualization = {};

/**
 * @param {Node} container
 * @constructor
 */
google.visualization.ColumnChart = function(container) {};

/**
 * @param {!Object} data
 * @param {Object=} opt_options
 */
google.visualization.ColumnChart.prototype.draw = function(
    data, opt_options) {};

/**
 * @param {string} dataSourceUrl
 * @param {Object=} opt_options
 * @constructor
 */
google.visualization.Query = function(dataSourceUrl, opt_options) {};

/**
 * @param {(string|Object)=} opt_data
 * @param {number=} opt_version
 * @constructor
 */
google.visualization.DataTable = function(opt_data, opt_version) {};

/**
 * @param {Object} dataTable
 * @constructor
 */
google.visualization.DataView = function(dataTable) {};

/**
 * @param {!Array.<!Object|number>} colIndices
 */
google.visualization.DataView.prototype.setColumns = function(colIndices) {};
