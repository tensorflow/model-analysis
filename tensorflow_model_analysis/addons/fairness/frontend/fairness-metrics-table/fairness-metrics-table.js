/**
 * Copyright 2019 Google LLC
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

import {PolymerElement} from '@polymer/polymer/polymer-element.js';
import {template} from './fairness-metrics-table-template.html.js';

const Util = goog.require('tensorflow_model_analysis.addons.fairness.frontend.Util');

/** @const {number} */
const FLOATING_POINT_PRECISION = 5;

/**
 * @enum {string}
 */
const BoundedValueFieldNames = {
  LOWER_BOUND: 'lowerBound',
  UPPER_BOUND: 'upperBound',
  VALUE: 'value',
};

/**
 * fairness-metrics-table renders a div-based table showing evaluation
 * metrics.
 *
 * @polymer
 */
export class FairnessMetricsTable extends PolymerElement {
  static get is() {
    return 'fairness-metrics-table';
  }

  /** @return {!HTMLTemplateElement} */
  static get template() {
    return template;
  }

  /** @return {!PolymerElementProperties} */
  static get properties() {
    return {
      /**
       * List of metrics to be shown in the table.
       * @type {!Array<string>}
       */
      metrics: {
        type: Array,
        value: () => ([]),
      },

      /**
       * Metrics table data.
       * @type {!Object}
       */
      data: {
        type: Object,
        value: () => ({}),
      },

      /**
       * A look up table to override header.
       * @type {!Object<string>}
       */
      headerOverride: {
        type: Object,
        value: () => ({}),
      },

      /**
       * List of example counts for each slice.
       * @type {!Array<string>}
       */
      exampleCounts: {
        type: Array,
        value: () => ([]),
      },

      /**
       * Generated plot data piped to google-chart. It should be 2d array
       * where the each element in the top level array represents a row in
       * the table.
       * @private {!Array<!Array>}
       */
      plotData_: {
        type: Array,
        computed: 'computePlotData_(data, metrics, headerOverride)',
      },
    };
  }

  /**
   * Computes the data table.
   * @param {!Object} data
   * @param {!Array<string>} metrics
   * @param {!Object<string>} headerOverride
   * @return {!Array<!Array>|undefined}
   * @private
   */
  computePlotData_(data, metrics, headerOverride) {
    if (!data || !metrics || !headerOverride) {
      return undefined;
    }
    if (Object.keys(data).length == 0) {
      // No need to compute plot data if data is empty.
      return [[]];
    }

    let headerRow = ['feature'];
    for (let i = 0; i < metrics.length; i++) {
      headerRow.push(Util.removePostExportMetrics(metrics[i]));
    }
    headerRow = headerRow.map(metric => headerOverride[metric] || metric);

    var plotData = [];
    Object.keys(data).forEach(key => {
      const metricsData = data[key]['metrics'];
      const slice = data[key]['slice'];

      if (metricsData === undefined || slice === undefined) {
        return;
      }

      var tableRow = [slice];
      metrics.forEach(entry => {
        tableRow.push(this.formatData_(metricsData[entry]));
      });
      plotData.push(tableRow);
    });

    return [headerRow].concat(plotData);
  }

  /**
   * Formats data so that it can be rendered in the table.
   * @param {number|string|!Object} data
   * @return {string}
   * @private
   */
  formatData_(data) {
    // TODO(b/137209618): Handle other data types as well.
    if (typeof data === 'object' && this.isBoundedValue_(data)) {
      return this.formatFloatValue_(data['value']) + ' (' +
          this.formatFloatValue_(data['lowerBound']) + ', ' +
          this.formatFloatValue_(data['upperBound']) + ')';
    } else if (typeof data === 'string') {
      return data;
    } else {
      return JSON.stringify(data);
    }
  }

  /**
   * Formats float value.
   * @param {string|number} value
   * @return {string} The given value formatted as a string.
   * @private
   */
  formatFloatValue_(value) {
    if (value == 'NaN') {
      return 'NaN';
    } else {
      return (typeof (value) == 'string') ?
          parseFloat(value).toFixed(FLOATING_POINT_PRECISION) :
          value.toFixed(FLOATING_POINT_PRECISION);
    }
  }

  /**
   * Convert decimal value to percentage.
   * @param {string|number} value
   * @return {string} The given value formatted as a string.
   * @private
   */
  toPercentage_(value) {
    return (100 * value).toFixed(tfma.FLOATING_POINT_PRECISION) + '%';
  }

  /**
   * @param {(string|number|?Object)} value
   * @return {boolean} Returns true if the given value represents a bounded
   *     value.
   * @private
   */
  isBoundedValue_(value) {
    return !!value && value[BoundedValueFieldNames.LOWER_BOUND] !== undefined &&
        value[BoundedValueFieldNames.UPPER_BOUND] !== undefined &&
        value[BoundedValueFieldNames.VALUE] !== undefined;
  }

  /**
   * @param {(number)} index
   * @return {boolean} Returns true if the given column index corresponds to a
   * Diff. w. baseline column.
   * @private
   */
  isDiffWithBaselineColumn_(index) {
    return index > 0 && !(index % 2);
  }

  /**
   * @param {(string)} rowNum
   * @param {(string)} exampleCounts
   * @return {boolean} Get example count for the corresponding row.
   * @private
   */
  getExampleCount_(rowNum, exampleCounts) {
    // We skip the first row, since it is a header row which does not correspond
    // to a slice.
    return exampleCounts[parseFloat(rowNum)-1];
  }

  /**
   * @param {(string)} s
   * @return {boolean} Returns true if string is 0.
   * @private
   */
  isZero_(s) {
    return parseFloat(s) === 0;
  }

  /**
   * @param {(string)} row_num
   * @return {boolean} Returns true if row_num is 0.
   * @private
   */
  isHeaderRow_(row_num) {
    return parseFloat(row_num) === 0;
  }

  /**
   * @param {(string)} row_num
   * @return {boolean} Returns true if row_num is 1.
   * @private
   */
  isBaselineRow_(row_num) {
    return parseFloat(row_num) === 1;
  }

  /**
   * @param {(string)} row_num
   * @return {boolean} Returns true if row_num is greater than 1.
   * @private
   */
  isSliceRow_(row_num) {
    return parseFloat(row_num) > 1;
  }

  /**
   * @param {(string)} s
   * @return {boolean} Returns true if string represents a positive number.
   * @private
   */
  isPositive_(s) {
    return s.charAt(0) != '-';
  }

  /**
   * @param {(string)} s
   * @return {boolean} Returns true if string represents a negative number.
   * @private
   */
  isNegative_(s) {
    return s.charAt(0) == '-';
  }
}

customElements.define('fairness-metrics-table', FairnessMetricsTable);
