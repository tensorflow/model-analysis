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

import '@polymer/iron-icons/iron-icons.js';

import {PolymerElement} from '@polymer/polymer/polymer-element.js';
import {template} from './fairness-metrics-table-template.html.js';

const Util = goog.require('tensorflow_model_analysis.addons.fairness.frontend.Util');

/** @const {number} */
const FLOATING_POINT_PRECISION = 3;

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
       * Metric name.
       * @type {string}
       */
      metric: {type: String, value: ''},

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
       * @type {!Array}
       */
      data: {
        type: Array,
        value: () => ([]),
      },

      /**
       * Metrics table data. Optional - used for model comparison.
       * @type {!Array}
       */
      dataCompare: {
        type: Array,
        value: () => ([]),
      },

      /**
       * Name of the first model.
       * @type {string}
       */
      evalName: {type: String, value: ''},

      /**
       * Name of the second model. Optional - used for model comparison.
       * @type {string}
       */
      evalNameCompare: {type: String, value: ''},

      /**
       * List of example counts for each slice.
       * @type {!Array<number>}
       */
      exampleCounts: {
        type: Array,
        value: () => ([]),
      },

      /**
       * Header of the table.
       * @private {!Array<string>}
       */
      headerRow_: {
        type: Array,
        computed:
            'populateHeaderRow_(data, dataCompare, metrics, evalName, evalNameCompare)',
        notify: true,
      },

      /**
       * Generated plot data piped to google-chart. It should be 2d array
       * where the each element in the top level array represents a row in
       * the table.
       * @private {!Array<!Array>}
       */
      tableData_: {
        type: Array,
        computed:
            'computeTableData_(data, dataCompare, metrics, evalName, evalNameCompare)',
      },
    };
  }

  /**
   * @return {boolean} Returns true if evals are being compared.
   * @private
   */
  evalComparison_() {
    return this.dataCompare && this.dataCompare.length > 0;
  }

  /**
   * Populate header row
   * @param {!Array} data
   * @param {!Array} dataCompare
   * @param {!Array<string>} metrics
   * @param {string} evalName
   * @param {string} evalCompareName
   * @return {!Array<string>}
   * @private
   */
  populateHeaderRow_(data, dataCompare, metrics, evalName, evalCompareName) {
    if (!metrics) {
      return [];
    }
    const metricCols = metrics.map(Util.removeMetricNamePrefix);

    if (!this.evalComparison_()) {
      return ['feature'].concat(metricCols);
    } else {
      const colName = (metricName, evalName) => {
        const threshold = metricName.split('@')[1];
        const baseline = metricName.split('against')[1];
        return threshold ?
            evalName.concat('@', threshold) :
            baseline ? evalName.concat(' against', baseline) : evalName;
      };

      const evalCols = [];
      const evalCompareCols = [];
      for (let j = 0; j < metricCols.length; j += 2) {
        evalCols.push(colName(metricCols[j], evalName));
        evalCompareCols.push(colName(metricCols[j], evalCompareName));
      }

      // +=2 to skip 'against Baseline' columns
      const againstCols = [];
      for (let j = 0; j < metrics.length; j += 2) {
        var againstCol = evalCompareName.concat(' against ', evalName);
        const threshold = metricCols[j].split('@')[1];
        againstCols.push(
            threshold ? againstCol.concat('@', threshold) : againstCol);
      }

      return ['feature'].concat(evalCols, evalCompareCols, againstCols);
    }
  }

  /**
   * Populate table rows
   * @param {!Array<string>} metrics
   * @param {!Array} data
   * @param {!Array} dataCompare
   * @return {!Array<string>}
   * @private
   */
  populateTableRows_(metrics, data, dataCompare) {
    var tableRows = [];
    for (let i = 0; i < data.length; i++) {
      var tableRow = [];

      // slice name
      tableRow.push(data[i]['slice']);

      const metricsData = data[i]['metrics'];

      // In comparison, skip over 'metric against Baseline' (+=2)
      if (this.evalComparison_()) {
        // First Eval column
        for (let j = 0; j < metrics.length; j += 2) {
          tableRow.push(this.formatCell_(metricsData[metrics[j]]));
        }

        // Second Eval column
        const metricsDataCompare = dataCompare[i]['metrics'];
        for (let j = 0; j < metrics.length; j += 2) {
          tableRow.push(this.formatCell_(metricsDataCompare[metrics[j]]));
        }

        // Comparison columns
        for (let j = 0; j < metrics.length; j += 2) {
          const evalMetric = this.isBoundedValue_(metricsData[metrics[j]]) ?
              metricsData[metrics[j]]['value'] :
              metricsData[metrics[j]];
          const evalCompareMetric =
              this.isBoundedValue_(metricsDataCompare[metrics[j]]) ?
              metricsDataCompare[metrics[j]]['value'] :
              metricsDataCompare[metrics[j]];
          const comparison = evalCompareMetric / evalMetric - 1;
          tableRow.push(comparison.toString());
        }
      }

      // In non-comparison, include both 'metric' and 'metric against Baseline'
      else {
        metrics.forEach(entry => {
          tableRow.push(this.formatCell_(metricsData[entry]));
        });
      }

      tableRows.push(tableRow);
    }
    return tableRows;
  }

  /**
   * Computes the data table.
   * @param {!Array} data
   * @param {!Array} dataCompare
   * @param {!Array<string>} metrics
   * @param {string} evalName
   * @param {string} evalNameCompare
   * @return {!Array<!Array<string>>|undefined}
   * @private
   */
  computeTableData_(data, dataCompare, metrics, evalName, evalNameCompare) {
    if (!data || !metrics) {
      return [];
    }
    if (data.length == 0) {
      // No need to compute plot data if data is empty.
      return [[]];
    }

    this.headerRow_ = this.populateHeaderRow_(
        data, dataCompare, metrics, evalName, evalNameCompare);
    let tableRows = this.populateTableRows_(metrics, data, dataCompare);
    tableRows.sort();
    return [this.headerRow_].concat(tableRows);
  }

  /**
   * Formats cell data so that it can be rendered in the table.
   * @param {number|string|!Object} cell_data
   * @return {string}
   * @private
   */
  formatCell_(cell_data) {
    // TODO(b/137209618): Handle other data types as well.
    if (typeof cell_data === 'object' && this.isBoundedValue_(cell_data)) {
      return this.formatFloatValue_(cell_data['value']) + ' (' +
          this.formatFloatValue_(cell_data['lowerBound']) + ', ' +
          this.formatFloatValue_(cell_data['upperBound']) + ')';
    } else if (typeof cell_data === 'string') {
      return cell_data;
    } else {
      return JSON.stringify(cell_data);
    }
  }

  /**
   * Formats float value.
   * @param {string|number} value
   * @return {string} The given value formatted as a string.
   * @private
   */
  formatFloatValue_(value) {
    if (value === undefined || value == 'NaN') {
      return 'NaN';
    } else {
      return this.toFixedNumber_(
          typeof (value) == 'string' ? parseFloat(value) : value,
          FLOATING_POINT_PRECISION);
    }
  }

  /**
   * Convert decimal value to percentage.
   * @param {string|number} value
   * @return {string} The given value formatted as a string.
   * @private
   */
  toPercentage_(value) {
    return this.toFixedNumber_(100 * value, FLOATING_POINT_PRECISION) + '%';
  }

  /**
   * Format a number with up to `digits` decimal places.
   *
   * We use this instead of JavaScript's built-in toFixed() function because
   *   toFixed() returns a string with exactly `digits` decimals, and pads with
   *   0's if needed.
   *
   * This function returns up to `digits` decimals, cutting off trailing 0's.
   *
   * @param {number} num
   * @param {number} digits
   * @return {string}
   * @private
   */
  toFixedNumber_(num, digits) {
    var pow = Math.pow(10, digits);
    return (Math.round(num * pow) / pow).toString();
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
   * @param {!Array} headerRow
   * @return {boolean} Returns true if the given column index corresponds to a
   * percentage column.
   * @private
   */
  isPercentageColumn_(index, headerRow) {
    return headerRow && index < headerRow.length &&
        headerRow[index].includes(' against ');
  }

  /**
   * @param {(string)} rowNum
   * @param {!Array} exampleCounts
   * @return {string} Get example count for the corresponding row.
   * @private
   */
  getExampleCount_(rowNum, exampleCounts) {
    if (!exampleCounts) {
      return '';
    }

    // We skip the first row, since it is a header row which does not correspond
    // to a slice.
    let value = exampleCounts[parseFloat(rowNum) - 1];
    if (typeof value === 'number') {
      return this.toFixedNumber_(value, FLOATING_POINT_PRECISION);
    } else {
      return '';
    }
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
   * @param {(string|number)} metric_diff
   * @return {boolean} Returns true if value is nonzero.
   * @private
   */
  isNonzero_(metric_diff) {
    return metric_diff != 0;
  }

  /**
   * @param {(string|number)} metric_diff value in a diff cell.
   * @return {string} Returns iron-icon string for metric_diff's value.
   * @private
   */
  arrow_(metric_diff) {
    if (parseFloat(metric_diff) > 0) {
      return 'arrow-upward';
    } else if (parseFloat(metric_diff) < 0) {
      return 'arrow-downward';
    } else {
      return '';
    }
  }

  /**
   * @param {(string|number)} metric_diff value in a diff cell.
   * @param {(string)} metric name of metric.
   * @return {string} Returns icon's class based on metric_diff's value.
   * @private
   */
  icon_class_(metric_diff, metric) {
    const unprefixed_metric = Util.removeMetricNamePrefix(metric);
    const parsed_metric_diff = parseFloat(metric_diff);
    if ((Util.POSITIVE_METRICS.includes(unprefixed_metric) &&
         parsed_metric_diff > 0) ||
        (Util.NEGATIVE_METRICS.includes(unprefixed_metric) &&
         parsed_metric_diff < 0)) {
      return 'green-icon';
    } else if (
        (Util.POSITIVE_METRICS.includes(unprefixed_metric) &&
         parsed_metric_diff < 0) ||
        (Util.NEGATIVE_METRICS.includes(unprefixed_metric) &&
         parsed_metric_diff > 0)) {
      return 'red-icon';
    } else if (
        !Util.POSITIVE_METRICS.includes(unprefixed_metric) &&
        !Util.NEGATIVE_METRICS.includes(unprefixed_metric)) {
      return 'blue-icon';
    } else {
      return '';
    }
  }
}

customElements.define('fairness-metrics-table', FairnessMetricsTable);
