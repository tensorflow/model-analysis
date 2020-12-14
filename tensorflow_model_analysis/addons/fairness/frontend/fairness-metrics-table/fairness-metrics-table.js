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
 * It's a class that contains following fields:
 *   'text': used to render on UI.
 *   'arrow': arrow iron-icon string. e.g. arrow-upward.
 *   'arrow_icon_css_class': css class defines the color of the arrow.
 */
/** @record */
class Cell {
  constructor() {
    /** @type {string} */
    this.text;

    /** @type {string|undefined} */
    this.arrow;

    /** @type {string|undefined} */
    this.arrow_icon_css_class;
  }
}


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
       * A dictionary of slice names and metrics values which itself is a
       * dictionary.
       * @type {!Array<!Object>}
       */
      data: {type: Array},

      /**
       * Data dictionary for the second eval. Optional.
       * @type {!Array<!Object>}
       */
      dataCompare: {type: Array},

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
       * The one slice that will be used as the baseline.
       * @type {string}
       */
      baseline: {type: String},

      /**
       * Only include the slices to plot. The baseline is not included.
       * @type {!Array<string>}
       */
      slices: {type: Array},

      /**
       * List of metrics to be shown in the table.
       * @type {!Array<string>}
       */
      metrics: {type: Array, value: undefined},

      /**
       * Header of the table.
       * @private {!Array<string>}
       */
      headerRow_: {
        type: Array,
        computed:
            'populateHeaderRow_(data, dataCompare, evalName, evalNameCompare, metrics, baseline)',
        notify: true,
      },
      /**
       * Content of the table.
       * @private {!Array<!Array<!Cell>>}
       */
      contentRows_: {
        type: Array,
        computed:
            'populateContentRows_(data, dataCompare, evalName, evalNameCompare, metrics, baseline, slices)',
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
   * @param {string} evalName
   * @param {string} evalCompareName
   * @param {!Array<string>} metrics
   * @param {string} baseline
   * @return {!Array<string>}
   * @private
   */
  populateHeaderRow_(
      data, dataCompare, evalName, evalCompareName, metrics, baseline) {
    if (!data || !metrics || !metrics.length) {
      return [];
    }
    const header = ['feature'];
    if (!this.evalComparison_()) {
      for (let i = 0; i < metrics.length; i++) {
        const metric = Util.removeMetricNamePrefix(metrics[i]);
        header.push(metric, metric + ' against ' + baseline);
      }
    } else {
      const thresholds = metrics.map(metric => {
        return metric.split('@')[1] || '';
      });
      const evalNames = [evalName, evalCompareName];
      for (const evaluation of evalNames) {
        for (const threshold of thresholds) {
          if (threshold) {
            header.push(evaluation + '@' + threshold);
          } else {
            header.push(evaluation);
          }
        }
      }
      for (const threshold of thresholds) {
        if (threshold) {
          header.push(
              evalName + ' against ' + evalCompareName + ' @' + threshold);
        } else {
          header.push(evalName + ' against ' + evalCompareName);
        }
      }
    }
    header.push('example count');
    return header;
  }

  /**
   * Populate header row
   * @param {!Array} data
   * @param {!Array} dataCompare
   * @param {string} evalName
   * @param {string} evalCompareName
   * @param {!Array<string>} metrics
   * @param {string} baseline
   * @param {!Array<string>} slices
   * @return {!Array<!Array<!Cell>>}
   * @private
   */
  populateContentRows_(
      data, dataCompare, evalName, evalCompareName, metrics, baseline, slices) {
    if (!data || !metrics || !metrics.length || !baseline || !slices) {
      return [];
    }

    slices = [baseline, ...slices];

    let tableRows = [];
    for (const slice of slices) {
      let tableRow = [];
      const sliceMetricsData = data.find(d => d['slice'] == slice);
      if (!sliceMetricsData) {
        continue;
      }

      // Add slice name cell.
      tableRow.push({text: slice});

      if (this.evalComparison_()) {
        const sliceMetricsDataCompare =
            dataCompare.find(d => d['slice'] == slice);
        for (const metricName of metrics) {
          const metricData = sliceMetricsData['metrics'][metricName];
          const metricDataCompare =
              sliceMetricsDataCompare['metrics'][metricName];
          // Add metric cell.
          tableRow.push(this.formatCell_(metricData));
          // Add metric of second Eval.
          tableRow.push(this.formatCell_(metricDataCompare));
        }
        for (const metricName of metrics) {
          const metricData = sliceMetricsData['metrics'][metricName];
          const metricDataCompare =
              sliceMetricsDataCompare['metrics'][metricName];
          // Add comparison cell (between first Eval and second Eval).
          tableRow.push(this.formatComparisonCell_(
              metricDataCompare, metricData, metricName));
        }
      } else {
        for (const metricName of metrics) {
          const metricData = sliceMetricsData['metrics'][metricName];

          // Add metric cell.
          tableRow.push(this.formatCell_(metricData));

          // Add comparison cell (between slice and baseline).
          const baselineMetricData =
              data.find(d => d['slice'] == baseline)['metrics'][metricName];
          tableRow.push(this.formatComparisonCell_(
              metricData, baselineMetricData, metricName));
        }
      }

      // Add example count cell.
      tableRow.push(this.formatCell_(
          Util.getMetricsValues(sliceMetricsData, 'example_count')));

      tableRows.push(tableRow);
    }
    return tableRows;
  }

  /**
   * Formats cell data so that it can be rendered in the table.
   * @param {number|string|!Object|undefined} cellData
   * @return {!Cell}
   * @private
   */
  formatCell_(cellData) {
    // TODO(b/137209618): Handle other data types as well.
    if (cellData === undefined) {
      return {
        text: 'NO_DATA',
        arrow: undefined,
        arrow_icon_css_class: undefined,
      };
    } else if (typeof cellData === 'number' && Number.isNaN(cellData)) {
      return {
        text: 'NaN',
        arrow: undefined,
        arrow_icon_css_class: undefined,
      };
    } else if (typeof cellData === 'object' && this.isBoundedValue_(cellData)) {
      return {
        text: this.formatFloatValue_(cellData['value']) + ' (' +
            this.formatFloatValue_(cellData['lowerBound']) + ', ' +
            this.formatFloatValue_(cellData['upperBound']) + ')',
        arrow: undefined,
        arrow_icon_css_class: undefined,
      };
    } else if (typeof cellData === 'string') {
      return {
        text: cellData,
        arrow: undefined,
        arrow_icon_css_class: undefined,
      };
    } else {
      return {
        text: JSON.stringify(cellData),
        arrow: undefined,
        arrow_icon_css_class: undefined,
      };
    }
  }


  /**
   * Formats cell data so that it can be rendered in the table.
   * @param {number|!Object|undefined} cellData1
   * @param {number|!Object|undefined} cellData2
   * @param {string} metricName
   * @return {!Cell}
   * @private
   */
  formatComparisonCell_(cellData1, cellData2, metricName) {
    // TODO(b/137209618): Handle other data types as well.
    if (cellData1 === undefined || cellData2 === undefined) {
      return {
        text: 'NO_DATA',
        arrow: undefined,
        arrow_icon_css_class: undefined,
      };
    } else {
      const diff = (tfma.CellRenderer.maybeExtractBoundedValue(cellData1) /
                    tfma.CellRenderer.maybeExtractBoundedValue(cellData2)) -
          1;
      return {
        text: this.toPercentage_(diff),
        arrow: this.arrow_(diff),
        arrow_icon_css_class: this.arrowIconCssClass_(diff, metricName)
      };
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
   * @param {number} value
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
   * @param {number} row_num
   * @return {string} Returns css class of the row.
   * @private
   */
  tableRowCssClass_(row_num) {
    if (parseFloat(row_num) === 0) {
      return 'baseline-row';
    }
    return 'table-row';
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
  arrowIconCssClass_(metric_diff, metric) {
    const unprefixed_metric = Util.removeMetricNamePrefix(metric).split('@')[0];
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
