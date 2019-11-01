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

import '@polymer/paper-button/paper-button.js';
import '@polymer/iron-label/iron-label.js';
import '@polymer/paper-dialog/paper-dialog.js';
import '@polymer/paper-icon-button/paper-icon-button.js';
import '@polymer/paper-item/paper-item.js';
import '@polymer/paper-listbox/paper-listbox.js';
import '../fairness-bounded-value-bar-chart/fairness-bounded-value-bar-chart.js';
import '../fairness-metrics-table/fairness-metrics-table.js';

import {PolymerElement} from '@polymer/polymer/polymer-element.js';
import {template} from './fairness-metric-summary-template.html.js';

export class FairnessMetricSummary extends PolymerElement {
  constructor() {
    super();
  }

  static get is() {
    return 'fairness-metric-summary';
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
       *
       */
      data: {type: Array},

      /**
       * The name of the metric.
       * @type {string}
       */
      metric: {type: String},

      /**
       * The list of selected thresholds.
       * @type {!Array<string>}
       */
      thresholds: {type: Array, value: []},

      /**
       * The slice that will be used as the baseline.
       * @type {string}
       */
      baseline: {type: String},

      /**
       * All available slices.
       * @type {!Array<string>}
       */
      slices: {type: Array},

      /**
       * The list of full metric names to plot in the bar chart. For regular
       * metrics, it should be a list containing only the property metric. For
       * fairness metrics, it would be the metric@threshold for all thresholds.
       * @private {!Array<string>}
       */
      metricsForBarChart_: {
        type: Array,
        computed:
            'computeMetricsForBarChart_(metric, thresholds, baseline, data)'
      },

      /**
       * The list of metrics to plot in the table view.
       * @private {!Array<string>}
       */
      metricsForTable_: {
        type: Array,
        computed: 'computeMetricsForTable_(metricsForBarChart_, baseline)'
      },

      /**
       * A cached object for the ratio of difference between each slice and the
       * baseline.
       * @private {!Object}
       */
      diffRatios_: {
        type: Object,
        computed:
            'computeDiffRatios_(baseline, data, slices, metricsForBarChart_)'
      },

      /**
       * The slices to plot.
       * @private {!Array<string>}
       */
      slicesToPlot_: {type: Array, value: []},

      /**
       * The header override for the table view.
       * @private {!Object}
       */
      headerOverride_: {
        type: Object,
        computed: 'computeHeaderOverride_(baseline, metricsForBarChart_)'
      },

      /**
       * The data backing the table view.
       * @private {!tfma.Data}
       */
      tableData_: {
        type: Object,
        computed: 'computeTableData_(baseline, data, metricsForBarChart_, ' +
            'slicesToPlot_, diffRatios_)'
      },

      /**
       * A list of names of available slices that can be selected in the config
       * menu.
       * @private {!Array<string>}
       */
      configSelectableSlices_: {type: Array},

      /**
       * A list of selected slice names in the config menu.
       * @private {!Array<string>}
       */
      configSelectedSlices_: {type: Array},

      /**
       * A list containing the number of examples for each slice.
       * @private {!Array<string>}
       */
      exampleCounts_: {
        type: Array,
        computed: 'computeExampleCounts_(baseline, data, ' +
            'slicesToPlot_)'
      }
    };
  }

  static get observers() {
    return [
      'initializeSlicesToPlot_(baseline, slices, metricsForBarChart_, ' +
          'diffRatios_)',
      'refreshSelectableSlices_(baseline, slices, ' +
          'configSelectedSlices_.length)',
    ];
  }

  /**
   * @param {string} metric
   * @param {!Array<string>} thresholds
   * @param {string} baseline
   * @param {!Object} data
   * @return {!Array<string>} The list of full metric names to be
   *     plotted.
   * @private
   */
  computeMetricsForBarChart_(metric, thresholds, baseline, data) {
    if (!data || !baseline) {
      return [];
    }
    const baselineSliceMetrics = data.find(d => d['slice'] == baseline);
    const metricsToPlot = [];
    thresholds.forEach(threshold => {
      const fairnessMetricName = metric + '@' + threshold;
      if (baselineSliceMetrics &&
          Object.keys(baselineSliceMetrics['metrics'])
              .includes(fairnessMetricName)) {
        metricsToPlot.push(fairnessMetricName);
      }
    });

    if (!metricsToPlot.length && baselineSliceMetrics &&
        Object.keys(baselineSliceMetrics['metrics']).includes(metric)) {
      metricsToPlot.push(metric);
    }
    return metricsToPlot;
  }

  /**
   * Constructs the header override for the table view.
   * @param {string} baseline
   * @param {!Array<string>} metrics
   * @return {!Array<string>}
   * @private
   */
  computeHeaderOverride_(baseline, metrics) {
    return metrics.reduce((acc, metric) => {
      acc[metric + '  against ' + baseline] = 'Diff. w. baseline';
      return acc;
    }, {});
  }

  /**
   * Computes the ratio of difference for all metrics between each slice and the
   * baseline.
   * @param {string} baseline
   * @param {!Object} data
   * @param {!Array<string>} slices
   * @param {!Array<string>} metrics
   * @return {!Object|undefined}
   * @private
   */
  computeDiffRatios_(baseline, data, slices, metrics) {
    if (!baseline || !data || !slices || !metrics) {
      return undefined;
    }
    return metrics.reduce((metricAcc, metric) => {
      const baselineValue = tfma.CellRenderer.maybeExtractBoundedValue(
          data.find(d => d['slice'] == baseline)['metrics'][metric]);
      metricAcc[metric] = slices.reduce((sliceAcc, slice) => {
        if (data.find(d => d['slice'] == slice)) {
          sliceAcc[slice] =
              (tfma.CellRenderer.maybeExtractBoundedValue(
                   data.find(d => d['slice'] == slice)['metrics'][metric]) /
               baselineValue) -
              1;
        }
        return sliceAcc;
      }, {});
      return metricAcc;
    }, {});
  }

  /**
   * Determines the slices to visualize. The slices are sorted by how noteworthy
   * they are when compared to the baseline and by default, the top 9 wil be
   * chosen.
   * @param {string} baseline
   * @param {!Array<string>} slices
   * @param {!Array<string>} metricsToPlot
   * @param {!Object} diffRatios
   * @private
   */
  initializeSlicesToPlot_(baseline, slices, metricsToPlot, diffRatios) {
    if (metricsToPlot && metricsToPlot.length && baseline && slices &&
        diffRatios) {
      // Use the first metrics to determine "interesting" slices to plot.
      const metric = metricsToPlot[0];
      this.slicesToPlot_ =
          slices.filter(slice => slice != baseline)
              .sort((sliceA, sliceB) => {
                return Math.abs(
                    diffRatios[metric][sliceB] - diffRatios[metric][sliceA]);
              })
              .slice(0, 9);  // Show up to 9 slices (plus baseline) by default.
    } else {
      this.slicesToPlot_ = [];
    }
  }

  /**
   * Computes the metrics that will be displayed in table view.
   * @param {!Array<string>} metrics The list of metrics to render in bar chart.
   * @param {string} baseline The baseline slice.
   * @return {!Array<string>}
   * @private
   */
  computeMetricsForTable_(metrics, baseline) {
    return metrics.reduce((acc, metric) => {
      acc.push(metric);
      acc.push(metric + ' against ' + baseline);
      return acc;
    }, []);
  }

  /**
   * @param {string} baseline
   * @param {!Array<!Object>} data
   * @param {!Array<string>} metrics
   * @param {!Array<string>} slices
   * @param {!Object<!Object<number>>} diffRatios
   * @return {!Array|undefined}
   * @private
   */
  computeTableData_(baseline, data, metrics, slices, diffRatios) {
    if (!baseline || !data || !metrics || !slices || !diffRatios) {
      return undefined;
    }

    try {
      const slicesInTable = [baseline, ...slices];
      const tableData = slicesInTable.map(slice => {
        const sliceMetrics = data.find(d => d['slice'] == slice);
        return !sliceMetrics ? {} : {
          'slice': slice,
          'metrics': metrics.reduce(
              (acc, metric) => {
                acc[metric] = sliceMetrics['metrics'][metric];
                const metricDiffName = metric + ' against ' + baseline;
                acc[metricDiffName] = diffRatios[metric][slice];
                return acc;
              },
              {})
        };
      });
      return tableData;
    } catch (error) {
      console.error(error);
      return undefined;
    }
  }

  /**
   * @param {string} baseline
   * @param {!Array<!Object>} data
   * @param {!Array<string>} slices
   * @return {!Array|undefined}
   * @private
   */
  computeExampleCounts_(baseline, data, slices) {
    if (!baseline || !data || !slices) {
      return undefined;
    }
    const slicesInTable = [baseline, ...slices];
    const exampleCounts = slicesInTable.map(slice => {
      const sliceMetrics = data.find(d => d['slice'] == slice);
      return !sliceMetrics ?
          {} :
          sliceMetrics['metrics']['post_export_metrics/example_count'];
    });
    return exampleCounts;
  }

  /**
   * Opens the settings dialog box.
   * @private
   */
  openSettings_() {
    this.configSelectedSlices_ = this.slicesToPlot_.slice();
    this.refreshSelectableSlices_(
        this.baseline, this.slices, this.configSelectedSlices_.length);
    this.$['settings'].open();
  }

  /**
   * Updates the config and closes the settings dialog.
   * @private
   */
  updateConfig_() {
    this.slicesToPlot_ = this.configSelectedSlices_.slice();
    this.$['settings'].close();
  }

  /**
   * Refreshes the selctable slices in the setting dialog.
   * @param {string} baseline The slice that will be used as the baseline.
   * @param {!Array<string>} slices All available slices.
   * @param {number} numOfSelectedSlices The number of selected slices.
   * @private
   */
  refreshSelectableSlices_(baseline, slices, numOfSelectedSlices) {
    if (!slices) {
      return;
    }
    this.configSelectableSlices_ = slices.map(slice => {
      return {
        'slice': slice,
        'disabled': (slice == baseline) ||
            // Only allow up to 16 (baseline + 15 selected) slices. So, if 15
            // is selected, disable all unselected slices.
            (numOfSelectedSlices > 14 &&
             this.configSelectedSlices_.indexOf(slice) == -1),
      };
    });
  }
}

customElements.define('fairness-metric-summary', FairnessMetricSummary);
