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
import '@polymer/iron-label/iron-label.js';
import '@polymer/paper-button/paper-button.js';
import '@polymer/paper-dialog/paper-dialog.js';
import '@polymer/paper-icon-button/paper-icon-button.js';
import '@polymer/paper-item/paper-item.js';
import '@polymer/paper-listbox/paper-listbox.js';
import '../fairness-bounded-value-bar-chart/fairness-bounded-value-bar-chart.js';
import '../fairness-metrics-table/fairness-metrics-table.js';

import {PolymerElement} from '@polymer/polymer/polymer-element.js';
import {template} from './fairness-metric-summary-template.html.js';

const Util = goog.require('tensorflow_model_analysis.addons.fairness.frontend.Util');

const DEFAULT_NUM_SLICES = 9;
const DEFAULT_NUM_SLICES_EVAL_COMPARE = 3;

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
       */
      data: {type: Array},

      /**
       * Data dictionary for the second eval. Optional.
       * @type {!Array<!Object>}
       */
      dataCompare: {type: Array},

      /**
       * The name of the metric.
       * @type {string}
       */
      metric: {type: String},

      /**
       * The list of available thresholds.
       * @type {!Array<string>}
       */
      thresholds: {type: Array, computed: 'computeThresholds_(data, metric)'},

      /**
       * The list of selected thresholds.
       * @type {!Array<string>}
       */
      selectedThresholds_: {type: Array},

      /**
       * A flag used to update the selected thresholds displayed in the UI.
       * @private {boolean}
       */
      thresholdsMenuOpened_:
          {type: Boolean, observer: 'thresholdsMenuOpenedChanged_'},

      /**
       * The slice that will be used as the baseline.
       * @type {string}
       */
      baseline: {type: String},

      /**
       * The name of the first eval.
       * @type {string}
       */
      evalName: {type: String},

      /**
       * The name of the second eval. Optional.
       * @type {string}
       */
      evalNameCompare: {type: String},

      /**
       * All available slices.
       * @type {!Array<string>}
       */
      slices: {type: Array},

      /**
       * The list of full metric names. For regular metrics, it should be an
       * array containing only the property metric. For thresholded metrics, it
       * would be "metric@threshold" for all thresholds.
       * @private {!Array<string>}
       */
      metrics_: {
        type: Array,
        computed:
            'computeMetrics_(metric, thresholds, selectedThresholds_.*, ' +
            'baseline, data)'
      },

      /**
       * The list of metrics to plot in the table view.
       * @private {!Array<string>}
       */
      metricsForTable_: {
        type: Array,
        computed: 'computeMetricsForTable_(metrics_, baseline)'
      },

      /**
       * A cached object for the ratio of difference between each slice and the
       * baseline.
       * @private {!Object}
       */
      diffRatios_: {
        type: Object,
        computed: 'computeDiffRatios_(baseline, data, slices, metrics_)'
      },

      /**
       * The diff ratios for the second eval.
       * @private {!Object}
       */
      diffRatiosCompare_: {
        type: Object,
        computed: 'computeDiffRatios_(baseline, dataCompare, slices, metrics_)'
      },

      /**
       * The slices to plot.
       * @private {!Array<string>}
       */
      slicesToPlot_: {type: Array, value: []},

      /**
       * The data backing the table view.
       * @private {!tfma.Data}
       */
      tableData_: {
        type: Object,
        computed: 'computeTableData_(baseline, data, metrics_, ' +
            'slicesToPlot_, diffRatios_)'
      },

      /**
       * The data backing the table view for the second eval.
       * @private {!tfma.Data}
       */
      tableDataCompare_: {
        type: Object,
        computed: 'computeTableData_(baseline, dataCompare, metrics_, ' +
            'slicesToPlot_, diffRatiosCompare_)'
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
      },

      /**
       * The selected parameter to sort on - can be 'Slice' or 'Eval'.
       * @private {string}
       */
      sort_: {
        type: String, value: 'Slice'
      },
    };
  }

  static get observers() {
    return [
      'initializeSlicesToPlot_(baseline, slices, metrics_, ' +
          'diffRatios_)',
      'refreshSelectableSlices_(baseline, slices, ' +
          'configSelectedSlices_.length)',
      'initializeSelectedThresholds_(thresholds.*)',
    ];
  }

  /**
   * @param {!Object} data
   * @param {string} metric
   * @private
   * @return {!Array<number>} The array of thresholds this metric supports. If
   *     the metric is not thresholded, return an empty array.
   */
  computeThresholds_(data, metric) {
    if (!data || !metric) {
      return [];
    }
    const thresholds = new Set();
    data.forEach(sliceData => {
      Object.keys(sliceData.metrics).forEach(metricName => {
        if (metricName.startsWith(metric)) {
          const fairnessMetric = Util.extractFairnessMetric(metricName);
          if (fairnessMetric) {
            thresholds.add(fairnessMetric.threshold);
          }
        }
      });
    });
    return Array.from(thresholds);
  }

  /**
   * Observer for property thresholds_. Automatically selects the median
   * thresholds as default.
   * @param {!Object} thresholdsEvent
   * @private
   */
  initializeSelectedThresholds_(thresholdsEvent) {
    this.selectedThresholds_ = [];
    if (thresholdsEvent.base.length) {
      this.$.thresholdsList.select(
          thresholdsEvent.base[Math.floor(thresholdsEvent.base.length / 2)]);
    }
  }

  /**
   * Observer for thresholdsMenuOpened_ flag. Updates the string for the
   * thresholds selected.
   * @param {boolean} open
   * @private
   */
  thresholdsMenuOpenedChanged_(open) {
    if (this.selectedThresholds_ && !open) {
      setTimeout(() => {
        // HACK: Fire off a fake iron-select event with fake label with multiple
        // selected thresholds so that they are displayed in the menu. In case
        // none is selected, use ' '.
        // Needed in order to display multiple thresholds in the paper-listbox's
        // label when multiple thresholds are selected.
        this.selectedThresholds_.sort();
        const label = this.selectedThresholds_.length > 0 ?
            this.selectedThresholds_.join(', ') :
            ' ';
        this.$.thresholdsList.fire('iron-select', {'item': {'label': label}});
      }, 0);
    }
  }

  /**
   * @param {string} metric
   * @param {!Array<string>} thresholds
   * @param {!Object} selectedThresholdsEvent
   * @param {string} baseline
   * @param {!Array<!Object>} data
   * @return {!Array<string>} The list of full metric names to be
   *     plotted.
   * @private
   */
  computeMetrics_(metric, thresholds, selectedThresholdsEvent, baseline, data) {
    if (!data || !baseline || thresholds === undefined ||
        (this.metricIsThresholded_() && !selectedThresholdsEvent.base)) {
      return [];
    }

    if (this.metricIsThresholded_()) {
      selectedThresholdsEvent.base.sort();
      const thresholdedMetrics = selectedThresholdsEvent.base.map(
          (threshold) => metric + '@' + threshold);

      const baselineSliceMetrics = data.find(d => d['slice'] == baseline);
      const hasBaselineSlice = (metricName) => baselineSliceMetrics &&
          Object.keys(baselineSliceMetrics['metrics']).includes(metricName);

      return thresholdedMetrics.filter(hasBaselineSlice);
    } else {
      return [metric];
    }
  }

  /**
   * The name of the metric when compared with baseline.
   * @param {string} metric
   * @param {string} baseline
   * @return {string}
   * @private
   */
  metricDiffName_(metric, baseline) {
    return metric + ' against ' + baseline;
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
    if (!baseline || !data || data.length == 0 || !slices || !metrics ||
        metrics.length == 0) {
      return undefined;
    }

    const getSliceValue = (metric, slice) =>
        tfma.CellRenderer.maybeExtractBoundedValue(
            data.find(d => d['slice'] == slice)['metrics'][metric]);

    const baselineValues = {};
    metrics.forEach(function(metric) {
      baselineValues[metric] = getSliceValue(metric, baseline);
    });

    return metrics.reduce((diffRatiosByMetric, metric) => {
      diffRatiosByMetric[metric] =
          slices.reduce((diffRatiosForMetricBySlice, slice) => {
            if (data.find(d => d['slice'] == slice)) {
              diffRatiosForMetricBySlice[slice] =
                  (getSliceValue(metric, slice) / baselineValues[metric]) - 1;
            }
            return diffRatiosForMetricBySlice;
          }, {});
      return diffRatiosByMetric;
    }, {});
  }

  /**
   * Determines the slices to visualize. The slices are sorted by how noteworthy
   * they are when compared to the baseline and by default, the top
   * DEFAULT_NUM_SLICES wil be chosen.
   * @param {string} baseline
   * @param {!Array<string>} slices
   * @param {!Array<string>} metricsToPlot
   * @param {!Object} diffRatios
   * @private
   */
  initializeSlicesToPlot_(baseline, slices, metricsToPlot, diffRatios) {

    // Show up to DEFAULT_NUM_SLICES slices (plus baseline) by default.
    const numSlices = this.evalComparison_() ? DEFAULT_NUM_SLICES_EVAL_COMPARE :
                                               DEFAULT_NUM_SLICES;

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
              .slice(0, numSlices);
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
    if (baseline) {
      return metrics.reduce((acc, metric) => {
        acc.push(metric);
        acc.push(this.metricDiffName_(metric, baseline));
        return acc;
      }, []);
    } else {
      return [];
    }
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
    if (!baseline || !data || !metrics || !slices || !diffRatios ||
        !Object.keys(diffRatios).length) {
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
                const metricDiffName = this.metricDiffName_(metric, baseline);
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
          tfma.CellRenderer.maybeExtractBoundedValue(
              Util.getMetricsValues(sliceMetrics, 'example_count'));
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

  /**
   * @return {boolean} Returns true if metric is thresholded.
   * @private
   */
  metricIsThresholded_() {
    return this.thresholds.length > 0;
  }

  /**
   * @return {boolean} Returns true if evals are being compared.
   * @private
   */
  evalComparison_() {
    return this.dataCompare && this.dataCompare.length > 0;
  }
}

customElements.define('fairness-metric-summary', FairnessMetricSummary);
