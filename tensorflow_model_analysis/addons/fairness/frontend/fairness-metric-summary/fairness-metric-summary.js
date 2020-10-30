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
import '@polymer/iron-dropdown/iron-dropdown.js';
import '@polymer/paper-button/paper-button.js';
import '@polymer/paper-item/paper-item.js';
import '@polymer/paper-listbox/paper-listbox.js';
import '../fairness-bounded-value-bar-chart/fairness-bounded-value-bar-chart.js';
import '../fairness-metrics-table/fairness-metrics-table.js';

import {PolymerElement} from '@polymer/polymer/polymer-element.js';
import {template} from './fairness-metric-summary-template.html.js';

const Util = goog.require('tensorflow_model_analysis.addons.fairness.frontend.Util');

/**
 * Name of master slice containing all data.
 * @private {string}
 * @const
 */
const OVERALL_SLICE_KEY = 'Overall';

/**
 * Number of default slices to be selected.
 * @private {number}
 * @const
 */
const DEFAULT_NUM_SLICES = 9;
const DEFAULT_NUM_SLICES_EVAL_COMPARE = 3;

/**
 * It's a map that contains following fields:
 *   'text': used to render on UI, e.g. "gender" or "male".
 *   'rawSlice': store the original slices, e.g. "gender:male", or "Overall".
 *      If this is a slice key, this field will be an empty string.
 *   'id': An unique id that's convenient for search purpose.
 *   'isSliceKey': used to indicate whether this is a slice key element or a
 *     general slice key value pair element.
 *   'representedSlices': a list of slices that this element presents. If it's a
 *      slice key, all the slices key value pairs under this slice key will be
 *      included. If it's a general slice element, this list will only include
 *      itself.
 *   'isSelected': indicates if this element is selected by users for plotting.
 *      This will also be used to change the checkbox status in front of the
 *      slices.
 *   'isDisabled': will be true if max number of slices is selected.
 */
/** @record */
class SlicesDropDownMenuCandidateType {
  constructor() {
    /** @type {string} */
    this.text;

    /** @type {string} */
    this.rawSlice;

    /** @type {number} */
    this.id;

    /** @type {boolean} */
    this.isSliceKey;

    /** @type {!Array<string>} */
    this.representedSlices;

    /** @type {boolean} */
    this.isSelected;

    /**
     * @export {boolean}
     * Export because it's used in unit test but didn't get renamed.
     * */
    this.isDisabled;
  }
}

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
       * The list of available thresholds.
       * @type {!Array<string>}
       */
      thresholds_: {type: Array, computed: 'computeThresholds_(data, metric)'},

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
       * The one slice that will be used as the baseline.
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
       * All available slices, including baseline. The user will select
       * necessary slices from this list to plot.
       * @type {!Array<string>}
       */
      slices: {type: Array},

      /**
       * The name of the metric, e.g. post_export_metrics/false_negative_rate
       * @type {string}
       */
      metric: {type: String},

      /**
       * The slices candidates in drop down menu.
       * @private {!Array<!SlicesDropDownMenuCandidateType>}
       */
      slicesDropDownMenuCandidates_: {
        type: Array,
        computed: 'computeSlicesDropDownCandidates_(slices)',

      },

      /**
       * The selected slices candidates in drop down menu.
       * @private {!Array<!SlicesDropDownMenuCandidateType>}
       */
      selectedSlicesDropDownMenuCandidates_: {
        type: Array,
      },

      /**
       * This field will be true if the max number of slices have been selected.
       * @private {boolean}
       */
      slicesDropDownMenuDisabled_: {
        type: Boolean,
        value: false,
        observer: 'slicesDropDownMenuDisabledChanged_'
      },

      /**
       * The slices to plot, the baseline is not included.
       * @private {!Array<string>}
       */
      slicesToPlot_: {
        type: Array,
        computed:
            'computeSlicesToPlot_(selectedSlicesDropDownMenuCandidates_.*, baseline)'
      },


      /**
       * The list of full metric names. For regular metrics, it should be an
       * array containing only the property metric. For thresholded metrics, it
       * would be "metric@threshold" for all thresholds.
       * @private {!Array<string>}
       */
      metrics_: {
        type: Array,
        computed:
            'computeMetrics_(data, baseline, metric, thresholds_, selectedThresholds_.*)'
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
      sort_: {type: String, value: 'Slice'},
    };
  }

  static get observers() {
    return [
      'initializeSelectedThresholds_(thresholds_.*)',
      'selectDefaultSlicesFromDropDownMenuCandidates_(slicesDropDownMenuCandidates_, baseline)',
    ];
  }

  /**
   *  @override
   */
  ready() {
    super.ready();

    /**
     * Number of max slices to be selected.
     * @export {number}
     * @const
     */
    this.MAX_NUM_SLICES = 15;
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
    return Array.from(thresholds).sort();
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
   * Generate the candidate for slices drop down menu.
   * @param {!Array<string>} slices
   * @return {!Array<!SlicesDropDownMenuCandidateType>}
   * @private
   */
  computeSlicesDropDownCandidates_(slices) {
    // First, group the slices by their slice key.
    // For example: gender:male, gender:female, will be grouped together under
    // 'gender' key.
    const slicesGroupedByKey = new Map();
    for (const slice of slices) {
      // Skip "Overall" slice.
      if (slice == OVERALL_SLICE_KEY) {
        continue;
      }
      const sliceKey = slice.split(':')[0];
      if (slicesGroupedByKey.has(sliceKey)) {
        slicesGroupedByKey.get(sliceKey).push(slice);
      } else {
        slicesGroupedByKey.set(sliceKey, [slice]);
      }
    }

    const candidates = [];
    let previousSliceKey = '';
    let id = 0;
    for (const slice of slices) {
      // Special process for "Overall" slice.
      if (slice == OVERALL_SLICE_KEY) {
        previousSliceKey = OVERALL_SLICE_KEY;
        candidates.push({
          text: OVERALL_SLICE_KEY,
          rawSlice: OVERALL_SLICE_KEY,
          isSliceKey: true,
          representedSlices: [OVERALL_SLICE_KEY],
          isSelected: false,
          isDisabled: false,
          id: id++,
        });
      } else {
        // Add slice key element to the candidates list.
        const currentSliceKey = slice.split(':')[0];
        if (currentSliceKey != previousSliceKey) {
          // Add slice key.
          candidates.push({
            text: currentSliceKey,
            rawSlice: '',
            isSliceKey: true,
            representedSlices: slicesGroupedByKey.get(currentSliceKey),
            isSelected: false,
            isDisabled: false,
            id: id++,
          });
          previousSliceKey = currentSliceKey;
        }
        // Add slice value to the candidates list.
        candidates.push({
          text: slice.split(':')[1],
          rawSlice: slice,
          isSliceKey: false,
          representedSlices: [slice],
          isSelected: false,
          isDisabled: false,
          id: id++,
        });
      }
    }

    return candidates;
  }

  /**
   * Compute the CSS class for the slices candidates.
   * @param {!SlicesDropDownMenuCandidateType} sliceCandidate
   * @return {string} css class based if slice is a key or value.
   * @private
   */
  slicesDropDownCandidatesClass_(sliceCandidate) {
    if (sliceCandidate.isSliceKey) {
      return 'slice-key-true';
    } else {
      return 'slice-key-false';
    }
  }

  /**
   * Initially select up to default number of slices. If it's eval comparison, 3
   * slices will be selected. Otherwise, 9 slices will be selected.
   * @param {!Array<!Object>} slicesDropDownMenuCandidates
   * @param {string} baseline
   * @private
   */
  selectDefaultSlicesFromDropDownMenuCandidates_(
      slicesDropDownMenuCandidates, baseline) {
    if (!slicesDropDownMenuCandidates || !baseline ||
        this.selectedSlicesDropDownMenuCandidates_ == undefined) {
      return;
    }

    // This function maybe triggered if baseline changes. But we only select the
    // default slices once at very beginning. Later if baseline changes, the
    // selected slices won't change.
    if (this.selectedSlicesDropDownMenuCandidates_.length == 0) {
      // Show up to DEFAULT_NUM_SLICES slices (plus baseline) by default.
      let numSlices = this.hasEvalComparison_() ?
          DEFAULT_NUM_SLICES_EVAL_COMPARE :
          DEFAULT_NUM_SLICES;
      for (let dropDownCandidatesPointer = 0; numSlices > 0 &&
           dropDownCandidatesPointer < slicesDropDownMenuCandidates.length;
           dropDownCandidatesPointer++) {
        if (!slicesDropDownMenuCandidates[dropDownCandidatesPointer]
                 .isSliceKey ||
            slicesDropDownMenuCandidates[dropDownCandidatesPointer].text ==
                OVERALL_SLICE_KEY) {
          // Process 'Overall' slice candidate or non slice key candidate.
          numSlices--;
          this.push(
              'selectedSlicesDropDownMenuCandidates_',
              slicesDropDownMenuCandidates[dropDownCandidatesPointer]);
        } else if (
            slicesDropDownMenuCandidates[dropDownCandidatesPointer]
                .representedSlices.length < numSlices) {
          // Process the slice key candidate.
          // Only select this slice key if the number of slice value under this
          // key is less than numSlices, because if a slice key is selected, all
          // the slice value under this slice key will be selected
          // automatically.
          numSlices -= slicesDropDownMenuCandidates[dropDownCandidatesPointer]
                           .representedSlices.length;
          this.push(
              'selectedSlicesDropDownMenuCandidates_',
              slicesDropDownMenuCandidates[dropDownCandidatesPointer]);
          dropDownCandidatesPointer +=
              slicesDropDownMenuCandidates[dropDownCandidatesPointer]
                  .representedSlices.length;
        }
      }
    }

    // Make sure the baseline is selected.
    const isBaselineSlice = (sliceCandidate) => {
      return sliceCandidate.rawSlice == baseline;
    };
    const index =
        this.selectedSlicesDropDownMenuCandidates_.findIndex(isBaselineSlice);
    if (index == -1) {
      this.push(
          'selectedSlicesDropDownMenuCandidates_',
          slicesDropDownMenuCandidates.find(isBaselineSlice));
    }
  }

  /**
   * Updates the isSelected status to true.
   * @private
   */
  slicesDropDownCandidatesSelected_(event) {
    const selectedItem = event.detail.item.slice;
    const selectedItemIndex = event.target.indexOf(event.detail.item);
    this.set(
        'slicesDropDownMenuCandidates_.' + selectedItemIndex + '.isSelected',
        true);
    this.disableOrUndisableSlicesDropDownMenu_();
    // If a slice key is selected, select all the slices under this slice key.
    if (selectedItem.isSliceKey && selectedItem.text != OVERALL_SLICE_KEY) {
      for (let i = selectedItemIndex + 1;
           i <= selectedItemIndex + selectedItem.representedSlices.length;
           i++) {
        let underSliceCandidate = this.slicesDropDownMenuCandidates_[i];
        // Stop early if no more slices can be selected.
        if (this.slicesDropDownMenuDisabled_) {
          break;
        }
        if (!underSliceCandidate.isSelected) {
          this.push(
              'selectedSlicesDropDownMenuCandidates_', underSliceCandidate);
        }
      }
    }
  }

  /**
   * Updates the isSelected status to false.
   * @private
   */
  slicesDropDownCandidatesUnselected_(event) {
    let selectedItem = event.detail.item.slice;
    let selectedItemIndex = this.slicesDropDownMenuCandidates_.findIndex(
        item => item.id == selectedItem.id);
    this.set(
        'slicesDropDownMenuCandidates_.' + selectedItemIndex + '.isSelected',
        false);
    this.disableOrUndisableSlicesDropDownMenu_();
    // If a slice key is unselected, unselect all the slices under this slice
    // key.
    if (selectedItem.isSliceKey && selectedItem.text != OVERALL_SLICE_KEY) {
      for (let i = selectedItemIndex + 1;
           i <= selectedItemIndex + selectedItem.representedSlices.length;
           i++) {
        const underSliceCandidate = this.slicesDropDownMenuCandidates_[i];
        const index = this.selectedSlicesDropDownMenuCandidates_.findIndex(
            candidate => candidate.id == underSliceCandidate.id);
        if (index != -1) {
          // Remove from selectedSlicesDropDownMenuCandidates_.
          this.splice('selectedSlicesDropDownMenuCandidates_', index, 1);
        }
      }
    }
  }

  /**
   * Disable slices drop down menu if max number of slices are selected. Or
   * Undiabled it if less number of slices are selected. changed.
   * @private
   */
  disableOrUndisableSlicesDropDownMenu_() {
    const numOfSelectedSlices =
        this.selectedSlicesDropDownMenuCandidates_
            .filter(
                sliceCandidate =>
                    sliceCandidate.isSelected && !sliceCandidate.isSliceKey)
            .length;

    // Disable the slices dropdown menu if max number of slices is selected.
    this.slicesDropDownMenuDisabled_ =
        numOfSelectedSlices >= this.MAX_NUM_SLICES;
  }

  /**
   * Update the isDisable status of the slices drop down menu candidates.
   * @param {boolean} newValue
   * @param {boolean} oldValue
   * @private
   */
  slicesDropDownMenuDisabledChanged_(newValue, oldValue) {
    if (newValue === oldValue || !this.slicesDropDownMenuCandidates_) {
      return;
    }
    for (let i = 0; i < this.slicesDropDownMenuCandidates_.length; i++) {
      if (!this.slicesDropDownMenuCandidates_[i].isSelected) {
        this.set(
            'slicesDropDownMenuCandidates_.' + i + '.isDisabled', newValue);
      }
    }
  }



  /**
   * Update the slicesToPlot_ if selectedSlicesDropDownMenuCandidates_
   * changed.
   * @param {!Object} selectedSlicesDropDownMenuCandidatesEvent
   * @param {string} baseline
   * @return {!Array<string>} The array of slices. The baseline is not
   *     included.
   * @private
   */
  computeSlicesToPlot_(selectedSlicesDropDownMenuCandidatesEvent, baseline) {
    if (!selectedSlicesDropDownMenuCandidatesEvent.base || !baseline) {
      return [];
    }
    return selectedSlicesDropDownMenuCandidatesEvent
        .base
        // Remove basline.
        .filter(
            sliceCandidate =>
                sliceCandidate && sliceCandidate.rawSlice != baseline)
        // Remove slice key element unless it's OVERALL.
        .filter(
            sliceCandidate => !sliceCandidate.isSliceKey ||
                (sliceCandidate.isSliceKey &&
                 sliceCandidate.text == OVERALL_SLICE_KEY &&
                 sliceCandidate.rawSlice == OVERALL_SLICE_KEY))
        // Map from slice candidates to slice.
        .map(sliceCandidate => sliceCandidate.rawSlice);
  }

  /**
   * @param {!Array<!Object>} data
   * @param {string} baseline
   * @param {string} metric
   * @param {!Array<string>} thresholds
   * @param {!Object} selectedThresholdsEvent
   * @return {!Array<string>} The list of full metric names to be
   *     plotted.
   * @private
   */
  computeMetrics_(data, baseline, metric, thresholds, selectedThresholdsEvent) {
    if (!data || !baseline || thresholds === undefined ||
        (this.isMetricThresholded_(thresholds) &&
         !selectedThresholdsEvent.base)) {
      return [];
    }

    if (this.isMetricThresholded_(thresholds)) {
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
   * Computes the ratio of difference for all metrics between each slice and
   * the baseline.
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
   * @param {!Array<string>} slicesToPlot
   * @param {!Object<!Object<number>>} diffRatios
   * @return {!Array|undefined}
   * @private
   */
  computeTableData_(baseline, data, metrics, slicesToPlot, diffRatios) {
    if (!baseline || !data || !metrics || !slicesToPlot || !diffRatios ||
        !Object.keys(diffRatios).length) {
      return undefined;
    }

    try {
      const slicesInTable = [baseline, ...slicesToPlot];
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
   * @param {!Array<string>} slicesToPlot
   * @return {!Array|undefined}
   * @private
   */
  computeExampleCounts_(baseline, data, slicesToPlot) {
    if (!baseline || !data || !slicesToPlot) {
      return undefined;
    }
    const slicesInTable = [baseline, ...slicesToPlot];
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
   * @param {!Array<string>} thresholds
   * @return {boolean} Returns true if metric is thresholded.
   * @private
   */
  isMetricThresholded_(thresholds) {
    return thresholds && thresholds.length > 0;
  }

  /**
   * @return {boolean} Returns true if evals are being compared.
   * @private
   */
  hasEvalComparison_() {
    return this.dataCompare && this.dataCompare.length > 0;
  }

  /**
   * Open Slices drop down memu.
   * @private
   */
  openSlicesDropDownMenu_() {
    this.$['SlicesDropDownMenu'].open();
  }

  /**
   * Display the info dialog.
   * @param {!Object} event
   * @private
   */
  openInfoDialog_(event) {
    event.stopPropagation();
    const dialog = event.target.parentElement.querySelector('paper-dialog');
    dialog.open();
  }

  /**
   * Strip prefix from metric name.
   * @param {string} metric
   * @return {string}
   */
  stripPrefix(metric) {
    return Util.removeMetricNamePrefix(metric);
  }
}

customElements.define('fairness-metric-summary', FairnessMetricSummary);
