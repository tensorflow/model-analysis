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

import {PolymerElement} from '@polymer/polymer/polymer-element.js';

import {template} from './tfma-graph-data-filter-template.html.js';

import '@polymer/paper-dropdown-menu/paper-dropdown-menu.js';

import '@polymer/iron-pages/iron-pages.js';
import '@polymer/paper-input/paper-input.js';
import '@polymer/paper-item/paper-item.js';
import '@polymer/paper-listbox/paper-listbox.js';
import '@polymer/paper-slider/paper-slider.js';
import '../tfma-metrics-histogram/tfma-metrics-histogram.js';
import '../tfma-slice-overview/tfma-slice-overview.js';

/** @enum {string} */
const ElementId = {
  HISTOGRAM: 'histogram',
  CHART_TYPE: 'chart-type',
  WEIGHTED_EXAMPLES_THRESHOLD: 'weighted-examples-threshold'
};

/** @enum {number} */
const ChartType = {
  SLICE_OVERVIEW: 0,
  METRICS_HISTOGRAM: 1,
};

/**
 * The cutoff which determines whetehr to use slice overview or metrics
 * histogram by default. If the number of slices falls under this cutoff,
 * slice overview will be used.
 * @type {number}
 */
const SLICE_OVERVIEW_SLICE_COUNT_CUT_OFF = 50;

/**
 * tfma-graph-data-filter provides a GUI for the user to filter data.
 *
 * @polymer
 */
export class GraphDataFilter extends PolymerElement {
  constructor() {
    super();
  }

  static get is() {
    return 'tfma-graph-data-filter';
  }

  /** @return {!HTMLTemplateElement} */
  static get template() {
    return template;
  }

  /** @return {!PolymerElementProperties} */
  static get properties() {
    return {
      /**
       * Input data to the component.
       * @type {!tfma.GraphData}
       */
      data: {type: Object, observer: 'dataChanged_'},

      /**
       * Name of the weighted examples column.
       * @type {string}
       */
      weightedExamplesColumn: {type: String, value: ''},

      /**
       * Slices with number of weighted examples fewer than this threshold will
       * be filtered out from the visualization.
       * @private {number}
       */
      weightedExamplesThreshold_: {type: Number, value: 0},

      /**
       * Lantern browser chart type. It should take on values defined in enum
       * ChartType.
       * @type {number}
       */
      chartType: {
        type: String,
        value: ChartType.METRICS_HISTOGRAM,
      },

      /**
       * Selected and highlighted features.
       * @type {!Array<string>}
       */
      selectedFeatures: {
        type: Array,
        value() {
          return [];
        }
      },

      /**
       * The current dataset that is being shown in the graphs to be paired
       * with a table view. This data is updated every time the user changes
       * the type of graph being displayed and every time the filtered data
       * is updated.
       * @type {!tfma.TableProviderExt}
       */
      tableData: {
        type: Object,
        notify: true,
        computed: 'computeTableData_(chartType, filteredData_, focusedData_)',
      },

      /**
       * Data filtered by number of weighted examples.
       * @private {!tfma.Data}
       */
      filteredData_: {
        type: Object,
        computed: 'computeFilteredData_(data, weightedExamplesColumn, ' +
            'weightedExamplesThreshold_)',
        observer: 'filteredDataChanged_'
      },

      /**
       * A subset of filteredData_ focused by the visualization.
       * @private {!tfma.Data}
       */
      focusedData_: {type: Object},

      /**
       * Range and step of weighted examples.
       * @private {{max: number, step: number}}
       */
      weightedExamplesInfo_: {
        type: Object,
        computed: 'computeWeightedExamplesInfo_(data, ' +
            'weightedExamplesColumn)'
      },

      /**
       * @private {boolean}
       */
      showSliceOverview_: {
        type: Boolean,
        computed: 'computeShowSliceOverview_(chartType)',
      },
    };
  }

  /** @override */
  ready() {
    super.ready();

    // Initialize UI control.
    this.initWeightedExamplesThreshold_();
  }

  /**
   * Initializes the input/slider handler for weighted examples threshold.
   * Since adjusting weighted examples threshold is a time costly operation,
   * we do not use two-way data binding here to update the threshold.
   * @private
   */
  initWeightedExamplesThreshold_() {
    const container = this.$[ElementId.WEIGHTED_EXAMPLES_THRESHOLD];
    const input = container.getElementsByTagName('input')[0];
    const slider = container.getElementsByTagName('paper-slider')[0];

    input.addEventListener(tfma.Event.CHANGE, (e) => {
      const val = +e.target.value;
      slider.setAttribute('value', val);
      this.weightedExamplesThreshold_ = val;
    });

    slider.addEventListener(tfma.Event.IMMEDIATE_VALUE_CHANGE, (e) => {
      const val = +e.target.shadowRoot.querySelector('paper-progress')
                       .getAttribute('value');
      input.value = val;
    });
  }

  /**
   * Computes the filtered data based on the weighted examples threshold.
   * @param {!tfma.GraphData} data
   * @param {string} weightedExamplesColumn
   * @param {number} weightedExamplesThreshold
   * @return {(!tfma.Data|undefined)}
   * @private
   */
  computeFilteredData_(
      data, weightedExamplesColumn, weightedExamplesThreshold) {
    if (!data || !weightedExamplesColumn) {
      return undefined;
    } else {
      return data.applyThreshold(
          weightedExamplesColumn, weightedExamplesThreshold);
    }
  }

  /**
   * Computes the range of number of weighted examples for a filtered data
   * table.
   * @param {!tfma.GraphData} data
   * @param {string} weightedExamplesColumn
   * @return {({max: number, step: number}|undefined)}
   *     max: The max value of the weighted examples threshold slider, that is
   *     the smallest multiple of the slider step that is larger than the max
   *     of weighted examples. step: The step the slider takes.
   * @private
   */
  computeWeightedExamplesInfo_(data, weightedExamplesColumn) {
    if (!data || !weightedExamplesColumn) {
      return undefined;
    } else {
      return data.getColumnSteppingInfo(weightedExamplesColumn);
    }
  }

  /**
   * Updates the chart type based on the new graph data, and the slice count
   * cut off property, so the correct chart is displayed.
   * @private
   */
  dataChanged_(data) {
    this.chartType =
        data && data.getFeatures().length > SLICE_OVERVIEW_SLICE_COUNT_CUT_OFF ?
        ChartType.METRICS_HISTOGRAM :
        ChartType.SLICE_OVERVIEW;
  }

  /**
   * When the filteredData_ changes because of a new weighted examples
   * threshold, we reset the focus range of the visualization, as
   * visualization would auto zoom to the range of the filteredData_. Not
   * resting the focus range would cause some confusion.
   * @private
   */
  filteredDataChanged_() {
    this.$[ElementId.HISTOGRAM].updateFocusRange(0, 1);
  }

  /**
   * @param {string} type
   * @return {boolean} Whether we should show slice overview.
   * @private
   */
  computeShowSliceOverview_(type) {
    return type == ChartType.SLICE_OVERVIEW;
  }

  /**
   * @param {number} chartType The value is defined in enum ChartType.
   * @param {!tfma.Data} filteredData
   * @param {!tfma.Data} focusedData
   * @return {(!tfma.TableProviderExt|undefined)} The tfma.TableProviderExt to
   *     use for backing the metrics-table.
   * @private
   */
  computeTableData_(chartType, filteredData, focusedData) {
    if (!filteredData || !focusedData) {
      return undefined;
    }
    const displayedData =
        (chartType == ChartType.SLICE_OVERVIEW ? filteredData : focusedData);
    return /** @type {!tfma.TableProviderExt} */ (
        this.data.getTableDataFromDataset(displayedData));
  }
}

customElements.define('tfma-graph-data-filter', GraphDataFilter);
