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
(() => {

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

  Polymer({

    is: 'tfma-graph-data-filter',

    properties: {
      /**
       * Input data to the component.
       * @type {!tfma.GraphData}
       */
      data: {
        type: Object,
        observer: 'dataChanged_'
      },

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
        observer: 'chartTypeChanged_'
      },

      /**
       * Selected and highlighted features.
       * @type {!Array<string>}
       */
      selectedFeatures: {
        type: Array,
        value: function() {
          return [];
        }
      },

      /**
       * The current dataset that is being shown in the graphs to be paired
       * with a table view. This data is updated every time the user changes
       * the type of graph being displayed and every time the filtered data
       * is updated.
       * @type {!tfma.TableProvider}
       */
      tableData: {
        type: Object,
        notify: true,
        computed:
            'computeTableData_(chartType, filteredData_, focusedData_)',
      },

      /**
       * Data filtered by number of weighted examples.
       * @private {!tfma.Data}
       */
      filteredData_: {
        type: Object,
        computed:
            'computeFilteredData_(data, weightedExamplesColumn, ' +
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
    },

    /** @override */
    ready: function() {
      // Initialize UI control.
      this.initWeightedExamplesThreshold_();
    },

    /**
     * Initializes the input/slider handler for weighted examples threshold.
     * Since adjusting weighted examples threshold is a time costly operation,
     * we do not use two-way data binding here to update the threshold.
     * @private
     */
    initWeightedExamplesThreshold_: function() {
      const container = this.$[ElementId.WEIGHTED_EXAMPLES_THRESHOLD];
      const input = container.getElementsByTagName('input')[0];
      const slider = container.getElementsByTagName('paper-slider')[0];

      input.addEventListener(tfma.Event.KEYUP, (e) => {
        const val = +e.target.value;
        slider.setAttribute('value', val);
        this.weightedExamplesThreshold_ = val;
      });

      slider.addEventListener(
          tfma.Event.IMMEDIATE_VALUE_CHANGE, (e) => {
            const val = +e.target.getElementsByTagName('paper-progress')[0]
                             .getAttribute('value');
            input.value = val;
          });

      slider.addEventListener(tfma.Event.CHANGE, (e) => {
        const val =
            +e.target.getElementsByTagName('paper-progress')[0].getAttribute(
                'value');
        this.weightedExamplesThreshold_ = val;
      });
    },

    /**
     * Computes the filtered data based on the weighted examples threshold.
     * @param {!tfma.GraphData} data
     * @param {string} weightedExamplesColumn
     * @param {number} weightedExamplesThreshold
     * @return {tfma.Data}
     * @private
     */
    computeFilteredData_: function(
        data, weightedExamplesColumn, weightedExamplesThreshold) {
      return data.applyThreshold(
          weightedExamplesColumn, weightedExamplesThreshold);
    },

    /**
     * Computes the range of number of weighted examples for a filtered data
     * table.
     * @param {!tfma.GraphData} data
     * @param {string} weightedExamplesColumn
     * @return {{max: number, step: number}}
     *     max: The max value of the weighted examples threshold slider, that is
     *     the smallest multiple of the slider step that is larger than the max
     *     of weighted examples. step: The step the slider takes.
     * @private
     */
    computeWeightedExamplesInfo_: function(data, weightedExamplesColumn) {
      return data.getColumnSteppingInfo(weightedExamplesColumn);
    },

    /**
     * Updates the chart type based on the new graph data, and the slice count
     * cut off property, so the correct chart is displayed.
     * @private
     */
    dataChanged_: function(data) {
      this.chartType = data &&
              data.getFeatures().length >
                  SLICE_OVERVIEW_SLICE_COUNT_CUT_OFF ?
          ChartType.METRICS_HISTOGRAM :
          ChartType.SLICE_OVERVIEW;
    },

    /**
     * When the filteredData_ changes because of a new weighted examples
     * threshold, we reset the focus range of the visualization, as
     * visualization would auto zoom to the range of the filteredData_. Not
     * resting the focus range would cause some confusion.
     * @private
     */
    filteredDataChanged_: function() {
      this.$[ElementId.HISTOGRAM].updateFocusRange(0, 1);
    },

    /**
     * Observer for property chartType.
     * @param {string} type
     * @private
     */
    chartTypeChanged_: function(type) {
      this.querySelector('tfma-slice-overview').displayed =
          type == ChartType.SLICE_OVERVIEW;
    },

    /**
     * @param {number} chartType The value is defined in enum ChartType.
     * @param {!tfma.Data} filteredData
     * @param {!tfma.Data} focusedData
     * @return {!tfma.TableProvider} The tfma.TableProvider to use for
     *     backing the metrics-table.
     */
    computeTableData_: function(chartType, filteredData, focusedData) {
      const displayedData = (chartType == ChartType.SLICE_OVERVIEW ?
          filteredData : focusedData);
      return /** @type {!tfma.TableProvider} */ (this.data
          .getTableDataFromDataset(displayedData));
    },
  });
})();
