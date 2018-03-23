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
    TABLE: 'table'
  };

  Polymer({

    is: 'tfma-slicing-metrics-browser',

    properties: {
      /**
       * Input data to the component.
       * @type {!Array<!Object>}
       */
      data: {type: Array},

      /**
       * Data metrics list.
       * @type {!Array<string>}
       */
      metrics: {type: Array},

      /**
       * Graph data object to be used in the graph component.
       * @private {!tfma.SingleSeriesGraphData}
       */
      graphData_: {type: Object, computed: 'computeGraphData_(metrics, data)'},

      /**
       * Name of the weighted examples column.
       * @type {string}
       */
      weightedExamplesColumn: {type: String, value: ''},

      /**
       * Selected and highlighted features.
       * @type {!Array<string>}
       */
      selectedFeatures_: {
        type: Array,
        value: function() {
          return [];
        }
      },

      /**
       * Metric value formats specification. The key of the object is the metric
       * name, and the value is the format specification.
       * @private {!Object<tfma.MetricValueFormatSpec>}
       */
      metricFormats_: {
        type: Object,
        computed: 'computeMetricFormats_(metricsTableData_, ' +
            'weightedExamplesColumn, formats)'
      },

      /**
       * The formats override.
       * @type {!Object}
       */
      formats: {type: Object, value: {}},

      /**
       * The metrics table data.
       * @private {!Object}
       */
      metricsTableData_: {type: Object}
    },

    /** @override */
    ready: function() {
      // Initialize UI control.
      this.initPlotInteraction_();
    },

    /**
     * Initializes the plot interaction event listeners.
     * @private
     */
    initPlotInteraction_: function() {
      const table = this.$[ElementId.TABLE];
      table.addEventListener(tfma.Event.SELECT, (e) => {
        this.selectedFeatures_ = [e.detail['feature']];
      });
    },

    /**
     * Computes the graph data to be used in the graph component.
     * @param {!Array<string>} metrics
     * @param {!Array<!Object>} data
     * @return {!Object<tfma.SingleSeriesGraphData>}
     * @private
     */
    computeGraphData_: function(metrics, data) {
      return new tfma.SingleSeriesGraphData(metrics, data);
    },

    /**
     * Computes the metric formats that are passed to the metrics-table.
     * Sets the weighted examples column to have type INT.
     * Sets the links column (if any) to have type HTML.
     * @param {!tfma.Data} metricsTableData
     * @param {string} weightedExamplesColumn
     * @param {!Object} formatsOverride
     * @return {!Object<tfma.MetricValueFormatSpec>}
     * @private
     */
    computeMetricFormats_: function(
        metricsTableData, weightedExamplesColumn, formatsOverride) {
      const formats = {};
      formats[weightedExamplesColumn] = {type: tfma.MetricValueFormat.INT};
      formats[tfma.Column.TOTAL_EXAMPLE_COUNT] = {
        type: tfma.MetricValueFormat.INT64
      };

      // Apply other overrides.
      for (let override in formatsOverride) {
        formats[override] = formatsOverride[override];
      }
      return formats;
    },
  });

})();
