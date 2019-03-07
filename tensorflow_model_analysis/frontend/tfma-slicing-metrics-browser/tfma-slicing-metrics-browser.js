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
import {template} from './tfma-slicing-metrics-browser-template.html.js';

import '../tfma-graph-data-filter/tfma-graph-data-filter.js';
import '../tfma-metrics-table/tfma-metrics-table.js';

/** @enum {string} */
const ElementId = {
  TABLE: 'table'
};

/**
 * tfma-slicing-metrics-browser visualizes the computed metrics for each slice
 * and provides some filtering funcitonality.
 *
 * @polymer
 */
export class SlicingMetricsBrowser extends PolymerElement {
  constructor() {
    super();
  }

  static get is() {
    return 'tfma-slicing-metrics-browser';
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
        value() {
          return [];
        }
      },

      /**
       * Metric value formats specification. The key of the object is the metric
       * name, and the value is the format specification.
       * @private {!Object<!tfma.MetricValueFormatSpec>}
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
    };
  }

  /** @override */
  ready() {
    super.ready();

    // Initialize UI control.
    this.initPlotInteraction_();
  }

  /**
   * Initializes the plot interaction event listeners.
   * @private
   */
  initPlotInteraction_() {
    const table = this.$[ElementId.TABLE];
    table.addEventListener(tfma.Event.SELECT, (e) => {
      this.selectedFeatures_ = [e.detail['feature']];
    });
  }

  /**
   * Computes the graph data to be used in the graph component.
   * @param {(!Array<string>|undefined)} metrics
   * @param {(!Array<!Object>|undefined)} data
   * @return {(!tfma.SingleSeriesGraphData|undefined)}
   * @private
   */
  computeGraphData_(metrics, data) {
    if (!metrics || !data) {
      return undefined;
    } else {
      return new tfma.SingleSeriesGraphData(metrics, data);
    }
  }

  /**
   * Computes the metric formats that are passed to the metrics-table.
   * Sets the weighted examples column to have type INT.
   * Sets the links column (if any) to have type HTML.
   * @param {!tfma.Data} metricsTableData
   * @param {string} weightedExamplesColumn
   * @param {!Object} formatsOverride
   * @return {!Object<!tfma.MetricValueFormatSpec>}
   * @private
   */
  computeMetricFormats_(
      metricsTableData, weightedExamplesColumn, formatsOverride) {
    if (!metricsTableData || !weightedExamplesColumn) {
      return {};
    }
    const formats = {};
    formats[weightedExamplesColumn] = {type: tfma.MetricValueFormat.INT};
    formats[tfma.Column.TOTAL_EXAMPLE_COUNT] = {
      'type': tfma.MetricValueFormat.INT64
    };

    // Apply other overrides.
    for (let override in formatsOverride) {
      formats[override] = formatsOverride[override];
    }
    return formats;
  }
}

customElements.define('tfma-slicing-metrics-browser', SlicingMetricsBrowser);
