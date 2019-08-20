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
import {template} from './tfma-time-series-browser-template.html.js';

import '../tfma-line-chart-grid/tfma-line-chart-grid.js';
import '../tfma-metrics-table/tfma-metrics-table.js';

/**
 * tfma-line-chart-grid renders the time series plot for a number of metrics in
 * a grid layout.
 *
 * @polymer
 */
export class TimeSeriesBrowser extends PolymerElement {
  constructor() {
    super();
  }

  static get is() {
    return 'tfma-time-series-browser';
  }

  /** @return {!HTMLTemplateElement} */
  static get template() {
    return template;
  }

  /** @return {!PolymerElementProperties} */
  static get properties() {
    return {
      /**
       * Series data being visualized, which will be passed to the
       * linechart-grid and the metrics-table.
       * @type {!tfma.SeriesData}
       */
      seriesData: {type: Object},

      /**
       * A comma separated string of metrics to skip.
       * @type {string}
       */
      blacklist: {
        type: String,
        value: '',
      },

      /**
       * Metrics available in the seriesData.
       * @private {!Array<string>}
       */
      metrics_: {type: Array, computed: 'computeMetrics_(seriesData)'},

      /**
       * An object that specifies desired format overrides.
       * @type {!Object}
       */
      formats: {
        type: Object,
        value: () => {
          return {};
        }
      },

      /**
       * An object containing the formats that will be used to render the
       * metrics table.
       * @type {!Object}
       */
      metricFormats_:
          {type: Object, computed: 'computeMetricFormats_(formats)'},
    };
  }


  /**
   * Initializes event listeners.
   * @override
   */
  ready() {
    super.ready();

    const grid = this.$.grid;
    const table = this.$.table;

    grid.addEventListener(tfma.Event.SELECT, (e) => {
      if (e.target == grid) {
        table.highlight(e.detail);
      }
    });
    grid.addEventListener(tfma.Event.CLEAR_SELECTION, () => {
      table.highlight(null);
    });

    table.addEventListener(tfma.Event.SELECT, (e) => {
      grid.highlight(e.detail);
    });
    table.addEventListener(tfma.Event.CLEAR_SELECTION, () => {
      grid.highlight(null);
    });
  }

  /**
   * Computes the metrics available in all models.
   * @param {!tfma.SeriesData|undefined} seriesData
   * @return {!Array<string>}
   * @private
   */
  computeMetrics_(seriesData) {
    return seriesData ? seriesData.getMetrics() : [];
  }

  /**
   * Appends necessary format override to the formats specified by the client.
   * @param {!Object} formats
   * @return {!Object}
   * @private
   */
  computeMetricFormats_(formats) {
    const desiredFormats = Object.assign({}, formats);
    desiredFormats[tfma.Column.TOTAL_EXAMPLE_COUNT] = {
      type: tfma.MetricValueFormat.INT64
    };
    return desiredFormats;
  }
}

customElements.define('tfma-time-series-browser', TimeSeriesBrowser);
