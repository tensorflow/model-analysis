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
import {template} from './tfma-nb-time-series-template.html.js';

import '../tfma-time-series-browser/tfma-time-series-browser.js';

/**
 *
 *
 * @polymer
 */
export class NotebookTimeSeriesWrapper extends PolymerElement {
  constructor() {
    super();
  }

  static get is() {
    return 'tfma-nb-time-series';
  }

  /** @return {!HTMLTemplateElement} */
  static get template() {
    return template;
  }

  /** @return {!PolymerElementProperties} */
  static get properties() {
    return {
      /**
       * A key value pair where the key is the name of the slice and the value
       * is the evaluation results for the slice.
       * @type {!Array<!Object>}
       */
      data: {type: Array},

      /**
       * A key value pair for the configuration.
       * @type {!Object}
       */
      config: {type: Object},

      /**
       * The data consumed by the time series browser.
       * @private {!tfma.SeriesData}
       */
      seriesData_: {type: Object},
    };
  }

  static get observers() {
    return ['refresh_(data, config)'];
  }

  /**
   * Refreshes the view by updating format override and the series data.
   * @param {!Array<!Object>} data
   * @param {!Object} config
   * @private
   */
  refresh_(data, config) {
    if (!data || !config) {
      return;
    }
    // Note that tfma.Data.flattenMetrics modifies its input in place so we compute the following in
    // an observer instead of making them computed properties.
    tfma.Data.flattenMetrics(data, 'metrics');
    const evalRuns = data.map(run => [{'metrics': run.metrics}]);
    const metricNames = tfma.Data.getAvailableMetrics(evalRuns, 'metrics');
    this.seriesData_ = new tfma.SeriesData(data.map(run => {
      return {
        'data': tfma.Data.build(metricNames, [{'metrics': run['metrics'], 'slice': ''}]),
        'config': run['config'],
      };
    }), config['isModelCentric']);
  }
}

customElements.define('tfma-nb-time-series', NotebookTimeSeriesWrapper);
