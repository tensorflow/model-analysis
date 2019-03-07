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
import {template} from './tfma-nb-slicing-metrics-template.html.js';

import '../tfma-slicing-metrics-browser/tfma-slicing-metrics-browser.js';

/**
 * tfma-nb-slicing-metrics provides a wrapper for tfma-slicing-metrics-browser
 * in the notebook environment. It performs the necessary data transformation.
 *
 * @polymer
 */
export class NotebookSlicingMetricsWrapper extends PolymerElement {
  constructor() {
    super();
  }

  static get is() {
    return 'tfma-nb-slicing-metrics';
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
       * @type {!Array<{slice: string, metrics:!Object}>}
       */
      data: {type: Array, observer: 'dataChanged_'},

      /**
       * A key value pair for the configuration.
       * @type {!Object}
       */
      config: {type: Object},

      /**
       * The data consumed by the slicing metrics browser.
       * @private {!Array<!Object>}
       */
      browserData_: {type: Array},

      /**
       * @private {!Array<string>}
       */
      metrics_: {type: Array},
    };
  }

  /**
   * @param {!Array<{slice: string, metrics:!Object}>} data
   * @private
   */
  dataChanged_(data) {
    // Note that tfma.Data.flattenMetrics modifies its input in place so we
    // compute the following in an observer instead of making them computed
    // properties.
    tfma.Data.flattenMetrics(data, 'metrics');
    this.metrics_ = tfma.Data.getAvailableMetrics([data], 'metrics');
    this.browserData_ = data;
  }

  getWeightedExampleColumn_(config) {
    return config['weightedExamplesColumn'];
  }
}

customElements.define('tfma-nb-slicing-metrics', NotebookSlicingMetricsWrapper);
