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
import {PolymerElement} from '@polymer/polymer/polymer-element.js';
import {template} from './fairness-metric-and-slice-selector-template.html.js';

import '@polymer/paper-checkbox/paper-checkbox.js';
import '@polymer/paper-item/paper-item.js';
import '@polymer/paper-listbox/paper-listbox.js';

const MULTIHEAD_METRIC_PREFIX_ = 'post_export_metrics/';


export class FairnessMetricAndSliceSelector extends PolymerElement {
  constructor() {
    super();
  }

  static get is() {
    return 'fairness-metric-and-slice-selector';
  }

  /** @return {!HTMLTemplateElement} */
  static get template() {
    return template;
  }

  /** @return {!PolymerElementProperties} */
  static get properties() {
    return {
      /**
       * A list string of all available metrics.
       * @type {!Array<string>}
       */
      availableMetrics: {type: Array, observer: 'availableMetricsChanged_'},

      /**
       * A list of metrics selected.
       * @type {!Array<string>}
       */
      selectedMetrics: {
        type: Array,
        notify: true,
      },

      /**
       * A list of objects which indicate if metrics have been selected.
       * @private {!Array<!Object>}
       */
      metricsSelectedStatus_: {
        type: Array,
        computed:
            'computedMetricsSelectedStatus_(availableMetrics, selectedMetrics.length)',
      },
    };
  }


  /**
   * Init the defult seleted metrics.
   * @param {!Array<string>} availableMetrics
   * @private
   */
  availableMetricsChanged_(availableMetrics) {
    if (availableMetrics) {
      this.selectedMetrics = availableMetrics.slice(0, 1);
    } else {
      this.selectedMetrics = [];
    }
  }


  /**
   * Generate a list of object to record the metrics select status.
   * @param {!Array<string>} availableMetrics
   * @param {number} length
   * @return {!Array<!Object>}
   * @private
   */
  computedMetricsSelectedStatus_(availableMetrics, length) {
    let status = [];
    if (!availableMetrics) {
      return status;
    }
    availableMetrics.forEach((metricsName, idx) => {
      if (metricsName.startsWith(MULTIHEAD_METRIC_PREFIX_)) {
        metricsName = metricsName.slice(MULTIHEAD_METRIC_PREFIX_.length);
      }
      status.push({
        'metricsName': metricsName,
        'selected':
            this.selectedMetrics && this.selectedMetrics.includes(metricsName)
      });
    });
    return status;
  }
}

customElements.define(
    'fairness-metric-and-slice-selector', FairnessMetricAndSliceSelector);
