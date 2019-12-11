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
import {template} from './fairness-nb-container-template.html.js';

import {SelectEventMixin} from '../../../../frontend/tfma-nb-event-mixin/tfma-nb-event-mixin.js';

import '@polymer/paper-card/paper-card.js';
import '@polymer/iron-flex-layout/iron-flex-layout.js';
import '../fairness-metrics-board/fairness-metrics-board.js';
import '../fairness-metric-and-slice-selector/fairness-metric-and-slice-selector.js';


/**
 * The list of fainress metrics. The order is used to determine which
 * metrics is displayed by default.
 * @const
 * @private
 */
const FAIRNESS_METRICS_ = [
  'false_negative_rate',
  'false_positive_rate',
  'true_positive_rate',
  'true_negative_rate',
  'positive_rate',
  'negative_rate',
];

/**
 * The prefix for fairness metric that will be applied under multihead
 * model.
 * @private {string}
 */
const MULTIHEAD_METRIC_PREFIX_ = 'post_export_metrics';

/**
 * @extends HTMLElement
 * @polymer
 */
export class FairnessNbContainer extends SelectEventMixin
(PolymerElement) {
  constructor() {
    super();
  }

  static get is() {
    return 'fairness-nb-container';
  }

  /** @return {!HTMLTemplateElement} */
  static get template() {
    return template;
  }


  /** @return {!PolymerElementProperties} */
  static get properties() {
    return {
      /**
       * The slicing metrics evaluation result. It's a list of dict with key
       * "slice" and "metrics". For example:
       * [
       *   {
       *     "slice":"Overall",
       *     "sliceValue": "Overall"
       *     "metrics": {
       *       "auc": {
       *         "doubleValue": 0.6
       *       }
       *     }
       *   }, {
       *     "slice":"feature:1",
       *     "sliceValue":"1",
       *     "metrics": {
       *       "auc": {
       *         "doubleValue": 0.6
       *       }
       *     }
       *   }
       * ]
       * @type {!Array<!Object>}
       */
      slicingMetrics: {type: Array, observer: 'slicingMetricsChanged_'},

      /**
       * The list of run numbers that's available to select.
       * @type {!Array<string>}
       */
      availableEvaluationRuns: {type: Array, value: []},

      /**
       * The full names of metrics available. eg: auc, negative_rate@0.25 or
       * post_export_metrics/head_1/negative_rate@0.25.
       * @private {!Array<string>|undefined}
       */
      availableMetricsNames_: {
        type: Array,
      },

      /**
       * A set containing metrics that are thresholded.
       * @private {!Object}
       */
      thresholdedMetrics_: {type: Set},

      /**
       * The short names of metrics available. eg: auc, negative_rate or
       * post_export_metrics/head_1/negative_rate.
       * @private {!Array<string>}
       */
      selectableMetrics_: {type: Array},

      /**
       * The thresholds at which fairness metrics are computed.
       * @private {!Array<string>}
       */
      fairnessThresholds_: {type: Array},

      /** @private {!Array<string>} */
      selectedMetrics_: {
        type: Array,
      },

      /** @type {string} */
      weightColumn: {type: String, value: 'totalWeightedExamples'},

      /** @type {string} */
      selectedEvaluationRun: {type: String, notify: true},
    };
  }

  /**
   * @param {!Array<!Object>} slicingMetrics
   * @return {undefined}
   * @private
   */
  slicingMetricsChanged_(slicingMetrics) {
    if (slicingMetrics) {
      tfma.Data.flattenMetrics(slicingMetrics, 'metrics');
    }
    this.availableMetricsNames_ =
        this.computeAvailableMetricsNames_(slicingMetrics);
    this.updateSelectableMetricsAndThresholds_(this.availableMetricsNames_);
  }

  /**
   * Extracts short fairness metric name if applicable; null. otherwise.
   * @param {string} metricName
   * @return {?Object} An object containing fairness metric name and
   *     threshold or null if the named metric is not a  fairness metric.
   * @private
   */
  extractFairnessMetric_(metricName) {
    for (let i = 0; i < FAIRNESS_METRICS_.length; i++) {
      const parts = metricName.split(FAIRNESS_METRICS_[i] + '@');
      const prefix = parts[0];
      if (parts.length == 2 &&
          (prefix == '' || prefix.indexOf(MULTIHEAD_METRIC_PREFIX_) == 0)) {
        return {name: parts[0] + FAIRNESS_METRICS_[i], threshold: parts[1]};
      }
    }
    return null;
  }

  /**
   * @param {!Array<!Object>} slicingMetrics
   * @return {!Array<string>|undefined} An array of names of all metrics
   *     suitable for the fairness view.
   * @private
   */
  computeAvailableMetricsNames_(slicingMetrics) {
    if (!slicingMetrics) {
      return [];
    }
    const allMetrics = new Set();
    slicingMetrics.forEach(slicingMetric => {
      Object.keys(slicingMetric['metrics']).forEach(metricName => {
        allMetrics.add(metricName);
      });
    });
    // Only support numeric value and bounded value metrics.
    const isSupportedMetricFormat = (entry, metricName) => {
      const value = entry[metricName];

      const isDef = v => v !== undefined;
      const isNumber = v => typeof v === 'number';
      return isDef(value) &&
          (isNumber(value) || tfma.CellRenderer.isBoundedValue(value) ||
           tfma.CellRenderer.isRatioValue(value));
    };
    return [...allMetrics].filter(
        metric => this.extractFairnessMetric_(metric) ||
            isSupportedMetricFormat(slicingMetrics[0]['metrics'], metric));
  }

  /**
   * Updates selectable metrics and available thresholds from available
   * metrics.
   * @param {!Array<string>|undefined} availableMetricsNames_
   * @private
   */
  updateSelectableMetricsAndThresholds_(availableMetricsNames_) {
    this.thresholdedMetrics_ = new Set();
    const otherMetrics = new Set();
    const thresholds = new Set();
    availableMetricsNames_.forEach(metricName => {
      const fairnessMetric = this.extractFairnessMetric_(metricName);
      if (fairnessMetric) {
        thresholds.add(fairnessMetric.threshold);
        this.thresholdedMetrics_.add(fairnessMetric.name);
      } else {
        otherMetrics.add(metricName);
      }
    });

    const setToArray = (s) => Array.from(s.entries()).map(entry => entry[0]);
    this.selectableMetrics_ = [
      ...setToArray(this.thresholdedMetrics_)
          .sort((a, b) => a.localeCompare(b)),
      ...setToArray(otherMetrics).sort((a, b) => a.localeCompare(b))
    ];
    this.fairnessThresholds_ = setToArray(thresholds).sort();
  }
};

customElements.define('fairness-nb-container', FairnessNbContainer);
