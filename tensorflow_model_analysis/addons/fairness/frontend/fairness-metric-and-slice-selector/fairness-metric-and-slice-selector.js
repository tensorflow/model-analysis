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

const Util = goog.require('tensorflow_model_analysis.addons.fairness.frontend.Util');

const METRIC_DEFINITIONS = {
    'false_negative_rate': 'The percentage of positive data points (as labeled in the ground truth) that are incorrectly classified as negative',
    'false_positive_rate': 'The percentage of negative data points (as labeled in the ground truth) that are incorrectly classified as positive',
    'negative_rate': 'The percentage of data points that are classified as negative, independent of ground truth',
    'positive_rate': 'The percentage of data points that are classified as positive, independent of ground truth',
    'true_negative_rate': 'The percentage of negative data points (as labeled in the ground truth) that are correctly classified as negative',
    'true_positive_rate': 'The percentage of positive data points (as labeled in the ground truth) that are correctly classified as positive',
    'false_discovery_rate': 'The percentage of negative data points (as labeled in the ground truth) that are incorrectly classified as positive out of all data points classified as positive',
    'false_omission_rate': 'The percentage of positive data points (as labeled in the ground truth) that are incorrectly classified as negative out of all data points classified as negative',
    'accuracy': 'The percentage of data points that are classified correctly',
    'accuracy_baseline': 'The percentage of data points that are classified correctly if the model always predicts one class',
    'auc': 'The area under the ROC curve',
    'auc_precision_recall': 'The area under the Precision-Recall curve',
    'average_loss': 'The mean loss per data point',
    'label/mean': 'The mean of all ground truth labels',
    'example_count': 'The number of examples processed',
    'precision': 'The percentage of positive data points (as labeled in the ground truth) that are correctly classified as positive out of all data points classified as positive',
    'prediction/mean': 'The mean of all predicted labels',
    'recall': 'The percentage of positive data points (as labeled in the ground truth) that are correctly classified as positive',
    'totalWeightedExamples': 'The number of weighted examples processed',
};

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
      selectedMetrics: {type: Array, notify: true},

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
   * Init the defult selected metrics.
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
      status.push({
        'metricsName': metricsName,
        'selected':
            this.selectedMetrics && this.selectedMetrics.includes(metricsName)
      });
    });
    return status;
  }

  /**
   * Strip prefix from metric name.
   * @param {string} metric
   * @return {string}
   */
  stripPrefix(metric) {
    return Util.removeMetricNamePrefix(metric);
  }

  /**
   * Get the defintion of a metric, or return the metric if not defined.
   * @param {string} metric
   * @return {string}
   */
  getDefinition(metric) {
    let strippedMetric = this.stripPrefix(metric);
    if (strippedMetric in METRIC_DEFINITIONS) {
      return METRIC_DEFINITIONS[strippedMetric];
    } else {
      return strippedMetric;
    }
  }

  /**
   * Handler listening to any change in "Select all" check box.
   */
  onSelectAllCheckedChanged_(event) {
    const checked = event.detail.value;
    if (checked) {
      this.selectedMetrics = this.availableMetrics.slice();
    } else {
      this.selectedMetrics = [];
    }
  }

  /**
   * Handler listening to any changes in selected item list.
   */
  onCheckedChanged_(event) {
    if (!this.availableMetrics) {
      return;
    }

    let selectedMetrics = this.selectedMetrics.slice();
    let availableMetrics = this.availableMetrics.slice();
    if (JSON.stringify(selectedMetrics.sort()) ==
        JSON.stringify(availableMetrics.sort())) {
      this.$.selectAll.checked = true;
    } else {
      this.$.selectAll.checked = false;
    }

    // To To re-select all the checked metrics. Un-checking "Select all"
    // checkbox in previous if else block, will de-select all the metrics
    // user has selected initially.
    this.selectedMetrics = selectedMetrics.slice();
  }
}

customElements.define(
    'fairness-metric-and-slice-selector', FairnessMetricAndSliceSelector);
