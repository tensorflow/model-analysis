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

import '@polymer/paper-checkbox/paper-checkbox.js';
import '@polymer/paper-item/paper-item.js';
import '@polymer/paper-listbox/paper-listbox.js';


import {PolymerElement} from '@polymer/polymer/polymer-element.js';
import {template} from './fairness-metric-and-slice-selector-template.html.js';

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

/**
 * It's a map that contains following fields:
 *   'metricsName': the name of the metric. Usually it has prefix, but not
 *     thresholds, e.g. 'post_export_metrics/false_negative_rate',
 *   'selected': indicates if this element is selected by users. This will
 *     also be used to change the checkbox status in front of the metrics.
 * @typedef {{
 *   metricsName: string,
 *   isSelected: boolean,
 * }}
 */
let MetricsListCandidateType;


/**
 * Type for Event.detail.item.
 * @typedef {{
 *   metric: !MetricsListCandidateType
 * }}
 */
let ItemType;

/**
 * Type for Event.detail.
 * @typedef {{
 *   item: !ItemType
 * }}
 */
let Detailype;

/**
 * Type for Event.
 * @typedef {{
 *   detail: !Detailype
 * }}
 */
let CustomEvent;

/**
 * FairnessMetricAndSliceSelector renders a list of metrics.
 *
 * @polymer
 */
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
      availableMetrics: {type: Array},

      /**
       * A list of metrics selected.
       * @type {!Array<string>}
       */
      selectedMetrics: {type: Array, notify: true, value: []},

      /**
       * A list of metrics candidates to be rendered on
       * @private {!Array<!MetricsListCandidateType>}
       */
      metricsListCandidates_: {
        type: Array,
        computed: 'computedMetricsListCandidates_(availableMetrics)',
      },

      /**
       * A list of objects which indicate if metrics have been selected.
       * @private {!Array<!MetricsListCandidateType>}
       */
      selectedMetricsListCandidates_: {type: Array, notify: true},
    };
  }

  /**
   * Generate the MetricsListCandidates_.
   * @param {!Array<string>} availableMetrics
   * @return {!Array<!MetricsListCandidateType>}
   * @private
   */
  computedMetricsListCandidates_(availableMetrics) {
    this.selectedMetricsListCandidates_ = [];
    this.selectedMetrics = [];

    if (!availableMetrics) {
      return [];
    }

    let candidates = [];
    for (const name of availableMetrics) {
      candidates.push({
        metricsName: name,
        isSelected: false,
      });
    }

    // Select 1st metric by default.
    setTimeout(() => {
      this.selectedMetricsListCandidates_ = [candidates[0]];
    }, 0);
    return candidates;
  }

  /**
   * Strip prefix from metric name.
   * @param {!MetricsListCandidateType} metricListCandidate
   * @return {string}
   */
  stripPrefix(metricListCandidate) {
    return Util.removeMetricNamePrefix(metricListCandidate.metricsName);
  }

  /**
   * Get the defintion of a metric, or return the metric if not defined.
   * @param {!MetricsListCandidateType} metricListCandidate
   * @return {string}
   */
  getDefinition(metricListCandidate) {
    let strippedMetric = this.stripPrefix(metricListCandidate);
    if (strippedMetric in METRIC_DEFINITIONS) {
      return METRIC_DEFINITIONS[strippedMetric];
    } else {
      return strippedMetric;
    }
  }

  /**
   * Handler listening to any change in 'Select all' check box.
   */
  onSelectAllCheckedChanged_(event) {
    if (!this.metricsListCandidates_) {
      return;
    }
    const checked = event.detail.value;
    if (checked) {
      // Select all.
      for (let i = 0; i < this.metricsListCandidates_.length; i++) {
        if (!this.selectedMetricsListCandidates_.includes(
                this.metricsListCandidates_[i])) {
          this.push(
              'selectedMetricsListCandidates_', this.metricsListCandidates_[i]);
        }
      }
    } else {
      // UnSelect all.
      const length = this.selectedMetricsListCandidates_.length;
      for (let i = 0; i < length; i++) {
        this.pop('selectedMetricsListCandidates_');
      }
    }
  }

  /**
   * This function will be triggered if user select a metric from metrics list.
   * It updates the metric 'isSelected' status (which will be used to change the
   * checkbox status) and selectedMetrics (which is a array of selected metrics
   * names).
   * @param {!CustomEvent} event
   * @private
   */
  metricsListCandidatesSelected_(event) {
    // Updates the selected status to true.
    let selectedItem = event.detail.item['metric'];
    let selectedItemIndex = this.metricsListCandidates_.findIndex(
        item => item.metricsName == selectedItem.metricsName);
    this.set(
        'metricsListCandidates_.' + selectedItemIndex + '.isSelected', true);

    // Updates the selectedMetrics.
    this.push('selectedMetrics', selectedItem.metricsName);
  }

  /**
   * This function will be triggered if user un-select a metric from metrics
   * list. It updates the metric 'isSelected' status (which will be used to
   * change the checkbox status) and selectedMetrics (which is a array of
   * selected metrics names).
   * @param {!CustomEvent} event
   * @private
   */
  metricsListCandidatesUnselected_(event) {
    // Updates the selected status to false.
    let selectedItem = event.detail.item['metric'];
    let selectedItemIndex = this.metricsListCandidates_.findIndex(
        item => item.metricsName == selectedItem.metricsName);
    this.set(
        'metricsListCandidates_.' + selectedItemIndex + '.isSelected', false);

    // Updates the selectedMetrics. Replace the unselected metric with undefined
    // to maintain the index of original selectedMetrics. In this way, the user
    // selected threshold in the fairness-metric-summary would not get
    // overwritten.
    this.splice(
        'selectedMetrics',
        this.selectedMetrics.indexOf(selectedItem.metricsName), 1, undefined);
  }

  /**
   * Display the info dialog.
   * @param {!Object} event
   * @private
   */
  openInfoDialog_(event) {
    event.stopPropagation();
    const dialog = event.target.parentElement.querySelector('paper-dialog');
    dialog.open();
  }
}


customElements.define(
    'fairness-metric-and-slice-selector', FairnessMetricAndSliceSelector);
