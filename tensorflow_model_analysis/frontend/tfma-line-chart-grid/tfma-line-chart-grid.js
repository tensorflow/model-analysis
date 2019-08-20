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
import {template} from './tfma-line-chart-grid-template.html.js';

import '@polymer/iron-icons/iron-icons.js';
import '@polymer/paper-card/paper-card.js';
import '@polymer/paper-dropdown-menu/paper-dropdown-menu.js';
import '@polymer/paper-icon-button/paper-icon-button.js';
import '@polymer/paper-item/paper-item.js';
import '@polymer/paper-listbox/paper-listbox.js';
import '../tfma-line-chart/tfma-line-chart.js';

/**
 * tfma-line-chart-grid renders the time series plot for a number of metrics in
 * a grid layout.
 *
 * @polymer
 */
export class LineChartGrid extends PolymerElement {
  constructor() {
    super();
  }

  static get is() {
    return 'tfma-line-chart-grid';
  }

  /** @return {!HTMLTemplateElement} */
  static get template() {
    return template;
  }

  /** @return {!PolymerElementProperties} */
  static get properties() {
    return {
      /**
       * List of all metrics available for linecharts.
       * @type {!Array<string>}
       */
      metrics: {
        type: Array,
        value() {
          return [];
        },
      },

      /**
       * A comma separated string of metrics to skip.
       * @type {string}
       */
      blacklist: {
        type: String,
        value: '',
      },

      /**
       * An array of selectable metrics.
       * @private {!Array<string>}
       */
      selectableMetrics_: {
        type: Array,
        computed: 'computeSelectableMetrics_(metrics, blacklist)',
        observer: 'selectableMetricsChanged_',
      },

      selectedMetric_: {type: String, observer: 'selectedMetricChanged_'},

      /**
       * List of metrics that are shown as linecharts.
       * The user may add/remove a metric linechart.
       * @type {!Array<string>}
       */
      selectedMetrics_: {type: Array},

      /**
       * The line chart provider object.
       * @type {!tfma.LineChartProvider}
       */
      provider: {type: Object},

      /**
       * List of metrics that can be added to linecharts, i.e. all metrics
       * excluding the selected metrics.
       * @private {!Array<string>}
       */
      addableMetrics_: {type: Array},

      /**
       * Label in the add-series dropdown.
       * @private {string}
       */
      addSeriesLabel_:
          {type: String, computed: 'computeAddSeriesLabel_(addableMetrics_)'},
    };
  }

  static get observers() {
    return [
      'setAddableMetrics_(selectableMetrics_, selectedMetrics_)',
      // Since array mutation does not trigger the compute method above, set a
      // separate observer for mutation.
      'updateAddableMetrics_(selectedMetrics_.splices)',
    ];
  }

  /**
   * @param {?Object} selection Key value pairs in the selection object specify
   *     the data point to highlight in the charts. If null, clear existing
   *     selection.
   */
  highlight(selection) {
    if (selection) {
      this.highlightSelection_(selection, false);
    } else {
      this.clearSelection_(false);
    }
  }

  /**
   * Internal implementation of model selection.
   * @param {!Object} selection
   * @param {boolean} fireEvent Whether to fire a model selection event from the
   *     component.
   * @private
   */
  highlightSelection_(selection, fireEvent) {
    const targetModelId = selection[this.provider.getModelColumnName()];
    const targetDataVersion = selection[this.provider.getDataColumnName()];
    const rowMatchingModel = [];
    this.provider.getModelIds().forEach((modelId, index) => {
      if (modelId == targetModelId) {
        rowMatchingModel.push(index);
      }
    });
    this.selectedMetrics_.forEach((metric) => {
      const row = rowMatchingModel.filter((rowId) => {
        const config = this.provider.getEvalConfig(rowId);
        return config && config.data == targetDataVersion;
      });

      if (row.length == 1) {
        const chart = this.getChartForMetric_(metric);
        if (chart) {
          chart.select({'row': row[0]});
        }
      }
    });

    if (fireEvent) {
      this.dispatchEvent(new CustomEvent(
          tfma.Event.SELECT,
          {detail: selection, composed: true, bubbles: true}));
    }
  }

  /**
   * Internal implementation of clearing model selection.
   * @param {boolean} fireEvent Whether to fire a model selection event from the
   *     component.
   * @private
   */
  clearSelection_(fireEvent) {
    this.selectedMetrics_.forEach((metric) => {
      const chart = this.getChartForMetric_(metric);
      if (chart != null) {
        chart.clearSelection();
      }
    });

    if (fireEvent) {
      this.dispatchEvent(new CustomEvent(
          tfma.Event.CLEAR_SELECTION, {composed: true, bubbles: true}));
    }
  }

  /**
   * Closes a line chart.
   * @param {!Event} e
   * @private
   */
  closeLineChart_(e) {
    const item = e['model']['item'];
    const index = item && this.selectedMetrics_.indexOf(item);
    if (index >= 0) {
      this.splice('selectedMetrics_', index, 1);
    }
  }

  /**
   * Computes the chart data for the named metric using the given provider.
   * @param {string} metric
   * @param {!tfma.LineChartProvider} provider
   * @return {!Array<!Object>|undefined}
   * @private
   */
  computeChartData_(metric, provider) {
    if (!provider) {
      return undefined;
    }
    const lineChartData = provider.getLineChartData(metric);
    return [
      [
        {'label': provider.getDataColumnName(), 'type': 'number'},
        {'role': 'annotation', 'type': 'string'},
        {'role': 'annotationText', 'type': 'string'},
        {'label': metric, 'type': 'number'}
      ]
    ].concat(lineChartData);
  }

  /**
   * Computes the addable metrics, which are (all) metrics excluding the
   * selectedMetrics.
   * @param {!Array<string>} metrics
   * @param {!Array<string>} selectedMetrics
   * @private
   */
  setAddableMetrics_(metrics, selectedMetrics) {
    this.addableMetrics_ =
        metrics.filter(metric => selectedMetrics.indexOf(metric) == -1);

    // Unselect since the list is of addable metrics is being changed.
    this.selectedMetric_ = '';
  }

  /**
   * Updates the addable metrics.
   * @private
   */
  updateAddableMetrics_() {
    this.setAddableMetrics_(this.selectableMetrics_, this.selectedMetrics_);
  }

  /**
   * Sets the add-series dropdown label message to a string reflecting the
   * number of remaining metrics that can be added. When all metrics are added,
   * the dropdown will not pop-out so it is necessary to have a hint presented.
   * @param {!Array<string>} addableMetrics_
   * @return {string} The add series label.
   * @private
   */
  computeAddSeriesLabel_(addableMetrics_) {
    return addableMetrics_.length ? 'Add metric series' :
                                    'All metrics have been added';
  }

  /**
   * Observer for selectedMetric_ property.
   * @param {string} metric
   * @private
   */
  selectedMetricChanged_(metric) {
    if (metric) {
      this.push('selectedMetrics_', metric);
    }
  }

  /**
   * Determines selectable metrics by filtering out all blacklisted metrics from
   * available metrics.
   * @param {!Array<string>} metrics
   * @param {string} blacklist A comma separated string of metrics that should
   *     be blacklisted.
   * @return {!Array<string>}
   * @private
   */
  computeSelectableMetrics_(metrics, blacklist) {
    const blacklisted = blacklist.split(',');
    return metrics.filter(metric => blacklisted.indexOf(metric) == -1);
  }

  /**
   * Resets the selectedMetrics to the first available metric. This is expected
   * to only occur when the input data changes.
   * @private
   */
  selectableMetricsChanged_() {
    // By default show the first available metric series.
    this.selectedMetrics_ = this.metrics.slice(0, 1);
  }

  /**
   * Handler for select event from a line chart.
   * @param {!Event} e
   * @private
   */
  onChartSelect_(e) {
    e.stopPropagation();
    const config = this.provider.getEvalConfig(e.detail.point['row']);
    if (config) {
      const selection = {};
      selection[this.provider.getModelColumnName()] = config.model;
      selection[this.provider.getDataColumnName()] = config.data;
      this.highlightSelection_(selection, true);
    }
  }

  /**
   * Handler for clear-selection event from any line chart.
   * @param {!Event} e
   * @private
   */
  onChartClearSelection_(e) {
    e.stopPropagation();
    this.clearSelection_(true);
  }

  /**
   * @param {string} metric
   * @return {?Element} The chart for the named metric if availale.
   * @private
   */
  getChartForMetric_(metric) {
    return this.shadowRoot.querySelector(
        'tfma-line-chart[metric="' + metric + '"]');
  }
}

customElements.define('tfma-line-chart-grid', LineChartGrid);
