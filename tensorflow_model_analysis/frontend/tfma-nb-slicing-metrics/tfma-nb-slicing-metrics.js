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
import {SelectEventMixin} from '../tfma-nb-event-mixin/tfma-nb-event-mixin.js';
import '../tfma-config-picker/tfma-config-picker.js';
import '../tfma-slicing-metrics-browser/tfma-slicing-metrics-browser.js';

/**
 * tfma-nb-slicing-metrics provides a wrapper for tfma-slicing-metrics-browser
 * in the notebook environment. It performs the necessary data transformation.
 * @extends HTMLElement
 * @polymer
 */
export class NotebookSlicingMetricsWrapper extends SelectEventMixin
(PolymerElement) {
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
      data: {type: Array},

      /**
       * A key value pair for the configuration.
       * @type {!Object}
       */
      config: {type: Object},

      /**
       * The data consumed by the slicing metrics browser.
       * @private {!Array<!Object>}
       */
      browserData_: {
        type: Array,
      },

      /**
       * @private {!Array<string>}
       */
      metrics_: {type: Array},

      /**
       * @private {string}
       */
      weightColumn_: {type: String},

      /**
       * A map of all available configs.
       * @private {!Object<!Array<string>>}}
       */
      availableConfigs_:
          {type: Object, computed: 'computeAvailableConfigs_(data)'},

      /**
       * A map of all selected configs.
       * @private {!Object<!Array<string>>}}
       */
      selectedConfigs_: {type: Object},
    };
  }

  static get observers() {
    return ['setUp_(data, config, selectedConfigs_)'];
  }

  /**
   * Sets up all fields based on data and config.
   * @param {!Array<{slice: string, metrics:!Object}>|undefined} data
   * @param {!Object|undefined} config
   * @param {!Object<!Array<string>>} selectedConfigs
   * @private
   */
  setUp_(data, config, selectedConfigs) {
    const configsList =
        selectedConfigs ? tfma.Util.createConfigsList(selectedConfigs) : [];

    if (!data || !config || !configsList.length) {
      return;
    }

    const processedData = data.map(function(entry) {
      return {
        'slice': entry['slice'],
        'metrics': tfma.Util.mergeMetricsForSelectedConfigsList(
            entry['metrics'], configsList),
      };
    });

    // Note that tfma.Data.flattenMetrics modifies its input in place so we
    // compute the following in an observer instead of making them computed
    // properties.
    tfma.Data.flattenMetrics(processedData, 'metrics');

    const metrics = tfma.Data.getAvailableMetrics([processedData], 'metrics');
    let weightColumn = config['weightedExamplesColumn'];
    // Look for exact match for weight column first.
    let absent = metrics.indexOf(weightColumn) < 0;
    if (absent) {
      // If no exact match found, try look for match after removing the config
      // prefix. We will use the first match.
      metrics.forEach(metricName => {
        if (absent && metricName.endsWith('/' + weightColumn)) {
          weightColumn = metricName;
          absent = false;
        }
      });
    }

    // If the weight column is missing, set it to 1.
    if (absent) {
      processedData.map(entry => {
        entry['metrics'][weightColumn] = 1;
      });
      metrics.push(weightColumn);
    }

    this.weightColumn_ = weightColumn;
    this.metrics_ = metrics;
    this.browserData_ = processedData;
  }

  /**
   * Constructs the map of available config where the key is the output name and
   * the value is an array of class ids for that output.
   * @param {!Array<!Object>} data
   * @return {!Object<!Array<string>>}
   * @private
   */
  computeAvailableConfigs_(data) {
    const configsMap = {};
    // Combines configs from different runs.
    data.forEach(entry => {
      const metricsMap = entry['metrics'];
      Object.keys(metricsMap).forEach(outputName => {
        const outputConfig = configsMap[outputName] || {};
        Object.keys(metricsMap[outputName]).forEach(classId => {
          outputConfig[classId] = 1;
        });
        configsMap[outputName] = outputConfig;
      });
    });
    return Object.keys(configsMap).reduce((acc, outputName) => {
      acc[outputName] = Object.keys(configsMap[outputName]).sort();
      return acc;
    }, {});
  }
}

customElements.define('tfma-nb-slicing-metrics', NotebookSlicingMetricsWrapper);
