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

import '../tfma-config-picker/tfma-config-picker.js';
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
       * is a nested dictionary containing the evaluation results for the slice
       * for different outputs and / or class id's.
       * @type {!Array<!Object>}
       */
      data: {type: Array},

      /**
       * A key value pair for the configuration.
       * @type {!Object}
       */
      config: {type: Object},

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

      /**
       * The data consumed by the time series browser.
       * @private {!tfma.SeriesData}
       */
      seriesData_: {type: Object},
    };
  }

  static get observers() {
    return ['refresh_(data, config, selectedConfigs_)'];
  }

  /**
   * Refreshes the view by updating format override and the series data.
   * @param {!Array<!Object>} data
   * @param {!Object} config
   * @param {!Object<!Array<string>>} selectedConfigs
   * @private
   */
  refresh_(data, config, selectedConfigs) {
    const configsList =
        selectedConfigs ? tfma.Util.createConfigsList(selectedConfigs) : [];

    if (!data || !config || !configsList.length) {
      return;
    }

    const processedData = data.map(function(entry) {
      return {
        'config': entry['config'],
        'metrics': tfma.Util.mergeMetricsForSelectedConfigsList(
            entry['metrics'], configsList),
      };
    });

    tfma.Data.flattenMetrics(processedData, 'metrics');
    const evalRuns = processedData.map(run => [{'metrics': run.metrics}]);
    const metricNames = tfma.Data.getAvailableMetrics(evalRuns, 'metrics');
    this.seriesData_ = new tfma.SeriesData(
        processedData.map(run => {
          return {
            'data': tfma.Data.build(
                metricNames, [{'metrics': run['metrics'], 'slice': ''}]),
            'config': run['config'],
          };
        }),
        config['isModelCentric']);
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

customElements.define('tfma-nb-time-series', NotebookTimeSeriesWrapper);
