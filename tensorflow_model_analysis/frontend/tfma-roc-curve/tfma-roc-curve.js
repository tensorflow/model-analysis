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
import {template} from './tfma-roc-curve-template.html.js';

import '../tfma-google-chart-wrapper/tfma-google-chart-wrapper.js';

/**
 * tfma-roc-curve plot renders the ROC curve.
 *
 * @polymer
 */
export class RocCurve extends PolymerElement {
  constructor() {
    super();
  }

  static get is() {
    return 'tfma-roc-curve';
  }

  /** @return {!HTMLTemplateElement} */
  static get template() {
    return template;
  }

  /** @return {!PolymerElementProperties} */
  static get properties() {
    return {
      /** @type {!Array<!Object>} */
      data: {type: Array},

      /**
       * Chart rendering options.
       * @type {!Object}
       * @private
       */
      options_: {
        type: Object,
        value: {
          'legend': {'position': 'bottom'},
          'hAxis': {'title': 'False Positive Rate'},
          'vAxis': {'title': 'True Positive Rate'},
          'series': {
            0: {
              'lineDashStyle': [3, 2],
              'visibleInLegend': false,
            },
            1: {'visibleInLegend': false},
          },
          'explorer': {actions: ['dragToZoom', 'rightClickToReset']},
        }
      },

      /**
       * The data to be plotted in the line chart.
       * @private {!Array<!Array<string|number>>}
       */
      plotData_: {type: Array, computed: 'computePlotData_(data)'},
    };
  }

  /**
   * @param {!Array<!Object>} data
   * @return {!Array<!Array<string|number>>} A 2d array representing the data
   *     that will be visualized in the ROC curve.
   * @private
   */
  computePlotData_(data) {
    const plotData = [[
      'FPR', '', {'type': 'string', 'role': 'tooltip'}, 'TPR',
      {'type': 'string', 'role': 'tooltip'}
    ]];
    data.forEach((entry) => {
      const threshold = Math.max(0, Math.min(1, entry['threshold'] || 0));
      const truePositives =
          tfma.CellRenderer.extractFloatValue(entry, 'truePositives');
      const falseNegatives =
          tfma.CellRenderer.extractFloatValue(entry, 'falseNegatives');
      const truePositiveRate = truePositives / (truePositives + falseNegatives);
      const trueNegatives =
          tfma.CellRenderer.extractFloatValue(entry, 'trueNegatives');
      const falsePositives =
          tfma.CellRenderer.extractFloatValue(entry, 'falsePositives');
      const falsePositiveRate =
          falsePositives / (trueNegatives + falsePositives);
      const tooltip = 'Prediction threshold: ' + threshold.toFixed(5) +
          '\nFPR: ' + falsePositiveRate.toFixed(5) +
          '\nTPR: ' + truePositiveRate.toFixed(5);
      plotData.push([
        falsePositiveRate, falsePositiveRate, 'Random', truePositiveRate,
        tooltip
      ]);
    });
    return plotData;
  }
}

customElements.define('tfma-roc-curve', RocCurve);
