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
import {template} from './tfma-precision-recall-curve-template.html.js';

import '../tfma-google-chart-wrapper/tfma-google-chart-wrapper.js';

/**
 * tfma-precision-recall-curve enders the precision recall curve.
 *
 * @polymer
 */
export class PrecisionRecallCurve extends PolymerElement {
  constructor() {
    super();
  }

  static get is() {
    return 'tfma-precision-recall-curve';
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
          'hAxis': {'title': 'Recall'},
          'vAxis': {'title': 'Precision'},
          'series': {0: {'visibleInLegend': false}},
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
   *     that will be visualized in the claibration plot.
   * @private
   */
  computePlotData_(data) {
    const plotData =
        [['Recall', 'Precision', {'type': 'string', 'role': 'tooltip'}]];
    data.forEach((entry) => {
      const threshold = Math.max(0, Math.min(1, entry['threshold'] || 0));
      const recall = tfma.CellRenderer.extractFloatValue(entry, 'recall');
      const precision = tfma.CellRenderer.extractFloatValue(entry, 'precision');
      const tooltip = 'Prediction threshold: ' + threshold.toFixed(5) +
          '\nRecall: ' + recall.toFixed(5) +
          '\nPrecision: ' + precision.toFixed(5);
      plotData.push([recall, precision, tooltip]);
    });
    return plotData;
  }
}

customElements.define('tfma-precision-recall-curve', PrecisionRecallCurve);
