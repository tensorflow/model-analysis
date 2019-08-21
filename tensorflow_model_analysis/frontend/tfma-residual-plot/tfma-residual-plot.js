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
import {template} from './tfma-residual-plot-template.html.js';

import '../tfma-google-chart-wrapper/tfma-google-chart-wrapper.js';

/**
 * tfma-residual-plot plot renders the residual plot.
 *
 * @polymer
 */
export class ResidualPlot extends PolymerElement {
  constructor() {
    super();
  }

  static get is() {
    return 'tfma-residual-plot';
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
          'legend': {'position': 'top'},
          'hAxis': {'title': 'Label'},
          'vAxes': {0: {'title': 'Residual'}, 1: {'title': 'Sample Count'}},
          'series': {
            0: {'visibleInLegend': true, 'targetAxisIndex': 0, 'type': 'line'},
            1: {
              'visibleInLegend': false,
              'targetAxisIndex': 0,
              'type': 'line',
              'lineDashStyle': [3, 2],
            },
            2: {
              'visibleInLegend': true,
              'targetAxisIndex': 1,
              'type': 'scatter',
              'pointShape': 'diamond',
            },
          },
          'explorer': {'actions': ['dragToZoom', 'rightClickToReset']},
        },
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
   * @return {!Array<!Array<string|number>>|undefined} A 2d array representing
   *     the data that will be visualized in the ROC curve.
   * @private
   */
  computePlotData_(data) {
    if (!data.length) {
      return undefined;
    }

    const plotData = [
      [
        'Label',
        'Residual',
        {'type': 'string', 'role': 'tooltip'},
        '',
        {'type': 'string', 'role': 'tooltip'},
        'Count',
        {'type': 'string', 'role': 'tooltip'},
      ],
    ];

    for (let i = 0; i < data.length; i++) {
      const entry = data[i];
      const count = entry['numWeightedExamples'] || 0;
      const upperBound = parseFloat(entry['upperThresholdExclusive'] || 0);
      const lowerBound = parseFloat(entry['lowerThresholdInclusive'] || 0);
      // We assume only one of upperBound and lowerBound can be infinite. When
      // that happens, use the other value as the label.
      const label = isFinite(upperBound) ?
          (isFinite(lowerBound) ? (upperBound + lowerBound) / 2 : upperBound) :
          lowerBound;
      const prediction =
          count ? entry['totalWeightedRefinedPrediction'] / count : 0;
      const residual = count ? label - prediction : 0;
      const predictionRange = '[' +
          lowerBound.toFixed(tfma.FLOATING_POINT_PRECISION) + ', ' +
          upperBound.toFixed(tfma.FLOATING_POINT_PRECISION) + ')';

      plotData.push([
        label,
        residual,
        'Residual is ' + residual.toFixed(tfma.FLOATING_POINT_PRECISION) +
            ' for label in ' + predictionRange,
        0,
        'Prediction range is ' + predictionRange,
        count,
        'There are ' + count + ' example(s) for label in ' + predictionRange,
      ]);
    }

    return plotData;
  }
}

customElements.define('tfma-residual-plot', ResidualPlot);
