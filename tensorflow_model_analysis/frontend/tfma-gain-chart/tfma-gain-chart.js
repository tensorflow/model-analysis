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
import {template} from './tfma-gain-chart-template.html.js';

import '../tfma-google-chart-wrapper/tfma-google-chart-wrapper.js';

/**
 * tfma-gain-chart renders cumulative gain in a plot.
 *
 * @polymer
 */
export class GainChart extends PolymerElement {
  constructor() {
    super();
  }

  static get is() {
    return 'tfma-gain-chart';
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

      /** @type {number} */
      steps: {type: Number, value: 10},

      /**
       * Chart rendering options.
       * @type {!Object}
       * @private
       */
      options_: {
        type: Object,
        value: {
          'legend': {'textStyle': {'fontSize': 9}},
          'hAxis': {'title': 'Percentile'},
          'vAxis': {'title': 'Gain'},
          'explorer': {'actions': ['dragToZoom', 'rightClickToReset']},
          'series': {
            0: {
              'lineDashStyle': [3, 2],
              'visibleInLegend': false,
            },
            1: {'visibleInLegend': false},
          },

        },
      },

      /**
       * The data to be plotted in the line chart.
       * @private {!Array<!Array<string|number>>}
       */
      plotData_: {type: Array, computed: 'computePlotData_(data, steps)'},
    };
  }

  /**
   * @param {!Array<!Object>} data
   * @param {number} steps
   * @return {!Array<!Array<string|number>>|undefined} A 2d array representing
   *     the data that will be visualized.
   * @private
   */
  computePlotData_(data, steps) {
    if (!data || !data.length || !steps) {
      return undefined;
    }

    const plotData = [
      [
        'Percentile',
        '',
        {'type': 'string', 'role': 'tooltip'},
        'Gain',
        {'type': 'string', 'role': 'tooltip'},
      ],
      [0, 0, 'Random', 0, 'Origin']
    ];

    // Create a copy of the input data and sort it by threshold in decreasing
    // order.
    const sortedData = data.slice().sort((entry1, entry2) => {
      const threshold1 =
          tfma.CellRenderer.extractFloatValue(entry1, 'threshold');
      const threshold2 =
          tfma.CellRenderer.extractFloatValue(entry2, 'threshold');
      return threshold2 - threshold1;
    });

    const totalPositives =
        tfma.CellRenderer.extractFloatValue(sortedData[0], 'truePositives') +
        tfma.CellRenderer.extractFloatValue(sortedData[0], 'falseNegatives');
    const totalCount = totalPositives +
        tfma.CellRenderer.extractFloatValue(sortedData[0], 'trueNegatives') +
        tfma.CellRenderer.extractFloatValue(sortedData[0], 'falsePositives');
    const stepSize = totalCount / steps;
    let currentStep = Math.round(stepSize);

    for (let i = 0; i < sortedData.length; i++) {
      const entry = sortedData[i];
      const truePositives =
          tfma.CellRenderer.extractFloatValue(entry, 'truePositives');
      const totalPredictedPositives = truePositives +
          tfma.CellRenderer.extractFloatValue(entry, 'falsePositives');

      if (totalPredictedPositives >= currentStep) {
        const threshold =
            tfma.CellRenderer.extractFloatValue(entry, 'threshold');
        const percentile = totalPredictedPositives / totalCount * 100;

        plotData.push([
          percentile,
          percentile,
          'Random',
          truePositives / totalPositives * 100,
          'True Positives: ' + truePositives + '\nPredicted Positives: ' +
              totalPredictedPositives + '\nThreshold: ' + threshold.toFixed(5) +
              '\nPercentile: ' + Math.round(percentile),
        ]);

        if (percentile >= 100) {
          // Stop if we reach 100% before visiting all buckets.
          break;
        }

        currentStep = Math.min(totalCount, Math.round(currentStep + stepSize));
      }
    }

    return plotData;
  }
}

customElements.define('tfma-gain-chart', GainChart);
