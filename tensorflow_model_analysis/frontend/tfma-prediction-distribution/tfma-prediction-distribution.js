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
import {template} from './tfma-prediction-distribution-template.html.js';

import '../tfma-google-chart-wrapper/tfma-google-chart-wrapper.js';

/**
 * tfma-prediciton-distribution renders the prediction distribution.
 *
 * @polymer
 */
export class PredicitonDistribution extends PolymerElement {
  constructor() {
    super();
  }

  static get is() {
    return 'tfma-prediction-distribution';
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
      numberOfBuckets: {type: Number, value: 16},

      /**
       * Chart rendering options.
       * @type {!Object}
       * @private
       */
      options_: {
        type: Object,
        value: {
          'hAxis': {'title': 'Prediction'},
          'vAxes': {0: {'title': 'Positive'}, 1: {'title': 'Negative'}},
          'series': {
            0: {'visibleInLegend': false, 'type': 'bars'},
            1: {
              'visibleInLegend': true,
              'targetAxisIndex': 1,
              'type': 'scatter'
            },
            2: {
              'visibleInLegend': true,
              'targetAxisIndex': 0,
              'type': 'scatter',
              'pointShape': 'diamond'
            },
          },
          'explorer': {actions: ['dragToZoom', 'rightClickToReset']},
        }
      },

      /**
       * The data to be plotted in the line chart.
       * @private {!Array<!Array<string|number>>}
       */
      plotData_: {
        type: Array,
        computed: 'computePlotData_(data, numberOfBuckets)',
      },
    };
  }

  /**
   * @param {!Array<!Object>|undefined} data
   * @param {number} numberOfBuckets
   * @return {(!Array<!Array<string|number>>|undefined)} A 2d array representing
   *     the data that will be visualized in the prediciton distribution.
   * @private
   */
  computePlotData_(data, numberOfBuckets) {
    if (!data) {
      return undefined;
    }

    const minValue = data && data[0] && data[0]['upperThresholdExclusive'] || 0;
    const maxValue = data && data[data.length - 1] &&
            data[data.length - 1]['lowerThresholdInclusive'] ||
        0;
    const bucketSize = (maxValue - minValue) / numberOfBuckets || 1;
    const plotData = [[
      'Prediction',
      'Count',
      {'type': 'string', 'role': 'tooltip'},
      'Positive',
      {'type': 'string', 'role': 'tooltip'},
      'Negative',
      {'type': 'string', 'role': 'tooltip'},
    ]];
    let currentBucketCenter = minValue + bucketSize / 2;
    do {
      // Initialize histogram with center x and zero count.
      plotData.push([currentBucketCenter, 0, '', 0, '', 0, '']);
      currentBucketCenter += bucketSize;
    } while (currentBucketCenter < maxValue);

    const maxIndex = plotData.length - 1;
    // For each entry, find the corresponding prediction and update weighted
    // example count. Note that index is 1-based since the 0-th entry is the
    // header.
    data.forEach((entry) => {
      const weightedExamples = entry['numWeightedExamples'];
      if (weightedExamples) {
        const totalLabel = entry['totalWeightedLabel'] || 0;
        const prediction =
            entry['totalWeightedRefinedPrediction'] / weightedExamples;
        const bucketIndex = Math.min(
            Math.trunc((prediction - minValue) / bucketSize) + 1, maxIndex);
        plotData[bucketIndex][1] = plotData[bucketIndex][1] + weightedExamples;
        plotData[bucketIndex][3] = plotData[bucketIndex][3] + totalLabel;
        plotData[bucketIndex][5] =
            plotData[bucketIndex][5] + weightedExamples - totalLabel;
      }
    });

    // Fill tooltip
    let lowerBound = minValue;
    let upperBound = minValue + bucketSize;
    for (let i = 1; i < plotData.length; i++) {
      const boundText = ' example(s) between ' + lowerBound.toFixed(4) +
          ' and ' + upperBound.toFixed(4);
      plotData[i][2] = plotData[i][1] + ' weighted' + boundText;
      plotData[i][4] = plotData[i][3] + ' positive' + boundText;
      plotData[i][6] = plotData[i][5] + ' negative' + boundText;
      lowerBound = upperBound;
      upperBound += bucketSize;
    }
    return plotData;
  }
}

customElements.define('tfma-prediction-distribution', PredicitonDistribution);
