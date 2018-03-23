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
Polymer({
  is: 'tfma-binary-confusion-matrices-from-regression',
  properties: {
    data: {type: String, observer: 'dataChanged_'},
    threshold_: {type: String},
    f1Score_: {type: String},
    precision_: {type: String},
    recall_: {type: String},
    accuracy_: {type: String},
    specificity_: {type: String},
  },

  /**
   * Parses the given data and sets up the component.
   * @param {string} data
   * @private
   */
  dataChanged_: function(data) {
    if (data) {
      let firstRow = {};
      try {
        // Data is expected to be a serialized array of JSON objects.
        const parsedData = JSON.parse(data);
        if (Array.isArray(parsedData)) {
          firstRow = parsedData[0] || {};
        }
      } catch (e) {
      }

      this.threshold_ = this.getValue_(
          firstRow['binaryClassificationThreshold'], 'predictionThreshold');

      const matrix = firstRow['matrix'];
      this.f1Score_ = this.getValue_(matrix, 'f1Score');
      this.precision_ = this.getValue_(matrix, 'precision');
      this.recall_ = this.getValue_(matrix, 'recall');
      this.accuracy_ = this.getValue_(matrix, 'accuracy');
      this.specificity_ = this.computeSpecificity_(matrix);
    }
  },

  /**
   * Helper function to help extract the metric value out of the given object
   * and convert it to string to the preferred decimal places.
   * @param {!Object} object
   * @param {string} key
   * @return {string}
   */
  getValue_: function(object, key) {
    return (object[key] || 0).toFixed(tfma.FLOATING_POINT_PRECISION);
  },

  /**
   * @param {!Object} matrix
   * @return {string} The specificity from the matrix.
   */
  computeSpecificity_: function(matrix) {
    const trueNegatives = matrix['trueNegatives'] || 0;
    const falsePositives = matrix['falsePositives'] || 0;
    const denominator = trueNegatives + falsePositives;
    return (denominator ? trueNegatives / (trueNegatives + falsePositives) : 0)
        .toFixed(tfma.FLOATING_POINT_PRECISION);
  },
});
