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
(() => {
  const VALUE_ABSENT_ = '-';
  Polymer({
    is: 'tfma-multi-values-threshold-based-binary-classification-metrics',
    properties: {
      data: {type: String},
      parsedData_: {type: Object, computed: 'parseData_(data)'},
      metrics_: {type: Array, computed: 'computeMetrics_(parsedData_)'},
    },

    /**
     * Parses the given data.
     * @param {string} data
     * @return {!Object}
     * @private
     */
    parseData_: function(data) {
      let firstRow = {};
      if (data) {
        try {
          const parsedData = JSON.parse(data);
          if (Array.isArray(parsedData)) {
            firstRow = parsedData[0] || {};
          }
        } catch (e) {
        }
      }
      return firstRow;
    },

    /**
     * @param {!Object} parsedData
     * @return {string} The prediction threshold rounded to desired precision if
     *     available; VALUE_ABSENT_, otherwise.
     * @private
     */
    getThreshold_: function(parsedData) {
      const threshold = parsedData['binaryClassificationThreshold'];
      return threshold ? this.getValue_(threshold, 'predictionThreshold') :
                         VALUE_ABSENT_;
    },

    /**
     * Tranforms metric results from parsed data into presentation data.
     * @param {!Object} parsedData
     * @return {!Array<!Object<string>>}
     */
    computeMetrics_: function(parsedData) {
      const metrics = ['Precision', 'Recall', 'F1Score', 'Accuracy'];
      const types = ['Macro', 'Micro', 'Weighted'];
      return types.map(type => {
        const typePrefix = type.toLowerCase();
        const typeNotConfigured =
            parsedData[typePrefix + metrics[0]] === undefined;
        const values = typeNotConfigured ?
            [VALUE_ABSENT_, VALUE_ABSENT_, VALUE_ABSENT_, VALUE_ABSENT_] :
            metrics.map(metric => {
              return this.getValue_(parsedData, typePrefix + metric);
            });
        return {
          'type': type,
          'precision': values[0],
          'recall': values[1],
          'f1': values[2],
          'accuracy': values[3]
        };
      });
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
  });
})();
