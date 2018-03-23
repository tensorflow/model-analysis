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

  is: 'tfma-bounded-value',

  properties: {
    /**
     * The upper bound of the estimate range.
     * @type {number}
     */
    upperBound: {type: Number},


    /**
     * The lower bound of the estimate range.
     * @type {number}
     */
    lowerBound: {type: Number},

    /**
     * The computed value.
     * @type {string}
     */
    value_: {type: String, computed: 'computeValue_(lowerBound, upperBound)'},

    /**
     * The range of values.
     * @type {string}
     */
    range_: {type: String, computed: 'computeRange_(lowerBound, upperBound)'},

    /**
     * The serialized form of the data.
     * @type {string}
     */
    data: {type: String, value: '', observer: 'dataChanged_'},
  },

  /**
   * Observer for the property data.
   * @param {string} serializedData
   * @private
   */
  dataChanged_: function(serializedData) {
    if (serializedData) {
      try {
        const parsedData = JSON.parse(serializedData);
        this.upperBound = parsedData['upperBound'];
        this.lowerBound = parsedData['lowerBound'];
      } catch (e) {
      }
    }
  },

  /**
   * @param {number} lowerBound
   * @param {number} upperBound
   * @return {string} The value range.
   */
  computeRange_: function(lowerBound, upperBound) {
    return ((upperBound - lowerBound) / 2)
        .toFixed(tfma.FLOATING_POINT_PRECISION);
  },

  /**
   * @param {number} lowerBound
   * @param {number} upperBound
   * @return {string} The value.
   */
  computeValue_: function(lowerBound, upperBound) {
    return ((upperBound + lowerBound) / 2)
        .toFixed(tfma.FLOATING_POINT_PRECISION);
  },

});
