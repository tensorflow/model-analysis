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

  is: 'tfma-multi-values-metric-at-k',

  properties: {
    /**
     * A serialized array of JSON objects.
     * @type {string}
     */
    data: {type: String},

    /**
     * @type {!Array<number|string>}
     * @private
     */
    formattedData_: {type: Object, computed: 'formatData_(data)'},

    /**
     * Whether the metric contains multiple row.
     * @private
     */
    multi: {
      type: Boolean,
      computed: 'computeMulti_(formattedData_)',
      reflectToAttribute: true,
    }
  },

  /**
   * @param {string} data
   * @return {!Array<number|string>|undefined} The formatted data
   */
  formatData_: function(data) {
    let parsedData;
    try {
      parsedData = JSON.parse(data);
    } catch (e) {
    }
    if (!parsedData || !Array.isArray(parsedData)) {
      return undefined;
    }

    const spacer = '-';
    return parsedData.map(function(precision) {
      return {
        'k': precision['k'] || 'All',
        'macroValue': precision['macroValue'] === undefined ?
            spacer :
            precision['macroValue'].toFixed(tfma.FLOATING_POINT_PRECISION),
        'microValue': precision['microValue'] === undefined ?
            spacer :
            precision['microValue'].toFixed(tfma.FLOATING_POINT_PRECISION),
        'weightedValue': precision['weightedValue'] === undefined ?
            spacer :
            precision['weightedValue'].toFixed(tfma.FLOATING_POINT_PRECISION),
      };
    });
  },

  /**
   * Determines if the metric contains multiple row.
   * @param {!Array<!Object>} formattedData
   * @return {boolean}
   * @private
   */
  computeMulti_: function(formattedData) {
    return formattedData.length > 1;
  },
});
