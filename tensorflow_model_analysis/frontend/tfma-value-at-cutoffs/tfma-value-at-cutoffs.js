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

  is: 'tfma-value-at-cutoffs',

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
    const values = parsedData && parsedData['values'];
    if (!values || !Array.isArray(values)) {
      return undefined;
    }

    return values.map(function(pair) {
      return {
        'cutoff': pair['cutoff'] || 'All',
        'value': tfma.CellRenderer.extractFloatValue(pair, 'value')
      };
    });
  },
});
