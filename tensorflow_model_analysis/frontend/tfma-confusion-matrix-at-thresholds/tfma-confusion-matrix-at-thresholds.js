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
  is: 'tfma-confusion-matrix-at-thresholds',
  properties: {
    /**
     * The serialized data.
     * @type {string}
     */
    data: {type: String},
    /**
     * An array of data transformed from the data string.
     * @private {!Array<!Object>}
     */
    transformedData_:
        {type: Array, computed: 'computeTransformedData_(data, expanded)'},

    /**
     * Whether the component can be expanded.
     * @private {boolean}
     */
    expandable_: {
      type: Boolean,
      computed: 'computeExpandable_(transformedData_, interactive)'
    },

    /**
     * The data to display.
     * @private {!Array<!Object>}
     */
    displayedData_: {
      type: Array,
      computed: 'computeDisplayedData_(transformedData_, expandable_, expanded)'
    },

    /**
     * Whether the metric is expanded.
     * @type {boolean}
     */
    expanded: {type: Boolean, value: false, reflectToAttribute: true},

    /**
     * Whether this component is interactive or not.
     * @type {boolean}
     */
    interactive: {type: Boolean, value: true},
  },

  /**
   * Parses the srialized data and transforms the result into expected objects.
   * @param {string} data
   * @param {boolean} expanded
   * @return {!Array<!Object>}
   * @private
   */
  computeTransformedData_: function(data, expanded) {
    if (data) {
      let parsedData = [];
      try {
        // Data is expected to be a serialized array of JSON objects.
        parsedData = JSON.parse(data);
      } catch (e) {
      }

      return parsedData['matrices'].map((matrix, index) => {
        return {
          'showTitle': index == 0,
          'threshold': this.getValue_(matrix, 'threshold', expanded),
          'precision': this.getValue_(matrix, 'precision', expanded),
          'recall': this.getValue_(matrix, 'recall', expanded),
          'truePositives': this.getValue_(matrix, 'truePositives', expanded),
          'trueNegatives': this.getValue_(matrix, 'trueNegatives', expanded),
          'falsePositives': this.getValue_(matrix, 'falsePositives', expanded),
          'falseNegatives': this.getValue_(matrix, 'falseNegatives', expanded),
        };
      });
    }
    return [];
  },

  /**
   * Determines whether the component is expandable.
   * @param {!Array<!Object>} data
   * @param {boolean} interactive
   * @return {boolean}
   * @private
   */
  computeExpandable_: function(data, interactive) {
    return interactive && data.length > 3;
  },

  /**
   * Determines the data to display.
   * @param {!Array<!Object>} data
   * @param {boolean} expandable
   * @param {boolean} expanded
   * @return {!Array<!Object>}
   * @private
   */
  computeDisplayedData_: function(data, expandable, expanded) {
    // If expanded or have no more than three records, show all. If collapsed
    // and have more than three records, show only the first three.
    return (expanded || data.length <= 3) ? data : [data[0], data[1], data[2]];
  },

  /**
   * Helper function to help extract the metric value out of the given object
   * and convert it to string to the preferred decimal places.
   * @param {!Object} object
   * @param {string} key
   * @param {boolean} expanded
   * @return {string}
   */
  getValue_: function(object, key, expanded) {
    return this.toFixed_(object[key] || 0, expanded);
  },

  /**
   * @param {!Object} matrix
   * @param {boolean} expanded
   * @return {string} The specificity from the matrix.
   */
  computeSpecificity_: function(matrix, expanded) {
    const trueNegatives = matrix['trueNegatives'] || 0;
    const falsePositives = matrix['falsePositives'] || 0;
    const denominator = trueNegatives + falsePositives;
    return this.toFixed_(
        denominator ? trueNegatives / denominator : 0, expanded);
  },

  /**
   * Round the given value to a fixed number of digits depending on whether the
   * component is in expanded mode or not.
   * @param {number} value
   * @param {boolean} expanded
   * @return {string}
   * @private
   */
  toFixed_: function(value, expanded) {
    // When in epxanded mode, show all 16 digits available in double precision.
    return value.toFixed(expanded ? 16 : tfma.FLOATING_POINT_PRECISION);
  },

  /**
   * Toggles the expanded flag of the component and fires expand-metric event.
   * @private
   */
  toggleExpanded_: function() {
    if (!this.expandable_) {
      return;
    }
    if (this.expanded) {
      this.expanded = false;
    } else {
      const event = this.fire(
          'expand-metric', {'data': this.data, 'metric': this.is},
          {'cancelable': true});
      this.expanded = !event.defaultPrevented;
    }
  },
});
