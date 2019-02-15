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

import {template} from './tfma-array-value-template.html.js';

import '@polymer/paper-tooltip/paper-tooltip.js';

/**
 * tfma-array-value renders an array value.
 *
 * @polymer
 */
export class ArrayValue extends PolymerElement {
  constructor() {
    super();
  }

  static get is() {
    return 'tfma-array-value';
  }

  /** @return {!HTMLTemplateElement} */
  static get template() {
    return template;
  }

  /** @return {!PolymerElementProperties} */
  static get properties() {
    return {
      /**
       * A string of a serialized JSON object representing ArrayValue defined in
       * metrics_for_slice.proto. Note that only one of data and arrayData
       * should be provided.
       * @type {string}
       */
      data: {type: String, observer: 'dataChanged_'},

      /**
       * A nested array representing the array value.
       * Note that only one of data and arrayData should be provided.
       * @type {!Array<number|!Object>}
       */
      arrayData: {type: Array},

      /**
       * The values to be rendered. Each element repreents a row. If the
       * original data is not a 1d array, the values are turned into a string
       * like "[[1, 2, 3], [4,5,6]]".
       * @type {!Array<number|string>}
       */
      values_: {type: Array, computed: 'computeValues_(arrayData, expanded)'},

      /**
       * Whether the component is expanded.
       * @type {boolean}
       */
      expanded: {type: Boolean, value: false},

      /**
       * Whether the component should be expandable.
       * @private {boolean}
       */
      expandable_: {type: Boolean, computed: 'computeExpandable_(arrayData)'},
    };
  }

  /**
   * Obsever for the property data. It will update the arrayData property.
   * @param {string} data
   */
  dataChanged_(data) {
    let parsedData = {};

    try {
      // Data is expected to be a serialized array of JSON objects.
      parsedData = JSON.parse(data);
    } catch (e) {
    }

    let values = [];
    const shape = parsedData['shape'];
    const shapeCount = shape && shape.length;
    if (shapeCount) {
      const dataType = parsedData['dataType'];
      const dataValuesKey = (dataType + '').toLowerCase() + 'Values';
      const dataValues = parsedData[dataValuesKey];
      const expectedElementCount = shape.reduce((acc, value) => acc * value, 1);
      if (dataValues && dataValues.length == expectedElementCount) {
        let temp1 = dataValues;
        let temp2;
        let temp3;
        for (let i = shapeCount - 1; i > 0; i--) {
          temp2 = [];
          const elementCount = shape[i];
          let count = 0;
          temp1.forEach(element => {
            if (!count) {
              temp3 = [];
            }
            temp3.push(element);
            count++;
            if (count == elementCount) {
              temp2.push(temp3);
              count = 0;
            }
          });
          temp1 = temp2;
        }
        values = temp1;
      }
    }

    this.arrayData = values;
  }

  /**
   * Determines the values to render.
   * @param {(!Array<number|!Object>|undefined)} arrayData
   * @param {boolean} expanded
   * @return {!Array<string>}
   * @private
   */
  computeValues_(arrayData, expanded) {
    return (arrayData || [])
        .filter((v, index) => {
          return expanded || index < 3;
        })
        .map(entry => JSON.stringify(entry).replace(/,/gi, ', '));
  }

  /**
   * @param {!Array<number|!Object>} arrayData
   * @return {boolean} Whether the cell should be expandable.
   */
  computeExpandable_(arrayData) {
    return arrayData.length > 3;
  }

  /**
   * Toggles if the cell is expanded or not.
   * @param {!Event} e
   */
  toggleExpanded_(e) {
    this.expanded = !this.expanded;
  }
}

customElements.define('tfma-array-value', ArrayValue);
