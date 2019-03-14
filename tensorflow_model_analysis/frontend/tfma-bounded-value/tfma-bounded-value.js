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
import {template} from './tfma-bounded-value-template.html.js';

/**
 * tfma-bounded-value renders a bounded value. The bounds can represent
 * confidence interval, numerical error, etc.
 *
 * @polymer
 */
export class BoundedValue extends PolymerElement {
  constructor() {
    super();
  }

  static get is() {
    return 'tfma-bounded-value';
  }

  /** @return {!HTMLTemplateElement} */
  static get template() {
    return template;
  }

  /** @return {!PolymerElementProperties} */
  static get properties() {
    return {
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
       * The value.
       * @type {string}
       */
      value: {type: Number},

      /**
       * The serialized form of the data.
       * @type {string}
       */
      data: {type: String, observer: 'dataChanged_'},
    };
  }

  /**
   * Observer for the property data.
   * @param {string} serializedData
   * @private
   */
  dataChanged_(serializedData) {
    if (serializedData) {
      try {
        const parsedData = JSON.parse(serializedData);
        this.upperBound = parsedData['upperBound'];
        this.lowerBound = parsedData['lowerBound'];
        this.value = parsedData['value'];
      } catch (e) {
      }
    }
  }

  /**
   * @param {string|number} value
   * @return {string} The given value formatted as a string.
   * @private
   */
  formatValue_(value) {
    return value == 'NaN' ? 'NaN' :
                            value.toFixed(tfma.FLOATING_POINT_PRECISION);
  }
}

customElements.define('tfma-bounded-value', BoundedValue);
