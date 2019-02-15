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

import {template} from './tfma-value-at-cutoffs-template.html.js';

/**
 * tfma-value-at-cutoffs renders a series of metric values at different cutoffs.
 * For example, precision at k with different k's.
 *
 * @polymer
 */
export class ValueAtCutoffs extends PolymerElement {
  constructor() {
    super();
  }

  static get is() {
    return 'tfma-value-at-cutoffs';
  }

  /** @return {!HTMLTemplateElement} */
  static get template() {
    return template;
  }

  /** @return {!PolymerElementProperties} */
  static get properties() {
    return {
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
    };
  }

  /**
   * @param {string} data
   * @return {!Array<number|string>|undefined} The formatted data
   */
  formatData_(data) {
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
        'value': (pair['value'] || 0).toFixed(tfma.FLOATING_POINT_PRECISION)
      };
    });
  }
}

customElements.define('tfma-value-at-cutoffs', ValueAtCutoffs);
