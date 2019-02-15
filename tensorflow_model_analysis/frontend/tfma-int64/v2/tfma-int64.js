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
import {html, PolymerElement} from '@polymer/polymer/polymer-element.js';

/**
 * tfma-int64 renders a string represening int64.
 *
 * @polymer
 */
export class Int64 extends PolymerElement {
  constructor() {
    super();
  }

  static get is() {
    return 'tfma-int64';
  }

  /** @return {!HTMLTemplateElement} */
  static get template() {
    return html`
    <style>
      #int64 {
        text-align: right;
      }
    </style>
    <div id="int64">
      [[data]]
    </div>
`;
  }

  /** @return {!PolymerElementProperties} */
  static get properties() {
    return {
      /**
       * The string representing the int64 value.
       * @type {string}
       */
      data: {type: String},
    };
  }
}

customElements.define('tfma-int64', Int64);
