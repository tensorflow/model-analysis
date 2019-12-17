/**
 * Copyright 2019 Google LLC
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

import '@polymer/paper-button/paper-button.js';
import '@polymer/paper-dialog/paper-dialog.js';
import '@polymer/paper-item/paper-item.js';
import '@polymer/paper-listbox/paper-listbox.js';

import {PolymerElement} from '@polymer/polymer/polymer-element.js';
import {template} from './fairness-privacy-container-template.html.js';

export class FairnessPrivacyContainer extends PolymerElement {
  constructor() {
    super();
  }

  static get is() {
    return 'fairness-privacy-container';
  }

  /** @return {!HTMLTemplateElement} */
  static get template() {
    return template;
  }

  /** @return {!PolymerElementProperties} */
  static get properties() {
    return {
      /** @type {!Array<string>} */
      omittedSlices: {type: Array},
    };
  }

  /**
   * Opens up the dialg box.
   * @private
   */
  openDialog_() {
    this.$['privacy-dialog'].open();
  }
}


customElements.define('fairness-privacy-container', FairnessPrivacyContainer);
