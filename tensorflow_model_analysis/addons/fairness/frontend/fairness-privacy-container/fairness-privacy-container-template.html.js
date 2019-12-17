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
const template = /** @type {!HTMLTemplateElement} */(document.createElement('template'));
template.innerHTML = `

<style>
  .popup {
    margin:  0 10px;
    padding: 16px;
  }
  #privacy-dialog {
    width: 50%;
  }
  #omitted-slices-list {
    max-height: 300px;
    overflow-x: hidden;
  }
  .omitted-slices-listitem {
    font-size: 14px;
    font-style: italic;
  }
</style>
<h3>
  Note: Some feature slices with smaller example count might have been omitted because of
  privacy concerns.
  <paper-button id='paper-button' raised on-tap="openDialog_">Click here</paper-button> to learn
  more.
</h3>
<paper-dialog id="privacy-dialog">
  <h2>Privacy: k-anonymity</h2>
  <div class="popup">
    <p>
      If the number of examples for a specific slice is smaller than the k_anonymization_count
      specified as part of your model evaluation, then aggregated data for that slice won't
      be displayed. This will be useful to ensure privacy.
    </p>
    Here is the list of all such slices being omitted:<br>
    <paper-listbox id="omitted-slices-list">
      <template is="dom-repeat" items="[[omittedSlices]]">
        <paper-item class="omitted-slices-listitem">[[item]]</paper-item>
      </template>
    </paper-listbox>
  </div>
</paper-dialog>

`;
export {template};
