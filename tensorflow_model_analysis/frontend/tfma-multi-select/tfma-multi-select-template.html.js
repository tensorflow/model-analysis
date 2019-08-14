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
const template = /** @type {!HTMLTemplateElement} */(document.createElement('template'));
template.innerHTML = `
<paper-dropdown-menu id="dropdown" value="[[selectedItemsString]]" label="[[label]]">
  <div slot="dropdown-content" on-iron-activate="stopEvent_" on-iron-select="stopEvent_">
    <paper-listbox id="listbox" multi selected-values="{{selectedIndices_}}">
      <template is="dom-repeat" items="[[items]]">
        <paper-item>[[item]]</paper-item>
      </template>
    </paper-listbox>
  </div>
</paper-dropdown-menu>
`;
export {template};
