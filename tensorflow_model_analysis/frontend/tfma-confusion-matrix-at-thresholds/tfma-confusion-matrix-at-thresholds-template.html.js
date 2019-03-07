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
<style>
  .table, .header {
    display: flex;
  }
  .column {
    flex-basis: 100%;
    padding: 0 8px;
    justify-content: center;
    text-align:center;
  }
  :host[expanded] .table:hover {
    background-color: rgba(0,0,0,0.125);
  }
</style>
<div on-tap="toggleExpanded_">
  <div class="header">
    <div class="column subheader title">Threshold</div>
    <div class="column subheader title">Precision</div>
    <div class="column subheader title">Recall</div>
    <div class="column subheader title">TP</div>
    <div class="column subheader title">TN</div>
    <div class="column subheader title">FP</div>
    <div class="column subheader title">FN</div>
  </div>
  <template is="dom-repeat" items="[[displayedData_]]">
    <div class="table">
      <div class="column">
        [[item.threshold]]
      </div>
      <div class="column">
        [[item.precision]]
      </div>
      <div class="column">
        [[item.recall]]
      </div>
      <div class="column">
        [[item.truePositives]]
      </div>
      <div class="column">
        [[item.trueNegatives]]
      </div>
      <div class="column">
        [[item.falsePositives]]
      </div>
      <div class="column">
        [[item.falseNegatives]]
      </div>
    </div>
  </template>
  <template is="dom-if" if="[[expandable_]]">
    <paper-tooltip>
      More data available. Click anywhere to expand / collapse.
    </paper-tooltip>
  </template>
</div>
`;
export {template};
