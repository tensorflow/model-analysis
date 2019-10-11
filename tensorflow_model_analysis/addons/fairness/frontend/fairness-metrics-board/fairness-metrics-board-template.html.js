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
  .config {
    margin: 0 16px;
  }
  .config > span {
    padding: 0 12px;
  }
</style>
<fairness-privacy-container hidden$="[[!omittedSlices_.length]]"
                            omitted-slices="[[omittedSlices_]]">
</fairness-privacy-container>
<div class="config">
  <span>
    <paper-dropdown-menu label="Baseline">
      <paper-listbox selected="{{baseline_}}" attr-for-selected="slice"
                     class="dropdown-content" slot="dropdown-content">
        <template is="dom-repeat" items="[[slices_]]">
          <paper-item slice="[[item]]">
            [[item]]
          </paper-item>
        </template>
      </paper-listbox>
    </paper-dropdown-menu>
  </span>
  <span>
    <paper-dropdown-menu opened="{{thresholdsMenuOpened_}}" label="Thresholds">
      <paper-listbox id="thresholdsList" multi selected-values="{{selectedThresholds_}}"
                     attr-for-selected="threshold"
                     class="dropdown-content" slot="dropdown-content">
        <template is="dom-repeat" items="[[thresholds]]">
          <paper-item threshold="[[item]]">
            [[item]]
          </paper-item>
        </template>
      </paper-listbox>
    </paper-dropdown-menu>
  </span>
</div>
<template is="dom-repeat" items="[[metrics]]">
  <fairness-metric-summary data="[[data]]"
                                   metric="[[item]]" slices="[[slices_]]" baseline="[[baseline_]]"
                                   thresholds="[[thresholdsToPlot_]]">
  </fairness-metric-summary>
</template>
`;
export {template};
