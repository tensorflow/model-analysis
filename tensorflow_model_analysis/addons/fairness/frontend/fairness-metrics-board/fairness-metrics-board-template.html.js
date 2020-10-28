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
  paper-dialog p {
    font-size: 18px;
  }
  paper-icon-button {
    color: #5f6368;
  }
</style>
<fairness-privacy-container hidden$="[[!omittedSlices_.length]]"
                            omitted-slices="[[omittedSlices_]]">
</fairness-privacy-container>
<div class="config">
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
  <span>
    <paper-dialog id="dialog">
      <h2>Baseline</h2>
      <p>
        Select a slice from the "Baseline" dropdown. Slices are the subsets of data for which metrics are evaluated. The baseline is the slice which all other slices are compared to. By default, baseline is "Overall", meaning the entire dataset.
      </p>
      <div class="buttons">
        <paper-button dialog-confirm autofocus>Close</paper-button>
      </div>
    </paper-dialog>
    <paper-icon-button icon="info-outline" id="Information" on-click="openInfoDialog_">
  </span>
</div>
<template is="dom-repeat" items="[[metrics]]">
  <!-- The metrics could contain "undefined" elements because we want to maintain the -->
  <!-- index of selected metrics. If a metric is unselected, we will replace it with  -->
  <!-- "undefined". Otherwise, the user selected thresholds in the fairness-metric-summary -->
  <!-- will get overwritten. -->
  <template is="dom-if" if="[[item]]">
    <fairness-metric-summary data="[[data]]" eval-name="[[evalName]]"
                             metric="[[item]]" slices="[[slices_]]" baseline="[[baseline_]]"
                             data-compare="[[dataCompare]]" eval-name-compare="[[evalNameCompare]]">
    </fairness-metric-summary>
  </template>
</template>
`;
export {template};
