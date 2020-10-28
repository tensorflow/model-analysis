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
  #metric-and-slice-selector-title {
    padding: 16px 16px 0 16px;
    /* We set font color to black because the Fairness widget background is always white.
     * Without explicitly setting it, the font color is selected by the Jupyter environment theme.
     */
    color: black
  }
  paper-dialog p {
    font-size: 18px;
  }
  paper-icon-button {
    color: #5f6368;
  }
  paper-listbox {
    width: 350px;
  }
  .metric-name {
    max-width: 290px;
    overflow: hidden;
    text-overflow: ellipsis;
    display: block;
  }
</style>

<div id="metric-and-slice-selector-title">
  Select metrics to display:
  <span>
    <paper-dialog id="dialog">
      <h2>Metrics</h2>
      <p>
        Select metrics from the leftside column. Each metric will generate a barchart and table.
      </p><p>
        Hovering over a metric name will display a metric description tooltip.
      </p>
      <div class="buttons">
        <paper-button dialog-confirm autofocus>Close</paper-button>
      </div>
    </paper-dialog>
    <paper-icon-button icon="info-outline" id="Information" on-click="openInfoDialog_">
  </span>
</div>

<paper-item>
  <paper-checkbox id="selectAll" on-checked-changed="onSelectAllCheckedChanged_">
     <span class="metric-name" title$="Select all">
       Select all
    </span>
  </paper-checkbox>
</paper-item>

<paper-listbox multi attr-for-selected="metric"
               selected-values="{{selectedMetricsListCandidates_}}"
               on-iron-select="metricsListCandidatesSelected_"
               on-iron-deselect="metricsListCandidatesUnselected_">
  <template is="dom-repeat" items="[[metricsListCandidates_]]">
    <paper-item metric="[[item]]">
      <paper-checkbox checked="[[item.isSelected]]">
        <span class="metric-name" title$="[[getDefinition(item)]]">
          [[stripPrefix(item)]]
        </span>
      </paper-checkbox>
    </paper-item>
  </template>
</paper-listbox>
`;
export {template};
