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
  paper-card {
    margin-top: 5px;
    margin-right: 5px;
  }

  .close-btn {
    position: absolute;
    top: 0;
    right: 0;
    padding: 12px;
  }

  #add-series {
    display: block;
    width: 300px;
  }

  .charts {
    display: flex;
    flex-wrap: wrap;
  }
</style>
<paper-dropdown-menu label="[[addSeriesLabel_]]" id="add-series">
  <paper-listbox class="dropdown-content" selected="{{selectedMetric_}}"
                 attr-for-selected="value" slot="dropdown-content">
    <template is="dom-repeat" items="[[addableMetrics_]]">
      <paper-item value="[[item]]">[[item]]</paper-item>
    </template>
  </paper-listbox>
</paper-dropdown-menu>
<div class="charts" on-select="onChartSelect_" on-clear-selection="onChartClearSelection_">
  <template is="dom-repeat" items="[[selectedMetrics_]]">
    <paper-card id="card-[[item]]">
      <div class="card-content">
        <tfma-line-chart metric$="[[item]]" data="[[computeChartData_(item, provider)]]"
                         title="[[item]]">
        </tfma-line-chart>
      </div>
      <paper-icon-button metric$="[[item]]" icon="close" on-tap="closeLineChart_"
                         class="close-btn">
      </paper-icon-button>
    </paper-card>
  </template>
</div>
`;
export {template};
