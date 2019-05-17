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
  #controls {
    margin: 0 auto 20px;
    width: 724px;
  }
  #controls paper-dropdown-menu {
    width: 320px;
    padding: 0 12px;
  }
</style>
<div id="controls">
  <paper-dropdown-menu label="Show">
    <paper-listbox class="dropdown-content" selected="{{metricToShow}}"
                   attr-for-selected="value" slot="dropdown-content">
      <template is="dom-repeat" items="[[metrics_]]">
        <paper-item value="[[item]]">
          [[item]]
        </paper-item>
      </template>
    </paper-listbox>
  </paper-dropdown-menu>
  <paper-dropdown-menu label="Sort by">
    <paper-listbox class="dropdown-content" selected="{{metricToSort_}}"
                   attr-for-selected="value" slot="dropdown-content">
      <template is="dom-repeat" items="[[metricsForSorting_]]">
        <paper-item value="[[item]]">
          [[item]]
        </paper-item>
      </template>
    </paper-listbox>
  </paper-dropdown-menu>
</div>
<google-chart-loader id="loader" packages="[[chartPackages_]]"
                     on-google-chart-select="handleSelect_">
</google-chart-loader>
<div id="chart"></div>
`;
export {template};
