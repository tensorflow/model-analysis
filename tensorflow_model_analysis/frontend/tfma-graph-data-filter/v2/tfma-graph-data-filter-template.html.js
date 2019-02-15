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
  #table {
    display: block;
    width: calc(100% - 240px);
    margin: 20px 120px 60px 120px;
  }

  .ui {
    margin: 20px auto 0;
    /* Two ui-inputs, each 260px (have 10px soft margin) */
    width: calc(260px * 2);
  }

  .ui-input {
    width: 250px;
  }

  .ui-element {
    display: inline-block;
    margin-right: 2px;
  }

  .placeholder {
    height: 32px;
    display: flex;
  }

  ::content .links {
    text-transform: none;
    font-weight: normal;
    text-decoration: underline;
    font-size: 13px;
    color: #337ab7;
  }
</style>
<div class="section ui">
  <span id="chart-type" class="ui-element ui-input">
    <paper-dropdown-menu label="Visualization">
      <paper-listbox class="dropdown-content" selected="{{chartType}}" slot="dropdown-content">
        <paper-item>Slices Overview</paper-item>
        <paper-item>Metrics Histogram</paper-item>
      </paper-listbox>
    </paper-dropdown-menu>
    <div class="placeholder"></div>
  </span>
  <span id="weighted-examples-threshold" class="ui-element ui-input">
    <paper-input-container always-float-label>
      <label slot="label">Examples (Weighted) Threshold</label>
      <input value="0" type="number" slot="input"></input>
    </paper-input-container>
    <paper-slider class="slider" min="0" max="[[weightedExamplesInfo_.max]]"
                  step="[[weightedExamplesInfo_.step]]" value="{{weightedExamplesThreshold_}}">
    </paper-slider>
  </span>
</div>
<iron-pages selected="[[chartType]]">
  <tfma-slice-overview slices="[[filteredData_]]" metric-to-show="[[weightedExamplesColumn]]" displayed="[[showSliceOverview_]]">
  </tfma-slice-overview>
  <div>
    <tfma-metrics-histogram id="histogram" data="[[filteredData_]]"
                            details-data="{{focusedData_}}"
                            weighted-examples-column="[[weightedExamplesColumn]]"
                            selected-features="[[selectedFeatures]]">
    </tfma-metrics-histogram>
  </div>
</iron-pages>
`;
export {template};
