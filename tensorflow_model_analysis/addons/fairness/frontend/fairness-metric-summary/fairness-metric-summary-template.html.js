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
  .header {
    font-weight: 600;
    padding: 20px 0 8px 30px;
    color: #666;
  }
  #settings-icon {
    padding: 0 12px;
  }
  .check {
    width:20px;
    height:20px;
    border:1px solid grey;
    margin-right: 6px;
  }
  .check iron-icon{
    display: none;
    width: 20px;
    height: 20px;
    margin-top: -6px;
  }
  .iron-selected .check iron-icon {
    display: inline-flex;
  }
  #table {
    margin-top: 10px;
    width: 1000px;
  }

</style>
<div id="metric-header" class="header">
  [[metric]]
  <paper-icon-button id="settings-icon" icon="settings" on-tap="openSettings_">
  </paper-icon-button>
</div>
<fairness-bounded-value-bar-chart id="bar-chart" metrics="[[metricsForBarChart_]]"
                                 data="[[data]]" slices="{{slicesToPlot_}}"
                                 baseline="[[baseline]]">
</fairness-bounded-value-bar-chart>
<fairness-metrics-table id="table" metrics="[[metricsForTable_]]" data="[[tableData_]]"
                        header-override="[[headerOverride_]]" example-counts="[[exampleCounts_]]">
</fairness-metrics-table>
<paper-dialog id="settings">
  <div class="header">
    Config for [[metric]]
  </div>
  <div style="display:flex;">
    <div>
      <iron-label>Slices to Compare</iron-label>
      <div style="max-height: 360px; overflow-y:scroll;">
        <paper-listbox multi selected-values="{{configSelectedSlices_}}" attr-for-selected="slice">
          <template is="dom-repeat" items="[[configSelectableSlices_]]">
            <paper-item slice="[[item.slice]]" disabled="[[item.disabled]]">
              <div class="check">
                <iron-icon icon="icons:check">
                </iron-icon>
              </div>
              [[item.slice]]
            </paper-item>
          </template>
        </paper-listbox>
      </div>
    </div>
  </div>
  <paper-button on-tap="updateConfig_">Update</paper-button>
</paper-dialog>

`;
export {template};
