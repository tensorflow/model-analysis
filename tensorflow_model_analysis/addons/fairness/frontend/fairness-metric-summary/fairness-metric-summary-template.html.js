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
  .slices-drop-down-menu {
    min-width: 200px;
  }
  .slices-drop-down-menu .slice-key-true{
    text-transform: uppercase;
  }
  .slices-drop-down-menu .slice-key-false{
    padding-left: 20px;
  }
  .slices-drop-down-candidates {
    --paper-item-min-height: 43px; /* It's 48px by default.*/
  }
  #table {
    margin-top: 10px;
    width: 1000px;
  }
  .config {
    margin: 0 30px;
  }
  paper-dropdown-menu[hidden] {
    display: none;
  }
</style>

<div id="metric-header" class="header">
  [[metric]]
</div>
<div class="config">
  <paper-button raised on-tap="openSlicesDropDownMenu_">
    Select Slices
    <iron-icon icon="icons:arrow-drop-down"></iron-icon>
  </paper-button>
  <iron-dropdown id="SlicesDropDownMenu">
    <paper-listbox multi selected-values="{{selectedSlicesDropDownMenuCandidates_}}"
                   attr-for-selected="slice"
                   class="dropdown-content slices-drop-down-menu"
                   slot="dropdown-content"
                   on-iron-select="slicesDropDownCandidatesSelected_"
                   on-iron-deselect="slicesDropDownCandidatesUnselected_"
                   >
      <template is="dom-repeat" items="[[slicesDropDownMenuCandidates_]]">
        <paper-item slice="[[item]]" disabled="[[item.isDisabled]]" class="slices-drop-down-candidates">
          <div class$="[[slicesDropDownCandidatesClass_(item)]]">
             <template is="dom-if" if="[[item.isSelected]]">
              <iron-icon icon="icons:check-box"></iron-icon>
            </template>
            <template is="dom-if" if="[[!item.isSelected]]">
              <iron-icon icon="icons:check-box-outline-blank"></iron-icon>
            </template>
            [[item.text]]
          </div>
        </paper-item>
      </template>
    </paper-listbox>
  </iron-dropdown>
  <paper-dropdown-menu opened="{{thresholdsMenuOpened_}}" label="Thresholds"
                       hidden$="[[!isMetricThresholded_(thresholds_)]]">
    <paper-listbox id="thresholdsList" multi selected-values="{{selectedThresholds_}}"
                   attr-for-selected="threshold"
                   class="dropdown-content" slot="dropdown-content">
      <template is="dom-repeat" items="[[thresholds_]]">
        <paper-item threshold="[[item]]">
          [[item]]
        </paper-item>
      </template>
    </paper-listbox>
  </paper-dropdown-menu>
  <paper-dropdown-menu label="Sort by" hidden$="[[!hasEvalComparison_()]]">
    <paper-listbox attr-for-selected="item-name" selected="{{sort_}}" class="dropdown-content" slot="dropdown-content">
      <paper-item item-name="Slice">Slice</paper-item>
      <paper-item item-name="Eval">Eval</paper-item>
    </paper-listbox>
  </paper-dropdown-menu>
</div>
<fairness-bounded-value-bar-chart id="bar-chart" metrics="[[metrics_]]"
                                  data="[[data]]" data-compare="[[dataCompare]]"
                                  slices="[[slicesToPlot_]]" baseline="[[baseline]]"
                                  eval-name="[[evalName]]" eval-name-compare="[[evalNameCompare]]"
                                  sort="[[sort_]]">
</fairness-bounded-value-bar-chart>
<fairness-metrics-table id="table" metric="[[metric]]" metrics="[[metricsForTable_]]"
                        data="[[tableData_]]" data-compare="[[tableDataCompare_]]"
                        eval-name="[[evalName]]" eval-name-compare="[[evalNameCompare]]"
                        example-counts="[[exampleCounts_]]">
</fairness-metrics-table>

`;
export {template};
