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
  paper-dialog ol {
    font-size: 18px;
    list-style-position: inside;
  }
  paper-dialog p {
    font-size: 18px;
  }
  paper-dropdown-menu[hidden] {
    display: none;
  }
  paper-icon-button {
    color: #5f6368;
  }
</style>

<div id="metric-header" class="header">
  [[stripPrefix(metric)]]
  <span>
    <paper-dialog id="dialog">
      <h2>Analysis</h2>
      <p>There are two visualizations generated for each metric - a <b>barchart</b> and a <b>table</b>.</p>
      <p>The <b>barchart</b> gives a visual representation of each slice's metric value. The baseline slice is a darker color than the other slices. If multiple thresholds are selected, each threshold is distinguished by a different color. Hovering over a bar provides information on the bar's slice threshold, exact value, confidence interval, and example count.</p>
      <p>The <b>table</b> contains a row for every slice, and a column for every metric. Some columns compare slices against the baseline (or between evals, if comparing evals). The final column gives the number of examples for a slice.</p>
      <div class="buttons">
        <paper-button dialog-confirm autofocus>Close</paper-button>
      </div>
    </paper-dialog>
    <paper-icon-button icon="info-outline" id="Information" on-click="openInfoDialog_">
  </span>
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
  <span>
    <paper-dialog id="dialog">
      <h2>Parameters</h2>
      <ol>
        <li><b>Slices</b>
          <p>
            For each metric, select a set of slices to evaluate using the "SELECT SLICES" dropdown.
            At most [[MAX_NUM_SLICES]] slices can be selected and rendered.
          </p>
        </li>
        <li><b>Thresholds</b>
          <p>
            If a metric is thresholded (meaning that its value depends on a model's classification threshold), it will have the option to select thresholds using the "Thresholds" dropdown. Multiple thresholds may be selected simultaneously.
          </p>
        </li>
        <li><b>Sort by</b>
          <p>
            This option is only available when comparing evaluations. Allows toggling the order of bars in the barchart. When "Slice" (default) is selected, the first pair of bars correspond to the evals' values for the first slice, the second pair correspond to the second slice, and so on. When "Eval" is selected, all the slices for the Eval 1 are displayed, followed by all the slices for Eval 2.
          </p>
        </li>
      </ol>
      <div class="buttons">
        <paper-button dialog-confirm autofocus>Close</paper-button>
      </div>
    </paper-dialog>
    <paper-icon-button icon="info-outline" id="Information" on-click="openInfoDialog_">
  </span>
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
