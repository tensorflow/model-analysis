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
  paper-card {
    margin: 10px;
    padding: 10px;
  }
  #metrics {
    width: 100%;
    min-width: 600px;
  }
  #metrics-and-slice-selector {
    height: 100%
  }
  .flex-row {
    display: flex;
    flex-direction: row;
    flex-wrap: nowrap;
  }
  .flex-column {
    display: flex;
    flex-direction: column;
    flex-wrap: nowrap;
  }
  .evaluation-run {
    max-width: 290px;
    word-wrap: break-word;
    display: block;
  }
  #run-selector > paper-dropdown-menu {
    margin-left: 16px;
    width: 90%;
    margin-right: 16px;
  }
  #div-to-compare {
    margin-left: 16px;
    width: 90%;
    margin-right: 16px;
  }
  #drop-down-to-compare {
    width: 100%;
  }
  #model-comparison {
    max-width: 290px;
    overflow: hidden;
    text-overflow: ellipsis;
    display: block;
    margin-top: 5px;
  }
</style>

<div class="flex-row">
  <div class="flex-column">
    <paper-card id="run-selector"
                hidden$="[[hideRunSelector_(hideSelectEvalRunDropDown, availableEvaluationRuns)]]">
        <paper-dropdown-menu label="Select evaluation run:" title$="[[selectedEvaluationRun]]">
          <paper-listbox selected="{{selectedEvaluationRun}}" attr-for-selected="run"
                         class="dropdown-content" slot="dropdown-content" title$="">
            <template is="dom-repeat" items="[[availableEvaluationRuns]]">
              <paper-item run="[[item]]">
                <span class="evaluation-run" title$="[[item]]">[[item]]</span>
              </paper-item>
            </template>
          </paper-listbox>
        </paper-dropdown-menu>
        <paper-item>
          <paper-checkbox id="model-comparison" checked="{{modelComparisonEnabled_}}">
             <span class="model-comparison" title$="Enable Model Comparison">
               Enable Model Comparison
            </span>
          </paper-checkbox>
        </paper-item>
        <div id="div-to-compare" hidden$="[[!modelComparisonEnabled_]]">
          <paper-dropdown-menu id="drop-down-to-compare"
                               label="Select evaluation run to compare:"
                               title$="[[selectedEvaluationRunToCompare]]">
            <paper-listbox selected="{{selectedEvaluationRunToCompare}}" attr-for-selected="run"
                           class="dropdown-content" slot="dropdown-content" title$="">
              <template is="dom-repeat" items="[[availableEvaluationRuns]]">
                <paper-item run="[[item]]">
                  <span class="evaluation-run" title$="[[item]]">[[item]]</span>
                </paper-item>
              </template>
            </paper-listbox>
          </paper-dropdown-menu>
        </div>
    </paper-card>
    <paper-card id="metrics-and-slice-selector">
      <fairness-metric-and-slice-selector available-metrics="[[selectableMetrics_]]"
                                   selected-metrics='{{selectedMetrics_}}'>
      </fairness-metric-and-slice-selector>
    </paper-card>
  </div>
  <paper-card id="metrics">
    <fairness-metrics-board data="[[flattenSlicingMetrics_]]"
                            data-compare="[[flattenSlicingMetricsCompare_]]"
                            eval-name="[[evalName]]" eval-name-compare="[[evalNameCompare]]"
                            weight-column="[[weightColumn]]" metrics="[[selectedMetrics_]]">
    </fairness-metrics-board>
  </paper-card>
</div>
`;
export {template};
