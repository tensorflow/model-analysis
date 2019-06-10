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
  .center {
    display: flex;
    justify-content: center;
  }
  :host([loading]) #plots, :host([error_]) #plots-container {
    visibility: hidden;
  }
  #spinner {
    width: 32px;
    height: 32px;
    position: absolute;
    top: 50%;
    left: 50%;
    margin: -16px;
  }
  :host(:not([error_])) #reload {
    display: none;
  }
  #reload {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
  }
  #button-container {
    text-align: center;
    position: absolute;
    bottom: 24px;
    width: 100%;
  }
  #error-message {
    position: absolute;
    top: 50%;
    width: 100%;
    text-align: center;
    font-weight: 500;
    color: red;
  }
  paper-button.btn {
    background-color: #3f51b5;
    color: #fff;
    font-weight: bolder;
    display: inline-block;
    text-align: center;
  }
  paper-button.btn[disabled] {
    background: #eaeaea;
    color: #a8a8a8;
    font-weight: normal;
  }
</style>
<div style="position:relative">
  <div class="center">
    <h4>[[heading]]</h4>
  </div>
  <div id="reload">
    <div id="error-message">
      Error loading plot data.
    </div>
    <div id="button-container">
      <paper-button class="btn" on-tap="reload_">Reload</paper-button>
    </div>
  </div>
  <div id="plots-container">
    <iron-pages id="plots" attr-for-selected="name" selected="{{selectedTab_}}">
      <div name="[[tabNames_.Calibration]]" class="center">
        <tfma-calibration-plot buckets="[[calibrationData_]]" bucket-size="0.01">
        </tfma-calibration-plot>
      </div>
      <div name="[[tabNames_.Macro]]" class="center">
        <tfma-precision-recall-curve id="mapr" data="[[macroPrecisionRecallCurveData_]]">
        </tfma-precision-recall-curve>
      </div>
      <div name="[[tabNames_.Micro]]" class="center">
        <tfma-precision-recall-curve id="mipr" data="[[microPrecisionRecallCurveData_]]">
        </tfma-precision-recall-curve>
      </div>
      <div name="[[tabNames_.Weighted]]" class="center">
        <tfma-precision-recall-curve id="wpr" data="[[weightedPrecisionRecallCurveData_]]">
        </tfma-precision-recall-curve>
      </div>
      <div name="[[tabNames_.Precision]]" class="center">
        <tfma-precision-recall-curve id="pr" data="[[precisionRecallCurveData_]]">
        </tfma-precision-recall-curve>
      </div>
      <div name="[[tabNames_.Prediction]]" class="center">
        <tfma-prediction-distribution data="[[calibrationData_]]">
        </tfma-prediction-distribution>
      </div>
      <div name="[[tabNames_.ROC]]" class="center">
        <tfma-roc-curve data="[[precisionRecallCurveData_]]">
        </tfma-roc-curve>
      </div>
      <div name="[[tabNames_.Gain]]" class="center">
        <tfma-gain-chart data="[[precisionRecallCurveData_]]">
        </tfma-gain-chart>
      </div>
      <div name="[[tabNames_.Accuracy]]" class="center">
        <tfma-accuracy-charts data="[[precisionRecallCurveData_]]">
        </tfma-accuracy-charts>
      </div>
    </iron-pages>
    <paper-spinner id="spinner" active="[[loading]]"></paper-spinner>
    <div>
      <template is="dom-if" if="[[availableTypes.length]]">
        <paper-tabs selected="{{selectedTab_}}" attr-for-selected="name">
          <template is="dom-repeat" items="[[availableTabs_]]">
            <paper-tab name="[[item.type]]">[[item.text]]</paper-tab>
          </template>
        </paper-tabs>
      </template>
    </div>
  </div>
</div>
`;
export {template};
