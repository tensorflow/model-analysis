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
  :host {
    display: block;
  }

  .section {
    margin: 5px 0;
  }

  .ui {
    margin: auto;
    width: 680px; /* Three ui-elements (210px each) + options toggle (50px) */
  }

  .ui-input {
    width: 200px;
  }

  .ui-element {
    display: inline-block;
    margin-right: 2px;
  }

  .placeholder {
    height: 32px;
    display: flex;
  }

  #overview path.blue {
    fill: #3366cc;
  }

  #overview path.red {
    fill: #dc3912;
  }

  #overview rect.overview {
    fill: #ccc;
    stroke: #000;
    opacity: .1;
  }

  #overview rect#focus {
    fill: gray;
    stroke: #000;
    opacity: .2;
  }

  #details rect.highlighted {
    fill: #ffde5a;
  }

  #details g.highlighted text {
    fill: #212121;
    font-weight: bold;
  }

  #empty {
    margin: 20px auto 0;
    width: calc(100% - 240px);
    text-align: center;
  }
</style>
<google-chart-loader id="loader" packages="[[chartPackages_]]"></google-chart-loader>
<div class="ui section">
  <span id="metric-select" class="ui-element ui-input">
    <paper-dropdown-menu label="Select Metric">
      <paper-listbox class="dropdown-content" selected="{{metric}}" attr-for-selected="value"
                     slot="dropdown-content">
        <template is="dom-repeat" items="[[selectableMetrics_]]">
          <paper-item value="[[item]]">[[item]]</paper-item>
        </template>
      </paper-listbox>
    </paper-dropdown-menu>
    <div class="placeholder"></div>
  </span>
  <span id="type-select" class="ui-element ui-input">
    <paper-dropdown-menu label="Histogram Type">
      <paper-listbox class="dropdown-content" selected="{{type}}" attr-for-selected="value"
                     slot="dropdown-content">
        <paper-item value="unweighted">Slice Counts</paper-item>
        <paper-item value="weighted">Example Counts</paper-item>
        <paper-item value="both">Both</paper-item>
      </paper-listbox>
    </paper-dropdown-menu>
    <div class="placeholder"></div>
  </span>
  <span id="num-buckets" class="ui-element ui-input">
    <paper-input label="Number of Buckets" always-float-label value="{{numBuckets}}" auto-validate
                 pattern="[1-9]|[1-4][0-9]|50" error-message="must be between [1, 50]">
    </paper-input>
    <paper-slider class="slider" min="1" max="50" value="{{numBuckets}}"
                  immediate-value="{{numBuckets}}">
    </paper-slider>
  </span>
  <span class="ui-element">
    <paper-icon-button id="options-toggle" icon="settings" data-dialog="options"
                       on-tap="openOptions_">
      More Options
    </paper-icon-button>
    <div class="placeholder"></div>
    <paper-dialog id="options">
      <h2>Options</h2>
      <div>
        <paper-toggle-button id="logarithm-scale" type="checkbox" checked="{{logarithmScale}}">
          Logarithm Scale
        </paper-toggle-button>
      </div>
    </paper-dialog>
  </span>
</div>
<!--
  NOTE: Need an extra container around the SVG #overview where mouse action is performed. Otherwise,
  the shadow root will get the mouse event and js error will result from d3 on mousedown.
-->
<div>
  <svg id="overview" class="section">
    <g class="unweighted"></g>
    <g class="weighted"></g>
  </svg>
</div>
<div id="details"></div>
<paper-card id="empty">
  <div class="card-content">
    <paper-icon-button icon="block"></paper-icon-button>
    <div>Empty Histogram</div>
  </div>
</paper-card>
`;
export {template};
