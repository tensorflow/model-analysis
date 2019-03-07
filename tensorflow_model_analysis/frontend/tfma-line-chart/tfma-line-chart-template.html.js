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
    background-color: white;
  }

  #title {
    padding-left: 0;
    padding-top: 0;
    font-size: 14px;
    font-family: 'Roboto', 'Noto', sans-serif;
    color: rgba(0,0,0,0.87);
  }

  google-chart {
    width: 400px;
    height: 225px;
  }
</style>
<div id="title" hidden$="[[!!title]]">
  {{title}}
</div>
<google-chart id="chart" type="line" events="[[events_]]"
              on-google-chart-onmouseover="onMouseOver_"
              on-google-chart-onmouseout="onMouseOut_" options="[[options_]]" data="[[data]]">
</google-chart>
`;
export {template};
