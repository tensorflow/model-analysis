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
  #metric-and-slice-selector-title {
    padding: 16px 16px 0 16px;
    /* We set font color to black because the Fairness widget background is always white.
     * Without explicitly setting it, the font color is selected by the Jupyter environment theme.
     */
    color: black
  }

  paper-listbox {
    width: 350px;
  }

  .metric-name {
    max-width: 290px;
    overflow: hidden;
    text-overflow: ellipsis;
    display: block;
  }
</style>
<div id="metric-and-slice-selector-title">
  Select metrics to display:
</div>
<paper-listbox multi attr-for-selected="item-name" selected-values="{{selectedMetrics}}" >
  <template is="dom-repeat" items="[[metricsSelectedStatus_]]">
    <paper-item item-name="[[item.metricsName]]">
      <paper-checkbox checked="[[item.selected]]">
         <span class="metric-name" title$="[[stripPostExport(item.metricsName)]]">
           [[stripPostExport(item.metricsName)]]
        </span>
      </paper-checkbox>
    </paper-item>
  </template>
</paper-listbox>
`;
export {template};
