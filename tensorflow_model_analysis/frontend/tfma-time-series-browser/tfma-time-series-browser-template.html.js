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
  tfma-metrics-table {
    margin-top: 10px;
    width: 100%;
  }
</style>
<div>
  <tfma-line-chart-grid id="grid" provider="[[seriesData]]" metrics="[[metrics_]]"
                        blacklist="[[blacklist]]">
  </tfma-line-chart-grid>
  <tfma-metrics-table id="table" data="[[seriesData]]" metrics="[[metrics_]]"
                      metric-formats="[[metricFormats_]]">
  </tfma-metrics-table>
</div>
`;
export {template};
