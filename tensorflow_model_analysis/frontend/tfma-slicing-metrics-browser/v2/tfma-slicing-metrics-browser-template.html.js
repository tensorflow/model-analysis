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
<tfma-graph-data-filter id="graph" data="[[graphData_]]"
                        weighted-examples-column="[[weightedExamplesColumn]]"
                        selected-features="[[selectedFeatures_]]"
                        table-data="{{metricsTableData_}}">
</tfma-graph-data-filter>
<tfma-metrics-table id="table" data="[[metricsTableData_]]" metrics="[[metrics]]"
                    metric-formats="[[metricFormats_]]">
</tfma-metrics-table>
`;
export {template};
