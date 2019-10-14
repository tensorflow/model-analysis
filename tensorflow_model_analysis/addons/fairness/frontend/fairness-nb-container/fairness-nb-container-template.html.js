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
    display:block;
    padding: 10px;
  }
  #metrics {
    width: 100%;
  }
  .flex-horizontal {
    @apply --layout-horizontal;
  }
</style>

<div class="container flex-horizontal">
  <paper-card>
    <fairness-metric-and-slice-selector available-metrics="[[selectableMetrics_]]"
                                 selected-metrics='{{selectedMetrics_}}'>
    </fairness-metric-and-slice-selector>
  </paper-card>
  <paper-card id="metrics">
    <fairness-metrics-board data="[[slicingMetrics]]" weight-column="[[weightColumn]]"
                      metrics="[[selectedMetrics_]]"
                      thresholds="[[fairnessThresholds_]]">
    </fairness-metrics-board>
  </paper-card>
</div>
`;
export {template};
