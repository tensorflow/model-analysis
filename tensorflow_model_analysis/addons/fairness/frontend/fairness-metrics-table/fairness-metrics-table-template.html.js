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
  .table-holder {
    width: 100%;
    display: flex;
    flex-direction: column;
    background: #f8f9fa;
    border: 1px solid black;
    margin: inherit;
  }
  .table-row,
  .table-head-row {
    background: white;
    color: #3c4043;
    letter-spacing: 0.25;
    display: flex;
    position: relative;
    border-bottom: 1px solid #63696e;
    min-height: min-content;
    padding-top: 6px;
    padding-bottom: 6px;
    padding-right: 6px;
    padding-left: 6px;
  }
  .baseline-row {
    color: #3ca58b;
    font-weight: bold;
  }
  .baseline-row,
  .table-head-row {
    background: white;
    color: #3ca58b;
    font-weight: bold;
    letter-spacing: 0.25;
    display: flex;
    position: relative;
    border-bottom: 1px solid #63696e;
    min-height: min-content;
    padding-top: 6px;
    padding-bottom: 6px;
    padding-right: 6px;
    padding-left: 6px;
  }
  .table-head-row {
    background: #e8e8e8;
    font-weight: 500;
    color: #63696e;
  }
  .table-entry,
  .table-feature-column {
    text-align: left;
    padding-top: 8px;
    padding-bottom: 8px;
    font-size: 12px;
    min-height: min-content;
    width: 100%;
    padding-left: 6px;
    padding-right: 6px;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .table-feature-column {
    text-align: left;
  }
  .baseline-row:hover {
    background-color: #f8f8f8;
  }
  .table-row:hover {
    background-color: #f8f8f8;
  }
  .blue-icon {
    --iron-icon-fill-color: blue;
  }
  .orange-icon {
    --iron-icon-fill-color: orange;
  }
</style>
<div class="table-holder" id="table">
  <template is="dom-repeat" items="[[plotData_]]" as="row">
    <template is="dom-if" if="[[isHeaderRow_(index)]]">
      <div class="table-head-row">
        <template is="dom-repeat" items="[[row]]">
          <template is="dom-if" if="[[!index]]">
            <div class="table-feature-column" title="[[item]]">[[item]]</div>
          </template>
          <template is="dom-if" if="[[index]]">
            <div class="table-entry" title="[[item]]">[[item]]</div>
          </template>
        </template>
        <div class="table-feature-column" title="[[item]]">example_count</div>
      </div>
    </template>
    <template is="dom-if" if="[[isBaselineRow_(index)]]">
      <div class="baseline-row" on-tap="togglePerfRow">
        <template is="dom-repeat" items="[[row]]">
          <template is="dom-if" if="[[!index]]">
            <div class="table-feature-column" title="[[item]]">[[item]]</div>
          </template>
          <template is="dom-if" if="[[index]]">
            <div class="table-entry" title="[[item]]">
              <template is="dom-if" if="[[isDiffWithBaselineColumn_(index)]]">
                <template is="dom-if" if="[[!isZero_(item)]]">
                  <template is="dom-if" if="[[isPositive_(item)]]">
                    <iron-icon icon="arrow-upward" class="orange-icon"></iron-icon>
                  </template>
                  <template is="dom-if" if="[[isNegative_(item)]]">
                    <iron-icon icon="arrow-downward" class="blue-icon"></iron-icon>
                  </template>
                  [[toPercentage_(item)]]
                </template>
              </template>
              <template is="dom-if" if="[[!isDiffWithBaselineColumn_(index)]]">
                [[formatFloatValue_(item)]]
              </template>
            </div>
          </template>
        </template>
        <div class="table-entry" title="[[item]]">
          [[getExampleCount_(index, exampleCounts)]]
        </div>
      </div>
    </template>
    <template is="dom-if" if="[[isSliceRow_(index)]]">
      <div class="table-row" on-tap="togglePerfRow">
        <template is="dom-repeat" items="[[row]]">
          <template is="dom-if" if="[[!index]]">
            <div class="table-feature-column" title="[[item]]">[[item]]</div>
          </template>
          <template is="dom-if" if="[[index]]">
            <div class="table-entry" title="[[item]]">
              <template is="dom-if" if="[[isDiffWithBaselineColumn_(index)]]">
                <template is="dom-if" if="[[!isZero_(item)]]">
                  <template is="dom-if" if="[[isPositive_(item)]]">
                    <iron-icon icon="arrow-upward" class="orange-icon"></iron-icon>
                  </template>
                  <template is="dom-if" if="[[isNegative_(item)]]">
                    <iron-icon icon="arrow-downward" class="blue-icon"></iron-icon>
                  </template>
                  [[toPercentage_(item)]]
                </template>
              </template>
              <template is="dom-if" if="[[!isDiffWithBaselineColumn_(index)]]">
                [[formatFloatValue_(item)]]
              </template>
            </div>
          </template>
        </template>
        <div class="table-entry" title="[[item]]">
          [[getExampleCount_(index, exampleCounts)]]
        </div>
      </div>
    </template>
  </template>
</div>
`;
export {template};
