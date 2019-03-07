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
  :host div.tr:nth-child(even) {
    background: #e4e4e4;
  }

  /**
   * There is a bad interaction between dom-repeat element, table tag and vulcanization. Use div
   * and equivalent display: table on class table, tr and td as a work-around.
   * @bug 22376520
   */
  :host div.table {
    text-align: center;
    display: table;
  }

  :host div.td {
    min-width: 15px;
    padding: 0 5px;
    display: table-cell;
  }

  :host div.tr {
    display: table-row;
  }
</style>
<div class="table">
    <template is="dom-repeat" items="{{formattedData_}}">
      <div class="tr">
        <div class="td">[[item.cutoff]]</div>
        <div class="td">[[item.value]]</div>
      </div>
    </template>
  </tbody>
</div>
`;
export {template};
