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
/**
 * @fileoverview Demo JS that provides the precision-at-k component with test
 *     data.
 */

(function() {
function makeData(count) {
  const precision = [];
  for (let i = 0; i <= count; i++) {
    const value = {'k': i, 'macroValue': 1 - 0.1 * i, 'microValue': 1 - 0.11 * i};
    if (i % 3) {
      value['weightedValue'] = 0.5 + 0.01 * i;
    }
    if (i % 4) {
      value['microValue'] = 0;
    }
    precision.push(        value);
  }
  return precision;
}

const precisionTable = document.getElementById('metric-at-k');
precisionTable.data = JSON.stringify(makeData(8));
}());
