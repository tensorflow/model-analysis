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
var makeData = function(count) {
  const values = [];
  for (let i = 0; i <= count; i++) {
    values.push({'cutoff': i, 'value': 1 - 0.1 * i});
  }
  return {'values': values};
};
const precisionTable = document.querySelector('#metric-at-k');
precisionTable.data = JSON.stringify(makeData(8));
}());
