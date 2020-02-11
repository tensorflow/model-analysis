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

const NUM_SLICES = 100;

(() => {
  var metrics = [];
  metrics.push('false_negative_rate@0.50');
  metrics.push('false_negative_rate@0.50 against Overall');

  const table = document.getElementById('table');
  table.metrics = metrics;

  var data = [];
  var exampleCounts = [];

  const baseline = Math.random();

  for (var i = 0; i < NUM_SLICES; i++) {
    var entry = {};
    entry['slice'] = 'col:' + i;

    var metric = {};
    metric['false_negative_rate@0.50'] = Math.random();
    metric['false_negative_rate@0.50 against Overall'] =
        metric['false_negative_rate@0.50'] - baseline;

    entry['metrics'] = metric;
    data.push(entry);
    exampleCounts.push(Math.floor(Math.random() * 100));
  }

  table.data = data;
  table.exampleCounts = exampleCounts;
})();
