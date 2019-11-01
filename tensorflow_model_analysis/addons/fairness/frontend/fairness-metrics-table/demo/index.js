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
  for (var i = 0; i < 7; i++) {
    metrics.push('auprc_' + i);
    metrics.push('count_' + i);
    metrics.push('boundedAuc_' + i);
  }

  const table = document.getElementById('table');
  table.metrics = metrics;

  var data = [];
  var exampleCounts = [];

  for (var i = 0; i < NUM_SLICES; i++) {
    var entry = {};
    entry['slice'] = 'col:' + i;

    var metric = {};
    for (var j = 0; j < 7; j++) {
      metric['auprc_' + j] = '100';
      metric['count_' + j] = 0.7;
      metric['boundedAuc_' + j] = {
        'value': 0.61,
        'lowerBound': 0.60,
        'upperBound': 0.62
      };
      metric['random_' + j] = 'rand';
    }

    entry['metrics'] = metric;
    data.push(entry);
    exampleCounts.push(Math.floor(Math.random() * 100));
  }

  table.data = data;
  table.exampleCounts = exampleCounts;
})();
