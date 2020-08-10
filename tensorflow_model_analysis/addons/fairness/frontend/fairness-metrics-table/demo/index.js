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

const NUM_SLICES = 10;

(() => {
  var metrics = [];
  metrics.push('false_negative_rate@0.50');
  metrics.push('false_negative_rate@0.50 against Overall');
  metrics.push('false_negative_rate@0.25');
  metrics.push('false_negative_rate@0.25 against Overall');

  var data = [];
  var dataCompare = [];
  var exampleCounts = [];

  const baseline = Math.random();

  function generateDoubleEntry(sliceName) {
    var entry = {};
    entry['slice'] = sliceName;

    var metric = {};
    metric['false_negative_rate@0.50'] = Math.random();
    metric['false_negative_rate@0.50 against Overall'] =
        metric['false_negative_rate@0.50'] - baseline;
    metric['false_negative_rate@0.25'] = Math.random();
    metric['false_negative_rate@0.25 against Overall'] =
        metric['false_negative_rate@0.25'] - baseline;
    entry['metrics'] = metric;

    return entry;
  }

  function generateBoundedEntry(sliceName) {
    var entry = {};
    entry['slice'] = sliceName;

    const valAt50 = Math.random();
    const valAt25 = Math.random();
    var metric = {};
    metric['false_negative_rate@0.50'] = {
      'lowerBound': valAt50 - 0.1,
      'upperBound': valAt50 + 0.1,
      'value': valAt50,
    };
    metric['false_negative_rate@0.50 against Overall'] = valAt50 - baseline;
    metric['false_negative_rate@0.25'] = {
      'lowerBound': valAt25 - 0.2,
      'upperBound': valAt25 + 0.2,
      'value': valAt25,
    };
    metric['false_negative_rate@0.25 against Overall'] = valAt25 - baseline;
    entry['metrics'] = metric;

    return entry;
  }

  for (var i = 0; i < NUM_SLICES; i++) {
    const slice = (i < NUM_SLICES-1) ?
        'col:' + i :
        'this_is_a_veeeeeeeeeeeeeeeeeeeeery_long_feature_name_aka_col:' + i;
    data.push(generateDoubleEntry(slice));
    dataCompare.push(generateBoundedEntry(slice));
    exampleCounts.push(Math.floor(Math.random() * 100));
  }

  // Table with one model
  var singleTable = document.getElementById('single_table');
  singleTable.metrics = metrics;
  singleTable.data = data;
  singleTable.exampleCounts = exampleCounts;
  singleTable.evalName = 'modelA';

  // Table comparing two models
  var doubleTable = document.getElementById('double_table');
  doubleTable.metrics = metrics;
  doubleTable.data = data;
  doubleTable.dataCompare = dataCompare;
  doubleTable.exampleCounts = exampleCounts;
  doubleTable.evalName = 'modelA';
  doubleTable.evalNameCompare = 'modelB';
})();
