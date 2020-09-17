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
    metric['custom_metric@0.50'] = {
      'lowerBound': valAt50 - 0.1,
      'upperBound': valAt50 + 0.1,
      'value': valAt50,
    };
    metric['custom_metric@0.50 against Overall'] = valAt50 - baseline;
    metric['custom_metric@0.25'] = {
      'lowerBound': valAt25 - 0.2,
      'upperBound': valAt25 + 0.2,
      'value': valAt25,
    };
    metric['custom_metric@0.25 against Overall'] = valAt25 - baseline;
    entry['metrics'] = metric;

    return entry;
  }

  // Populate table data
  var dataFNR = [];
  var dataCustom = [];
  var dataCustomCompare = [];
  var exampleCounts = [];
  for (var i = 0; i < NUM_SLICES; i++) {
    const slice = (i < NUM_SLICES-1) ?
        'col:' + i :
        'this_is_a_veeeeeeeeeeeeeeeeeeeeery_long_feature_name_aka_col:' + i;
    dataFNR.push(generateDoubleEntry(slice));
    dataCustom.push(generateBoundedEntry(slice));
    dataCustomCompare.push(generateBoundedEntry(slice));
    exampleCounts.push(Math.floor(Math.random() * 100));
  }

  // Create single-model table
  var singleTable = document.getElementById('single_table');
  singleTable.metric = 'false_negative_rate';
  singleTable.metrics = [
    'false_negative_rate@0.50', 'false_negative_rate@0.50 against Overall',
    'false_negative_rate@0.25', 'false_negative_rate@0.25 against Overall'
  ];
  singleTable.data = dataFNR;
  singleTable.exampleCounts = exampleCounts;
  singleTable.evalName = 'modelA';

  // Create model-comparison table
  var doubleTable = document.getElementById('double_table');
  doubleTable.metric = 'custom_metric';
  doubleTable.metrics = [
    'custom_metric@0.50', 'custom_metric@0.50 against Overall',
    'custom_metric@0.25', 'custom_metric@0.25 against Overall'
  ];
  doubleTable.data = dataCustom;
  doubleTable.dataCompare = dataCustomCompare;
  doubleTable.exampleCounts = exampleCounts;
  doubleTable.evalName = 'modelA';
  doubleTable.evalNameCompare = 'modelB';
})();
