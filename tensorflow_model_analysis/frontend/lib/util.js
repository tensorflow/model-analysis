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
goog.module('tfma.Util');

/**
 * @typedef {{
 *   outputName: string,
 *   classId: string,
 * }}
 */
let ConfigListItem;

/**
 * Parses the nest configs object and convert it into an list of ConfigListItem.
 * @param {!Object} configs
 * @return {!Array<!ConfigListItem>}
 */
function createConfigsList(configs) {
  const configsList = [];
  for (let output in configs) {
    const classIds = configs[output] || [];
    classIds.forEach(classId => {
      configsList.push({outputName: output, classId: classId});
    });
  }
  return configsList;
}

/**
 * @param {!ConfigListItem} config
 * @param {boolean} compact
 * @return {string} Builds the metric prefix for the given config.
 */
function buildPrefix(config, compact) {
  const prefixArray = [];
  if (config.outputName != '') {
    prefixArray.push(config.outputName);
  }
  if (config.classId) {
    prefixArray.push(compact ? config.classId.split(':')[1] : config.classId);
  }
  if (prefixArray.length) {
    // If we are adding prefix, add an empty string so that we add an extra
    // slash just before the metric name.
    prefixArray.push('');
  }
  return prefixArray.join('/');
}

/**
 * @param {!Object} metricsMap The object containing metrics to be merged. We
 *     assume each level of the nested structure corresponds to one field in the
 *     metric key like output name and multi-class key. eg:
 * {
 *   output1: {
 *     classId:1: {
 *       auc: 0.81,
 *       accuracy: 0.71,
 *       ...
 *     },
 *   },
 *   output2: {
 *     topK:3: {
 *       auc: 0.82,
 *       accuracy: 0.72,
 *       ...
 *     }
 *   },
 *   ...
 * }
 * @param {!Array<!ConfigListItem>} configsList The list of configs to use.
 * @param {!Object<string>=} blacklist The metrics to omit in the merged result.
 * @return {!Object} The merged metrics. If only only one config is selected,
 *     there is no change to the metric names. When more than one config is
 *     selected, a prefix containig the metric key will be added to help
 *     disambiguate which configuration the metric is for. eg:
 * {
 *   output1/classId:1/auc: 0.81,
 *   output2/topK:3/auc: 0.82,
 *   ...
 * }
 */
function mergeMetricsForSelectedConfigsList(
    metricsMap, configsList, blacklist) {
  // Only add prefix if there are more than on config selected.
  const addPrefix = configsList.length > 1;
  const noBlacklist = !blacklist;
  const mergedMetrics = {};
  const classIdPrefixUsed = {};
  configsList.forEach(config => {
    const classIdParts = config.classId.split(':');
    if (classIdParts[0]) {
      classIdPrefixUsed[classIdParts[0]] = 1;
    }
  });
  // Use compact prefix if selected config have the same non-empty multi-class
  // key. eg: all class id.
  const useCompactPrefix = Object.keys(classIdPrefixUsed).length <= 1;
  configsList.forEach(config => {
    const outputMap = metricsMap[config.outputName] || {};
    const classResult = outputMap[config.classId] || {};
    const prefix = addPrefix ? buildPrefix(config, useCompactPrefix) : '';
    for (let metricName in classResult) {
      if (noBlacklist || !blacklist[metricName]) {
        mergedMetrics[prefix + metricName] = classResult[metricName];
      }
    }
  });
  return mergedMetrics;
}

goog.exportSymbol('tfma.Util.createConfigsList', createConfigsList);
goog.exportSymbol(
    'tfma.Util.mergeMetricsForSelectedConfigsList',
    mergeMetricsForSelectedConfigsList);

exports = {
  createConfigsList,
  mergeMetricsForSelectedConfigsList,
};
