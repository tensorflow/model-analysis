
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

goog.module('tfma.TestUtil');

/**
 * Creates a default data object for testing.
 * @return {!tfma.Data}
 */
function createDefaultTestData() {
  return tfma.Data.build(
      createDefaultMetricsList(), createDefaultMetricsResult());
}

/**
 * @return {!Array<string>} An array containing names of the default metrics.
 */
function createDefaultMetricsList() {
  return ['weightedExamples', 'metricA', 'metricB'];
}

/**
 * Creates default resutls for testing.
 * @return {!Array<!Object>}
 */
function createDefaultMetricsResult() {
  return [
    {
      'slice': 'col:1',
      'metrics': {'weightedExamples': 4, 'metricA': 0.333, 'metricB': 0.002}
    },
    {
      'slice': 'col:2',
      'metrics': {'weightedExamples': 9, 'metricA': 0.997, 'metricB': 0.753}
    },
    {
      'slice': 'col:3',
      'metrics': {'weightedExamples': 1, 'metricA': 1.140, 'metricB': 0.198}
    },
    {
      'slice': 'col:4',
      'metrics': {'weightedExamples': 12, 'metricA': 0.180, 'metricB': 0.332}
    },
  ];
}

/**
 * Calls querySelectorAll.
 * @param {!Element} element
 * @param {string} selector
 * @return {!NodeList}
 */
function selectAll(element, selector) {
  return (element.shadowRoot || element).querySelectorAll(selector);
}

/**
 * @param {!Element} container
 * @param {string} selector
 * @param {string} text
 * @return {!Array<!Element>} All elements under container matching the selector
 *     with the given text.
 */
function getElementsWithText(container, selector, text) {
  const elements = selectAll(container, selector);
  const match = [];
  for (let element, i = elements.length - 1; element = elements[i]; i--) {
    if (element.textContent.trim() == text) {
      match.push(element);
    }
  }
  return match;
}

exports = {
  createDefaultTestData,
  createDefaultMetricsList,
  createDefaultMetricsResult,
  selectAll,
  getElementsWithText
};
