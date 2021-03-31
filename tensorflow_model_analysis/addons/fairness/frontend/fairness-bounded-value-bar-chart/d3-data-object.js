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

goog.module('tensorflow_model_analysis.addons.fairness.frontend.fairness_bounded_value_bar_chart.D3DataObject');

class D3DataObject {
  constructor(metricsData, labelNameParams) {
    this.fullSliceName = metricsData.length ? metricsData[0].fullSliceName : '';
    this.sliceValue = metricsData.length ? metricsData[0].sliceValue : '';
    this.evalName = metricsData.length ? metricsData[0].evalName : '';
    this.metricsData = metricsData;
    this.labelName = metricsData.length ? this.labelName_(labelNameParams) : '';
  }

  /**
   * Label name for a bar cluster.
   * @param {!Object} labelNameParams
   * @return {string}
   * @private
   */

  labelName_(labelNameParams) {
    if (!labelNameParams['evalComparison']) {
      return labelNameParams['sliceValue'];
    } else if (labelNameParams['sort'] === 'Slice') {
      return labelNameParams['evalName'] + '-' + labelNameParams['sliceValue'];
    } else {  // sort === 'Eval'
      return labelNameParams['sliceValue'] + '-' + labelNameParams['evalName'];
    }
  }
}

/**
 * Returns sorting function for D3DataObjects.
 * @param {string} baseline
 * @param {string} sort
 * @param {boolean} evalComparison
 * @return {function(!Object, !Object): number}
 */
exports.sortFunction = function(baseline, sort, evalComparison) {
  function sliceCompare(a, b) {
    // Ensure that the baseline slice always appears first.
    if (a.fullSliceName === baseline && b.fullSliceName === baseline) {
      return 0;
    } else if (a.fullSliceName === baseline) {
      return -1;
    } else if (b.fullSliceName === baseline) {
      return 1;
    } else {
      return a.fullSliceName.localeCompare(b.fullSliceName);
    }
  }

  if (evalComparison) {
    if (sort === 'Slice') {
      return (a, b) => {
        return sliceCompare(a, b) || a.evalName.localeCompare(b.evalName);
      };
    } else {  // Sort by eval name
      return (a, b) => {
        return a.evalName.localeCompare(b.evalName) || sliceCompare(a, b);
      };
    }
  } else {
    return (a, b) => {
      // Ensure that the baseline slice always appears on the left side.
      if (a.fullSliceName == baseline) {
        return -1;
      } else if (b.fullSliceName == baseline) {
        return 1;
      }
      // Sort by the first threshold value if multiple thresholds are present.
      for (let i = 0; i < a.metricsData.length; i++) {
        const diff = a.metricsData[i]['value'] - b.metricsData[i]['value'];
        if (diff != 0) {
          return diff;
        }
      }
      // If metrics are equal for both slices, go by alphabetical order.
      if (a['fullSliceName'] <= b['fullSliceName']) {
        return -1;
      } else {
        return 1;
      }
    };
  }
};

/**
 * Create a D3DataObject
 * @param {!Array<!Object>} metricsData a list of objects containing the metric
 *   values for a slice, including fullSliceName, sliceValue, evalName,
 *   metricName, value, upperBound, lowerBound, and exampleCount
 * @param {!Object} labelNameParams an object containing parameters to
 *   establish a slice's label, including sliceValue, evalName, evalComparison,
 *   and sort.
 * @return {!Object}
 */
exports.create = function(metricsData, labelNameParams){
   return new D3DataObject(metricsData, labelNameParams);
};
