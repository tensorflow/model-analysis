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
goog.module('tfma.SingleSeriesGraphData');

const Constants = goog.require('tfma.Constants');
const Data = goog.require('tfma.Data');
const GraphData = goog.require('tfma.GraphData');

/**
 * The ratio to define the step size.
 * @private {number}
 */
const EXAMPLES_STEP_RATIO = 100;

/**
 * An implementation of {GraphData} for a single data series.
 * @implements {GraphData}
 */
class SingleSeriesGraphData {
  /**
   * @param {!Array<string>} metrics Names of the metrics. Note that index of
   *     the metric names must match to the index of corresponding metric value
   *     in the array of metric values list returned by any element in data.
   * @param {!Array<!Data.Series>} data An array of Series. Each element can
   *     represent a slice or a version of a model.
   */
  constructor(metrics, data) {
    /**
     * The stored Data object that represents the data set.
     * @private {Data.Data}
     */
    this.tfmaData_ = Data.build(metrics, data);
  }

  /**
   * Delegates the call to the filter method in the underlying data object.
   * @override
   */
  applyThreshold(columnName, threshold) {
    return this.tfmaData_.filter((dataSeries) => {
      const value = this.tfmaData_.getMetricValue(
                        dataSeries.getFeatureString(), columnName) ||
          0;
      return value >= threshold;
    });
  }

  /** @override */
  getColumnSteppingInfo(columnName) {
    const range = this.tfmaData_.getColumnRange(columnName);
    const step =
        Math.min(Constants.PlotDataDisplay.EXAMPLES_MAX_STEP,
          Math.ceil(range.max / EXAMPLES_STEP_RATIO));
    const max = Math.ceil((range.max + 1) / step) * step;
    return {max: max, step: step};
  }

  /** @override */
  getFeatures() {
    return this.tfmaData_.getFeatures();
  }

  /**
   * This method just returns the given argument, as the table repesentation
   * of the single series data is exactly the same as the data itself.
   * @override
   */
  getTableDataFromDataset(data) {
    return data;
  }
}

goog.exportSymbol('tfma.SingleSeriesGraphData', SingleSeriesGraphData);

exports = SingleSeriesGraphData;
