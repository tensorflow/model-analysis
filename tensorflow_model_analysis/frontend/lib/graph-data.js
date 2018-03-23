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
goog.module('tfma.GraphData');

const TableProvider = goog.require('tfma.TableProvider');

/**
 * The GraphData interface is used as a layer on top of {Data} to get
 * the correct objects required by the graph components, in order to plot the
 * visualizations correctly.
 * @interface
 */
class GraphData {
  /**
   * @param {string} columnName
   * @param {number} threshold
   * @return {tfma.Data} the data after applying the threshold for the given
   *      column. All the values that have values below the threshold
   *      should be removed from the returned data object.
   * @export
   */
  applyThreshold(columnName, threshold) {}

  /**
   * @param {string} columnName
   * @return {{max: number, step: number}}
   *     max: The max value of the given column value to be displayed on a
   *     slider. This is the smallest multiple of the slider step that is
   *     larger or equal than the max of examples, in order to cover all
   *     example values in the slider.
   *     step: The step the slider takes.
   * @export
   */
  getColumnSteppingInfo(columnName) {}

  /**
   * @return {!Array<string>} All the features of the dataset.
   * @export
   */
  getFeatures() {}

  /**
   * @param {tfma.Data} data
   * @return {TableProvider} A reference to the data in table format.
   * @export
   */
  getTableDataFromDataset(data) {}
}

exports = GraphData;
