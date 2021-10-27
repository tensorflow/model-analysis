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
goog.module('tfma.SeriesDataHelper');

/**
 * @typedef {{
 *   config: !Object,
 *   data: !tfma.Data
 * }}
 *    config: The configuration of the evaluation run. Each implementation of
 *        the Helper should know about how to process the config object.
 *    data: The model's metric values along with a list of metrics. See
 *        tfma.Data definition.
 */
let EvalRun;

/**
 * Defines an interface that a client could implement to suit its own work
 * environment so that model and data information would show up properly in time
 * series view.
 * @interface
 */
class SeriesDataHelper {
  constructor() {}

  /**
   * @param {!Object<!Object>} specifiedFormats
   * @return {!Object<!Object>} A formats object based on the specified formats
   *     given. Each value should be of type
   *     tfma.Constants.MetricValueFormatSpec.
   * @export
   */
  getFormats(specifiedFormats) {}

  /**
   * @param {!Object} config
   * @return {string|number}
   * @export
   */
  getModelId(config) {}

  /**
   * @param {!Object} config
   * @return {string}
   * @export
   */
  getModelDisplayText(config) {}

  /**
   * @param {!Object} config
   * @return {string|number}
   * @export
   */
  getDataVersion(config) {}

  /**
   * @param {!Object} config
   * @return {string}
   * @export
   */
  getDataDisplayText(config) {}

  /**
   * Sorts the eval runs with lexical order.
   * @param {!Array<EvalRun>} evalRuns
   * @param {boolean} modelCentric
   * @return {!Array<EvalRun>}
   * @export
   */
  sortEvalRuns(evalRuns, modelCentric) {}

  /**
   * @return {string} The header that should be used for model column in metrics
   *     table.
   * @export
   */
  getModelHeader() {}

  /**
   * @return {string} The header that should be used for data column in metrics
   *     table.
   * @export
   */
  getDataHeader() {}

  /**
   * @return {!Array<string>} The header that should be used for additional column in metrics
   *     table.
   * @export
   */
  getAdditionalHeaders() {}

  /**
   * @param {!Object} config
   * @return {!Array<string>}
   * @export
   */
  getAdditionalDisplayTexts(config) {}
}

exports = {
  EvalRun,
  SeriesDataHelper
};
