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
goog.module('tfma.SeriesData');

const LineChartProvider = goog.require('tfma.LineChartProvider');
const SeriesDataHelper = goog.require('tfma.SeriesDataHelper');
const TableProvider = goog.require('tfma.TableProvider');

/**
 * Lantern metric series data that contains the data for a list of model
 * versions. Each model version has its own lantern.Data.
 * @implements {TableProvider}
 * @implements {LineChartProvider}
 */
class SeriesData {
  /**
   * Lantern metric series data that contains the data for a list of model
   * versions. Each model version has its own lantern.Data.
   * @param {!Array<!SeriesDataHelper.EvalRun>} evalRuns
   * @param {boolean} modelCentric True if the data should be presented in
   *     a model centric way which shows how a model progresses over each run.
   *     ie: the main axis is model run id. If the value is false, the data
   *     should be presented in a data centric way which shows how the latest
   *     version of model performs as new evaluation data become available. ie:
   *     main axisis last data span.
   * @param {!SeriesDataHelper.SeriesDataHelper=} opt_helperImpl
   */
  constructor(evalRuns, modelCentric, opt_helperImpl) {
    /**
     * @private {boolean}
     */
    this.modelCentric_ = modelCentric;

    /**
     * @private {!SeriesDataHelper.SeriesDataHelper}
     */
    this.helper_ = opt_helperImpl || getDefaultHelper();

    /**
     * @private {!Array<SeriesDataHelper.EvalRun>}
     */
    this.evalRuns_ = this.helper_.sortEvalRuns(evalRuns, modelCentric);
  }

  /**
   * @override
   * @suppress {missingProperties} getMetricValue is unknown on 'data'
   */
  getLineChartData(metric) {
    const totalEntries = this.evalRuns_.length;
    return this.evalRuns_.map((evalRun, index) => {
      const config = /** @type {!Object} */ (evalRun.config);
      const metricValue = evalRun.data.getMetricValue('', metric);
      return [
        {
          // In the case the eval runs are sorted lexically, the runs at the
          // beginning of the list should show up at the end of the time series
          // plot. As a result, totalEntries - index is used as back up value in
          // time series plot.
          // @bug 76103045
          'v': this.getLineChartXCoord_(config, totalEntries - index),
          'f': this.helper_.getModelHeader() + ' ' +
              this.helper_.getModelDisplayText(config) + ' at ' +
              this.helper_.getDataHeader() + ' ' +
              this.helper_.getDataDisplayText(config)
        },
        this.helper_.getModelId(config),
        this.helper_.getModelHeader() + ': ' +
            this.helper_.getModelDisplayText(config),
        // GViz automatically rounds the values displayed in tooltip. Force it
        // to show the raw value.
        {'v': metricValue, 'f': metricValue}
      ];
    });
  }

  /**
   * @param {!Object} config
   * @param {number} index
   * @return {number} The x coordinate in the time series graph. If the
   *     information in the config cannot be parsed properly, use provided
   *     index instead.
   * @private
   */
  getLineChartXCoord_(config, index) {
    const value = this.modelCentric_ ? this.helper_.getModelId(config) :
                                       this.helper_.getDataVersion(config);
    return typeof value == 'number' ? value : index;
  }

  /**
   * @override
   * @suppress {missingProperties} getAllMetricValues is unknown on 'data'
   */
  getDataTable() {
    const helper = this.helper_;
    return this.evalRuns_.map(evalRun => {
      const values = evalRun.data.getAllMetricValues('');
      const config = evalRun.config;
      const modelText = helper.getModelDisplayText(config);
      const dataText = helper.getDataDisplayText(config);
      const column1 = this.modelCentric_ ? modelText : dataText;
      const column2 = this.modelCentric_ ? dataText : modelText;
      return [column1, column2].concat(values);
    });
  }

  /**
   * Generates a list of metrics that are available in all the evaluation runs.
   * @return {!Array<string>}
   * @export
   */
  getMetrics() {
    const metrics = {};
    this.evalRuns_.forEach((evalRun) => {
      const modelMetrics = evalRun.data.getMetrics();
      modelMetrics.forEach(function(metric) {
        if (metric in metrics) {
          metrics[metric]++;
        } else {
          metrics[metric] = 1;
        }
      });
    });
    return Object.keys(metrics).filter((metric) => {
      return metrics[metric] == this.evalRuns_.length;
    });
  }

  /** @override */
  getModelIds() {
    return this.evalRuns_.map(
        evalRun => this.helper_.getModelId(evalRun.config));
  }

  /**
   * @return {boolean} Whether the series data is empty, i.e. has no models or
   *     the models contains empty data.
   */
  isEmpty() {
    return !this.evalRuns_.length || this.evalRuns_[0].data.isEmpty();
  }

  /** @override */
  readyToRender() {
    return !this.isEmpty();
  }

  /** @override */
  getHeader(requiredColumns) {
    const model = this.helper_.getModelHeader();
    const data = this.helper_.getDataHeader();
    const column1 = this.modelCentric_ ? model : data;
    const column2 = this.modelCentric_ ? data : model;
    return [column1, column2].concat(requiredColumns);
  }

  /** @override */
  getFormats(specifiedFormats) {
    return this.helper_.getFormats(specifiedFormats);
  }

  /** @override */
  applyOverride(value, override) {
    const transform = override['transform'];
    return transform ? transform(value) : value;
  }

  /** @override */
  getEvalConfig(index) {
    const chosenEvalRun = this.evalRuns_[index];
    const chosenConfig = chosenEvalRun && chosenEvalRun.config;
    return chosenConfig ? {
      model: this.helper_.getModelId(chosenConfig),
      data: this.helper_.getDataVersion(chosenConfig)
    } :
                          null;
  }

  /** @override */
  getModelColumnName() {
    return this.helper_.getModelHeader();
  }

  /** @override */
  getDataColumnName() {
    return this.helper_.getDataHeader();
  }
}

/**
 * @typedef {{
 *   modelIdentifier: (string|number),
 *   dataIdentifier: (string|number)
 * }}
 */
let DefaultEvalRunConfig;

/** @enum {string} */
const ConfigFieldNames = {
  DATA_IDENTIFIER: 'dataIdentifier',
  MODEL_IDENTIFIER: 'modelIdentifier',
};

/** @enum {string} */
const Headers = {
  MODEL: 'Model',
  DATA: 'Data',
};

/**
 * An instantance of DefaultSeriesDataHelperImpl that can be reused across
 * SeriesData objects.
 * @type {?SeriesDataHelper.SeriesDataHelper}
 */
let defaultHelper;

/**
 * Lazily initialize a DefaultSeriesDataHelperImpl and returns a reference to
 * it.
 * @return {!SeriesDataHelper.SeriesDataHelper}
 */
function getDefaultHelper() {
  defaultHelper = defaultHelper || new DefaultSeriesDataHelperImpl();
  return defaultHelper;
}

/**
 * Default implementation for SeriesDataHelper
 * @implements {SeriesDataHelper.SeriesDataHelper}
 */
class DefaultSeriesDataHelperImpl {
  /** @override */
  getFormats(specifiedFormats) {
    return specifiedFormats;
  }

  /** @override */
  getModelId(config) {
    return parseAsNumber(this.getModelDisplayText(config));
  }

  /** @override */
  getModelDisplayText(config) {
    return config[ConfigFieldNames.MODEL_IDENTIFIER] || 'Model ID not set';
  }

  /** @override */
  getDataVersion(config) {
    return parseAsNumber(this.getDataDisplayText(config));
  }

  /** @override */
  getDataDisplayText(config) {
    return config[ConfigFieldNames.DATA_IDENTIFIER] || 'Data ID not set';
  }

  /** @override */
  sortEvalRuns(evalRuns, modelCentric) {
    return evalRuns.sort((a, b) => {
      const configA = /** @type {!DefaultEvalRunConfig} */ (a.config);
      const configB = /** @type {!DefaultEvalRunConfig} */ (b.config);
      return modelCentric ? sortByModelThenData(configA, configB) :
                            sortByDataThenModel(configA, configB);
    });
  }

  /** @override */
  getModelHeader() {
    return Headers.MODEL;
  }

  /** @override */
  getDataHeader() {
    return Headers.DATA;
  }
}

/**
 * Given input, if it can be parsed into an integer return its integer value. If
 * it can be parsed as float, return its float value. Otherwise, return its
 * original value.
 * @param {string|number} input
 * @return {string|number}
 */
function parseAsNumber(input) {
  const parsedInput = parseFloat(input);
  if (isNaN(parsedInput) || input != parsedInput) {
    // Cannot be parsed as numbers exactly. Return as is.
    return input;
  } else {
    const inputFloor = Math.floor(parsedInput);
    return parsedInput == inputFloor ? inputFloor : parsedInput;
  }
}


/**
 * A comparator function that tries to parse input as number and then perform
 * comparison. The array will be sorted in descending order.
 * @param {number|string} a
 * @param {number|string} b
 * @return {number}
 */
function descendingOrder(a, b) {
  const valueA = parseAsNumber(a);
  const valueB = parseAsNumber(b);
  return valueA == valueB ? 0 : (valueA > valueB ? -1 : 1);
}

/**
 * Comparator function that sorts the eval runs in descending order by data
 * identifier then by model identifier.
 * @param {!DefaultEvalRunConfig} a
 * @param {!DefaultEvalRunConfig} b
 * @return {number}
 * @private
 */
function sortByDataThenModel(a, b) {
  return descendingOrder(
             a[ConfigFieldNames.DATA_IDENTIFIER],
             b[ConfigFieldNames.DATA_IDENTIFIER]) ||
      descendingOrder(
             a[ConfigFieldNames.MODEL_IDENTIFIER],
             b[ConfigFieldNames.MODEL_IDENTIFIER]);
}

/**
 * Comparator function that sorts the eval runs in descending order by model
 * identifier then by data identifier.
 * @param {!DefaultEvalRunConfig} a
 * @param {!DefaultEvalRunConfig} b
 * @return {number}
 * @private
 */
function sortByModelThenData(a, b) {
  return descendingOrder(
             a[ConfigFieldNames.MODEL_IDENTIFIER],
             b[ConfigFieldNames.MODEL_IDENTIFIER]) ||
      descendingOrder(
             a[ConfigFieldNames.DATA_IDENTIFIER],
             b[ConfigFieldNames.DATA_IDENTIFIER]);
}

goog.exportSymbol('tfma.SeriesData', SeriesData);

exports = SeriesData;
