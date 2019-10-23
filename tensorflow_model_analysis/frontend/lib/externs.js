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
/**
 * @externs
 */

/**
 * @const
 */
var tfma = {};

/**
 * @const
 */
tfma.BucketsWrapper = {};

/**
 * @param {!Array<!Object>} buckets
 * @param {string} fit
 * @param {string} scale
 * @param {string} color
 * @param {string} size
 * @param {number|undefined} newBucketSize
 * @param {!Array<!Array<string|number>>} outputArray
 */
tfma.BucketsWrapper.getCalibrationPlotData = function(
    buckets, fit, scale, color, size, newBucketSize, outputArray) {};

/**
 * @const
 */
tfma.CellRenderer = {};

/**
 * @param {string} type
 * @param {Function} renderer
 */
tfma.CellRenderer.registerOverrideRenderer = function(type, renderer) {};

/**
 * Registers a renderer and a type checker for the given type. No-op if the
 * given type already has a renderer.
 * @param {string} type
 * @param {Object} renderer
 * @param {function(!Object):boolean} typeChecker
 */
tfma.CellRenderer.registerRenderer = function(type, renderer, typeChecker) {};

/**
 * @param {number|string|?Object} value
 * @param {!Object<!tfma.TableProviderExt>=} opt_tableProvider
 * @param {!tfma.MetricValueFormatSpec=} opt_override
 */
tfma.CellRenderer.renderValueWithFormatOverride = function(
    value, opt_tableProvider, opt_override) {};

/**
 * @param {number|!Object} value
 * @return {number}
 */
tfma.CellRenderer.maybeExtractBoundedValue = function(value) {};

/**
 * @param {!Object} value
 * @return {boolean}
 */
tfma.CellRenderer.isBoundedValue = function(value) {};

/**
 * @param {!Object} value
 * @return {boolean}
 */
tfma.CellRenderer.isRatioValue = function(value) {};

/**
 * @param {!Object} value
 * @param {string} key
 * @return {number}
 */
tfma.CellRenderer.extractFloatValue = function(value, key) {};

/**
 * @constructor
 */
tfma.Data = function() {};

/**
 * @param {!tfma.Data} data
 * @return {boolean}
 */
tfma.Data.prototype.equals = function(data) {};

/**
 * @param {function(!tfma.Data.Series): boolean} filterFn
 * @return {!tfma.Data} Filtered data.
 */
tfma.Data.prototype.filter = function(filterFn) {};

/**
 * @param {string} metric
 * @return {{min: number, max: number}}
 */
tfma.Data.prototype.getColumnRange = function(metric) {};

/**
 * @return {!Array<string>}
 */
tfma.Data.prototype.getFeatureDataSeriesKeys = function() {};

/**
 * @param {string} feature
 * @return {(string|number)}
 */
tfma.Data.prototype.getFeatureId = function(feature) {};

/**
 * @return {!Array<string>}
 */
tfma.Data.prototype.getFeatures = function() {};

/**
 * @return {!Array<string>}
 */
tfma.Data.prototype.getMetrics = function() {};

/**
 * @param {string} metricName
 * @return {number}
 */
tfma.Data.prototype.getMetricIndex = function(metricName) {};

/**
 * @param {string} feature
 * @param {string} metric
 * @return {number}
 */
tfma.Data.prototype.getMetricValue = function(feature, metric) {};

/**
 * @return {!Array<tfma.Data.Series>}
 */
tfma.Data.prototype.getSeriesList = function() {};

/**
 * @param {string} feature
 * @return {!Array<number>}
 * @export
 */
tfma.Data.prototype.getAllMetricValues = function(feature) {};

/**
 * Factory method for the data object.
 * @param {!Array<string>} metricNames
 * @param {!Array<{slice: string, metrics: !Object}>} results
 */
tfma.Data.build = function(metricNames, results) {};

/**
 * @param {!Array<!Object>} runs
     @param {string} metricsKey
 */
tfma.Data.flattenMetrics = function(runs, metricsKey) {};

/**
 * @param {!Array<!Array<!Object>>} dataArrays
 * @param {string} metricsFieldKey
 * @return {!Array<string>}
 */
tfma.Data.getAvailableMetrics = function(dataArrays, metricsFieldKey) {};

/**
 * @param {!Object} plotData
 * @return {!Array<string>}
 */
tfma.Data.getAvailablePlotTypes = function(plotData) {};

/**
 * @constructor
 */
tfma.Data.Series = function() {};

/**
 * @return {string}
 */
tfma.Data.Series.prototype.getFeatureString = function() {};

/**
 * @return {string}
 */
tfma.Data.Series.prototype.getFeatureIdForMatching

/**
 * @enum {string}
 */
tfma.Event = {
  CHANGE: '',
  CLEAR_SELECTION: '',
  DOUBLE_CLICK: '',
  IMMEDIATE_VALUE_CHANGE: '',
  KEYUP: '',
  RELOAD_PLOT_DATA: '',
  SELECT: '',
  UPDATE_FOCUS_RANGE: '',
};

/**
 * @enum {string}
 */
tfma.Column = {
  TOTAL_EXAMPLE_COUNT: '',
};

/**
 * @type {number}
 */
tfma.FLOATING_POINT_PRECISION;

/** @type {string} */
tfma.KEY;

/**
 * @constructor
 */
tfma.LineChartProvider = function() {};

/**
 * @param {string} metric
 * @return {!Array<!Array<string|number|!tfma.GVizCell>>}
 */
tfma.LineChartProvider.prototype.getLineChartData = function(metric) {};

/**
 * @return {!Array<string|number>}
 */
tfma.LineChartProvider.prototype.getModelIds = function() {};

/**
 * @param {number} index
 * @return {?{
 *   model: (string|number),
 *   data: (string|number)
 * }}
 */
tfma.LineChartProvider.prototype.getEvalConfig = function(index) {};

/**
 * @return {string}
 */
tfma.LineChartProvider.prototype.getModelColumnName = function() {};

/**
 * @return {string}
 */
tfma.LineChartProvider.prototype.getDataColumnName = function() {};

/**
 * @enum {string}
 */
tfma.MetricValueFormat = {
  INT: '',
  INT64: '',
  FLOAT: '',
  ROW_ID: '',
  VALUE_AT_CUTOFFS: '',
};

/**
 * @constructor
 */
tfma.MetricValueFormatSpec = function() {};

/**
 * @type {tfma.MetricValueFormat}
 */
tfma.MetricValueFormatSpec.prototype.type;

/**
 * @type {(undefined|function((string|number|!Object),(string|number|!Object)):(number|string|!Object))}
 */
tfma.MetricValueFormatSpec.prototype.transform;

/** @type {string} */
tfma.METRIC_KEYS_AND_VALUES;

/**
 * @enum {string}
 */
tfma.PlotDataFieldNames = {
  CALIBRATION_BUCKETS: '',
  CALIBRATION_DATA: '',
  CONFUSION_MATRICES: '',
  MACRO_PRECISION_RECALL_CURVE_DATA: '',
  MICRO_PRECISION_RECALL_CURVE_DATA: '',
  MULTI_CLASS_CONFUSION_MATRIX_DATA: '',
  MULTI_LABEL_CONFUSION_MATRIX_DATA: '',
  PRECISION_RECALL_CURVE_DATA: '',
  WEIGHTED_PRECISION_RECALL_CURVE_DATA: '',
};

/**
 * @enum {string}
 */
tfma.PlotFit = {
  PERFECT: '',
  LEAST_SQUARE: '',
};

/**
 * @enum {string}
 */
tfma.PlotHighlight = {
  ERROR: '',
  WEIGHTS: '',
};

/**
 * @enum {string}
 */
tfma.PlotScale = {
  LINEAR: '',
  LOG: '',
};

/**
 * @enum {string}
 */
tfma.PlotTypes = {
  ACCURACY_CHARTS: '',
  CALIBRATION_PLOT: '',
  GAIN_CHART: '',
  PREDICTION_DISTRIBUTION: '',
  MACRO_PRECISION_RECALL_CURVE: '',
  MICRO_PRECISION_RECALL_CURVE: '',
  MULTI_CLASS_CONFUSION_MATRIX: '',
  MULTI_LABEL_CONFUSION_MATRIX: '',
  PRECISION_RECALL_CURVE: '',
  RESIDUAL_PLOT: '',
  ROC_CURVE: '',
  WEIGHTED_PRECISION_RECALL_CURVE: '',
};

/**
 * @enum {number}
 */
tfma.PlotDataDisplay = {
  EXAMPLES_MAX_STEP: 0
};

/** @type {string} */
tfma.PLOT_KEYS_AND_VALUES;

/**
 * @constructor
 */
tfma.TableProviderExt = function() {};

/**
 * @return {!Array<!Array<(string|number)>>}
 */
tfma.TableProviderExt.prototype.getDataTable = function() {};

/**
 * @param {!Array<string>} requiredColumns
 * @return {!Array<string>}
 * @export
 */
tfma.TableProviderExt.prototype.getHeader = function(requiredColumns) {};

/**
 * @param {!Object<!Object>} specifiedFormats
 * @return {!Object<!tfma.MetricValueFormatSpec>}
 * @export
 */
tfma.TableProviderExt.prototype.getFormats = function(specifiedFormats) {};

/**
 * @return {boolean}
 * @export
 */
tfma.TableProviderExt.prototype.readyToRender = function() {};

/**
 * @param {number|string|!Object} value
 * @param {!Object} override
 * @return {number|string|!Object}
 */
tfma.TableProviderExt.prototype.applyOverride = function(value, override) {};

/**
 * @constructor
 */
tfma.GVizCell = function() {};

/**
 * @type {number|string}
 */
tfma.GVizCell.prototype.v;

/**
 * @type {number|string}
 */
tfma.GVizCell.prototype.f;

/**
 * @constructor
 */
tfma.GraphData = function() {};

/**
 * @param {string} columnName
 * @param {number} threshold
 * @return {!tfma.Data}
 */
tfma.GraphData.prototype.applyThreshold = function(
    columnName, threshold) {};

/**
 * @param {string} columnName
 * @return {{max: number, step: number}}
 *     max: The max value of the given column value to be displayed on a
 *     slider, that is the smallest multiple of the slider step that is
 *     larger than the max of examples.
 *     step: The step the slider takes.
 */
tfma.GraphData.prototype.getColumnSteppingInfo = function(
    columnName) {};

/**
 * @return {!Array<string>} All the features of the data set.
 */
tfma.GraphData.prototype.getFeatures = function() {};

/**
 * @return {!tfma.TableProviderExt} A reference to the data in table format.
 */
tfma.GraphData.prototype.getTableDataFromDataset = function(data) {};

/**
 * @param {!Array<!Object>} series
 * @param {boolean} modelCentric
 * @param {tfma.SeriesDataHelper=} opt_helper
 * @constructor
 */
tfma.SeriesData = function(series, modelCentric, opt_helper) {};

/**
 * @return {!Array<string>}
 */
tfma.SeriesData.prototype.getMetrics = function() {};

/**
 * @constructor
 */
tfma.SeriesDataHelper = function() {};

/**
 * @param {!Object<!Object>} specifiedFormats
 * @return {!Object<!Object>}
 */
tfma.SeriesDataHelper.prototype.getFormats = function(specifiedFormats) {};

/**
 * @param {!Object} config
 * @return {string|number}
 */
tfma.SeriesDataHelper.prototype.getModelId = function(config) {};

/**
 * @param {!Object} config
 * @return {string}
 */
tfma.SeriesDataHelper.prototype.getModelDisplayText = function(config) {};

/**
 * @param {!Object} config
 * @return {string|number}
 */
tfma.SeriesDataHelper.prototype.getDataVersion = function(config) {};

/**
 * @param {!Object} config
 * @return {string}
 */
tfma.SeriesDataHelper.prototype.getDataDisplayText = function(config) {};

/**
 * @param {!Array<tfma.SeriesDataHelper.EvalRun>} evalRuns
 * @param {boolean} modelCentric
 * @return {!Array<tfma.SeriesDataHelper.EvalRun>}
 */
tfma.SeriesDataHelper.prototype.sortEvalRuns = function(
    evalRuns, modelCentric) {};

/**
 * @return {string}
 */
tfma.SeriesDataHelper.prototype.getModelHeader = function() {};

/**
 * @return {string}
 */
tfma.SeriesDataHelper.prototype.getDataHeader = function() {};

/**
 * @constructor
 */
tfma.SeriesDataHelper.EvalRun = function() {};

/**
 * @type {!Object}
 */
tfma.SeriesDataHelper.EvalRun.prototype.config;

/**
 * @type {!tfma.Data}
 */
tfma.SeriesDataHelper.EvalRun.prototype.data;

/**
 * @param {!Array<string>} metrics
 * @param {!Array<{slice: string, metrics: !Object}>} data
 * @constructor
 */
tfma.SingleSeriesGraphData = function(metrics, data) {};

/**
 * @const
 */
tfma.Util = {};

/**
 * @param {!Object} configs
 * @return {!Array<!Object>}
 */
tfma.Util.createConfigsList = function(configs) {};

/**
 * @param {!Object} metrics
 * @param {!Array<!Object>} configsList
 * @param {!Object<string>=} blacklist
 * @return {!Object}
 */
tfma.Util.mergeMetricsForSelectedConfigsList = function(
    metrics, configsList, blacklist) {};

/** @type {string} */
tfma.VALUE;
