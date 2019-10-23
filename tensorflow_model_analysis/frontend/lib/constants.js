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
 * @fileoverview Common tfma definitions and static functions.
 */

goog.module('tfma.Constants');

/** @enum {string} */
const Event = {
  CHANGE: 'change',
  CLICK: 'click',
  DOUBLE_CLICK: 'dblclick',
  KEYUP: 'keyup',
  IMMEDIATE_VALUE_CHANGE: 'immediate-value-change',
  UPDATE_FOCUS_RANGE: 'update-focus-range',
  SELECT: 'select',
  CLEAR_SELECTION: 'clear-selection',
  IRON_SELECT: 'iron-select',
  ON_MOUSEOVER: 'onmouseover',
  RELOAD_PLOT_DATA: 'reload-plot-data',
};

/**
 * Column display names.
 * @enum {string}
 */
const Column = {
  OVERALL: 'Overall',
  FEATURE: 'feature',
  WEIGHTED_EXAMPLES: 'weighted examples',
  CALIBRATION: 'calibration',
  LINKS: 'links',
  TOTAL_EXAMPLE_COUNT: 'totalExampleCount',
};

/** @const {number} */
const FLOATING_POINT_PRECISION = 5;

/** @enum {string} */
const MetricValueFormat = {
  FLOAT: 'float',
  HTML: 'html',
  INT: 'int',
  INT64: 'int64',
  MULTI_VALUES_METRIC_AT_K: 'multiValueMetricAtK',
  ROW_ID: 'rowId',
  STRING: 'string',
  VALUE_AT_CUTOFFS: 'valueAtCutoffs',
};

/** @type {string} */
const KEY = 'key';

/** @type {string} */
const METRIC_KEYS_AND_VALUES = 'metricKeysAndValues';

/** @type {string} */
const VALUE = 'value';

/** @enum {string}*/
const PlotTypes = {
  ACCURACY_CHARTS: 'accuracyPrecisionRecallF1Charts',
  CALIBRATION_PLOT: 'calibrationPlot',
  GAIN_CHART: 'gainChart',
  MACRO_PRECISION_RECALL_CURVE: 'macroPrecisionRecallCurve',
  MICRO_PRECISION_RECALL_CURVE: 'microPrecisionRecallCurve',
  MULTI_CLASS_CONFUSION_MATRIX: 'multiClassConfusionMatrix',
  MULTI_LABEL_CONFUSION_MATRIX: 'multiLabelConfusionMatrix',
  PREDICTION_DISTRIBUTION: 'predictionDistribution',
  PRECISION_RECALL_CURVE: 'precisionRecallCurve',
  RESIDUAL_PLOT: 'residualPlot',
  ROC_CURVE: 'rocCurve',
  WEIGHTED_PRECISION_RECALL_CURVE: 'weightedPrecisionRecallCurve',
};

/** @enum {string} */
const PlotDataFieldNames = {
  CALIBRATION_BUCKETS: 'buckets',
  CALIBRATION_DATA: 'bucketByRefinedPrediction',
  CONFUSION_MATRICES: 'matrices',
  MACRO_PRECISION_RECALL_CURVE_DATA: 'macroValuesByThreshold',
  MICRO_PRECISION_RECALL_CURVE_DATA: 'microValuesByThreshold',
  MULTI_CLASS_CONFUSION_MATRIX_DATA: 'multiClassConfusionMatrixAtThresholds',
  MULTI_LABEL_CONFUSION_MATRIX_DATA: 'multiLabelConfusionMatrixAtThresholds',
  PRECISION_RECALL_CURVE_DATA: 'binaryClassificationByThreshold',
  WEIGHTED_PRECISION_RECALL_CURVE_DATA: 'weightedValuesByThreshold',
};

/**
 * How to fit the points in the plots to determine error at each point.
 * @enum {string}
 */
const PlotFit = {
  // A perfect fit. For calibration, this is y = x.
  PERFECT: 'perfect',
  // Fits prediction / label pairs to a line using least square method.
  LEAST_SQUARE: 'leastSquare'
};

/**
 * What to highlight in each dot.
 * @enum {string}
 */
const PlotHighlight = {
  ERROR: 'error',
  WEIGHTS: 'weights'
};

/**
 * The scale in the x and y axis.
 * @enum {string}
 */
const PlotScale = {
  LINEAR: 'linear',
  LOG: 'log'
};

/**
 * The display config constants for the graphs.
 * @enum {number}
 */
const PlotDataDisplay = {
  EXAMPLES_MAX_STEP: 10
};

/** @type {string} */
const PLOT_KEYS_AND_VALUES = 'plotKeysAndValues';

/**
 * @typedef {
 *   function((string|number|!Object),(string|number|!Object)):(number|string|!Object)
 * }
 */
let MetricValueTransformer;

/**
 * Object that specifies the metric value format and its accessor function.
 * Currently, accessor is only required for HTML format to provide links
 * in the metrics-table.
 * @typedef {{
 *   type: MetricValueFormat,
 *   transform: (!MetricValueTransformer|undefined)
 * }}
 *     accessor: Required only for type HTML. The input to the accessor is the
 *       data series corresponding to the table row. The output of the accessor
 *       should be the HTML output of that table cell.
 */
let MetricValueFormatSpec;

goog.exportSymbol('tfma.Event.CHANGE', Event.CHANGE);
goog.exportSymbol('tfma.Event.CLEAR_SELECTION', Event.CLEAR_SELECTION);
goog.exportSymbol('tfma.Event.DOUBLE_CLICK', Event.DOUBLE_CLICK);
goog.exportSymbol(
    'tfma.Event.IMMEDIATE_VALUE_CHANGE', Event.IMMEDIATE_VALUE_CHANGE);
goog.exportSymbol('tfma.Event.KEYUP', Event.KEYUP);
goog.exportSymbol('tfma.Event.RELOAD_PLOT_DATA', Event.RELOAD_PLOT_DATA);
goog.exportSymbol('tfma.Event.SELECT', Event.SELECT);
goog.exportSymbol('tfma.Event.UPDATE_FOCUS_RANGE', Event.UPDATE_FOCUS_RANGE);

goog.exportSymbol(
    'tfma.Column.TOTAL_EXAMPLE_COUNT', Column.TOTAL_EXAMPLE_COUNT);

goog.exportSymbol('tfma.FLOATING_POINT_PRECISION', FLOATING_POINT_PRECISION);

goog.exportSymbol('tfma.KEY', KEY);

goog.exportSymbol('tfma.MetricValueFormat.INT', MetricValueFormat.INT);
goog.exportSymbol('tfma.MetricValueFormat.INT64', MetricValueFormat.INT64);
goog.exportSymbol('tfma.MetricValueFormat.FLOAT', MetricValueFormat.FLOAT);
goog.exportSymbol('tfma.MetricValueFormat.ROW_ID', MetricValueFormat.ROW_ID);
goog.exportSymbol(
    'tfma.MetricValueFormat.VALUE_AT_CUTOFFS',
    MetricValueFormat.VALUE_AT_CUTOFFS);

goog.exportSymbol('tfma.METRIC_KEYS_AND_VALUES', METRIC_KEYS_AND_VALUES);

goog.exportSymbol(
    'tfma.PlotDataFieldNames.CALIBRATION_BUCKETS',
    PlotDataFieldNames.CALIBRATION_BUCKETS);
goog.exportSymbol(
    'tfma.PlotDataFieldNames.CALIBRATION_DATA',
    PlotDataFieldNames.CALIBRATION_DATA);
goog.exportSymbol(
    'tfma.PlotDataFieldNames.CONFUSION_MATRICES',
    PlotDataFieldNames.CONFUSION_MATRICES);
goog.exportSymbol(
    'tfma.PlotDataFieldNames.MACRO_PRECISION_RECALL_CURVE_DATA',
    PlotDataFieldNames.MACRO_PRECISION_RECALL_CURVE_DATA);
goog.exportSymbol(
    'tfma.PlotDataFieldNames.MICRO_PRECISION_RECALL_CURVE_DATA',
    PlotDataFieldNames.MICRO_PRECISION_RECALL_CURVE_DATA);
goog.exportSymbol(
    'tfma.PlotDataFieldNames.MULTI_CLASS_CONFUSION_MATRIX_DATA',
    PlotDataFieldNames.MULTI_CLASS_CONFUSION_MATRIX_DATA);
goog.exportSymbol(
    'tfma.PlotDataFieldNames.MULTI_LABEL_CONFUSION_MATRIX_DATA',
    PlotDataFieldNames.MULTI_LABEL_CONFUSION_MATRIX_DATA);
goog.exportSymbol(
    'tfma.PlotDataFieldNames.PRECISION_RECALL_CURVE_DATA',
    PlotDataFieldNames.PRECISION_RECALL_CURVE_DATA);
goog.exportSymbol(
    'tfma.PlotDataFieldNames.WEIGHTED_PRECISION_RECALL_CURVE_DATA',
    PlotDataFieldNames.WEIGHTED_PRECISION_RECALL_CURVE_DATA);

goog.exportSymbol('tfma.PlotFit.PERFECT', PlotFit.PERFECT);
goog.exportSymbol('tfma.PlotFit.LEAST_SQUARE', PlotFit.LEAST_SQUARE);

goog.exportSymbol('tfma.PlotHighlight.ERROR', PlotHighlight.ERROR);
goog.exportSymbol('tfma.PlotHighlight.WEIGHTS', PlotHighlight.WEIGHTS);

goog.exportSymbol('tfma.PlotScale.LINEAR', PlotScale.LINEAR);
goog.exportSymbol('tfma.PlotScale.LOG', PlotScale.LOG);

goog.exportSymbol('tfma.PlotTypes.ACCURACY_CHARTS', PlotTypes.ACCURACY_CHARTS);
goog.exportSymbol(
    'tfma.PlotTypes.CALIBRATION_PLOT', PlotTypes.CALIBRATION_PLOT);
goog.exportSymbol('tfma.PlotTypes.GAIN_CHART', PlotTypes.GAIN_CHART);
goog.exportSymbol(
    'tfma.PlotTypes.MACRO_PRECISION_RECALL_CURVE',
    PlotTypes.MACRO_PRECISION_RECALL_CURVE);
goog.exportSymbol(
    'tfma.PlotTypes.MICRO_PRECISION_RECALL_CURVE',
    PlotTypes.MICRO_PRECISION_RECALL_CURVE);
goog.exportSymbol(
    'tfma.PlotTypes.MULTI_CLASS_CONFUSION_MATRIX',
    PlotTypes.MULTI_CLASS_CONFUSION_MATRIX);
goog.exportSymbol(
    'tfma.PlotTypes.MULTI_LABEL_CONFUSION_MATRIX',
    PlotTypes.MULTI_LABEL_CONFUSION_MATRIX);
goog.exportSymbol(
    'tfma.PlotTypes.PREDICTION_DISTRIBUTION',
    PlotTypes.PREDICTION_DISTRIBUTION);
goog.exportSymbol(
    'tfma.PlotTypes.PRECISION_RECALL_CURVE', PlotTypes.PRECISION_RECALL_CURVE);
goog.exportSymbol('tfma.PlotTypes.RESIDUAL_PLOT', PlotTypes.RESIDUAL_PLOT);
goog.exportSymbol('tfma.PlotTypes.ROC_CURVE', PlotTypes.ROC_CURVE);
goog.exportSymbol(
    'tfma.PlotTypes.WEIGHTED_PRECISION_RECALL_CURVE',
    PlotTypes.WEIGHTED_PRECISION_RECALL_CURVE);

goog.exportSymbol('tfma.PlotDataDisplay.EXAMPLES_MAX_STEP',
    PlotDataDisplay.EXAMPLES_MAX_STEP);

goog.exportSymbol('tfma.PLOT_KEYS_AND_VALUES', PLOT_KEYS_AND_VALUES);

goog.exportSymbol('tfma.VALUE', VALUE);

exports = {
  Column,
  Event,
  FLOATING_POINT_PRECISION,
  KEY,
  MetricValueFormat,
  MetricValueFormatSpec,
  MetricValueTransformer,
  METRIC_KEYS_AND_VALUES,
  PlotDataFieldNames,
  PlotFit,
  PlotHighlight,
  PlotScale,
  PlotTypes,
  PlotDataDisplay,
  VALUE,
};
