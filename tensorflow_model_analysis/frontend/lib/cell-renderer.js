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
goog.module('tfma.CellRenderer');


const Constants = goog.require('tfma.Constants');
const TableProvider = goog.require('tfma.TableProvider');
const googString = goog.require('goog.string');

/**
 * @typedef {{
 *   value: number,
 *   lowerBound: number,
 *   upperBound: number,
 * }}
 */
let BoundedValue;

/**
 * Json representation of Int32Value.
 * @typedef {{
 *   value: number
 * }}
 */
let Int32ValueProto;

/**
 * Json representation of the PrecisionAtK.
 * @typedef {{
 *   k: !Int32ValueProto,
 *   value: number,
 *   totalPositives: number,
 * }}
 */
let PrecisionAtKProto;

/**
 * Json representation of the recall at k metric.
 * @typedef {{
 *   k: !Int32ValueProto,
 *   value: number,
 *   totalActualPositives: number,
 * }}
 */
let RecallAtKProto;

/**
 * @typedef {!Array<PrecisionAtKProto>|!Array<RecallAtKProto>}
 */
let MetricAtK;

/**
 * Json representation of the multi value precision or recall at k metric.
 * @typedef {{
 *   k: !Int32ValueProto,
 *   macroValue: number,
 *   microValue: number,
 * }}
 */
let MultiValuesMetricAtKProto;

/**
 * @typedef {!Array<!MultiValuesMetricAtKProto>}
 */
let MultiValuesMetricAtK;

/**
 * @enum {string}
 */
const PrecisionAtKFieldNames = {
  K: 'k',
  TOTAL_POSITIVES: 'totalPositives',
  VALUE: 'value',
};

/**
 * @enum {string}
 */
const RecallAtKFieldNames = {
  K: 'k',
  TOTAL_ACTUAL_POSITIVES: 'totalActualPositives',
  VALUE: 'value'
};

/**
 * @enum {string}
 */
const MetricAtKFieldNames = {
  K: 'k',
  VALUE: 'value',
};

/**
 * @enum {string}
 */
const MultiValuesMetricAtKFieldNames = {
  K: 'k',
  MACRO_VALUE: 'macroValue',
  MICRO_VALUE: 'microValue',
  WEIGHTED_VALUE: 'weightedValue',
};

/**
 * Json representation of the BinaryClassificationThreshold.
 * @typedef {{
 *    predictionThreshold: number,
 * }}
 */
let BinaryClassificationThresholdProto;

/**
 * Json representation of the BinaryConfusionMatrix.
 * @typedef {{
 *   f1Score: number,
 *   precision: number,
 *   recall: number,
 *   accuracy: number,
 * }}
 */
let BinaryConfusionMatrixProto;

/**
 * Json representation of the BinaryConfusionMatricesFromRegression.
 * @typedef {{
 *   binaryClassificationThreshold : !BinaryClassificationThresholdProto,
 *   matrix: !BinaryConfusionMatrixProto
 * }}
 */
let BinaryConfusionMatricesFromRegressionProto;

/**
 * @typedef {!Array<!BinaryConfusionMatricesFromRegressionProto>}
 */
let BinaryConfusionMatricesFromRegression;


/**
 * @enum {string}
 */
const BinaryConfusionMatricesFromRegressionFieldNames = {
  ACCURACY: 'accuracy',
  BINARY_CLASSIFICATION_THRESHOLD: 'binaryClassificationThreshold',
  F1_SCORE: 'f1Score',
  MATRIX: 'matrix',
  PRECISION: 'precision',
  PREDICTION_THRESHOLD: 'predictionThreshold',
  RECALL: 'recall',
};

/**
 * @typedef {{
 *   availableTypes: !Array<string>,
 *   slice: (string|undefined),
 *   span: (string|undefined)
 * }}
 */
let PlotTrigger;

/**
 * @enum {string}
 */
const PlotTriggerFieldNames = {
  TYPES: 'types',
  SLICE: 'slice',
  SPAN: 'span',
};

/**
 * Json representation of the BinaryConfusionMatricesFromRegression.
 * @typedef {{
 *   binaryClassificationThreshold : !BinaryClassificationThresholdProto,
 *   macroPrecision: number,
 *   macroRecall: (number|undefined),
 *   macroF1Score: (number|undefined),
 *   macroAccuracy: (number|undefined),
 *   microPrecision: (number|undefined),
 *   microRecall: (number|undefined),
 *   microF1Score: (number|undefined),
 *   microAccuracy: (number|undefined),
 *   weightedPrecision: (number|undefined),
 *   weighteRecall: (number|undefined),
 *   weighteF1Score: (number|undefined),
 *   weighteAccuracy: (number|undefined),
 * }}
 */
let MultiValuesThresholdBasedBinaryClassificationMetricsProto;

/**
 * @enum {string}
 */
const MultiValuesThresholdBasedBinaryClassificationMetricsFieldNames = {
  BINARY_CLASSIFICATION_THRESHOLD: 'binaryClassificationThreshold',
  PREDICTION_THRESHOLD: 'predictionThreshold',
};

/**
 * @typedef {!Array<!MultiValuesThresholdBasedBinaryClassificationMetricsProto>}
 */
let MultiValuesThresholdBasedBinaryClassificationMetrics;

/**
 * Json representation of the MultiClassConfusionMatrix.
 * @typedef {{
 *   actualClass : string,
 *   predictedClass: string,
 *   weight: number,
 * }}
 */
let MultiClassConfusionMatrixEntry;

/**
 * @typedef {{
 *   entries: !Array<!MultiClassConfusionMatrixEntry>
 * }}
 */
let MultiClassConfusionMatrix;

/**
 * @enum {string}
 */
const MultiClassConfusionMatrixFieldNames = {
  ACTUAL_CLASS: 'actualClass',
  ENTRIES: 'entries',
  PREDICTED_CLASS: 'predictedClass',
  WEIGHT: 'weight',
};

/**
 * @enum {string}
 */
const ValueType = {
  BINARY_CONFUSION_MATRICES_FROM_REGRESSION:
      'binaryConfusionMatricesFromRegression',
  BOUNDED_VALUE: 'boundedValue',
  FLOAT: 'float',
  MULTI_CLASS_CONFUSION_MATRIX: 'MultiClassConfusionMatrix',
  MULTI_VALUES_METRIC_AT_K: 'MultiValuesMetricAtK',
  MULTI_VALUES_THRESHOLD_BASED_BINARY_CLASSIFICATION_METRICS:
      'MultiValuesThresholdBasedBinaryClassificationMetrics',
  PRECISION_AT_K: 'precisionAtK',
  RECALL_AT_K: 'recallatK',
  PLOT_TRIGGER: 'plotTrigger',
  STRING: 'string',
  UNKNOWN: 'unknown',
};

/**
 * @enum {string}
 */
const BoundedValueFieldNames = {
  LOWER_BOUND: 'lowerBound',
  UPPER_BOUND: 'upperBound',
  VALUE: 'value',
};

/**
 * @param {?BinaryConfusionMatricesFromRegression}
 *     value
 * @return {number} The f1 score, if available, or NaN, otherwise.
 */
function getF1Score(value) {
  const fieldNames = BinaryConfusionMatricesFromRegressionFieldNames;
  const matrix = value && value[0] && value[0][fieldNames.MATRIX];
  return matrix ? matrix[fieldNames.F1_SCORE] || 0 : NaN;
}

/**
 * JS represents numbers using 64-bit floating point representation. To avoid
 * loss of accuracy, int64 is returned to client side as strings. To allow int64
 * fields to be sorted accurately, pad the string representation with leading
 * zeros.
 * @param {string} value
 * @return {string}
 */
function padInt64ForSort(value) {
  // Int64 represents value between -9,223,372,036,854,775,808 and
  // 9,223,372,036,854,775,807. Since this method need to support the difference
  // between two int64, we will have at most 20-digit long numbers and we will
  // need to pad at most 19.
  const padding = '000000000000000000000';
  const negative = value.substring(0, 1) == '-';
  const number = negative ? value.substring(1) : value;
  return (negative ? '-' : '') + padding.substring(0, 20 - number.length) +
      number;
}

/**
 * @param {string} value
 * @return{!TableProvider.GvizCell} A gviz cell for a data span
 *     range.
 */
function renderDataSpanRange(value) {
  const range = value.split(' - ');
  const lastDataSpan = parseInt(range[range.length - 1], 10);
  return {f: value, v: lastDataSpan};
}

/**
 * @param {!MetricAtK} value
 * @param {number} position Note that position is 0-based while k is 1-based.
 * @return {number} Returns the precision value at the given position.
 */
function getValueAt(value, position) {
  return value[position][MetricAtKFieldNames.VALUE];
}

/**
 * @param {!MultiValuesMetricAtK} value
 * @param {number} position Note that position is 0-based while k is 1-based.
 * @return {number} Returns the macro value at the given position or zero if not
 *     defined.
 */
function getMacroValueAt(value, position) {
  return value[position][MultiValuesMetricAtKFieldNames.MACRO_VALUE] || 0;
}

/**
 * @param {string} name
 * @param {string} value
 * @return {string} Returns the html snippet for the named attribute with its
 *     value properly escaped and wrapped in quotes. Returns empty string if
 *     value is empty string.
 */
function createAttributeHtml(name, value) {
  return value ? name + '="' + googString.htmlEscape(value) + '"' : '';
}

/**
 * @param {!Object} value
 * @return {!TableProvider.GvizCell} A gviz cell for an
 *     unsupported metric format.
 */
function renderUnsupported(value) {
  return {
    'f': 'Unsupported: ' + JSON.stringify(value),
    'v': 0,
  };
}

/**
 * @param {number} value
 * @return {!TableProvider.GvizCell} A gviz cell for a float.
 */
function renderFloat(value) {
  return {
    // If value is non-zero, trim it to specified number of digits. If it is
    // zero, simply show zero.
    'f': value ? trimFloat(value) : '0',
    'v': value,
  };
}

/**
 * @param {number} value
 * @return {!TableProvider.GvizCell} A gviz cell for an integer.
 */
function renderInteger(value) {
  return {
    'f': '' + value,
    'v': value,
  };
}

/**
 * @param {string} value
 * @return {!TableProvider.GvizCell} A gviz cell for an int64.
 */
function renderInt64(value) {
  return {
    'f': '<tfma-int64 ' + createAttributeHtml('data', value) + '></tfma-int64>',
    'v': padInt64ForSort(value),
  };
}

/**
 * @param {string} value Currently supports "123", "foo:123" and "bar_123".
 * @return {number} The parsed row id or NaN if we failed to parse it.
 */
function parseRowId(value) {
  let result = parseInt(value, 10);
  if (result.toString() == value) {
    // Only return if value is a pure integer.
    return result;
  }
  let separator = value.lastIndexOf('_') > value.lastIndexOf(':') ? '_' : ':';
  let parts = value.split(separator);
  return parts.length > 1 ? parseRowId(parts[parts.length - 1]) : NaN;
}

/**
 * @param {string} value
 * @return {!TableProvider.GvizCell} A gviz cell for row id. We
 *     assume a row id is of the form feature:id or version_id and check if
 *     the id portion can be converted into a number. If so, we use it as the
 *     value to allow more natural sorting. ie: age:1, age:2, age11 instead of
 *     age:1, age:11, and:2.
 */
function renderRowId(value) {
  let id = parseRowId(value);
  return {
    'f': value,
    'v': isNaN(id) ? value : id,
  };
}

/**
 * @param {string} value
 * @return {!TableProvider.GvizCell} A gviz cell containing the
 *     given string.
 */
function renderString(value) {
  return {
    'f': value,
    'v': value,
  };
}

/**
 * @param {!BoundedValue} value
 * @return {!TableProvider.GvizCell} A gviz cell for a bounded
 *     value.
 */
function renderBoundedValue(value) {
  const estimatedValue = value[BoundedValueFieldNames.VALUE];
  return {
    'f': '<tfma-bounded-value value=' + trimFloat(estimatedValue) +
        ' lower-bound=' + trimFloat(value[BoundedValueFieldNames.LOWER_BOUND]) +
        ' upper-bound=' + trimFloat(value[BoundedValueFieldNames.UPPER_BOUND]) +
        '></tfma-bounded-value>',
    'v': estimatedValue,
  };
}

/**
 * Renders null for metrics that are absent.
 * @return {!TableProvider.GvizCell} The rendered cell.
 */
function renderNotAvailable() {
  return {
    'f': 'n/a',
    'v': 0,
  };
}

/**
 * Trims a float to a predefined number of decimal places. If the absolute value
 * is too small, use scientific notation to avoid loss of information.
 * @param {number} value
 * @return {string}
 */
function trimFloat(value) {
  // If we are losing two or more digits of information, use scientific notation
  // instead.
  return value && Math.abs(value) < 0.01 ?
      value.toExponential(Constants.FLOATING_POINT_PRECISION - 1) :
      value.toFixed(Constants.FLOATING_POINT_PRECISION);
}

/**
 * @param {PrecisionAtKProto|RecallAtKProto|MultiValuesMetricAtKProto} a
 * @param {PrecisionAtKProto|RecallAtKProto|MultiValuesMetricAtKProto} b
 * @return {number}
 */
function sortByK(a, b) {
  const ka = a[MetricAtKFieldNames.K] || 0;
  const kb = b[MetricAtKFieldNames.K] || 0;
  return ka - kb;
}

/**
 * @param {!MetricAtK} value
 * @return {!TableProvider.GvizCell} A gviz cell for a series of
 *     precision at k.
 */
function renderMetricAtK(value) {
  value.sort(sortByK);
  return {
    'f': '<tfma-metric-at-k ' +
        createAttributeHtml('data', JSON.stringify(value)) +
        '></tfma-metric-at-k>',
    // Use the precision at the first position for sorting.
    'v': getValueAt(value, 0),
  };
}

/**
 * @param {!MultiValuesMetricAtK} value
 * @return {!TableProvider.GvizCell} A gviz cell for a series of
 *     precision at k.
 */
function renderMultiValuesMetricAtK(value) {
  value.sort(sortByK);
  return {
    'f': '<tfma-multi-values-metric-at-k ' +
        createAttributeHtml('data', JSON.stringify(value)) +
        '></tfma-multi-values-metric-at-k>',
    // Use the macro value at the first position for sorting.
    'v': getMacroValueAt(value, 0),
  };
}

/**
 * @param {!BinaryConfusionMatricesFromRegression} value
 * @return {!TableProvider.GvizCell} A gviz cell for a series of
 *     binary confusion matrices from regression.
 */
function renderBinaryConfusionMatricesFromRegression(value) {
  return {
    'f': '<tfma-binary-confusion-matrices-from-regression ' +
        createAttributeHtml('data', JSON.stringify(value)) +
        '></tfma-binary-confusion-matrices-from-regression>',
    // Use the f1 score for sorting.
    'v': getF1Score(value)
  };
}

/**
 * @param {!PlotTrigger} value
 * @return {!TableProvider.GvizCell} A gviz cell for a plot
 *     trigger.
 */
function renderPlotTrigger(value) {
  return {
    'f': '<tfma-plot-trigger ' +
        createAttributeHtml('data', JSON.stringify(value)) +
        '></tfma-plot-trigger>',
    'v': 0
  };
}

/**
 * @param {!MultiValuesThresholdBasedBinaryClassificationMetrics} value
 * @return {!TableProvider.GvizCell} A gviz cell for a series of
 *     multi values threshold based binary classification metrics.
 */
function renderMultiValuesThresholdBasedBinaryClassificationMetrics(value) {
  return {
    'f': '<tfma-multi-values-threshold-based-binary-classification-metrics ' +
        createAttributeHtml('data', JSON.stringify(value)) +
        '></tfma-multi-values-threshold-based-binary-classification-metrics>',
    // Use the macro f1 score for sorting.
    'v': value[0]['macroF1Score'] || 0
  };
}

/**
 * @param {!MultiClassConfusionMatrix} value
 * @return {!TableProvider.GvizCell} A gviz cell for a multi-class confusion
 *     matrix.
 */
function renderMultiClassConfusionMatrix(value) {
  return {
    'f': '<tfma-multi-class-confusion-matrix ' +
        createAttributeHtml('data', JSON.stringify(value)) +
        '></tfma-multi-class-confusion-matrix>',
    // It does not make sense to sort by confusion matrix. Making the value of
    // the cell always 0.
    'v': 0,
  };
}

/**
 * A map where the key is the cell type and the value is its corresponding
 * renderer.
 * @type {!Object<function((number|string|?Object)):!TableProvider.GvizCell>}
 */
const rendererMap = {};

/**
 * An array of objects where field check is a method that returns boolean and
 * the field type is the corresponding type if check returns true.
 * @type {!Array<{type: string, check:function((number|string|?Object)):boolean}>}
 */
const typeCheckers = [];

/**
 * @typedef {function(string):!TableProvider.GvizCell}
 */
let stringRenderer;

/**
 * @typedef {function(number):!TableProvider.GvizCell}
 */
let numberRenderer;

/**
 * @typedef {function(?Object):!TableProvider.GvizCell}
 */
let objectRenderer;

/**
 * @typedef {(stringRenderer|numberRenderer|objectRenderer)}
 */
let metricRenderer;

/**
 * Registers a renderer and a type checker for the given type. No-op if the
 * given type already has a renderer.
 * @param {string} type
 * @param {metricRenderer} renderer
 * @param {function((string|number|?Object)):boolean} typeChecker
 */
function registerRenderer(type, renderer, typeChecker) {
  if (!rendererMap[type]) {
    rendererMap[type] = renderer;
    typeCheckers.push({type: type, check: typeChecker});
  }
}

/**
 * Renders the given value into a TableProvider.GvizCell depending on its type.
 * @param {string|number|!Object} value
 * @return {!TableProvider.GvizCell} The rendered cell.
 */
function renderValue(value) {
  if (goog.isDefAndNotNull(value)) {
    const renderer = rendererMap[getValueType(value)];
    return renderer ? renderer(value) :
                      renderUnsupported(/** @type {!Object} */ (value));
  } else {
    return renderNotAvailable();
  }
}

/**
 * @param {string|number|!Object} value
 * @return {string} The type of the metric.
 */
function getValueType(value) {
  let type = ValueType.UNKNOWN;
  for (let typeChecker, i = 0;
       type == ValueType.UNKNOWN && (typeChecker = typeCheckers[i]); i++) {
    if (typeChecker.check(value)) {
      type = typeChecker.type;
    }
  }
  return type;
}

/**
 * @param {(string|number|?Object)} value
 * @return {boolean} Returns true if the given value represents a bounded value.
 */
function isBoundedValue(value) {
  return !!value && goog.isDef(value[BoundedValueFieldNames.LOWER_BOUND]) &&
      goog.isDef(value[BoundedValueFieldNames.UPPER_BOUND]) &&
      goog.isDef(value[BoundedValueFieldNames.VALUE]);
}

/**
 * @param {string|number|?Object} value
 * @param {function(!Object):boolean} checkCallback
 * @return {boolean} True if value is a non-empty array and all of its items
 *     satifies the checkCallback.
 */
function checkRepeatedMetric(value, checkCallback) {
  return Array.isArray(value) && value.length && value.reduce((acc, item) => {
    return acc && checkCallback(item);
  }, true);
}

/**
 * @param {(string|number|?Object)} value
 * @return {boolean} Returns true if the given value is an array of json
 *     representation of PrecisionAtK.
 */
function isPrecisionAtK(value) {
  return checkRepeatedMetric(
      value,
      item => goog.isDefAndNotNull(item[PrecisionAtKFieldNames.K]) &&
          item[PrecisionAtKFieldNames.TOTAL_POSITIVES]);
}

/**
 * @param {(string|number|?Object)} value
 * @return {boolean} Returns true if the given value is an array of json
 *     representation of RecallAtK.
 */
function isRecallAtK(value) {
  return checkRepeatedMetric(
      value,
      item => goog.isDefAndNotNull(item[RecallAtKFieldNames.K]) &&
          item[RecallAtKFieldNames.TOTAL_ACTUAL_POSITIVES]);
}

/**
 * @param {(string|number|?Object)} value
 * @return {boolean} Returns true if the given value is an array of json
 *     representation of RecallAtK.
 */
function isMultiValuesMetricAtK(value) {
  return checkRepeatedMetric(
      value,
      item => goog.isDefAndNotNull(item[MultiValuesMetricAtKFieldNames.K]) &&
          (goog.isDefAndNotNull(
               item[MultiValuesMetricAtKFieldNames.MACRO_VALUE]) ||
           goog.isDefAndNotNull(
               item[MultiValuesMetricAtKFieldNames.MICRO_VALUE]) ||
           goog.isDefAndNotNull(
               item[MultiValuesMetricAtKFieldNames.WEIGHTED_VALUE])));
}

/**
 * @param {(string|number|?Object)} value
 * @return {boolean} Returns true if the given value is an array of json
 *     representation of
 *     intelligence.lantern.BinaryConfusionMatricesFromRegression proto.
 */
function isBinaryConfusionMatricesFromRegression(value) {
  return checkRepeatedMetric(
      value,
      item =>
          goog.isDef(
              item[BinaryConfusionMatricesFromRegressionFieldNames.MATRIX]) &&
          goog.isDef(item[BinaryConfusionMatricesFromRegressionFieldNames
                              .BINARY_CLASSIFICATION_THRESHOLD]));
}

/**
 * @param {(string|number|?Object)} value
 * @return {boolean} True if the given value has expected fields to construct a
 *     plot-trigger widget.
 */
function isPlotTrigger(value) {
  return !!value && goog.isDef(value[PlotTriggerFieldNames.TYPES]) &&
      (goog.isDef(value[PlotTriggerFieldNames.SPAN]) ||
       goog.isDef(value[PlotTriggerFieldNames.SLICE]));
}

/**
 * @param {(string|number|?Object)} value
 * @return {boolean} Returns true if the given value is an array of json
 *     representation of MultiValuesThresholdBasedBinaryClassificationMetrics.
 */
function isMultiValuesThresholdBasedBinaryClassificationMetrics(value) {
  return checkRepeatedMetric(value, item => {
    const types = ['macro', 'micro', 'weighted'];
    const metrics = ['Precision', 'Recall', 'Accuracy', 'F1Score'];
    const threshold =
        MultiValuesThresholdBasedBinaryClassificationMetricsFieldNames
            .BINARY_CLASSIFICATION_THRESHOLD;
    return goog.isDef(item[threshold]) &&
        types.reduce(
            // We expect that when one type is configured, all four metrics will
            // be computed.
            (acc, type) => {
              return acc || metrics.reduce((acc, metric) => {
                return acc && goog.isDef(item[type + metric]);
              }, true);
            },
            false);
  });
}

/**
 * @param {(string|number|?Object)} value
 * @return {boolean} Returns true if the given value is json representation of
 *     MultiClassConfusionMatrix.
 */
function isMultiClassConfusionMatrix(value) {
  const entries =
      value && value[MultiClassConfusionMatrixFieldNames.ENTRIES] || [];
  return checkRepeatedMetric(
      entries,
      item => item[MultiClassConfusionMatrixFieldNames.ACTUAL_CLASS] &&
          item[MultiClassConfusionMatrixFieldNames.PREDICTED_CLASS]);
}

// Registers all built-in renderers.
registerRenderer(ValueType.FLOAT, renderFloat, goog.isNumber);
registerRenderer(ValueType.BOUNDED_VALUE, renderBoundedValue, isBoundedValue);
registerRenderer(ValueType.STRING, renderString, goog.isString);
registerRenderer(ValueType.PRECISION_AT_K, renderMetricAtK, isPrecisionAtK);
registerRenderer(ValueType.RECALL_AT_K, renderMetricAtK, isRecallAtK);
registerRenderer(
    ValueType.MULTI_VALUES_METRIC_AT_K, renderMultiValuesMetricAtK,
    isMultiValuesMetricAtK);
registerRenderer(
    ValueType.BINARY_CONFUSION_MATRICES_FROM_REGRESSION,
    renderBinaryConfusionMatricesFromRegression,
    isBinaryConfusionMatricesFromRegression);
registerRenderer(ValueType.PLOT_TRIGGER, renderPlotTrigger, isPlotTrigger);
registerRenderer(
    ValueType.MULTI_VALUES_THRESHOLD_BASED_BINARY_CLASSIFICATION_METRICS,
    renderMultiValuesThresholdBasedBinaryClassificationMetrics,
    isMultiValuesThresholdBasedBinaryClassificationMetrics);
registerRenderer(
    ValueType.MULTI_CLASS_CONFUSION_MATRIX, renderMultiClassConfusionMatrix,
    isMultiClassConfusionMatrix);

/**
 * A map containing all format override renderers.
 * @type {!Object<function((number|string|?Object)):!TableProvider.GvizCell>}
 */
const overrideRendererMap = {};

/**
 * Registers the given renderer to render the provided format override type. No-
 * op if the given format override type already has a renderer.
 * @param {string} formatOverrideType
 * @param {metricRenderer} renderer
 */
function registerOverrideRenderer(formatOverrideType, renderer) {
  if (!overrideRendererMap[formatOverrideType]) {
    overrideRendererMap[formatOverrideType] = renderer;
  }
}

/**
 * Renders the given value with the overridden type.
 * @param {!TableProvider.RawCellData} value
 * @param {!TableProvider=} opt_tableProvider
 * @param {!tfma.MetricValueFormatSpec=} opt_override
 * @return {!TableProvider.GvizCell}
 */
function renderValueWithFormatOverride(value, opt_tableProvider, opt_override) {
  if (goog.isDefAndNotNull(value)) {
    if (opt_tableProvider && opt_override) {
      try {
        return overrideRendererMap[opt_override.type](
            opt_tableProvider.applyOverride(value, opt_override));
      } catch (e) {
        return renderUnsupported({'override': opt_override, 'value': value});
      }
    } else {
      return renderValue(value);
    }
  } else {
    return renderNotAvailable();
  }
}

// Register all built-in override renderer.
registerOverrideRenderer(Constants.MetricValueFormat.ROW_ID, renderRowId);
registerOverrideRenderer(Constants.MetricValueFormat.INT, renderInteger);
registerOverrideRenderer(Constants.MetricValueFormat.FLOAT, renderFloat);
registerOverrideRenderer(Constants.MetricValueFormat.INT64, renderInt64);
registerOverrideRenderer(Constants.MetricValueFormat.HTML, renderString);
registerOverrideRenderer(Constants.MetricValueFormat.STRING, renderString);
registerOverrideRenderer(
    Constants.MetricValueFormat.METRIC_AT_K, renderMetricAtK);
registerOverrideRenderer(
    Constants.MetricValueFormat.MULTI_VALUES_METRIC_AT_K,
    renderMultiValuesMetricAtK);

goog.exportSymbol(
    'tfma.CellRenderer.renderValueWithFormatOverride',
    renderValueWithFormatOverride);
goog.exportSymbol('tfma.CellRenderer.parseRowId', parseRowId);
goog.exportSymbol('tfma.CellRenderer.registerRenderer', registerRenderer);
goog.exportSymbol(
    'tfma.CellRenderer.registerOverrideRenderer', registerOverrideRenderer);

exports = {
  BoundedValue,
  MetricAtK,
  MultiValuesMetricAtK,
  renderValue,
  parseRowId,
  renderValueWithFormatOverride,
  registerRenderer,
  registerOverrideRenderer,
};
