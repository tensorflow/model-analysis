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
 * Json representation of the BinaryClassificationThreshold.
 * @typedef {{
 *    predictionThreshold: number,
 * }}
 */
let BinaryClassificationThresholdProto;

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
 * Json representation of the value cutoff pair.
 * @typedef {{
 *   cutoff: number,
 *   value: number,
 * }}
 */
let ValueCutoffPair;

/**
 * @typedef {{
 *   values: !Array<!ValueCutoffPair>
 * }}
 */
let ValueAtCutoffs;

/**
 * @enum {string}
 */
const ValueAtCutoffsFieldNames = {
  BOUNDED_VALUE: 'boundedValue',
  CUTOFF: 'cutoff',
  VALUE: 'value',
  VALUES: 'values',
};

/**
 * @typedef {{
 *   threshold: number,
 *   falseNegatives: number,
 *   trueNegatives: number,
 *   falsePositives: number,
 *   truePositives: number,
 *   precision: number,
 *   recall: number,
 * }}
 */
let ConfusionMatrixAtThreshold;

/**
 * @typedef {{
 *   matrices: !Array<!ConfusionMatrixAtThreshold>
 * }}
 */
let ConfusionMatrixAtThresholds;

/**
 * @enum {string}
 */
const ConfusionMatrixAtThresholdsFieldNames = {
  BOUNDED_FALSE_NEGATIVES: 'boundedFalseNegatives',
  BOUNDED_FALSE_POSITIVES: 'boundedFalsePositives',
  BOUNDED_PRECISION: 'boundedPrecision',
  BOUNDED_RECALL: 'boundedRecall',
  BOUNDED_TRUE_NEGATIVES: 'boundedTrueNegatives',
  BOUNDED_TRUE_POSITIVES: 'boundedTruePositives',
  FALSE_NEGATIVES: 'falseNegatives',
  FALSE_POSITIVES: 'falsePositives',
  MATRICES: 'matrices',
  PRECISION: 'precision',
  RECALL: 'recall',
  THRESHOLD: 'threshold',
  TRUE_NEGATIVES: 'trueNegatives',
  TRUE_POSITIVES: 'truePositives',
};

/**
 * @typedef {{
 * dataType: string,
 * shape: !Array<number>,
 * bytesValues: (!Array<string>|undefined),
 * int32Values: (!Array<number>|undefined),
 * int64Values: (!Array<string>|undefined),
 * float32Values: (!Array<number>|undefined),
 * float64Values: (!Array<number>|undefined),
 * }}
 */
let ArrayValue;

/**
 * @enum {string}
 */
const ValueType = {
  ARRAY_VALUE: 'arrayValue',
  BOUNDED_VALUE: 'boundedValue',
  CONFUSION_MATRIX_AT_THRESHOLDS: 'confusionMatrixAtThresholds',
  FLOAT: 'float',
  MULTI_CLASS_CONFUSION_MATRIX: 'MultiClassConfusionMatrix',
  RATIO_VALUE: 'ratioValue',
  SCALAR_IN_VALUE: 'scalarInValue',
  STRING: 'string',
  VALUE_AT_CUTOFFS: 'valueAtCutoffs',
  UNKNOWN: 'unknown',
};

/**
 * @enum {string}
 */
const ArrayValueFieldNames = {
  DATA_TYPE: 'dataType',
  SHAPE: 'shape',
  BYTES_VALUES: 'bytesValues',
  INT32_VALUES: 'int32Values',
  INT64_VALUES: 'int64Values',
  FLOAT32_VALUES: 'float32Values',
  FLOAT64_VALUES: 'float64Values',
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
 * @enum {string}
 */
const RatioValueFieldNames = {
  DENOMINATOR: 'denominator',
  NUMERATOR: 'numerator',
  RATIO: 'ratio',
};

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
 * The prefix for the bounded value version of a scalar metric.
 * @const @private
 */
const BOUNDED_VALUE_METRIC_NAME_PREFIX_ = 'bounded';

/**
 * Extracts float value with the given key out of the data object. It is assumed
 * that there could be two fields of the form xyz and boundedXyz while xyz is
 * the float version of the value and boundedXyz is the BoundedValue version of
 * the value. Preference is given to the bounded value version.
 * @param {!Object} data
 * @param {string} key
 * @return {number}
 */
function extractFloatValue(data, key) {
  const boundedKey = BOUNDED_VALUE_METRIC_NAME_PREFIX_ +
      key.charAt(0).toUpperCase() + key.slice(1);
  const boundedValue = data[boundedKey];
  const value =
      (boundedValue !== undefined ? boundedValue[BoundedValueFieldNames.VALUE] :
                                    data[key]) ||
      0;
  // Since NaN and Infinity is not in JSON spec, they are serialized as strings
  // when sent to the client side. When that happens, we need to parse them as
  // float to restore them.
  return typeof value == 'string' ? parseFloat(value) : value;
}

/**
 * @param {!ValueAtCutoffs} value
 * @param {number} position Note that position is 0-based while k is 1-based.
 * @return {number} Returns the precision value at the given position.
 */
function getValueAt(value, position) {
  return extractFloatValue(value[position], ValueAtCutoffsFieldNames.VALUE);
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
 * Extracts the value from the value field of BoundedValue strcuture. If it's a
 * number, just return it.
 * @param {number|!Object} value Can be a number or BoundedValue.
 * @return {number}
 */
function maybeExtractBoundedValue(value) {
  let extractedValue = value;
  if (isRatioValue(value)) {
    extractedValue = value['ratio']['value'];
  } else if (isBoundedValue(value)) {
    extractedValue = value['value'];
  }
  return extractedValue;
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
    // zero, simply show zero. This helps the user to differentiate really small
    // values that get rounded off to 0.00000 from 0.
    // @bug 32777052 (comment 37)
    'f': value === 0 ? '0' : trimFloat(value),
    'v': value,
  };
}

/**
 * @param {!Object} value
 * @return {!TableProvider.GvizCell} A gviz cell for a float.
 */
function renderScalarInValue(value) {
  return renderFloat(value[BoundedValueFieldNames.VALUE]);
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
    // @bug 64568820
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
    'f': '<tfma-bounded-value value=' + estimatedValue +
        ' lower-bound=' + value[BoundedValueFieldNames.LOWER_BOUND] +
        ' upper-bound=' + value[BoundedValueFieldNames.UPPER_BOUND] +
        '></tfma-bounded-value>',
    'v': estimatedValue,
  };
}

/**
 * @param {!BoundedValue} value
 * @return {!TableProvider.GvizCell} A gviz cell for a bounded
 *     value.
 */
function renderRatioValue(value) {
  // Render RatioValue as bounded value if the confidence interval is computed.
  // Otherwise, render it as float.
  const ratio = value['ratio'];
  return isBoundedValue(ratio) ? renderBoundedValue(ratio) :
                                 renderFloat(parseFloat(ratio['value']));
}

/**
 * @param {!ArrayValue} value
 * @return {!TableProvider.GvizCell} A gviz cell for a bounded
 *     value.
 */
function renderArrayValue(value) {
  return {
    'f': '<tfma-array-value ' +
        createAttributeHtml('data', JSON.stringify(value)) +
        '></tfma-array-value>',
    // It does not make sense to sort by confusion matrix. Making the value of
    // the cell always 0.
    'v': 0,
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
 * @param {!ValueCutoffPair} a
 * @param {!ValueCutoffPair} b
 * @return {number}
 */
function sortByCutoff(a, b) {
  const ka = a[ValueAtCutoffsFieldNames.CUTOFF] || 0;
  const kb = b[ValueAtCutoffsFieldNames.CUTOFF] || 0;
  return ka - kb;
}

/**
 * @param {!ValueAtCutoffs} value
 * @return {!TableProvider.GvizCell} A gviz cell for a series of
 *     precision at k.
 */
function renderValueAtCutoffs(value) {
  const values = value[ValueAtCutoffsFieldNames.VALUES];
  values.sort(sortByCutoff);
  return {
    'f': '<tfma-value-at-cutoffs ' +
        createAttributeHtml('data', JSON.stringify(value)) +
        '></tfma-value-at-cutoffs>',
    // Use the precision at the first position for sorting.
    'v': getValueAt(values, 0),
  };
}

/**
 * @param {!ConfusionMatrixAtThresholds} value
 * @param {number} index
 * @return {number} The precision with the given index.
 */
function getPrecisionAt(value, index) {
  return extractFloatValue(
      value[ConfusionMatrixAtThresholdsFieldNames.MATRICES][index],
      ConfusionMatrixAtThresholdsFieldNames.PRECISION);
}

/**
 * @param {!ConfusionMatrixAtThresholds} value
 * @return {!TableProvider.GvizCell} A gviz cell for a series of confusion
 *     matrix at thresholds.
 */
function renderConfusionMatrixAtThresholds(value) {
  return {
    'f': '<tfma-confusion-matrix-at-thresholds ' +
        createAttributeHtml('data', JSON.stringify(value)) +
        '></tfma-confusion-matrix-at-thresholds>',
    // Use the precision at the first position for sorting.
    'v': getPrecisionAt(value, 0),
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
 * @type {!Array<{type: string,
 *     check:function((number|string|?Object)):boolean}>}
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
  if (value != null) {
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
 * @return {boolean} Returns true if the given value contains a number in its
 *     only field named "value"
 */
function isScalarInValue(value) {
  const scalar = value[BoundedValueFieldNames.VALUE];
  return scalar !== undefined && typeof scalar === 'number' &&
      (Object.keys(/** @type{!Object} */ (value)).length === 1);
}

/**
 * @param {(string|number|?Object)} value
 * @return {boolean} Returns true if the given value represents a bounded value.
 */
function isBoundedValue(value) {
  return !!value && value[BoundedValueFieldNames.LOWER_BOUND] !== undefined &&
      value[BoundedValueFieldNames.UPPER_BOUND] !== undefined &&
      value[BoundedValueFieldNames.VALUE] !== undefined;
}

/**
 * @param {(string|number|?Object)} value
 * @return {boolean} Returns true if the given value represents a ratio value.
 */
function isRatioValue(value) {
  return !!value && value[RatioValueFieldNames.NUMERATOR] !== undefined &&
      value[RatioValueFieldNames.DENOMINATOR] !== undefined &&
      value[RatioValueFieldNames.RATIO] !== undefined;
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

/**
 * @param {(string|number|?Object)} value
 * @return {boolean} Returns true if the given value is an array of json
 *     representation of ValueAtCutoffs.
 */
function isValueAtCutoffs(value) {
  return value && value[ValueAtCutoffsFieldNames.VALUES] &&
      checkRepeatedMetric(
             value[ValueAtCutoffsFieldNames.VALUES],
             item => item[ValueAtCutoffsFieldNames.CUTOFF] != null);
}


/**
 * @param {(string|number|?Object)} value
 * @return {boolean} Returns true if the given value is an array of json
 *     representation of ValueAtCutoffs.
 */
function isConfusionMatrixAtThresholds(value) {
  const hasMatrixData = (item) =>
      item[ConfusionMatrixAtThresholdsFieldNames.FALSE_NEGATIVES] != null ||
      item[ConfusionMatrixAtThresholdsFieldNames.FALSE_POSITIVES] != null ||
      item[ConfusionMatrixAtThresholdsFieldNames.TRUE_NEGATIVES] != null ||
      item[ConfusionMatrixAtThresholdsFieldNames.TRUE_POSITIVES] != null;
  const hasMatrixDataWithConfidenceInterval = (item) =>
      item[ConfusionMatrixAtThresholdsFieldNames.BOUNDED_FALSE_NEGATIVES] !=
          null &&
      item[ConfusionMatrixAtThresholdsFieldNames.BOUNDED_FALSE_POSITIVES] !=
          null &&
      item[ConfusionMatrixAtThresholdsFieldNames.BOUNDED_PRECISION] != null &&
      item[ConfusionMatrixAtThresholdsFieldNames.BOUNDED_RECALL] != null &&
      item[ConfusionMatrixAtThresholdsFieldNames.BOUNDED_TRUE_NEGATIVES] !=
          null &&
      item[ConfusionMatrixAtThresholdsFieldNames.BOUNDED_TRUE_POSITIVES] !=
          null;

  return value && value[ConfusionMatrixAtThresholdsFieldNames.MATRICES] &&
      checkRepeatedMetric(
             value[ConfusionMatrixAtThresholdsFieldNames.MATRICES],
             item => item[ConfusionMatrixAtThresholdsFieldNames.THRESHOLD] !=
                     null &&
                 (hasMatrixData(item) ||
                  hasMatrixDataWithConfidenceInterval(item)));
}

/**
 * @param {(string|number|?Object)} value
 * @return {boolean} Returns true if the given value is an array of json
 *     representation of ArrayValue.
 */
function isArrayValue(value) {
  return value[ArrayValueFieldNames.SHAPE] !== undefined &&
      value[ArrayValueFieldNames.DATA_TYPE] !== undefined &&
      (value[ArrayValueFieldNames.BYTES_VALUES] !== undefined ||
       value[ArrayValueFieldNames.INT32_VALUES] !== undefined ||
       value[ArrayValueFieldNames.INT64_VALUES] !== undefined ||
       value[ArrayValueFieldNames.FLOAT32_VALUES] !== undefined ||
       value[ArrayValueFieldNames.FLOAT64_VALUES] !== undefined);
}

// Registers all built-in renderers.
registerRenderer(ValueType.FLOAT, renderFloat, x => typeof x === 'number');
registerRenderer(
    ValueType.SCALAR_IN_VALUE, renderScalarInValue, isScalarInValue);
registerRenderer(ValueType.BOUNDED_VALUE, renderBoundedValue, isBoundedValue);
registerRenderer(ValueType.STRING, renderString, x => typeof x === 'string');
registerRenderer(
    ValueType.MULTI_CLASS_CONFUSION_MATRIX, renderMultiClassConfusionMatrix,
    isMultiClassConfusionMatrix);
registerRenderer(
    ValueType.VALUE_AT_CUTOFFS, renderValueAtCutoffs, isValueAtCutoffs);
registerRenderer(
    ValueType.CONFUSION_MATRIX_AT_THRESHOLDS, renderConfusionMatrixAtThresholds,
    isConfusionMatrixAtThresholds);
registerRenderer(ValueType.RATIO_VALUE, renderRatioValue, isRatioValue);
registerRenderer(ValueType.ARRAY_VALUE, renderArrayValue, isArrayValue);

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
  if (value != null) {
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
    Constants.MetricValueFormat.VALUE_AT_CUTOFFS, renderValueAtCutoffs);

goog.exportSymbol(
    'tfma.CellRenderer.renderValueWithFormatOverride',
    renderValueWithFormatOverride);
goog.exportSymbol('tfma.CellRenderer.parseRowId', parseRowId);
goog.exportSymbol('tfma.CellRenderer.registerRenderer', registerRenderer);
goog.exportSymbol(
    'tfma.CellRenderer.registerOverrideRenderer', registerOverrideRenderer);

goog.exportSymbol('tfma.CellRenderer.isBoundedValue', isBoundedValue);
goog.exportSymbol('tfma.CellRenderer.isRatioValue', isRatioValue);
goog.exportSymbol('tfma.CellRenderer.extractFloatValue', extractFloatValue);
goog.exportSymbol(
    'tfma.CellRenderer.maybeExtractBoundedValue', maybeExtractBoundedValue);


exports = {
  BoundedValue,
  renderValue,
  parseRowId,
  renderValueWithFormatOverride,
  registerRenderer,
  registerOverrideRenderer,
  extractFloatValue,
};
