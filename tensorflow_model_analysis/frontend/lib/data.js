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
goog.module('tfma.Data');

const CellRenderer = goog.require('tfma.CellRenderer');
const Constants = goog.require('tfma.Constants');
const TableProvider = goog.require('tfma.TableProvider');

/**
 * The Data class represents the results of a particular evaluation. It provides
 * data to be rendered in the metrics table and metrics histogram.
 * @implements {TableProvider}
 */
class Data {
  /**
   * Do not call the constructor directly. Instead, please use the factory
   * method "build" to instantiate this class.
   * @param {!Array<string>} metrics Names of the metrics. Note that index of
   *     the metric names must match to the index of corresponding metric value
   *     in the array of metric values list returned by any element in data.
   * @param {SeriesList} data An array of Series. Each element can
   *     represent a slice or a version of a model.
   */
  constructor(metrics, data) {
    /** @private {!Array<string>} */
    this.metrics_ = metrics;

    /** @private {SeriesList} */
    this.data_ = data;

    /**
     * A lazily initialized object caching numerical metric values.
     * @private {!Object<!Object<number>>}
     */
    this.cachedNumericalMetricValue_ = {};

    /**
     * The lazily initalized feature strings.
     * @private {!Array<string>}
     */
    this.cachedFeatureStrings_ = [];

    /**
     * Mapping from a feature name to its data series.
     * This is used to quickly fetch a metric value for a given feature.
     * @private {!Object<!Series>}
     */
    this.featureDataSeries_ = {};
    this.data_.forEach((/** !Series */ dataSeries) => {
      const feature = dataSeries.getFeatureString();
      this.featureDataSeries_[feature] = dataSeries;
      this.cachedNumericalMetricValue_[feature] = {};
    });
  }

  /**
   * @return {!Array<string>} The keys of the data series feature object.
   *     Empty array in case 'featureDataSeries_' is not defined.
   * @export
   */
  getFeatureDataSeriesKeys() {
    if (this.featureDataSeries_) {
      return Object.keys(this.featureDataSeries_);
    }
    return [];
  }

  /**
   * @param {string} metric
   * @return {number} Index of a given metric in the metric list, or -1 if not
   *     found.
   * @export
   */
  getMetricIndex(metric) {
    return this.metrics_.indexOf(metric);
  }

  /**
   * @return {!Array<string>} The metrics list. The returned list should not be
   *     mutated and is treated as read-only.
   * @export
   */
  getMetrics() {
    return this.metrics_;
  }

  /**
   * @return {SeriesList} The Series list. The returned list should not be
   *     mutated and is treated as read-only.
   * @export
   */
  getSeriesList() {
    return this.data_;
  }

  /**
   * Filters the data series based on the given criterion function.
   * @param {function(!Series): boolean} filterFn
   * @return {!Data} Filtered data.
   * @export
   */
  filter(filterFn) {
    return new Data(this.metrics_, this.data_.filter(filterFn));
  }

  /**
   * Computes the value range for a specified column from the DataSeries list.
   * @param {string} metric Metric name.
   * @return {{min: number, max: number}} Value range for the specified column.
   * @export
   */
  getColumnRange(metric) {
    return this.data_.reduce((prev, cur) => {
      const feature = cur.getFeatureString();
      const value = this.getMetricValue(feature, metric);
      if (!isNaN(value) && value != null) {
        return {
          'min': Math.min(prev.min, value),
          'max': Math.max(prev.max, value)
        };
      } else {
        // If the metric is not set, exclude it when determining range.
        return prev;
      }
    }, {'min': Infinity, 'max': -Infinity});
  }

  /**
   * Gets the data series corresponding to a feature.
   * @param {string} feature Feature of which the data series is to be fetched.
   *     The feature must exist in the data.
   * @return {!Series|undefined} The data series of the feature.
   * @private
   */
  getSeries_(feature) {
    return this.featureDataSeries_[feature];
  }

  /**
   * @param {string} feature Feature of which the data series is to be fetched.
   *     The feature must exist in the data.
   * @return {(string|number)} The id for the series with the named feature. The
   *     return value will be an integer if it can be parsed properly or a
   *     string if not. If there is no series matching the feature, simply echo
   *     the feature back.
   * @export
   */
  getFeatureId(feature) {
    const series = this.getSeries_(feature);
    return series && series.getFeatureIdForMatching() || feature;
  }

  /**
   * Gets all the metric values for a given feature.
   * @param {string} feature Feature of which the metric values are to be
   *     fetched. The feature must exist in the data.
   * @return {!Array<number>} The metric values for the feature or empty array
   *     if feature not available
   * @export
   */
  getAllMetricValues(feature) {
    const series = this.getSeries_(feature);
    return series ? series.getMetricValuesList() : [];
  }

  /**
   * Gets one metric value for a given feature. If it is a complex type, use
   * CellRenderer to get the value we use for sorting in table view. If it is
   * string, parse it as float. The result is cached.
   * @param {string} feature Feature of which the metric value is to be fetched.
   *     The feature must exist in the data.
   * @param {string} metric Metric (name) of which the value is to be fetched.
   *     The metric must exist in the data's metric list.
   * @return {number} The metric value for a feature or NaN if the metric is not
   *     available.
   * @export
   */
  getMetricValue(feature, metric) {
    if (this.cachedNumericalMetricValue_[feature][metric] === undefined) {
      const metricIndex = this.getMetricIndex(metric);
      let value = this.getAllMetricValues(feature)[metricIndex];
      value = value != null ? value : NaN;
      if (goog.isObject(value)) {
        value = CellRenderer.renderValue(value)['v'];
      } else if (typeof value === 'string') {
        value = parseFloat(value);
      }
      this.cachedNumericalMetricValue_[feature][metric] = value;
    }
    return this.cachedNumericalMetricValue_[feature][metric];
  }

  /**
   * @return {!Array<string>} The features of all slices.
   * @export
   */
  getFeatures() {
    if (!this.cachedFeatureStrings_.length) {
      this.cachedFeatureStrings_ = this.data_.map((slice) => {
        return slice.getFeatureString();
      });
    }
    return this.cachedFeatureStrings_;
  }

  /**
   * @param {string} feature
   * @return {!Array<number|string>} A data table row that contains the feature
   *     in the beginning and then the metric values.
   * @private
   */
  getDataTableRow_(feature) {
    const values = this.getAllMetricValues(feature);
    return [feature].concat(values);
  }

  /**
   * @override
   * @export
   */
  getDataTable() {
    return this.data_.map(
        dataSeries => this.getDataTableRow_(dataSeries.getFeatureString()));
  }

  /**
   * @return {boolean} Whether the data is empty, i.e. has no features or has no
   *     metrics.
   */
  isEmpty() {
    return !this.data_.length || !this.metrics_.length;
  }

  /**
   * Checks if the two Data's are equal. Two data objects are equal if one is
   * out from the other but has all the data series as the other one. They
   * should have the same references to the metrics array and the data series in
   * the data series list (this.data_).
   * @param {!Data} data
   * @return {boolean} Whether the two Data are equal.
   * @export
   */
  equals(data) {
    const dataSeriesList = data.getSeriesList();
    if (this.metrics_ !== data.getMetrics() ||
        this.data_.length != dataSeriesList.length) {
      return false;
    }
    for (let i = 0; i < this.data_.length; i++) {
      if (this.data_[i] !== dataSeriesList[i]) {
        return false;
      }
    }
    return true;
  }

  /**
   * Metrics data is always ready to render.
   * @override
   * @export
   */
  readyToRender() {
    return !this.isEmpty();
  }

  /**
   * @override
   * @export
   */
  getHeader(requiredColumns) {
    return [Constants.Column.FEATURE].concat(requiredColumns);
  }

  /**
   * @override
   * @export
   */
  getFormats(specifiedFormats) {
    const desiredFormats = Object.assign({}, specifiedFormats);
    desiredFormats[Constants.Column.FEATURE] = {
      'type': Constants.MetricValueFormat.ROW_ID
    };
    return desiredFormats;
  }

  /**
   * @override
   * @export
   */
  applyOverride(value, override) {
    const transform = override['transform'];
    return transform ? transform(value) : value;
  }
}

/**
 * An array of Series object. Each can represent a slice or a version of a
 * model.
 * @typedef {!Array<!Series>}
 */
let SeriesList;

/**
 * @param {!Array<string>} metrics
 * @param {!Object} values A key value pair of metric names and their values.
 * @param {string} slice
 * @return {!Series} The Series object containing metrics extracted from values.
 */
function buildSeriesFromJson(metrics, values, slice) {
  const featureTokens = slice.split(':');
  const metricValuesList = metrics.map((metric) => {
    // Compute calibration from label and prediction if both are available.
    if (metric == Constants.Column.CALIBRATION &&
        values['averageLabel'] !== undefined &&
        values['averageRefinedPrediction'] !== undefined) {
      return values['averageRefinedPrediction'] / values['averageLabel'];
    }
    return values[metric];
  });

  return new Series(featureTokens[0], slice, metricValuesList);
}

/**
 * Factory method for Data.
 * @param {!Array<string>} metrics
 * @param {!Array<!Object>} data
 * @return {!Data}
 */
function build(metrics, data) {
  const inputSeries = data.map((json) => {
    return buildSeriesFromJson(metrics, json['metrics'], json['slice'] || '');
  });

  return new Data(metrics, inputSeries);
}

/**
 * Extracts all metrics defined in the data arrays.
 * @param {!Array<!Array<!Object>>} dataArrays An array of arrays of
 *     DataStatistics in JSON format.
 * @param {string} metricsFieldKey
 * @return {!Array<string>}
 */
function getAvailableMetrics(dataArrays, metricsFieldKey) {
  const seenMetrics = {};
  dataArrays.forEach((data) => {
    data.forEach((evaluation) => {
      const statistics = evaluation[metricsFieldKey] || {};
      for (let key in statistics) {
        seenMetrics[key] = true;
      }
    });
  });
  const metrics = Object.keys(seenMetrics);
  // If we can compute calibration, make it available.
  if (seenMetrics['averageRefinedPrediction'] && seenMetrics['averageLabel']) {
    metrics.push('calibration');
  }
  metrics.sort();
  return metrics;
}

/**
 * @param {!Object} plotMap A map where the keys are the base plot types that we
 *   have data for.
 * @return {!Array<string>} An array of all plot types that we can support. Many
 *   plot types shares the same data so with a base type, other plot types might
 *   be supported.
 */
function getAvailablePlotTypes(plotMap) {
  const plotsToShow = [];
  if (plotMap) {
    const supportedPlotTypes = [
      {
        type: Constants.PlotTypes.CALIBRATION_PLOT,
        additional: [
          Constants.PlotTypes.RESIDUAL_PLOT,
          Constants.PlotTypes.PREDICTION_DISTRIBUTION,
        ]
      },
      {
        type: Constants.PlotTypes.PRECISION_RECALL_CURVE,
        additional: [
          Constants.PlotTypes.ROC_CURVE, Constants.PlotTypes.ACCURACY_CHARTS,
          Constants.PlotTypes.GAIN_CHART
        ]
      },
      {
        type: Constants.PlotTypes.MACRO_PRECISION_RECALL_CURVE,
      },
      {
        type: Constants.PlotTypes.MICRO_PRECISION_RECALL_CURVE,
      },
      {
        type: Constants.PlotTypes.WEIGHTED_PRECISION_RECALL_CURVE,
      }, {
        type: Constants.PlotTypes.MULTI_CLASS_CONFUSION_MATRIX,
      }, {
        type: Constants.PlotTypes.MULTI_LABEL_CONFUSION_MATRIX,
      }
    ];

    supportedPlotTypes.forEach((plot) => {
      if (plotMap[plot.type]) {
        plotsToShow.push(plot.type);
        if (plot.additional) {
          plot.additional.forEach(additionalPlot => {
            plotsToShow.push(additionalPlot);
          });
        }
      }
    });
  }
  return plotsToShow;
}

/**
 * Extracts the metric values from a map of
 * tensorflow_model_analysis.MetricValue.
 * @param {!Object<!Object>} metricsMap The metrics map.
 * @return {!Object<!Object>}
 */
function flattenMetricsMap(metricsMap) {
  return Object.keys(metricsMap).reduce((acc, metricName) => {
    const metricValue = metricsMap[metricName];
    const metricFields = Object.keys(metricValue);
    // Since tensorflow_model_analysis.MetricValue is a oneof, there can only be
    // one field defined. Use it to extract the actual metric value.
    acc[metricName] = metricValue[metricFields[0]];
    return acc;
  }, {});
}


/**
 * Flattens all metrics in the given array of runs.
 * @param {!Array<!Object>} runs The runs.
 * @param {string} metricsKey The key for metrics in a run object.
 */
function flattenMetrics(runs, metricsKey) {
  runs.forEach(run => {
    run[metricsKey] = flattenMetricsMap(run[metricsKey]);
  });
}


/**
 * An inner class that represents a set of metric values. The set could be for a
 * slice or a model version. It returns the metrics as a series of values for
 * easy rendering into a table.
 */
class Series {
  /**
   * @param {string} columnName
   * @param {string} featureString
   * @param {!Array<SeriesValue>} values
   */
  constructor(columnName, featureString, values) {
    /**
     * @private {string}
     */
    this.columnName_ = columnName;

    /**
     * @private {string}
     */
    this.featureString_ = featureString;

    /**
     * @private {number}
     */
    this.parsedFeatureId_ = CellRenderer.parseRowId(featureString);

    /**
     * The feature id used for matching.
     * @private {string|number}
     */
    this.featureIdForMatching_ = isNaN(this.parsedFeatureId_) ?
        this.featureString_ :
        this.parsedFeatureId_;
    /**
     * @private {!Array<SeriesValue>}
     */
    this.values_ = values;
  }
  /**
   * @return {string} Returns the column name associated with this row.
   */
  getColumnName() {
    return this.columnName_;
  }

  /**
   * @return {string} Returns the feature id of this row.
   * @export
   */
  getFeatureString() {
    return this.featureString_;
  }

  /**
   * @return {string|number} Returns the parsed feature id.
   * @export
   */
  getFeatureIdForMatching() {
    return this.featureIdForMatching_;
  }

  /**
   * @return {!Array<SeriesValue>} Returns the metric value list.
   * @export
   */
  getMetricValuesList() {
    return this.values_;
  }
}

/**
 * @typedef {string|number|!Object}
 * @private
 */
let SeriesValue;

goog.exportSymbol('tfma.Data.build', build);
goog.exportSymbol('tfma.Data.flattenMetrics', flattenMetrics);
goog.exportSymbol('tfma.Data.getAvailableMetrics', getAvailableMetrics);
goog.exportSymbol('tfma.Data.getAvailablePlotTypes', getAvailablePlotTypes);

exports = {
  build,
  util: {
    getAvailableMetrics,
    flattenMetrics,
    getAvailablePlotTypes,
  },
  Data,
  Series,
};
