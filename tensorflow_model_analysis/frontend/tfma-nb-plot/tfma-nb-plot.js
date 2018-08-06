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
Polymer({
  /**
   * tfma-nb-plot is a wrapper component for tfma-plot to be used inside jupyter
   * notebook.
   */
  is: 'tfma-nb-plot',

  properties: {
    /** @type {!Object} */
    data: {type: Object},

    /** @type {!Object} */
    config: {type: Object},

    /**
     * Available plot types.
     * @type {!Array<tfma.PlotTypes>|undefined}
     */
    availableTypes_:
        {type: Array, computed: 'computeAvailableTypes_(plotData_, config)'},

    /** @type {!Object} */
    plotData_: {type: Object, computed: 'computePlotData_(data, config)'},

    /** @type {string} */
    heading_: {type: String, computed: 'computeHeading_(config, initialType_)'},

    /** @type {string} */
    initialType_:
        {type: String, computed: 'computeInitialType_(availableTypes_)'},
  },

  /**
   * Computes the plot data based on raw data and config.
   * @param {!Object} data
   * @param {!Object} config
   * @return {!Object}
   * @private
   */
  computePlotData_: function(data, config) {
    const plotData = {};
    const metricKeys = config['metricKeys'];
    this.maybeSetPlotData_(
        data, metricKeys['calibrationPlot'], plotData,
        tfma.PlotDataFieldNames.CALIBRATION_DATA);
    this.maybeSetPlotData_(
        data, metricKeys['aucPlot'], plotData,
        tfma.PlotDataFieldNames.PRECISION_RECALL_CURVE_DATA);
    return {'plotData': plotData};
  },

  /**
   * Finds plot data as specified in plotConfig and if available, adds it to the
   * output at the specified key.
   * @param {!Object} data
   * @param {!Object|undefined} plotConfig
   * @param {!Object} output
   * @param {string} outputKey
   * @private
   */
  maybeSetPlotData_: function(data, plotConfig, output, outputKey) {
    const plotDataName = plotConfig && plotConfig['metricName'];
    if (plotDataName) {
      const dataSeriesName = plotConfig['dataSeries'];
      const dataSeries =
          data[plotDataName] && data[plotDataName][dataSeriesName];
      if (dataSeries) {
        output[outputKey] = data[plotDataName];
      }
    }
  },
  /**
   * @param {string|number} value
   * @return {number} value or NaN.
   * @private
   */
  getValue_: function(value) {
    // When NaN goes across the python / js bridge, it becomes the string "nan".
    // Convert it back to NaN.
    return value == 'nan' ? NaN : /** @type {number} */(value);
  },

  /**
   * Determines the list of available plot based on the data.
   * @param {!Object} plotData
   * @return {!Array<string>}
   * @private
   */
  computeAvailableTypes_: function(plotData) {
    const availableTypes = [];
    const data = plotData && plotData['plotData'];
    if (data) {
      if (data[tfma.PlotDataFieldNames.CALIBRATION_DATA]) {
        availableTypes.push(tfma.PlotTypes.CALIBRATION_PLOT);
        availableTypes.push(tfma.PlotTypes.PREDICTION_DISTRIBUTION);
      }
      if (data[tfma.PlotDataFieldNames.PRECISION_RECALL_CURVE_DATA]) {
        availableTypes.push(tfma.PlotTypes.PRECISION_RECALL_CURVE);
        availableTypes.push(tfma.PlotTypes.ROC_CURVE);
      }
    }
    return availableTypes;
  },

  /**
   * Determines the heading of the plot.
   * @param {!Object} config
   * @param {string} initialType
   * @return {string}
   * @private
   */
  computeHeading_: function(config, initialType) {
    const sliceName = config['sliceName'];
    if (!initialType) {
      return 'Plot data not available';
    }
    return sliceName ? 'Plot for ' + sliceName : '';
  },

  /**
   * @param {!Array<string>} availableTypes
   * @return {string} The initial type the plot should display.
   * @private
   */
  computeInitialType_: function(availableTypes) {
    return availableTypes[0] || '';
  },
});
