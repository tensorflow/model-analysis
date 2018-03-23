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

    if (metricKeys['calibrationPlot']) {
      const calibrationPlot =
          this.extractCalibrationPlotData_(data, metricKeys['calibrationPlot']);
      if (calibrationPlot.length) {
        plotData[tfma.PlotDataFieldNames.CALIBRATION_DATA] = calibrationPlot;
      }
    }

    if (metricKeys['aucPlot']) {
      const aucPlot = this.extractAucPlotData_(data, metricKeys['aucPlot']);
      if (aucPlot.length) {
        plotData[tfma.PlotDataFieldNames.PRECISION_RECALL_CURVE_DATA] = aucPlot;
      }
    }

    return {'plotData': plotData};
  },

  /**
   * Extracts the calibration data and transforms it into expected format.
   * @param {!Object} data
   * @param {!Object} metricKeys
   * @return {!Array<!Object>}
   */
  extractCalibrationPlotData_: function(data, metricKeys) {
    const matricesKey = metricKeys['matrices'];
    const boundariesKey = metricKeys['boundaries'];
    const calibrationPlot = [];
    const matricesData = matricesKey && data[matricesKey];
    const boundariesData = boundariesKey && data[boundariesKey];

    if (matricesData && boundariesData) {
      let i = 0;
      let lowerBound = -Infinity;
      let upperBound;
      do {
        upperBound =
            boundariesData[i] !== undefined ? boundariesData[i] : Infinity;

        if (matricesData[i]) {
          calibrationPlot.push({
            'lowerThresholdInclusive': lowerBound,
            'upperThresholdExclusive': upperBound,
            'totalWeightedRefinedPrediction': matricesData[i][0],
            'totalWeightedLabel': matricesData[i][1],
            'numWeightedExamples': matricesData[i][2],
          });
        }
        lowerBound = upperBound;
        i++;
      } while (i <= boundariesData.length);
    }

    return calibrationPlot;
  },

  /**
   * Extracts the AUC plot data and transforms it into expected format.
   * @param {!Object} data
   * @param {!Object} metricKeys
   * @return {!Array<!Object>}
   */
  extractAucPlotData_: function(data, metricKeys) {
    const matricesKey = metricKeys['matrices'];
    const thresholdsKey = metricKeys['thresholds'];
    const aucData = [];
    const matrices = matricesKey && data[matricesKey];
    const thresholds = thresholdsKey && data[thresholdsKey];

    if (matrices && thresholds) {
      matrices.forEach((matrix, i) => {
        aucData.push({
          'binaryClassificationThreshold':
              {'predictionThreshold': thresholds[i]},
          'matrix': {
            'falseNegatives': matrix[0],
            'trueNegatives': matrix[1],
            'falsePositives': matrix[2],
            'truePositives': matrix[3],
            'precision': this.getValue_(matrix[4]),
            'recall': this.getValue_(matrix[5]),
          }
        });
      });
    }

    return aucData;
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
