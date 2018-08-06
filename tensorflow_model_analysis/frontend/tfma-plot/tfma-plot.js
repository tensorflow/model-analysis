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
(() => {
  const TABS = {
    CALIBRATION_PLOT: 'cp',
    MACRO_PRECISION_RECALL: 'mapr',
    MICRO_PRECISION_RECALL: 'mipr',
    PRECISION_RECALL: 'pr',
    PREDICTION_DISTRIBUTION: 'pd',
    ROC: 'roc',
    WEIGHTED_PRECISION_RECALL: 'wpr',
  };

  const SUPPORTED_VISUALIZATION_ = {};
  SUPPORTED_VISUALIZATION_[tfma.PlotTypes.CALIBRATION_PLOT] = {
    type: TABS.CALIBRATION_PLOT,
    text: 'Calibration Plot'
  };
  SUPPORTED_VISUALIZATION_[tfma.PlotTypes.PRECISION_RECALL_CURVE] = {
    type: TABS.PRECISION_RECALL,
    text: 'Precision-Recall Curve',
  };
  SUPPORTED_VISUALIZATION_[tfma.PlotTypes.MACRO_PRECISION_RECALL_CURVE] = {
    type: TABS.MACRO_PRECISION_RECALL,
    text: 'Macro PR Curve',
  };
  SUPPORTED_VISUALIZATION_[tfma.PlotTypes.MICRO_PRECISION_RECALL_CURVE] = {
    type: TABS.MICRO_PRECISION_RECALL,
    text: 'Micro PR Curve',
  };
  SUPPORTED_VISUALIZATION_[tfma.PlotTypes.WEIGHTED_PRECISION_RECALL_CURVE] = {
    type: TABS.WEIGHTED_PRECISION_RECALL,
    text: 'Weighted PR Curve',
  };
  SUPPORTED_VISUALIZATION_[tfma.PlotTypes.PREDICTION_DISTRIBUTION] = {
    type: TABS.PREDICTION_DISTRIBUTION,
    text: 'Prediction Distribution',
  };
  SUPPORTED_VISUALIZATION_[tfma.PlotTypes.ROC_CURVE] = {
    type: TABS.ROC,
    text: 'ROC Curve',
  };

  Polymer({
    is: 'tfma-plot',
    properties: {
      /**
       * Available plot types.
       * @type {!Array<tfma.PlotTypes>|undefined}
       */
      availableTypes: {type: Array},

      /**
       * The initial chosen plot type. Resets to null after the user switches
       * to a new tab.
       * @type {?tfma.PlotTypes}
       */
      initialType: {type: String, observer: 'initialTypeChanged_'},

      /** @type {!Object} */
      data: {type: Object},

      /** @type {string} */
      heading: {type: String},

      /**
       * An array of configuration for available tabs.
       * @type {!Array<!Object>}
       * @private
       */
      availableTabs_:
          {type: Array, computed: 'computeAvailableTabs_(availableTypes)'},

      /**
       * @type {boolean}
       */
      loading: {type: Boolean, reflectToAttribute: true},

      error_: {
        type: Boolean,
        reflectToAttribute: true,
        computed: 'computeError_(loading, data)'
      },

      /**
       * The name of the selected tab.
       * @type {string}
       * @private
       */
      selectedTab_: {type: String, observer: 'selectedTabChanged_'},

      /**
       * A map of all tab names.
       * @type {!Object<string>}
       * @private
       */
      tabNames_: {
        type: Object,
        value: {
          Calibration: TABS.CALIBRATION_PLOT,
          Prediction: TABS.PREDICTION_DISTRIBUTION,
          Macro: TABS.MACRO_PRECISION_RECALL,
          Micro: TABS.MICRO_PRECISION_RECALL,
          Precision: TABS.PRECISION_RECALL,
          ROC: TABS.ROC,
          Weighted: TABS.WEIGHTED_PRECISION_RECALL,
        }
      },

      /**
       * The selected page.
       * @type {!Element|undefined}
       * @private
       */
      selectedPage_: {type: Object, observer: 'selectedPageChanged_'},

      /**
       * The data used by the calibration plot.
       * @type {!Array<!Object>}
       * @private
       */
      calibrationData_:
          {type: Array, computed: 'computeCalibrationData_(data)'},

      /**
       * The data used by the precision-recall-curve.
       * @type {!Array<!Object>}
       * @private
       */
      precisionRecallCurveData_:
          {type: Array, computed: 'computePrecisionRecallCurveData_(data)'},

      /**
       * The data used by the macro precision-recall-curve.
       * @type {!Array<!Object>}
       * @private
       */
      macroPrecisionRecallCurveData_: {
        type: Array,
        computed: 'computeMacroPrecisionRecallCurveData_(data)'
      },

      /**
       * The data used by the micro precision-recall-curve.
       * @type {!Array<!Object>}
       * @private
       */
      microPrecisionRecallCurveData_: {
        type: Array,
        computed: 'computeMicroPrecisionRecallCurveData_(data)'
      },

      /**
       * The data used by the weighted precision-recall-curve.
       * @type {!Array<!Object>}
       * @private
       */
      weightedPrecisionRecallCurveData_: {
        type: Array,
        computed: 'computeWeightedPrecisionRecallCurveData_(data)'
      },
    },

    /**
     * Extracts an array of calibration data out of the raw data.
     * @param {?Object} data
     * @return {!Array<!Object>}
     */
    computeCalibrationData_: function(data) {
      const plotData = data && data['plotData'] || {};
      return plotData[tfma.PlotDataFieldNames.CALIBRATION_DATA] &&
          plotData[tfma.PlotDataFieldNames
                       .CALIBRATION_DATA][tfma.PlotDataFieldNames
                                              .CALIBRATION_BUCKETS] ||
          [];
    },

    /**
     * Extracts an array of precision recall curve data out of the raw data.
     * @param {?Object} data
     * @return {!Array<!Object>}
     */
    computePrecisionRecallCurveData_: function(data) {
      const plotData = data && data['plotData'] || {};
      return plotData[tfma.PlotDataFieldNames.PRECISION_RECALL_CURVE_DATA] &&
          plotData[tfma.PlotDataFieldNames
                       .PRECISION_RECALL_CURVE_DATA][tfma.PlotDataFieldNames
                                                         .CONFUSION_MATRICES] ||
          [];
    },

    /**
     * Extracts an array of macro precision recall curve data out of the raw
     * data.
     * @param {?Object} data
     * @return {!Array<!Object>}
     */
    computeMacroPrecisionRecallCurveData_: function(data) {
      const plotData = data && data['plotData'] || {};
      return plotData[tfma.PlotDataFieldNames
                          .MACRO_PRECISION_RECALL_CURVE_DATA] ||
          [];
    },

    /**
     * Extracts an array of micro precision recall curve data out of the raw
     * data.
     * @param {?Object} data
     * @return {!Array<!Object>}
     */
    computeMicroPrecisionRecallCurveData_: function(data) {
      const plotData = data && data['plotData'] || {};
      return plotData[tfma.PlotDataFieldNames
                          .MICRO_PRECISION_RECALL_CURVE_DATA] ||
          [];
    },

    /**
     * Extracts an array of weighted precision recall curve data out of the
     * raw data.
     * @param {?Object} data
     * @return {!Array<!Object>}
     */
    computeWeightedPrecisionRecallCurveData_: function(data) {
      const plotData = data && data['plotData'] || {};
      return plotData[tfma.PlotDataFieldNames
                          .WEIGHTED_PRECISION_RECALL_CURVE_DATA] ||
          [];
    },

    /**
     * Observer for chonsenType. Sets selectedTab_ property upon
     * initialization.
     * @param {?string} initialType
     * @private
     */
    initialTypeChanged_: function(initialType) {
      if (initialType) {
        this.selectedTab_ = SUPPORTED_VISUALIZATION_[initialType].type;
      }
    },

    /**
     * Observer for selectedTab_.
     * @private
     */
    selectedTabChanged_: function() {
      // Clears initial chonsen type.
      this.initialType = null;
    },

    /**
     * Determines if we failed to fetch data from the backend.
     * @param {boolean} loading
     * @param {?Object} data null if encountered error.
     * @return {boolean}
     * @private
     */
    computeError_: function(loading, data) {
      return !loading && data == null;
    },

    /**
     * Fires a reload-plot-data event.
     * @private
     */
    reload_: function() {
      this.fire(tfma.Event.RELOAD_PLOT_DATA);
    },

    /**
     * @param {!Array<tfma.PlotTypes>} availableTypes
     * @return {!Array<!Object>} An aray of configuration for each type of plot
     *     specified.
     * @private
     */
    computeAvailableTabs_: function(availableTypes) {
      const supported = [];
      availableTypes.forEach((type) => {
        if (SUPPORTED_VISUALIZATION_[type]) {
          supported.push(SUPPORTED_VISUALIZATION_[type]);
        }
      });
      return supported;
    },

    /**
     * Observer for property selectedPage_. Makes sure the newly selected page
     * is properly rendered by calling redraw on all google-chart under it.
     * @param {!Element|undefined} page
     * @private
     */
    selectedPageChanged_: function(page) {
      if (page) {
        const charts = page.querySelectorAll('google-chart');
        for (let i = charts.length - 1, chart; chart = charts[i]; i--) {
          chart.redraw();
        }
      }
    },
  });
})();
