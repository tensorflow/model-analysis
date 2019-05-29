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
import {PolymerElement} from '@polymer/polymer/polymer-element.js';
import {template} from './tfma-plot-template.html.js';

import '@polymer/iron-pages/iron-pages.js';
import '@polymer/paper-button/paper-button.js';
import '@polymer/paper-spinner/paper-spinner.js';
import '@polymer/paper-tabs/paper-tabs.js';
import '../tfma-accuracy-charts/tfma-accuracy-charts.js';
import '../tfma-calibration-plot/tfma-calibration-plot.js';
import '../tfma-precision-recall-curve/tfma-precision-recall-curve.js';
import '../tfma-prediction-distribution/tfma-prediction-distribution.js';
import '../tfma-roc-curve/tfma-roc-curve.js';

const TABS = {
  CALIBRATION_PLOT: 'cp',
  MACRO_PRECISION_RECALL: 'mapr',
  MICRO_PRECISION_RECALL: 'mipr',
  PRECISION_RECALL: 'pr',
  PREDICTION_DISTRIBUTION: 'pd',
  ROC: 'roc',
  WEIGHTED_PRECISION_RECALL: 'wpr',
};

const SUPPORTED_VISUALIZATION_ = {
  [tfma.PlotTypes.CALIBRATION_PLOT]:
      {type: TABS.CALIBRATION_PLOT, text: 'Calibration Plot'},
  [tfma.PlotTypes.PRECISION_RECALL_CURVE]: {
    type: TABS.PRECISION_RECALL,
    text: 'Precision-Recall Curve',
  },
  [tfma.PlotTypes.MACRO_PRECISION_RECALL_CURVE]: {
    type: TABS.MACRO_PRECISION_RECALL,
    text: 'Macro PR Curve',
  },
  [tfma.PlotTypes.MICRO_PRECISION_RECALL_CURVE]: {
    type: TABS.MICRO_PRECISION_RECALL,
    text: 'Micro PR Curve',
  },
  [tfma.PlotTypes.WEIGHTED_PRECISION_RECALL_CURVE]: {
    type: TABS.WEIGHTED_PRECISION_RECALL,
    text: 'Weighted PR Curve',
  },
  [tfma.PlotTypes.PREDICTION_DISTRIBUTION]: {
    type: TABS.PREDICTION_DISTRIBUTION,
    text: 'Prediction Distribution',
  },
  [tfma.PlotTypes.ROC_CURVE]: {
    type: TABS.ROC,
    text: 'ROC Curve',
  },
};

/**
 * tfma-plot can render a number of supported plots.
 *
 * @polymer
 */
export class Plot extends PolymerElement {
  constructor() {
    super();
  }

  static get is() {
    return 'tfma-plot';
  }

  /** @return {!HTMLTemplateElement} */
  static get template() {
    return template;
  }

  /** @return {!PolymerElementProperties} */
  static get properties() {
    return {
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
          'Calibration': TABS.CALIBRATION_PLOT,
          'Prediction': TABS.PREDICTION_DISTRIBUTION,
          'Macro': TABS.MACRO_PRECISION_RECALL,
          'Micro': TABS.MICRO_PRECISION_RECALL,
          'Precision': TABS.PRECISION_RECALL,
          'ROC': TABS.ROC,
          'Weighted': TABS.WEIGHTED_PRECISION_RECALL,
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
    };
  }

  /**
   * Extracts an array of calibration data out of the raw data.
   * @param {?Object} data
   * @return {!Array<!Object>}
   */
  computeCalibrationData_(data) {
    const plotData = data && data['plotData'] || {};
    return plotData[tfma.PlotDataFieldNames.CALIBRATION_DATA] &&
        plotData[tfma.PlotDataFieldNames
                     .CALIBRATION_DATA][tfma.PlotDataFieldNames
                                            .CALIBRATION_BUCKETS] ||
        [];
  }

  /**
   * Extracts an array of precision recall curve data out of the raw data.
   * @param {?Object} data
   * @return {!Array<!Object>}
   */
  computePrecisionRecallCurveData_(data) {
    return this.getMatricesForPRCurve_(
        data, tfma.PlotDataFieldNames.PRECISION_RECALL_CURVE_DATA);
  }

  /**
   * Extracts the matrices data from the curve plot data with the given key.
   * @param {?Object} data
   * @param {string} curveKey
   * @return {!Array<!Object>}
   * @private
   */
  getMatricesForPRCurve_(data, curveKey) {
    const plotData = data && data['plotData'] || {};
    const curveData = plotData[curveKey];
    return curveData && curveData[tfma.PlotDataFieldNames.CONFUSION_MATRICES] ||
        [];
  }

  /**
   * Extracts an array of macro precision recall curve data out of the raw
   * data.
   * @param {?Object} data
   * @return {!Array<!Object>}
   */
  computeMacroPrecisionRecallCurveData_(data) {
    return this.getMatricesForPRCurve_(
        data, tfma.PlotDataFieldNames.MACRO_PRECISION_RECALL_CURVE_DATA);
  }

  /**
   * Extracts an array of micro precision recall curve data out of the raw
   * data.
   * @param {?Object} data
   * @return {!Array<!Object>}
   */
  computeMicroPrecisionRecallCurveData_(data) {
    return this.getMatricesForPRCurve_(
        data, tfma.PlotDataFieldNames.MICRO_PRECISION_RECALL_CURVE_DATA);
  }

  /**
   * Extracts an array of weighted precision recall curve data out of the
   * raw data.
   * @param {?Object} data
   * @return {!Array<!Object>}
   */
  computeWeightedPrecisionRecallCurveData_(data) {
    return this.getMatricesForPRCurve_(
        data, tfma.PlotDataFieldNames.WEIGHTED_PRECISION_RECALL_CURVE_DATA);
  }

  /**
   * Observer for chonsenType. Sets selectedTab_ property upon
   * initialization.
   * @param {?string} initialType
   * @private
   */
  initialTypeChanged_(initialType) {
    if (initialType) {
      this.selectedTab_ = SUPPORTED_VISUALIZATION_[initialType].type;
    }
  }

  /**
   * Observer for selectedTab_.
   * @private
   */
  selectedTabChanged_() {
    // Clears initial chonsen type.
    this.initialType = null;
  }

  /**
   * Determines if we failed to fetch data from the backend.
   * @param {boolean} loading
   * @param {?Object} data null if encountered error.
   * @return {boolean}
   * @private
   */
  computeError_(loading, data) {
    return !loading && data == null;
  }

  /**
   * Fires a reload-plot-data event.
   * @private
   */
  reload_() {
    this.dispatchEvent(new CustomEvent(tfma.Event.RELOAD_PLOT_DATA));
  }

  /**
   * @param {!Array<tfma.PlotTypes>} availableTypes
   * @return {!Array<!Object>} An aray of configuration for each type of plot
   *     specified.
   * @private
   */
  computeAvailableTabs_(availableTypes) {
    const supported = [];
    availableTypes.forEach((type) => {
      if (SUPPORTED_VISUALIZATION_[type]) {
        supported.push(SUPPORTED_VISUALIZATION_[type]);
      }
    });
    return supported;
  }

  /**
   * Observer for property selectedPage_. Makes sure the newly selected page
   * is properly rendered by calling redraw on all google-chart under it.
   * @param {!Element|undefined} page
   * @private
   */
  selectedPageChanged_(page) {
    if (page) {
      const charts = page.querySelectorAll('google-chart');
      for (let i = charts.length - 1, chart; chart = charts[i]; i--) {
        chart.redraw();
      }
    }
  }
}

customElements.define('tfma-plot', Plot);
