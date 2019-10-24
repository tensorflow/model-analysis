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
import '@polymer/paper-checkbox/paper-checkbox.js';
import '@polymer/paper-spinner/paper-spinner.js';
import '@polymer/paper-tabs/paper-tabs.js';
import '../tfma-accuracy-charts/tfma-accuracy-charts.js';
import '../tfma-calibration-plot/tfma-calibration-plot.js';
import '../tfma-gain-chart/tfma-gain-chart.js';
import '../tfma-multi-class-confusion-matrix-at-thresholds/tfma-multi-class-confusion-matrix-at-thresholds.js';
import '../tfma-precision-recall-curve/tfma-precision-recall-curve.js';
import '../tfma-prediction-distribution/tfma-prediction-distribution.js';
import '../tfma-residual-plot/tfma-residual-plot.js';
import '../tfma-roc-curve/tfma-roc-curve.js';

const TABS = {
  ACCURACY_CHARTS: 'acc',
  CALIBRATION_PLOT: 'cp',
  GAIN_CHART: 'gain',
  MACRO_PRECISION_RECALL: 'mapr',
  MICRO_PRECISION_RECALL: 'mipr',
  MULTI_CLASS_CONFUSION_MATRIX: 'mccm',
  MULTI_LABEL_CONFUSION_MATRIX: 'mlcm',
  PRECISION_RECALL: 'pr',
  PREDICTION_DISTRIBUTION: 'pd',
  RESIDUAL_PLOT: 'res',
  ROC: 'roc',
  WEIGHTED_PRECISION_RECALL: 'wpr',
};

const TITLES = {
  ACCURACY_CHARTS: 'Acc/P/R/F1',
  CALIBRATION_PLOT: 'Calibration Plot',
  GAIN_CHART: 'Gain',
  MACRO_PRECISION_RECALL: 'Macro PR Curve',
  MICRO_PRECISION_RECALL: 'Micro PR Curve',
  MULTI_CLASS_CONFUSION_MATRIX: 'Multi-class Confusion Matrix',
  MULTI_LABEL_CONFUSION_MATRIX: 'Multi-label Confusion Matrix',
  PRECISION_RECALL: 'Precision-Recall Curve',
  PREDICTION_DISTRIBUTION: 'Prediction Distribution',
  RESIDUAL_PLOT: 'Residual Plot',
  ROC: 'ROC Curve',
  WEIGHTED_PRECISION_RECALL: 'Weighted PR Curve',
};

const SUPPORTED_VISUALIZATION = {
  [tfma.PlotTypes.CALIBRATION_PLOT]:
      {type: TABS.CALIBRATION_PLOT, text: TITLES.CALIBRATION_PLOT},
  [tfma.PlotTypes.PRECISION_RECALL_CURVE]: {
    type: TABS.PRECISION_RECALL,
    text: TITLES.PRECISION_RECALL,
  },
  [tfma.PlotTypes.MACRO_PRECISION_RECALL_CURVE]: {
    type: TABS.MACRO_PRECISION_RECALL,
    text: TITLES.MACRO_PRECISION_RECALL,
  },
  [tfma.PlotTypes.MICRO_PRECISION_RECALL_CURVE]: {
    type: TABS.MICRO_PRECISION_RECALL,
    text: TITLES.MICRO_PRECISION_RECALL,
  },
  [tfma.PlotTypes.WEIGHTED_PRECISION_RECALL_CURVE]: {
    type: TABS.WEIGHTED_PRECISION_RECALL,
    text: TITLES.WEIGHTED_PRECISION_RECALL,
  },
  [tfma.PlotTypes.PREDICTION_DISTRIBUTION]: {
    type: TABS.PREDICTION_DISTRIBUTION,
    text: TITLES.PREDICTION_DISTRIBUTION,
  },
  [tfma.PlotTypes.RESIDUAL_PLOT]: {
    type: TABS.RESIDUAL_PLOT,
    text: TITLES.RESIDUAL_PLOT,
  },
  [tfma.PlotTypes.ROC_CURVE]: {
    type: TABS.ROC,
    text: TITLES.ROC,
  },
  [tfma.PlotTypes.ACCURACY_CHARTS]: {
    type: TABS.ACCURACY_CHARTS,
    text: TITLES.ACCURACY_CHARTS,
  },
  [tfma.PlotTypes.GAIN_CHART]: {
    type: TABS.GAIN_CHART,
    text: TITLES.GAIN_CHART,
  },
  [tfma.PlotTypes.MULTI_CLASS_CONFUSION_MATRIX]: {
    type: TABS.MULTI_CLASS_CONFUSION_MATRIX,
    text: TITLES.MULTI_CLASS_CONFUSION_MATRIX,
  },
  [tfma.PlotTypes.MULTI_LABEL_CONFUSION_MATRIX]: {
    type: TABS.MULTI_LABEL_CONFUSION_MATRIX,
    text: TITLES.MULTI_LABEL_CONFUSION_MATRIX,
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
          'Accuracy': TABS.ACCURACY_CHARTS,
          'Calibration': TABS.CALIBRATION_PLOT,
          'Gain': TABS.GAIN_CHART,
          'Prediction': TABS.PREDICTION_DISTRIBUTION,
          'Macro': TABS.MACRO_PRECISION_RECALL,
          'Micro': TABS.MICRO_PRECISION_RECALL,
          'MULTI_CLASS_CONFUSION_MATRIX': TABS.MULTI_CLASS_CONFUSION_MATRIX,
          'MULTI_LABEL_CONFUSION_MATRIX': TABS.MULTI_LABEL_CONFUSION_MATRIX,
          'Precision': TABS.PRECISION_RECALL,
          'Residual': TABS.RESIDUAL_PLOT,
          'ROC': TABS.ROC,
          'Weighted': TABS.WEIGHTED_PRECISION_RECALL,
        }
      },

      /**
       * A map of all chart titles.
       * @type {!Object<string>}
       * @private
       */
      chartTitles_: {
        type: Object,
        value: {
          'Accuracy': TITLES.ACCURACY_CHARTS,
          'Calibration': TITLES.CALIBRATION_PLOT,
          'Gain': TITLES.GAIN_CHART,
          'Prediction': TITLES.PREDICTION_DISTRIBUTION,
          'Macro': TITLES.MACRO_PRECISION_RECALL,
          'Micro': TITLES.MICRO_PRECISION_RECALL,
          'MultiClassConfusionMatrix': TITLES.MULTI_CLASS_CONFUSION_MATRIX,
          'MultiLabelConfusionMatrix': TITLES.MULTI_LABEL_CONFUSION_MATRIX,
          'Precision': TITLES.PRECISION_RECALL,
          'Residual': TITLES.RESIDUAL_PLOT,
          'ROC': TITLES.ROC,
          'Weighted': TITLES.WEIGHTED_PRECISION_RECALL,
        }
      },

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
       * The data for multi-class confusion matrix.
       * @type {!Array<!Object>}
       * @private
       */
      multiClassConfusionMatrixData_: {
        type: Object,
        computed: 'computeMultiClassConfusionMatrixData_(data)',
      },

      /**
       * The data for multi-label confusion matrix.
       * @type {!Array<!Object>}
       * @private
       */
      multiLabelConfusionMatrixData_: {
        type: Object,
        computed: 'computeMultiLabelConfusionMatrixData_(data)',
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

      /**
       * Whether the component should be rendered in flat layout. When in flat
       * layout, the user cannot go back to tabbed view.
       * @type {boolean}
       */
      flat: {type: Boolean, value: false, observer: 'flatChanged_'},

      /**
       * Whether to show all charts at once.
       * @private {boolean}
       */
      showAll_: {type: Boolean, value: false, notify: true},


      /**
       * A map of charts where the key is the name attribute that is also used
       * for tab selection.
       * @private {!Object<!Element>}
       */
      chartsMap_: {type: Object},

      /**
       * A map where the key is tfma.PlotTypes and the value is the
       * desired subtitle for that plot.
       * @type {!Object<string>}
       */
      subtitles: {
        type: Object,
      },
    };
  }

  /** @override */
  connectedCallback() {
    super.connectedCallback();
    if (!this.chartsMap_) {
      // Build charts map the first time the component is added to the dom tree.
      this.chartsMap_ = this.buildChartsMap_();
    }
  }

  static get observers() {
    return ['layoutCharts_(chartsMap_, showAll_, availableTabs_)'];
  }

  /**
   * Extracts an array of calibration data out of the raw data.
   * @param {?Object} data
   * @return {!Array<!Object>}
   * @private
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
   * @private
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
   * @private
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
   * @private
   */
  computeMicroPrecisionRecallCurveData_(data) {
    return this.getMatricesForPRCurve_(
        data, tfma.PlotDataFieldNames.MICRO_PRECISION_RECALL_CURVE_DATA);
  }

  /**
   * Extracts confusion matrix data out of raw data for multi-class
   * single-label model.
   * @param {?Object} data
   * @return {!Array<!Object>|undefined}
   * @private
   */
  computeMultiClassConfusionMatrixData_(data) {
    const plotData = data && data['plotData'] || {};
    return plotData[tfma.PlotDataFieldNames.MULTI_CLASS_CONFUSION_MATRIX_DATA];
  }

  /**
   * Extracts confusion matrix data out of raw data for multi-class
   * multi-label model.
   * @param {?Object} data
   * @return {!Array<!Object>|undefined}
   * @private
   */
  computeMultiLabelConfusionMatrixData_(data) {
    const plotData = data && data['plotData'] || {};
    return plotData[tfma.PlotDataFieldNames.MULTI_LABEL_CONFUSION_MATRIX_DATA];
  }

  /**
   * Extracts an array of weighted precision recall curve data out of the
   * raw data.
   * @param {?Object} data
   * @return {!Array<!Object>}
   * @private
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
      this.selectedTab_ = SUPPORTED_VISUALIZATION[initialType].type;
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
   * @param {!Array<!tfma.PlotTypes>} availableTypes
   * @return {!Array<!Object>} An aray of configuration for each type of plot
   *     specified.
   * @private
   */
  computeAvailableTabs_(availableTypes) {
    const supported = [];
    availableTypes.forEach((type) => {
      if (SUPPORTED_VISUALIZATION[type]) {
        supported.push(SUPPORTED_VISUALIZATION[type]);
      }
    });
    return supported;
  }

  /**
   * Observer for the property flat.
   * @param {boolean} flat
   * @private
   */
  flatChanged_(flat) {
    if (flat) {
      // Show all plots if the component is in the flat view.
      this.showAll_ = true;
    }
  }

  /**
   * Builds a map where charts are keyed off of their name property.
   * @return {!Object<!Element>}
   * @private
   */
  buildChartsMap_() {
    return Array.from(this.$['plots'].querySelectorAll('.plot-holder'))
        .reduce((acc, chart) => {
          acc[chart.name] = chart;
          return acc;
        }, {});
  }

  /**
   * Lays out the charts based on the input. If showing all components at once,
   * filter and sort the charts to show based on the tabs. Otherwise, show the
   * charts in tabbed view.
   * @param {!Object<!Element>} charts
   * @param {boolean} showAll
   * @param {!Array<!Object>} tabs
   * @private
   */
  layoutCharts_(charts, showAll, tabs) {
    if (!charts || !tabs) {
      return;
    }

    if (showAll) {
      const flatViewContainer = this.$['flat-view-container'];
      while (flatViewContainer.lastChild) {
        flatViewContainer.removeChild(flatViewContainer.lastChild);
      }
      tabs.forEach(tab => {
        flatViewContainer.appendChild(charts[tab.type]);
      });
    } else {
      const tabbedViewContainer = this.$['plots'];
      for (let chartName in charts) {
        tabbedViewContainer.appendChild(charts[chartName]);
      }
    }

    this.dispatchEvent(new CustomEvent('iron-resize'));
  }

  /**
   * Gets the subtitle for the named chart from the given map. If not set,
   * return empty string.
   * @param {!Object<string>|undefined} subtitles
   * @param {string} chart
   * @return {string}
   * @private
   */
  getSubTitle_(subtitles, chart) {
    if (subtitles) {
      for (let visualization in SUPPORTED_VISUALIZATION) {
        if (SUPPORTED_VISUALIZATION[visualization].type == chart) {
          return subtitles[visualization] || '';
        }
      }
    }
    return '';
  }
}

customElements.define('tfma-plot', Plot);
