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
  is: 'tfma-plot-trigger',
  properties: {
    /** @type {string} */
    data: {type: String, observer: 'dataChanged_'},

    /**
     * The slice for which we should show plots for.
     * @type {string|undefined}
     * @private
     */
    slice_: {type: String},

    /**
     * The data span for which we should show plots for.
     * @type {number|undefined}
     * @private
     */
    dataSpan_: {type: Number},

    /**
     * The model run id for which we should show plots for.
     * @type {number|undefined}
     * @private
     */
    runId_: {type: Number},

    /**
     * The types of available plots.
     * @type {!Array<tfma.PlotTypes>}
     * @private
     */
    availableTypes_: {type: Array},

    /**
     * The text to show on the link that triggers the plot to be displayed.
     * @type {string}
     * @private
     */
    text_: {
      type: String,
      computed: 'computeText_(availableTypes_, supportedPlots_)'
    },

    /**
     * The tooltip to show on the link.
     * @type {string}
     * @private
     */
    title_: {
      type: String,
      computed: 'computeTitle_(availableTypes_, supportedPlots_)'
    },

    supportedPlots_: {
      type: Object,
      value: () => {
        const supportedPlots = {};
        supportedPlots[tfma.PlotTypes.CALIBRATION_PLOT] = {
          text: 'Calibration',
          title: 'Calibration Plot',
        };
        supportedPlots[tfma.PlotTypes.MACRO_PRECISION_RECALL_CURVE] = {
          text: 'Macro PRC',
          title: 'Macro Precision-Recall Curve',
        };
        supportedPlots[tfma.PlotTypes.MICRO_PRECISION_RECALL_CURVE] = {
          text: 'Micro PRC',
          title: 'Micro Precision-Recall Curve',
        };
        supportedPlots[tfma.PlotTypes.PREDICTION_DISTRIBUTION] = {
          text: 'Pred. dist.',
          title: 'Prediction Distribution',
        };
        supportedPlots[tfma.PlotTypes.PRECISION_RECALL_CURVE] = {
          text: 'P-R curve',
          title: 'Precision-Recall Curve',
        };
        supportedPlots[tfma.PlotTypes.ROC_CURVE] = {
          text: 'ROC curve',
          title: 'Receiver Operating Characteristic Curve',
        };
        supportedPlots[tfma.PlotTypes.WEIGHTED_PRECISION_RECALL_CURVE] = {
          text: 'Weighted PRC',
          title: 'Weighted Precision-Recall Curve',
        };
        return supportedPlots;
      }
    },
  },

  /**
   * Observer for the property data. Parses the serialized data and initializes
   * the component.
   * @param {string} dataString
   * @private
   */
  dataChanged_: function(dataString) {
    try {
      const data = JSON.parse(dataString);

      this.availableTypes_ = data['types'] || [];

      if (data['slice']) {
        // The field slice will be available if the plot-trigger is under a
        // slicing metrics view.
        this.slice_ = data['slice'];
      } else {
        // The field span will be available if the plot-trigger is under a
        // time series view.
        this.dataSpan_ = data['span'];
        this.runId_ = data['runId'];
      }
    } catch (e) {
    }
  },

  /**
   * Determines the text to display in the link based on available types.
   * @param {!Array<tfma.PlotTypes>} availableTypes
   * @param {!Object} supportedPlots
   * @return {string}
   */
  computeText_: function(availableTypes, supportedPlots) {
    if (availableTypes.length == 1) {
      return supportedPlots[availableTypes[0]].text;
    } else {
      return 'Plots';
    }
  },

  /**
   * Determines the tooltip to use for the link based on available types.
   * @param {!Array<tfma.PlotTypes>} availableTypes
   * @param {!Object} supportedPlots
   * @return {string}
   */
  computeTitle_: function(availableTypes, supportedPlots) {
    const titles = [];
    availableTypes.forEach(type => {
      titles.push(supportedPlots[type].title);
    });
    return (availableTypes.length > 1 ? 'Available: ' : '') + titles.join(', ');
  },

  /**
   * Handler for tap event on the trigger link. Fires off a "show-plot" event
   * that will bubble up and be handled by other components up in the
   * hierarchy.
   * @param {!Event} e
   * @private
   */
  showPlot_: function(e) {
    e.preventDefault();
    const detail = {availableTypes: this.availableTypes_};
    if (this.slice_) {
      detail.slice = this.slice_;
    } else {
      detail.dataSpan = this.dataSpan_;
      detail.runId = this.runId_;
    }
    this.fire('show-plot', detail);
  },
});
