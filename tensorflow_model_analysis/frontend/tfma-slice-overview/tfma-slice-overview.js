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
  /** @type {string} */
  const DEFAULT_METRIC_TO_SORT = 'Slice';

  /** @type {number} */
  const MIN_CHART_WIDTH_PX = 680;

  /** @type {number} */
  const CHART_HEIGHT_PX = 200;

  Polymer({
    is: 'tfma-slice-overview',

    properties: {
      /**
       * The tfma.Data instance.
       * @type {tfma.Data}
       */
      slices: {type: Object},

      /**
       * The ColumnChart object created via gviz api.
       * @private {google.visualization.ColumnChart}
       */
      chart_: {type: Object},

      /**
       * An array of string used for rendering the list of metrics to show.
       * @private {Array<string>}
       */
      metrics_: {type: Array, computed: 'computeMetrics_(slices)'},

      /**
       * An array of string used for rendering the list of metrics to sort with.
       * @private {Array<string>}
       */
      metricsForSorting_:
          {type: Array, computed: 'computeMetricsForSorting_(metrics_)'},

      /**
       * The chosen metric to show.
       * @type {string}
       */
      metricToShow: {type: String},

      /**
       * The chosen metrics to sort the data with.
       * @private {string}
       */
      metricToSort_: {type: String, value: DEFAULT_METRIC_TO_SORT},

      /**
       * Google chart packages required for rendering the column chart. This is
       * used by google-chart-loader to download the necessary code. We do not
       * expect this property to change.
       * @private {!Array<string>}
       */
      chartPackages_: {type: Array, value: ['corechart']},

      /**
       * If the component is visible or not.
       *
       * This property is important because we are plotting the chart using all
       * available space. When the component is display:none, the chart will
       * always use the minimum width for rendering.
       * @type {boolean}
       */
      displayed: {type: Boolean},

      /**
       * The data view object. Needed for testing.
       * @private {!google.visualization.DataView|undefined}
       */
      dataView_: {type: Object},
    },

    observers:
        ['plot_(displayed, slices, metricToShow, metricToSort_, chart_)'],

    /**
     * @param {tfma.Data} slices
     * @return {!Array<string>} metrics that can be
     *     visualized by this component.
     * @private
     */
    computeMetrics_: function(slices) {
      return !slices ? [] : slices.getMetrics();
    },

    /**
     * @param {!Array<string>} metrics
     * @return {!Array<string>} all metrics that can
     *     be used for sorting the chart
     * @private
     */
    computeMetricsForSorting_: function(metrics) {
      return [DEFAULT_METRIC_TO_SORT].concat(metrics);
    },

    /**
     * Plots the chart when all data become available.
     * @param {boolean} displayed
     * @param {tfma.Data} slices
     * @param {string} metricToShow
     * @param {string} metricToSort
     * @param {google.visualization.ColumnChart} chart
     * @private
     */
    plot_: function(displayed, slices, metricToShow, metricToSort, chart) {
      if (!displayed || !slices || !chart || !metricToShow || !metricToSort) {
        return;
      }

      const metricToShowIndex = slices.getMetricIndex(metricToShow);
      if (metricToShowIndex == -1) {
        return;
      }

      const metricToSortIndex = slices.getMetricIndex(metricToSort);
      const table = [['feature', metricToShow, metricToSort]];
      const sortByFeatureId = metricToSortIndex == -1;
      const isSortingByMetric =
          !sortByFeatureId && metricToSort != metricToShow;

      slices.getFeatures().forEach(feature => {
        let featureCell = feature;
        const valueToSort = sortByFeatureId ?
            slices.getFeatureId(feature) :
            slices.getMetricValue(feature, metricToSort);
        if (isSortingByMetric) {
          featureCell = {
            'v': feature,
            'f': feature + ', ' + metricToSort + ':' +
                slices.getMetricValue(feature, metricToSort)
                    .toFixed(tfma.FLOATING_POINT_PRECISION)
          };
        }
        table.push([
          featureCell, slices.getMetricValue(feature, metricToShow), valueToSort
        ]);
      });

      this.$.loader.dataTable(table).then(dataTable => {
        dataTable.sort([{'column': 2}]);
        this.$.loader.dataView(dataTable).then(dataView => {
          dataView.setColumns([0, 1]);
          this.dataView_ = dataView;
          this.chart_.draw(dataView, {
            'bar': {'groupWidth': '75%'},
            'hAxis': {'ticks': []},
            'legend': {'position': 'top'},
            'width': Math.max(
                this.getBoundingClientRect().width, MIN_CHART_WIDTH_PX),
            'height': CHART_HEIGHT_PX,
          });
        });
      });
    },

    ready: function() {
      // Start loading the visualization API and instantiate a column chart
      // object as soon as we can.
      this.$.loader.create('column', this.$.chart)
          .then(chart => this.chart_ = chart);
      this.displayed = this.getBoundingClientRect().width > 0;
    },
  });

})();
