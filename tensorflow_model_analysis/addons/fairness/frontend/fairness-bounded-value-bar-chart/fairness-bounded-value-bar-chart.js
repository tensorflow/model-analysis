/**
 * Copyright 2019 Google LLC
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
import {template} from './fairness-bounded-value-bar-chart-template.html.js';

const Util = goog.require('tensorflow_model_analysis.addons.fairness.frontend.Util');

const HEIGHT = 360;
const WIDTH = 600;
const MARGIN = {
  top: 10,
  right: 10,
  bottom: 20,
  left: 40
};
const GRAPH_BOUND = {
  top: MARGIN.top,
  right: WIDTH - MARGIN.right,
  bottom: HEIGHT - MARGIN.bottom,
  left: MARGIN.left,
};
const METRICS_PADDING = 0.1;
const SLICES_PADDING = 0.2;

const CHART_BAR_COLOR_ = [
  '#F0BD80', '#61AFF7', '#FFE839', '#9B86EF', '#FF777B', '#7DDAD3', '#EF96CD'
];
const BASELINE_BAR_COLOR_ = [
  '#FF9230', '#3C7DBF', '#FFC700', '#7647EA', '#FC4F61', '#1F978A', '#B22A72'
];
const DARKEN_COMPARE = 80;

const NUM_DECIMAL_PLACES = 5;

const TOOLTIP_Y_OFFSET = 20;

/**
 * Tooltip text for a bar.
 * @param {!Object} d Slice data for a bar.
 * @return {string}
 * @private
 */
function tooltipText_(d) {
  const sliceRow = '<td>Slice</td><td>$(fullSliceName)</td>'.replace(
      '$(fullSliceName)', d.fullSliceName);
  const evalRow = d.evalName ?
      '<td>Eval</td><td>$(evalName)</td>'.replace('$(evalName)', d.evalName) :
      '';
  const metricRow = '<td>Metric</td><td>$(metric)</td>'.replace(
      '$(metric)', Util.removeMetricNamePrefix(d.metricName));
  const valueRow = '<td>Value</td><td>$(value)</td>'.replace(
      '$(value)', d.value.toFixed(NUM_DECIMAL_PLACES));

  const confInt = (upperBound, lowerBound) => ' (' +
      d.upperBound.toFixed(NUM_DECIMAL_PLACES) + ', ' +
      d.lowerBound.toFixed(NUM_DECIMAL_PLACES) + ')';
  const confIntRow = (d.upperBound && d.lowerBound) ?
      '<td>Confidence Interval</td><td>$(confInt)</td>'.replace(
          '$(confInt)', confInt(d.upperBound, d.lowerBound)) :
      '';

  const exampleCountRow = d.exampleCount ?
      '<td>Example Count</td><td>$(exampleCount)</td>'.replace(
          '$(exampleCount)', d.exampleCount) :
      '';

  const rows =
      [sliceRow, evalRow, metricRow, valueRow, confIntRow, exampleCountRow];

  const tablestart = '<table><tbody>';
  const tablerows = rows.map((row) => '<tr>' + row + '</tr>').join('');
  const tablestop = '</table></tbody>';

  return tablestart + tablerows + tablestop;
}

/**
 * Darken a color.
 * @param {string} color
 * @return {string}
 */
function darken_(color) {
  const bounded = (num) => Math.min(Math.max(num, 0), 255);
  const num = parseInt(color.slice(1), 16);
  const r = bounded((num >> 16) - DARKEN_COMPARE);
  const b = bounded(((num >> 8) & 0x00FF) - DARKEN_COMPARE);
  const g = bounded((num & 0x0000FF) - DARKEN_COMPARE);
  return '#' + (g | (b << 8) | (r << 16)).toString(16);
}


export class FairnessBoundedValueBarChart extends PolymerElement {
  constructor() {
    super();
  }

  static get is() {
    return 'fairness-bounded-value-bar-chart';
  }

  /** @return {!HTMLTemplateElement} */
  static get template() {
    return template;
  }

  /** @return {!PolymerElementProperties} */
  static get properties() {
    return {
      /**
       * A dictionary where the key is a slice name and the value is another
       * dictionary of metric names and values.
       * @type {!Array<!Object>}
       */
      data: {type: Array},

      /**
       * A dictionary to compare.
       * @type {!Array<!Object>}
       */
      dataCompare: {type: Array},

      /**
       * The name of the first eval. Optional - only needed for eval comparison.
       * @type {string}
       */
      evalName: {type: String},

      /**
       * The name of the second eval. Optional - only needed for eval
       * comparison.
       * @type {string}
       */
      evalNameCompare: {type: String},

      /** @type {!Array<string>} */
      metrics: {type: Array},

      /** @type {string} */
      baseline: {type: String},

      /** @type {!Array<string>} */
      slices: {type: Array, notify: true},

      /** @private {boolean} */
      isAttached_: {type: Boolean, value: false},

      /**
       * The selected parameter to sort on - can be 'Slice' or 'Eval'.
       * @private {string}
       */
      sort: {type: String},
    };
  }

  static get observers() {
    return [
      'renderClusteredBarChart_(data, dataCompare, metrics, slices, baseline, ' +
          'evalName, evalNameCompare, sort, isAttached_)',
    ];
  }

  /** @override */
  connectedCallback() {
    super.connectedCallback();
    this.isAttached_ = true;
  }

  /**
   * Render clustered bar chart.
   * @param {!Array<!Object>} data for plot.
   * @param {!Array<!Object>} dataCompare for plot.
   * @param {!Array<string>} metrics list.
   * @param {!Array<string>} slices list.
   * @param {string} baseline slice of the metrics.
   * @param {string} evalName to distinguish evals.
   * @param {string} evalNameCompare to distinguish evals.
   * @param {string} sort how to sort the clusters
   * @param {boolean} isAttached Whether this element is attached to
   *     the DOM.
   * @private
   */
  renderClusteredBarChart_(
      data, dataCompare, metrics, slices, baseline, evalName, evalNameCompare,
      sort, isAttached) {
    if (!data || !metrics || !slices || !baseline || !isAttached ||
        (this.evalComparison_() && (!evalName || !evalNameCompare))) {
      return;
    }

    slices = [baseline, ...slices];
    var absentSlices =
        slices.filter(slice => !(data.find(d => d['slice'] == slice)));
    if (absentSlices.length) {
      return;
    }

    // Make sure that data and dataCompare have the same slices.
    if (this.evalComparison_()) {
      data.forEach((slicingMetric) => {
        if (!dataCompare.find(d => d['slice'] == slicingMetric['slice'])) {
          let value = {
            'slice': slicingMetric['slice'],
            'sliceValue': slicingMetric['sliceValue'],
            'metrics': Object.keys(slicingMetric['metrics'])
                           .reduce(
                               (acc, metricName) => {
                                 acc[metricName] = NaN;
                                 return acc;
                               },
                               {})
          };
          dataCompare.push(value);
        }
      });
    }

    const d3Data = this.createD3Data_(
        data, dataCompare, metrics, slices, baseline, evalName, evalNameCompare,
        sort);
    const graphConfig = this.buildGraphConfig_(d3Data);
    this.drawGraph_(d3Data, baseline, graphConfig);
  }

  /**
   * Convert the data into the d3 friendly data structure.
   * @param {!Object} data for plot.
   * @param {!Object} dataCompare for plot.
   * @param {!Array<string>} metrics list.
   * @param {!Array<string>} slices list.
   * @param {string} baseline slice of the metrics.
   * @param {string} evalName to distinguish evals.
   * @param {string} evalNameCompare to distinguish evals.
   * @param {string} sort how to sort the clusters
   * @return {!Array<!Object>}
   * @private
   */
  createD3Data_(
      data, dataCompare, metrics, slices, baseline, evalName, evalNameCompare,
      sort) {
    // d3DataObjects = metrics data for each slice
    const d3DataObject = (metricsData) => {
      return {
        fullSliceName: metricsData.length ? metricsData[0].fullSliceName : '',
        sliceValue: metricsData.length ? metricsData[0].sliceValue : '',
        evalName: metricsData.length ? metricsData[0].evalName : '',
        metricsData: metricsData,
      };
    };
    // d3Data = array of d3DataObjects, returned by createD3Data_()
    const d3Data = [];

    // metricDataObjects = the values in a slice-eval-metric-threshold tuple
    const metricDataObject = (sliceMetrics, evalName, metricName) => {
      const bounds = this.getMetricBounds_(sliceMetrics['metrics'][metricName]);
      return {
        fullSliceName: sliceMetrics['slice'],
        sliceValue: sliceMetrics['sliceValue'],
        evalName: evalName,
        metricName: metricName,
        value: tfma.CellRenderer.maybeExtractBoundedValue(
            sliceMetrics['metrics'][metricName]),
        upperBound: bounds.min,
        lowerBound: bounds.max,
        exampleCount: tfma.CellRenderer.maybeExtractBoundedValue(
            Util.getMetricsValues(sliceMetrics, 'example_count'))
      };
    };

    slices.forEach((slice) => {
      // sliceMetrics = all the metrics for one slice
      const sliceMetrics = data.find(d => d['slice'] == slice);
      const sliceMetricsCompare = this.evalComparison_() ?
          dataCompare.find(d => d['slice'] == slice) :
          undefined;

      var undefinedMetrics = metrics.filter(
          metric => sliceMetrics['metrics'][metric] === undefined);
      if (undefinedMetrics.length) {
        return;
      }

      // metricsData = array of metrics data for all thresholds
      const metricsData = [];
      const metricsDataCompare = [];
      metrics.forEach((metricName) => {
        metricsData.push(metricDataObject(sliceMetrics, evalName, metricName));
        if (this.evalComparison_()) {
          metricsDataCompare.push(metricDataObject(
              sliceMetricsCompare, evalNameCompare, metricName));
        }
      });

      d3Data.push(d3DataObject(metricsData));
      if (this.evalComparison_()) {
        d3Data.push(d3DataObject(metricsDataCompare));
      }
    });

    d3Data.sort(this.sortD3_(baseline, sort));
    return d3Data;
  }

  /**
   * Configure graph.
   * @param {!Array<!Object>} d3Data for plot.
   * @return {!Object}
   * @private
   */
  buildGraphConfig_(d3Data) {
    // Build slice scale - these are groups of bars.
    const slicesX = d3.scaleBand()
                        .domain(d3Data.map(d => this.labelName_(d)))
                        .rangeRound([GRAPH_BOUND.left, GRAPH_BOUND.right])
                        .padding(SLICES_PADDING);

    // Build metrics scale - these are individual bars.
    const metricNames = d3Data.reduce((set, slice) => {
      slice.metricsData.forEach(metricData => set.add(metricData.metricName));
      return set;
    }, new Set());
    const metricsX = d3.scaleBand()
                         .domain(Array.from(metricNames))
                         .rangeRound([0, slicesX.bandwidth()])
                         .paddingInner(METRICS_PADDING);
    let metricsValueMin = 0;
    let metricsValueMax = 0;
    d3Data.forEach(d => {
      d.metricsData.forEach(m => {
        // The lowerBound upperBound will be zero when
        // confidence interval is not enabled.
        metricsValueMin = d3.min([metricsValueMin, m.value, m.lowerBound]);
        metricsValueMax = d3.max([metricsValueMax, m.value, m.upperBound]);
      });
    });
    const y = d3.scaleLinear()
                  .domain([metricsValueMin, metricsValueMax])
                  .nice()
                  .rangeRound([GRAPH_BOUND.bottom, GRAPH_BOUND.top]);

    const configureXAxis = g =>
        g.attr('transform', `translate(0,${HEIGHT - MARGIN.bottom})`)
            .call(d3.axisBottom(slicesX).tickSizeOuter(0))
            .selectAll('Text')  // Corp the text
            .each((d, i, n) => {
              while (n[i].getComputedTextLength() > slicesX.bandwidth()) {
                n[i].textContent = n[i].textContent.slice(0, -4) + '...';
              }
            });
    const configureYAxis = g =>
        g.attr('transform', `translate(${MARGIN.left},0)`).call(d3.axisLeft(y));
    const metricsColor = d3.scaleOrdinal().range(CHART_BAR_COLOR_);
    const baselineColor = d3.scaleOrdinal().range(BASELINE_BAR_COLOR_);
    const confidenceIntervalColor =
        d3.scaleOrdinal().range(CHART_BAR_COLOR_.map(darken_));

    return {
      'slicesX': slicesX,
      'metricsX': metricsX,
      'y': y,
      'baselineColor': baselineColor,
      'metricsColor': metricsColor,
      'confidenceIntervalColor': confidenceIntervalColor,
      'configureXAxis': configureXAxis,
      'configureYAxis': configureYAxis
    };
  }

  /**
   * Render clustered bar chart.
   * @param {!Array<!Object>} d3Data
   * @param {string} baseline
   * @param {!Object} graphConfig
   * @private
   */
  drawGraph_(d3Data, baseline, graphConfig) {

    const svg = d3.select(this.$['bar-chart']);
    svg.html('');

    // Create a group of bars for every cluster
    const bars =
        svg.append('g')
            .attr('id', 'bars')
            .selectAll('g')
            .data(d3Data)
            .enter()
            .append('g')
            .attr(
                'transform',
                d => `translate(${
                    graphConfig['slicesX'](this.labelName_(d))},0)`);

    const tooltip = d3.select(this.$['tooltip']);

    const mouseover = function(d) {
      tooltip.html(tooltipText_(d))
          .style('display', 'inline')
          .style('left', d3.event.clientX + 'px')
          .style('top', (d3.event.clientY + TOOLTIP_Y_OFFSET) + 'px')
          .style('position', 'fixed');
    };

    const mouseout = function(d) {
      tooltip.html('').style('display', 'none').style('position', 'static');
    };

    // For every cluster, add a bar for every eval-threshold pair
    bars.selectAll('rect')
        .data(d => d.metricsData)
        .enter()
        .append('rect')
        .attr('x', d => graphConfig['metricsX'](d.metricName))
        .attr(
            'y',
            d => isNaN(d.value) ? graphConfig['y'](0) :
                                  graphConfig['y'](d.value))
        .attr('width', graphConfig['metricsX'].bandwidth())
        .attr(
            'height',
            d => isNaN(d.value) ?
                0 :
                graphConfig['y'](0) - graphConfig['y'](d.value))
        .attr(
            'fill',
            d => d.fullSliceName === baseline ?
                graphConfig['baselineColor'](d.metricName) :
                graphConfig['metricsColor'](d.metricName))
        .on('mouseover', mouseover)
        .on('mouseout', mouseout)
        .on('click', (d, i) => {
          this.dispatchEvent(new CustomEvent(
              tfma.Event.SELECT,
              {detail: d.fullSliceName, composed: true, bubbles: true}));
        });

    // Add confidence interval lines if bounded value
    bars.selectAll('line')
        .data(d => d.metricsData)
        .enter()
        .append('line')
        .attr(
            'x1',
            d => graphConfig['metricsX'](d.metricName) +
                (graphConfig['metricsX'].bandwidth() / 2))
        .attr(
            'y1',
            d => isNaN(d.upperBound) ? graphConfig['y'](0) :
                                       graphConfig['y'](d.upperBound))
        .attr(
            'x2',
            d => graphConfig['metricsX'](d.metricName) +
                (graphConfig['metricsX'].bandwidth() / 2))
        .attr(
            'y2',
            d => isNaN(d.lowerBound) ? graphConfig['y'](0) :
                                       graphConfig['y'](d.lowerBound))
        .attr(
            'stroke', d => graphConfig['confidenceIntervalColor'](d.metricName))
        .attr('stroke-width', 1);

    // Draw X Y axis.
    svg.append('g').attr('id', 'xaxis').call(graphConfig['configureXAxis']);
    svg.append('g').attr('id', 'yaxis').call(graphConfig['configureYAxis']);
  }

  /**
   * Extracts the bounds of metric.
   * @param {!Object} metric
   * @return {!Object}
   * @private
   */
  getMetricBounds_(metric) {
    let upperBound = 0;
    let lowerBound = 0;
    if (tfma.CellRenderer.isBoundedValue(metric)) {
      upperBound = metric['upperBound'];
      lowerBound = metric['lowerBound'];
    }

    return {min: lowerBound, max: upperBound};
  }

  /**
   * @return {boolean} Returns true if evals are being compared.
   * @private
   */
  evalComparison_() {
    return this.dataCompare && this.dataCompare.length > 0;
  }

  /**
   * Returns sorting function for d3 data.
   * @param {string} baseline
   * @param {string} sort
   * @return {function(!Object, !Object): number}
   * @private
   */
  sortD3_(baseline, sort) {
    function sliceCompare(a, b) {
      // Ensure that the baseline slice always appears first.
      if (a.fullSliceName === baseline && b.fullSliceName === baseline) {
        return 0;
      } else if (a.fullSliceName === baseline) {
        return -1;
      } else if (b.fullSliceName === baseline) {
        return 1;
      } else {
        return a.fullSliceName.localeCompare(b.fullSliceName);
      }
    }

    if (this.evalComparison_()) {
      if (sort === 'Slice') {
        return (a, b) => {
          return sliceCompare(a, b) || a.evalName.localeCompare(b.evalName);
        };
      } else {  // Sort by eval name
        return (a, b) => {
          return a.evalName.localeCompare(b.evalName) || sliceCompare(a, b);
        };
      }
    } else {
      return (a, b) => {
        // Ensure that the baseline slice always appears on the left side.
        if (a.fullSliceName == baseline) {
          return -1;
        } else if (b.fullSliceName == baseline) {
          return 1;
        }
        // Sort by the first threshold value if multiple thresholds are present.
        for (let i = 0; i < a.metricsData.length; i++) {
          const diff = a.metricsData[i]['value'] - b.metricsData[i]['value'];
          if (diff != 0) {
            return diff;
          }
        }
        // If metrics are equal for both slices, go by alphabetical order.
        if (a['fullSliceName'] <= b['fullSliceName']) {
          return -1;
        } else {
          return 1;
        }
      };
    }
  }

  /**
   * Label name for a bar cluster.
   * @param {!Object} d a d3DataObject, used to create a bar cluster.
   * @return {string}
   * @private
   */
  labelName_(d) {
    if (!this.evalComparison_()) {
      return d.sliceValue;
    }
    else if (this.sort === 'Slice') {
      return d.evalName + '-' + d.sliceValue;
    }
    else {  // this.sort === 'Eval'
      return d.sliceValue + '-' + d.evalName;
    }
  }
}

customElements.define(
    'fairness-bounded-value-bar-chart', FairnessBoundedValueBarChart);
