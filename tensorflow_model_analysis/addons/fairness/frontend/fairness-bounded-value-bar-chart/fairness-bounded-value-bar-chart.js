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
  '#f0bd80', '#61aff7', '#ffe839', '#9b86ef', '#ff777b', '#7ddad3', '#ef96cd',
  '#b5bcc3'
];
const BASELINE_BAR_COLOR_ = [
  '#4ccfae', '#f761af', '#39ffe8', '#ef9b86', '#7bff77', '#d37dda', '#cdef96',
  '#c3b5bc'
];
const ERROR_BAR_COLOR_ = [
  '#ba4a0d', '#184889', '#b48014', '#270086', '#b12d33', '#006067', '#632440',
  '#515050'
];
const NUM_DECIMAL_PLACES = 5;


/**
 * Build bar tips.
 * @return {!d3.tip.x}
 */
function buildTooltips() {
  return d3.tip()
      .style('font-size', '10px')
      .style('padding', '2px')
      .style('background-color', '#616161')
      .style('color', '#fff')
      .style('border-radius', '2px')
      .html(d => {
        let html = '<table><tbody>';
        html += '<tr><td>Slice</td><td>' + d.fullSliceName + '</td></tr>';
        const metricName = Util.removePostExportMetrics(d.metricName);
        html += '<tr><td>Metric</td><td>' + metricName + '</td></tr>';
        html += '<tr><td>Value</td><td>' + d.value.toFixed(NUM_DECIMAL_PLACES) +
            '</td></tr>';
        if (d.upperBound && d.lowerBound) {
          const conf_int = ' (' + d.upperBound.toFixed(NUM_DECIMAL_PLACES) +
              ', ' + d.lowerBound.toFixed(NUM_DECIMAL_PLACES) + ')';
          html +=
              '<tr><td>Confidence Interval</td><td>' + conf_int + '</td></tr>';
        }
        if (d.exampleCount) {
          html +=
              '<tr><td>Example Count</td><td>' + d.exampleCount + '</td></tr>';
        }
        html += '</tbody></table>';
        return html;
      });
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

      /** @type {!Array<string>} */
      metrics: {type: Array},

      /** @type {string} */
      baseline: {type: String},

      /** @type {!Array<string>} */
      slices: {type: Array, notify: true},

      /** @private {boolean} */
      isAttached_: {type: Boolean, value: false},

      /**
       * @private {!d3.tip.x}
       */
      tip_: {value: buildTooltips},
    };
  }

  static get observers() {
    return [
      'initializePlotGraph_(data, metrics, slices, baseline, isAttached_, tip_)',
    ];
  }

  /** @override */
  connectedCallback() {
    super.connectedCallback();
    this.isAttached_ = true;
  }

  /**
   * Render clustered bar chart.
   * @param {!Object} data for plot.
   * @param {!Array<string>} metrics list.
   * @param {!Array<string>} slices list.
   * @param {string} baseline slice of the metrics.
   * @param {boolean} isAttached Whether this element is attached to
   *     the DOM.
   * @param {!d3.tip.x} tip for barc chart.
   * @private
   */
  initializePlotGraph_(data, metrics, slices, baseline, isAttached, tip) {
    if (!data || !metrics || !slices || !baseline || !isAttached || !tip) {
      return;
    }

    slices = [baseline, ...slices];
    var absentSlices =
        slices.filter(slice => !(data.find(d => d['slice'] == slice)));
    if (absentSlices.length) {
      return;
    }

    // Covert the data into the d3 friendly data structure.
    const d3Data = [];
    slices.forEach((slice) => {
      const sliceMetrics = data.find(d => d['slice'] == slice);
      const metricsData = [];

      var undefinedMetrics = metrics.filter(
          metric => sliceMetrics['metrics'][metric] === undefined);
      if (undefinedMetrics.length) {
        return;
      }

      metrics.forEach((metric) => {
        const bounds = this.getMetricBounds_(sliceMetrics['metrics'][metric]);
        metricsData.push({
          fullSliceName: sliceMetrics['slice'],
          sliceValue: sliceMetrics['sliceValue'],
          metricName: metric,
          value: tfma.CellRenderer.maybeExtractBoundedValue(
              sliceMetrics['metrics'][metric]),
          upperBound: bounds.min,
          lowerBound: bounds.max,
          exampleCount:
              sliceMetrics['metrics']['post_export_metrics/example_count']
        });
      });
      d3Data.push({
        fullSliceName: sliceMetrics['slice'],
        sliceValue: sliceMetrics['sliceValue'],
        metricsData: metricsData
      });
    });
    d3Data.sort(function(a, b) {
      // Ensure that the baseline slice always appears on the left side.
      if (a.fullSliceName == baseline) {
        return -1;
      }
      if (b.fullSliceName == baseline) {
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
    });

    // Build slice scale.
    const slicesX = d3.scaleBand()
                        .domain(d3Data.map(d => d.sliceValue))
                        .rangeRound([GRAPH_BOUND.left, GRAPH_BOUND.right])
                        .padding(SLICES_PADDING);
    // Build metrics scale.
    const metricsX = d3.scaleBand()
                         .domain(metrics)
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
    const confidenceIntervalColor = d3.scaleOrdinal().range(ERROR_BAR_COLOR_);

    // Draw Graph
    const svg = d3.select(this.$['bar-chart']);
    svg.html('');
    const bars =
        svg.append('g')
            .attr('id', 'bars')
            .selectAll('g')
            .data(d3Data)
            .enter()
            .append('g')
            .attr('transform', d => `translate(${slicesX(d.sliceValue)},0)`);
    bars.selectAll('rect')
        .data(d => d.metricsData)
        .enter()
        .append('rect')
        .attr('x', d => metricsX(d.metricName))
        .attr('y', d => isNaN(d.value) ? y(0) : y(d.value))
        .attr('width', metricsX.bandwidth())
        .attr('height', d => isNaN(d.value) ? 0 : y(0) - y(d.value))
        .attr(
            'fill',
            d => d.fullSliceName == baseline ? baselineColor(d.metricName) :
                                               metricsColor(d.metricName))
        .on('mouseover', this.tip_.show)
        .on('mouseout', this.tip_.hide)
        .on('click', (d, i) => {
          this.dispatchEvent(new CustomEvent(
              tfma.Event.SELECT,
              {detail: d.fullSliceName, composed: true, bubbles: true}));
        });
    bars.selectAll('line')
        .data(d => d.metricsData)
        .enter()
        .append('line')
        .attr('x1', d => metricsX(d.metricName) + (metricsX.bandwidth() / 2))
        .attr('y1', d => isNaN(d.upperBound) ? y(0) : y(d.upperBound))
        .attr('x2', d => metricsX(d.metricName) + (metricsX.bandwidth() / 2))
        .attr('y2', d => isNaN(d.lowerBound) ? y(0) : y(d.lowerBound))
        .attr('stroke', d => confidenceIntervalColor(d.metricName))
        .attr('stroke-width', 1);

    // Draw X Y axis.
    svg.append('g').attr('id', 'xaxis').call(configureXAxis);
    svg.append('g').attr('id', 'yaxis').call(configureYAxis);
    svg.call(tip);
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
}

customElements.define(
    'fairness-bounded-value-bar-chart', FairnessBoundedValueBarChart);
