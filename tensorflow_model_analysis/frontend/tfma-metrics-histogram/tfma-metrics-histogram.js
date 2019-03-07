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

import {template} from './tfma-metrics-histogram-template.html.js';

import '@polymer/iron-icons/iron-icons.js';
import {IronResizableBehavior} from '@polymer/iron-resizable-behavior/iron-resizable-behavior.js';
import '@polymer/paper-button/paper-button.js';
import '@polymer/paper-card/paper-card.js';
import '@polymer/paper-dialog/paper-dialog.js';
import '@polymer/paper-dropdown-menu/paper-dropdown-menu.js';
import '@polymer/paper-icon-button/paper-icon-button.js';
import '@polymer/paper-input/paper-input.js';
import '@polymer/paper-item/paper-item.js';
import '@polymer/paper-listbox/paper-listbox.js';
import '@polymer/paper-slider/paper-slider.js';
import '@polymer/paper-toggle-button/paper-toggle-button.js';
import {mixinBehaviors} from '@polymer/polymer/lib/legacy/class.js';
import '../../../../javascript/google_chart/google-chart-loader.js';

/**
 * The types of metrics histograms supported.
 * @enum {string}
 */
const Type = {
  UNWEIGHTED: 'unweighted',
  WEIGHTED: 'weighted',
  BOTH: 'both'
};

/**
 * @enum {string}
 */
const XLabel = {
  UNWEIGHTED: 'Number of slices in bucket',
  WEIGHTED: 'Number of (weighted) examples for slices in bucket'
};

/** @enum {string} */
const ElementId = {
  DETAILS: 'details',
  EMPTY: 'empty',
  OVERVIEW: 'overview',
  FOCUS: 'focus',
  METRIC_SELECT: 'metric-select',
  TYPE_SELECT: 'type-select',
  NUM_BUCKETS: 'num-buckets',
  LOGARITHM_SCALE: 'logarithm-scale',
  OPTIONS: 'options',
  OPTIONS_TOGGLE: 'options-toggle'
};

/** @const {number} */
const MIN_DETAILS_WIDTH_PX = 680;

/** @const {number} */
const HEIGHT_PX = 200;

/** @const {number} */
const OVERVIEW_HEIGHT_PX = 30;

/** @const {number} */
const OVERVIEW_MARGIN_LEFT_PX = 120;

/** @const {number} */
const OVERVIEW_MARGIN_RIGHT_PX = 120;

/** @const {number} */
const OVERVIEW_PADDING_TOP_PX = 2;

/** @const {string} */
const HIGHLIGHTED_CLASS = 'highlighted';

/**
 * Number of buckets used in the histogram overview. We use a constant number
 * instead of compute it proportionally from the overview size so as to obtain a
 * table looking of the overview. If the number of buckets changes for
 * different width, then it is likely that some elements fall in nearby bins
 * resulting in the overall shape of the overview keeping changing.
 * @const {number}
 */
const OVERVIEW_NUM_BUCKETS = 600;

/** @const {number} */
const MIN_NUM_BUCKETS = 1;

/** @const {number} */
const MAX_NUM_BUCKETS = 50;

/** @const {number} */
const DEFAULT_NUM_BUCKETS = 10;

/**
 * Computes the value range for a specified column from a data table.
 * @param {!Array<!Array<string|number>>} dataTable Input data table.
 * @param {number} index Column index.
 * @return {{min: number, max: number}} Value range for the specified column.
 */
function getDataTableColumnRange(dataTable, index) {
  // Skip the label row.
  const rows = dataTable.slice(1);
  return rows.reduce((prev, cur) => {
    return {
      min: Math.min(prev.min, cur[index]),
      max: Math.max(prev.max, cur[index])
    };
  }, {min: Infinity, max: -Infinity});
}

/**
 * tfma-metrics-histogram shows the distribution of each slice based on their
 * meitrc values.
 *
 * @polymer
 * @extends PolymerElement
 */
export class MetricsHistogram extends mixinBehaviors
([IronResizableBehavior], PolymerElement) {
  constructor() {
    super();
  }

  static get is() {
    return 'tfma-metrics-histogram';
  }

  /** @return {!HTMLTemplateElement} */
  static get template() {
    return template;
  }

  /** @return {!PolymerElementProperties} */
  static get properties() {
    return {
      /**
       * Data for the histogram.
       * Histogram overview directly renders this data, regardless of the
       * current focus range.
       * @type {!tfma.Data}
       */
      data: {
        type: Object,
        value: () => tfma.Data.build([], []),
      },

      /**
       * Focus range specified by two endpoints' relative positions between
       * [0, 1]. 0/1 are the left/right endpoints that maps to the
       * minimum/maximum metric value of the data.
       * @type {!Array<number>}
       */
      focusRange: {type: Array, value: [0, 1]},

      /**
       * Data for the histogram details. The details data excludes the data
       * outside the focus range.
       * Note that we use focusRange to auto compute the detailsData rather than
       * two arguments focusLeft and focusRight. Because if we use two variables
       * the computed function would be triggered twice if we have two lines
       * setting focusLeft and focusRight separately.
       * @type {!tfma.Data}
       */
      detailsData: {
        type: Object,
        computed: 'computeDetailsData_(data, metric, focusRange)',
        notify: true
      },

      /**
       * Number of histogram buckets.
       * @type {number}
       */
      numBuckets: {
        type: Number,
        value: DEFAULT_NUM_BUCKETS,
      },

      /**
       * Histogram type. Values are defined in enum Type.
       * @type {string}
       */
      type: {
        type: String,
        value: Type.UNWEIGHTED,
      },

      /**
       * Metrics to be visualized in the histogram.
       */
      metric: {type: String, value: ''},

      /**
       * Metrics available for visualization in the metric selection dropdown.
       * @private {!Array<string>}
       */
      selectableMetrics_: {
        type: Array,
        computed: 'computeSelectableMetrics_(data, weightedExamplesColumn)',
        observer: 'selectableMetricsChanged_'
      },

      /**
       * Name of the weighted examples column in the metrics list.
       * @type {string}
       */
      weightedExamplesColumn: {type: String, value: ''},

      /**
       * Whether to log the bin count value.
       * @type {boolean}
       */
      logarithmScale: {type: Boolean, value: false},

      /**
       * Selected (highlighted features).
       * @type {!Array<number|string>}
       */
      selectedFeatures: {
        type: Array,
        value: () => [],
      },

      /**
       * If true, then dragging the focus range would immediatley update the
       * visualization. This will significantly slow down the performance in
       * environment with complex element handling, such as Colab.
       */
      realTimeFocus: {type: Boolean, value: false},

      /**
       * Google chart packages required.
       * @private {!Array<string>}
       */
      chartPackages_: {type: Array, value: ['corechart']},

      /**
       * Previous width used to check whether the size has really changed or is
       * affected by some pop-out elements (e.g. dropdown).
       * @private {number}
       */
      previousWidth_: {type: Number, value: 0, observer: 'render_'},

      /**
       * The ColumnChart object created via gviz api.
       * @private {(!google.visualization.ColumnChart|undefined)}
       */
      chart_: {type: Object},
    };
  }

  static get observers() {
    return [
      // Observe all rendering related properties and update when necessary.
      'reRender_(data, metric, type, focusRange, weightedExamplesColumn, ' +
      'logarithmScale, numBuckets, selectedFeatures)'
    ];
  }

  /** @override */
  ready() {
    super.ready();

    this.addEventListener('iron-resize', () => {
      this.previousWidth_ = this.getDetailsWidth_();
    });

    this.$.loader.create('column', this.$[ElementId.DETAILS]).then(chart => {
      this.chart_ = chart;
      this.render_();
    });
  }

  /**
   * Updates the focus range to a given range and re-renders.
   * @param {number} min Range min.
   * @param {number} max Range max.
   */
  updateFocusRange(min, max) {
    this.focusRange = [min, max];
  }

  /**
   * Renders the metrics histogram.
   * @private
   */
  render_() {
    if (!this.renderable_()) {
      return;
    }
    this.renderDetails_();
    // Since overview width depends on the details width, we render details
    // first before rendering overview.
    this.renderOverview_();
  }

  /**
   * @return {number} Histogram details width in pixels.
   * @private
   */
  getDetailsWidth_() {
    const width = this.getBoundingClientRect().width;
    return Math.max(width, MIN_DETAILS_WIDTH_PX);
  }

  /**
   * @return {number} Histogram overview width in pixels.
   * @private
   */
  getOverviewWidth_() {
    const width = this.getDetailsWidth_();
    return width - OVERVIEW_MARGIN_LEFT_PX - OVERVIEW_MARGIN_RIGHT_PX;
  }

  /**
   * Renders the histogram overview.
   * @private
   */
  renderOverview_() {
    const svg = d3.select(this.$[ElementId.OVERVIEW]);
    const width = this.getOverviewWidth_();
    const svgNode = svg.node();
    svgNode.setAttribute('width', width);
    svgNode.setAttribute(
        'height', OVERVIEW_HEIGHT_PX * (this.type == Type.BOTH ? 2 : 1));
    svgNode.style.marginLeft = OVERVIEW_MARGIN_LEFT_PX + 'px';

    const height = OVERVIEW_HEIGHT_PX;
    const range = this.data.getColumnRange(this.metric);
    const plotData = this.prepareHistogram_(
        this.data, OVERVIEW_NUM_BUCKETS, range.min, range.max);
    const dataTableArray = plotData.dataTableArray;

    const unweightedSvg = svg.select('.' + Type.UNWEIGHTED);
    const weightedSvg = svg.select('.' + Type.WEIGHTED);
    // Clear the previous drawing.
    unweightedSvg.selectAll('*').remove();
    weightedSvg.selectAll('*').remove();

    if (this.type != Type.WEIGHTED) {
      this.drawOverview_(
          unweightedSvg, dataTableArray.map((row) => {
            return [row[0], row[1]];
          }),
          getDataTableColumnRange(dataTableArray, 1));
    }
    if (this.type != Type.UNWEIGHTED) {
      this.drawOverview_(
          weightedSvg, dataTableArray.map((row) => {
            return [row[0], row[2]];
          }),
          getDataTableColumnRange(dataTableArray, 2));
    }

    unweightedSvg.select('path').attr('class', 'blue');
    weightedSvg.select('path').attr(
        'class', this.type == Type.BOTH ? 'red' : 'blue');

    if (this.type == Type.BOTH) {
      weightedSvg.attr('transform', 'translate(0,' + height + ')');
    }

    this.drawOverviewFocus_(this.focusRange[0], this.focusRange[1]);

    let dragRange = [];
    const drag = d3.drag();
    let focusLeft;
    let focusRight;

    /**
     * Retrieves the beginning and ending positions of the mouse drag, and
     * computes the corresponding focus range coordinates on the histogram
     * overview.
     * @this {!Element}
     */
    const dragHandler = () => {
      const x = d3.mouse(/** @type {!Element} */ (svg.node()))[0];
      if (dragRange[0] == undefined) {
        dragRange[0] = x;
      } else {
        dragRange[1] = x;
        focusLeft = Math.max(0, Math.min(dragRange[0], dragRange[1]) / width);
        focusRight = Math.min(1, Math.max(dragRange[0], dragRange[1]) / width);
        this.drawOverviewFocus_(focusLeft, focusRight);
        if (this.realTimeFocus) {
          // Updating this.focusRange will immediately update
          // detailsData and visualization because of data binding.
          this.updateFocusRange(focusLeft, focusRight);
        }
      }
    };

    /**
     * Resets the drag range to empty range when the mouse drag is released.
     * @this {!Element}
     */
    const dragendHandler = () => {
      if (dragRange[0] == undefined || dragRange[1] == undefined) {
        // Mouse did not move.
        return;
      }
      this.updateFocusRange(focusLeft, focusRight);
      dragRange = [];
      this.dispatchEvent(new CustomEvent(tfma.Event.UPDATE_FOCUS_RANGE, {
        detail: {'focusLeft': focusLeft, 'focusRight': focusRight},
        bubbles: true,
        composed: true
      }));
    };

    drag.on('drag', dragHandler).on('end', dragendHandler);
    svg.call(drag);

    svg.on(tfma.Event.DOUBLE_CLICK, () => {
      this.resetFocusRange_();
      this.dispatchEvent(new CustomEvent(tfma.Event.UPDATE_FOCUS_RANGE, {
        detail: {'focusLeft': 0, 'focusRight': 1},
        bubbles: true,
        composed: true
      }));
    });
  }

  /**
   * Renders the histogram details in column chart.
   * @private
   */
  renderDetails_() {
    const range = this.data.getColumnRange(this.metric);
    const rangeMin = (range.max - range.min) * this.focusRange[0] + range.min;
    const rangeMax = (range.max - range.min) * this.focusRange[1] + range.min;
    const plotData = this.prepareHistogram_(
        this.detailsData, this.numBuckets, rangeMin, rangeMax);
    const series = [];
    const columns = [0];
    let vAxisCounter = 0;
    if (this.type != Type.WEIGHTED) {
      // unweighted or both
      series.push({'targetAxisIndex': vAxisCounter++});
      columns.push(1);
      columns.push({
        'calc': 'stringify',
        'sourceColumn': 1,
        'type': 'string',
        'role': 'annotation'
      });
    }
    if (this.type != Type.UNWEIGHTED) {
      // weighted or both
      series.push({'targetAxisIndex': vAxisCounter++});
      columns.push(2);
      columns.push({
        'calc': 'stringify',
        'sourceColumn': 2,
        'type': 'string',
        'role': 'annotation'
      });
    }

    const emptyElement = this.$[ElementId.EMPTY];
    const detailsElement = this.$[ElementId.DETAILS];
    emptyElement.style.display = plotData.isEmpty ? '' : 'none';
    detailsElement.style.display = !plotData.isEmpty ? '' : 'none';
    if (plotData.isEmpty) {
      return;
    }
    const data = google.visualization.arrayToDataTable(plotData.dataTableArray);
    const tempView = new google.visualization.DataView(data);
    // Render the column value within the column.
    tempView.setColumns(columns);
    this.chart_.draw(tempView, {
      'enableInteractivity': false,
      'bar': {'groupWidth': '99%'},
      'hAxis': {'ticks': plotData.hAxisTicks},
      // We must specify vertical ticks to work around a gviz bug. Set the
      // label to x axis with an empty label so that the UI remains unchanged.
      // @bug 34237301
      'vAxis': {'ticks': [{'v': 0, 'f': ''}]},
      'legend': {'position': 'top'},
      'tooltip': {'trigger': 'none'},
      'series': series,
      'width': this.getDetailsWidth_(),
      'height': HEIGHT_PX
    });

    this.highlightHistogramBuckets_(plotData.highlightedBuckets);
  }

  /**
   * Creates the data table array for the metrics histogram. The buckets are
   * evenly distributed between [rangeMin, rangeMax].
   * @param {!tfma.Data} data Input data.
   * @param {number} numBuckets Number of desired histogram buckets.
   * @param {number} rangeMin Range left coordinate.
   * @param {number} rangeMax Range right coordinate.
   * @return {{
   *   dataTableArray: !Array<!Array<string|number>>,
   *   hAxisTicks: !Array<{v: number, f: string}>,
   *   isEmpty: boolean,
   *   highlightedBuckets: !Array<number>
   * }}
   * @private
   */
  prepareHistogram_(data, numBuckets, rangeMin, rangeMax) {
    const dataSeriesList = data.getSeriesList();
    const metricRange = data.getColumnRange(this.metric);

    // Add the data column labels to the first row of dataTableArray.
    const header = [this.metric, XLabel.UNWEIGHTED, XLabel.WEIGHTED];
    const dataTableArray = [header];

    if (numBuckets <= 0 || rangeMin > rangeMax || metricRange.min > rangeMax ||
        metricRange.max < rangeMin || !dataSeriesList.length) {
      // Histogram range is empty (this.data is empty)
      // or does not contain any slices (no data in the focus range).
      // Return empty flag.
      return {
        dataTableArray: [header],
        hAxisTicks: [],
        highlightedBuckets: [],
        isEmpty: true
      };
    }

    // Adjust rangeMax to be slightly larger than rangeMin when
    // rangeMin == rangeMax, to avoid span being zero.
    if (rangeMax == rangeMin) {
      rangeMax =
          rangeMin + Math.pow(.1, tfma.FLOATING_POINT_PRECISION) * numBuckets;
    }

    // Initialize the data table array (one empty entry per bucket).
    const bucketSize = (rangeMax - rangeMin) / numBuckets;

    const hAxisTicks = [];
    for (let i = 0; i <= numBuckets; i++) {
      const value = rangeMin + i * bucketSize;
      hAxisTicks.push(
          {'v': value, 'f': value.toFixed(tfma.FLOATING_POINT_PRECISION)});
    }

    // Create the empty buckets (i.e. find the mid-point of each bucket).
    for (let i = 0; i < numBuckets; i++) {
      dataTableArray.push([rangeMin + bucketSize * (i + 0.5), 0, 0]);
    }

    const weightColumn = this.weightedExamplesColumn;
    const metric = this.metric;
    const highlightedBuckets = {};
    // Add values for each bucket. Skip the label row.
    dataSeriesList.forEach((dataSeries) => {
      const featureString = dataSeries.getFeatureString();
      const numExamples = data.getMetricValue(featureString, weightColumn);
      const metricValue = data.getMetricValue(featureString, metric);

      if (!isNaN(metricValue) && metricValue != null) {
        let index = Math.floor((metricValue - rangeMin) / bucketSize);
        // Add one to skip the label row.
        index = Math.min(index, numBuckets - 1) + 1;
        dataTableArray[index][1] += 1;
        dataTableArray[index][2] += numExamples;

        const feature = dataSeries.getFeatureIdForMatching();
        if (this.selectedFeatures.indexOf(feature) != -1) {
          // Use 0-based index for buckets.
          highlightedBuckets[index - 1] = true;
          if (this.type == Type.BOTH) {
            // When both unweighted and weighted histograms are shown, there
            // are two lists of rects that are siblings.
            highlightedBuckets[index + numBuckets - 1] = true;
          }
        }
      }
    });

    if (this.logarithmScale) {
      for (let i = 1; i < dataTableArray.length; i++) {
        dataTableArray[i][1] = Math.log(dataTableArray[i][1] + 1);
        dataTableArray[i][2] = Math.log(dataTableArray[i][2] + 1);
      }
    }

    return {
      dataTableArray: dataTableArray,
      hAxisTicks: hAxisTicks,
      isEmpty: false,
      highlightedBuckets:
          Object.keys(highlightedBuckets).map(key => parseInt(key, 10))
    };
  }

  /**
   * Highlights the currently focused region in the histogram overview.
   * @param {number} focusLeft Focus left endpoint (ratio in [0, 1]).
   * @param {number} focusRight Focus right endpoint (ratio in [0, 1]).
   * @private
   */
  drawOverviewFocus_(focusLeft, focusRight) {
    const svg = d3.select(this.$[ElementId.OVERVIEW]);
    let focus = svg.select('rect#focus');
    if (focus.empty()) {
      focus = svg.append('rect').attr('id', ElementId.FOCUS);
    }
    const width = this.getOverviewWidth_();
    focus.attr('x', width * focusLeft)
        .attr('width', width * (focusRight - focusLeft))
        .attr('height', OVERVIEW_HEIGHT_PX * (this.type == Type.BOTH ? 2 : 1));
  }

  /**
   * Renders a single histogram overview track.
   * @param {!d3.selection} svg SVG container of the histogram track.
   * @param {!Array<!Array<string|number>>} dataTable The entire data
   * table, with
   *     only 2 columns for the bucket range and the count.
   * @param {{min: number, max: number}} range Value range of the histogram.
   * @private
   */
  drawOverview_(svg, dataTable, range) {
    const data = dataTable.slice(1);
    if (!data.length) {
      // empty histogram
      return;
    }
    const width = this.getOverviewWidth_();
    const height = OVERVIEW_HEIGHT_PX;
    const xDomain = [data[0][0], data[data.length - 1][0]];
    const xScale = d3.scaleLinear().domain(xDomain).range([0, width]);
    const yScale = d3.scaleLinear().domain([range.min, range.max]).range([
      0, height - OVERVIEW_PADDING_TOP_PX
    ]);
    const line = d3.line();

    let lastX = 0;
    const points = data.map((row) => {
      lastX = xScale(row[0]);
      return [lastX, height - yScale(row[1])];
    });
    // Add the points at the bottom-left and bottom-right.
    points.push([lastX, height]);
    points.push([xScale(xDomain[0]), height]);

    svg.append('path').attr('d', line(points));
    svg.append('rect')
        .attr('class', 'overview')
        .attr('height', height)
        .attr('width', width);
  }

  /**
   * Highlights histogram buckets, currently based on selected feature in the
   * table.
   * @param {!Array<number>} buckets Bucket indices.
   * @private
   */
  highlightHistogramBuckets_(buckets) {
    const gs = this.$[ElementId.DETAILS].getElementsByTagName('g');
    let clippathg;
    for (let i = 0; i < gs.length; i++) {
      if (gs[i].getAttribute('clip-path') != null) {
        clippathg = gs[i];
        break;
      }
    }
    // The bars group is the 2nd child of the clip-path group.
    const barsg = clippathg.children[1];
    // The bar text group is the 3rd sibling of the clip-path group.
    const textg = clippathg.nextSibling.nextSibling.nextSibling;
    for (let i = 0; i < barsg.children.length; i++) {
      barsg.children[i].classList.remove(HIGHLIGHTED_CLASS);
      textg.children[i].classList.remove(HIGHLIGHTED_CLASS);
    }
    buckets.forEach((bucketIndex) => {
      barsg.children[bucketIndex].classList.add(HIGHLIGHTED_CLASS);
      textg.children[bucketIndex].classList.add(HIGHLIGHTED_CLASS);
    });
  }

  /**
   * Resets the focus range to ratio [0, 1] and re-renders.
   * @private
   */
  resetFocusRange_() {
    this.updateFocusRange(0, 1);
  }

  /**
   * @return {boolean} Whether the chart is ready to render, i.e. google-chart
   *     is ready and metric has been set.
   * @private
   */
  renderable_() {
    return !!this.chart_ && this.metric !== '' &&
        this.numBuckets >= MIN_NUM_BUCKETS &&
        this.numBuckets <= MAX_NUM_BUCKETS;
  }

  /**
   * Opens the options dialog.
   * @private
   */
  openOptions_() {
    this.$[ElementId.OPTIONS].open();
  }

  /**
   * Computes the details histogram data, which is the data within the focus
   * range.
   * @param {!tfma.Data} data Full histogram data.
   * @param {string} metric Currently selected metric.
   * @param {!Array<number>} focusRange Left and right endpoints of the focus
   *     range.
   * @return {!tfma.Data} Data filtered by focus range.
   * @private
   */
  computeDetailsData_(data, metric, focusRange) {
    const range = data.getColumnRange(metric);
    let metricSpan = range.max - range.min;
    if (metricSpan == 0) {
      // When metricSpan == 0, focusRange will have no effect on filtering.
      // This will make filtering not working for data with exactly 1 slice.
      // We set span to a small number so that even when there is 1 slice and
      // the user selects an empty focus range in the overview, the single
      // slice can be filtered out.
      metricSpan = Math.pow(.1, tfma.FLOATING_POINT_PRECISION);
    }
    const metricMin =
        Math.max(range.min, range.min + metricSpan * focusRange[0]);
    const metricMax =
        Math.min(range.max, range.min + metricSpan * focusRange[1]);
    const filteredData = data.filter((dataSeries) => {
      const value = data.getMetricValue(dataSeries.getFeatureString(), metric);
      return value >= metricMin && value <= metricMax;
    });
    if (this.detailsData && this.detailsData.equals(filteredData)) {
      // Details data unchanged. We return the original data to avoid
      // triggering unnecessary observers via notify.
      return this.detailsData;
    }
    return filteredData;
  }

  /**
   * @param {!tfma.Data} data
   * @param {string} weightedExamplesColumn
   * @return {!Array<string>} The selectable metrics, which are all metrics
   *     except the weighted examples.
   * @private
   */
  computeSelectableMetrics_(data, weightedExamplesColumn) {
    const metrics = data.getMetrics();
    if (!metrics.length || weightedExamplesColumn === '') {
      // When metrics and weightedExamplesColumn have not been set, we shall
      // not derive any selectable metrics.
      return [];
    }
    const selectableMetrics = metrics.slice();
    const weightedExamplesIndex =
        selectableMetrics.indexOf(weightedExamplesColumn);
    if (weightedExamplesIndex != -1) {
      // Remove weighted examples from selectable metrics.
      selectableMetrics.splice(weightedExamplesIndex, 1);
    }
    return selectableMetrics;
  }

  /** @private */
  selectableMetricsChanged_() {
    if (!this.selectableMetrics_.length) {
      return;
    }
    // Use the first metric available.
    this.metric = this.selectableMetrics_[0];
  }

  /**
   * Observers rendering related variable changes and re-renders when
   * necessary.
   * @param {!tfma.Data} data
   * @param {string} metric
   * @param {string} type
   * @param {!Array<number>} focusRange
   * @param {string} weightedExamplesColumn
   * @param {boolean} logarithmScale
   * @param {number} numBuckets
   * @param {!Array<number>} selectedFeatures
   * @private
   */
  reRender_(
      data, metric, type, focusRange, weightedExamplesColumn, logarithmScale,
      numBuckets, selectedFeatures) {
    this.render_();
  }
}

customElements.define('tfma-metrics-histogram', MetricsHistogram);
