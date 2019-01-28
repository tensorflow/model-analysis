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

  is: 'tfma-metrics-table',

  properties: {
    /**
     * List of metrics to be shown in the table.
     * @type {!Array<string>}
     */
    metrics: {
      type: Array,
      value: () => {
        return [];
      },
    },

    /**
     * Dictionary with metric name as key and metric format as value.
     * If a metric does not have a format given by this object, its format will
     * default to lantern.MetricValueFormat.FLOAT.
     * @type {!Object<tfma.MetricValueFormatSpec>}
     */
    metricFormats: {
      type: Object,
      value: () => {
        return {};
      },
    },

    /**
     * Metrics table data.
     * @type {!tfma.TableProviderExt}
     */
    data: Object,

    /** @type {number} */
    pageSize: {type: Number, value: 20},

    /**
     * A look up table to override header.
     * @type {!Object<string>}
     */
    headerOverride: {
      type: Object,
      value: () => {
        return {};
      }
    },

    /**
     * Google-chart options.
     * @type {!Object}
     * @private
     */
    options_: {type: Object, computed: 'computeOptions_(pageSize)'},

    /**
     * An array of extra events to listen to on the chart object.
     * @private {!Array<string>}
     */
    chartEvents_: {type: Array, value: () => ['page', 'sort']},

    /**
     * Selection of google-chart table.
     * @type {!Array<{row: (number|undefined), column: (number|undefined)}>}
     */
    selection: {type: Array, observer: 'selectionChanged_', notify: true},

    /**
     * This flag is set to true when the google-chart component is first
     * initialized notified via google-chart-ready event.
     * @private {boolean}
     */
    tableReady_: {type: Boolean, value: false},

    /**
     * Generated plot data piped to google-chart. It should be 2d array where
     * the each element in the top level array represents a row in the table.
     * @private {!Array<!Array>}
     */
    plotData_: {
      type: Array,
      computed:
          'computePlotData_(data, metrics, metricFormats, headerOverride, tableReady_)',
    },

    /**
     * Index of the current page in the table.
     * @private {number}
     */
    currentPage_: {type: Number, value: 0},

    /**
     * The array containing sorted indices of the rows of data if the user
     * applied any sorting; null, otherwise.
     * @private {?Array<number>}
     */
    sortedIndexes_: {type: Array, value: null},

    /**
     * The indices of rows of data in the current table view.
     * @type {!Array<number>}
     */
    visibleRows: {type: Array, notify: true},
  },

  observers: [
    'updateVisibleRows_(currentPage_, pageSize, sortedIndexes_, plotData_.length)'
  ],

  /**
   * Adds event listeners to elements after the table component is ready.
   * @override
   */
  ready: function() {
    const table = this.$.table;

    /**
     * If true, selecting the table row will trigger a select event. This is
     * set to false when the selection is made programmatically for linked data
     * items across components.
     * This flag defaults to true, when no programmatic control is in progress.
     * @private {boolean}
     */
    this.selectEventEnabled_ = true;

    const listener = () => {
      table.removeEventListener('google-chart-ready', listener);
      this.tableReady_ = true;
    };
    table.addEventListener('google-chart-ready', listener);
  },

  /**
   * If selection is given, finds and highlights it. If null, clears current
   * highlight.
   * @param {Object} selection
   */
  highlight: function(selection) {
    if (selection) {
      this.highlightSelection_(selection);
    } else {
      this.clearSelection_();
    }
  },

  /**
   * Finds the given selection and highlights it.
   * @param {!Object} selection Key value pairs in the selection object specify
   *     the row that should be highlighted.
   * @private
   */
  highlightSelection_: function(selection) {
    const table = this.$.table;
    let rowIndex = -1;
    const dataTable = this.plotData_;
    const fieldsToMatch = Object.keys(selection).map((field) => {
      return {id: dataTable[0].indexOf(field), value: selection[field]};
    });

    for (let i = 1; i < dataTable.length; i++) {
      let trying = true;
      for (let j = 0; trying && j < fieldsToMatch.length; j++) {
        // The v field is more consistently defined across other components.
        trying = trying &&
            dataTable[i][fieldsToMatch[j].id]['v'] == fieldsToMatch[j].value;
      }
      if (trying) {
        rowIndex = i - 1;  // Skip the label row.
        break;
      }
    }
    if (rowIndex != -1) {
      this.selectEventEnabled_ = false;
      table.selection = [{row: rowIndex}];
      this.selectEventEnabled_ = true;
    }
  },


  /**
   * Internal implementation of clearing selection.
   * @private
   */
  clearSelection_: function() {
    const table = this.$.table;
    this.selectEventEnabled_ = false;
    table.selection = [];
    this.selectEventEnabled_ = true;
  },

  /**
   * Fires a selection changed event when table row is clicked.
   * @private
   */
  selectionChanged_: function() {
    if (!this.selectEventEnabled_) {
      return;
    }
    const table = this.$.table;
    const selection = table.selection;
    if (!selection.length) {
      this.fire(tfma.Event.CLEAR_SELECTION);
    } else {
      const dataTable = this.plotData_;
      const selectedRow = dataTable[table.selection[0].row + 1];
      const selected = dataTable[0].reduce((acc, column, index) => {
        acc[column] = selectedRow[index]['v'];
        return acc;
      }, {});
      this.fire(tfma.Event.SELECT, selected);
    }
  },

  /**
   * Computes the data table.
   * @param {!tfma.TableProviderExt} data
   * @param {!Array<string>} metrics
   * @param {!Object<!tfma.MetricValueFormatSpec>} metricFormats
   * @param {!Object<string>} headerOverride
   * @param {boolean} tableReady
   * @return {!Array<!Array>}
   * @private
   */
  computePlotData_: function(
      data, metrics, metricFormats, headerOverride, tableReady) {
    if (!tableReady || !data.readyToRender()) {
      // No need to compute plot data if the table is not ready since it will
      // likely get ignored.
      // The header must contain at least one string to make google-chart show
      // up properly.
      return [[]];
    }
    const header =
        data.getHeader(metrics).map(metric => headerOverride[metric] || metric);

    const dataTable = data.getDataTable();
    const formats = data.getFormats(metricFormats);
    const renderedTable = dataTable.map(
        row => header.map(
            (column, index) => tfma.CellRenderer.renderValueWithFormatOverride(
                row[index], data,
                /** @type {!tfma.MetricValueFormatSpec|undefined} */(
                    formats[column]))));
    return [header].concat(renderedTable);
  },

  /**
   * @param {number} pageSize
   * @return {!Object} The default config options for the table.
   * @private
   */
  computeOptions_: function(pageSize) {
    return {
      'allowHtml': true,
      'width': '100%',
      'page': 'enable',
      'pageSize': pageSize,
      'pageButtons': 'auto'
    };
  },

  /**
   * Handler for the underlying chart's page event. Updates the index of the
   * current page.
   * @param {!Event} pageEvent The event object.
   * @private
   */
  onPage_: function(pageEvent) {
    this.currentPage_ = pageEvent['detail']['data']['page'];
  },

  /**
   * Handler for the underlying chart's sort event. Updates how the indices of
   * the newly sorted table should map to the original data.
   * @param {!Event} sortEvent The event object.
   * @private
   */
  onSort_: function(sortEvent) {
    this.sortedIndexes_ = sortEvent['detail']['data']['sortedIndexes'];
  },

  /**
   * Updates the array of visible rows based on current page index and how the
   * table is sorted.
   * @param {number} pageIndex The current page index.
   * @param {number} pageSize The current page size.
   * @param {?Array<number>} sortedIndices An array containing the indices of
   *     how the data is sorted. Null if extra sorting is applied.
   * @param {number} rowCount The total number of rows in the table.
   * @private
   */
  updateVisibleRows_: function(pageIndex, pageSize, sortedIndices, rowCount) {
    const pageStartIndex = pageIndex * pageSize;
    const pageEndIndex = Math.min(pageStartIndex + pageSize, rowCount - 1);
    this.visibleRows = sortedIndices ?
        sortedIndices.slice(pageStartIndex, pageEndIndex) :
        Array.from(
            {'length': pageEndIndex - pageStartIndex},
            (value, index) => index + pageStartIndex);
  },
});
