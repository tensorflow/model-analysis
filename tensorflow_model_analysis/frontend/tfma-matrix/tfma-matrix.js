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
   * A simple matrix visualization designed to support up to 32 rows / columns.
   */
  is: 'tfma-matrix',

  properties: {
    /** @type {!Object} */
    data: {type: Object, observer: 'dataChanged_'},

    /** @type {string} */
    rowLabel: {type: String},

    /** @type {string} */
    columnLabel: {type: String},

    /** @type {number} */
    minValue: {type: Number},

    /** @type {number} */
    maxValue: {type: Number},

    /**
     * If not null, the color of a cell is determined between either
     * [minValue, pivot) or [pivot, maxValue].
     * @type {?number}
     */
    pivot: {type: Number, value: null},

    /**
     * Whether the component is in expanded mode.
     * @type {boolean}
     */
    expanded: {type: Boolean, value: false, reflectToAttribute: true},

    /** @private {!Array<string>} */
    rowNames_: {type: Array},

    /** @private {!Array<string>} */
    columnNames_: {type: Array},

    /**
     * A mapping between row names and assigned color.
     * @private {!Object}
     */
    rowColors_: {type: Object},

    /**
     * A mapping between column names and assigned color.
     * @private {!Object}
     */
    columnColors_: {type: Object},

    /**
     * A two dimension array containing the matrix that will be rendered.
     * @private {!Array<!Array<!Object>>}
     */
    matrix_: {
      type: Array,
      computed: 'computeMatrix_(data, rowNames_, columnNames_, rowColors_, ' +
          'columnColors_, minValue, maxValue, pivot, expanded)'
    },

    /** @private {?Object} */
    selectedCell_: {type: Object},
  },

  /**
   * Updates row and column names based on data.
   * @param {!Object} data
   * @private
   */
  dataChanged_: function(data) {
    const rows = Object.keys(data).sort();
    const rowColors = {};

    const columns = {};
    const columnColors = {};
    rows.forEach((row, index) => {
      Object.keys(data[row]).forEach(column => {
        columns[column] = 1;
      });
      rowColors[row] = 'c' + (index % 16);
    });

    const columnNames = Object.keys(columns).sort();
    columnNames.forEach((column, index) => {
      columnColors[column] = 'c' + (index % 16);
    });

    this.rowNames_ = rows;
    this.columnNames_ = columnNames;
    this.rowColors_ = rowColors;
    this.columnColors_ = columnColors;
    this.selectedCell_ = null;
  },


  /**
   * Creates an object containing all the information necessary to render a cell
   * in the confusion matrix and the details section.
   * @param {string} row
   * @param {string} column
   * @param {!Object} dataMap
   * @param {function(number):number} getScale
   * @return {!Object}
   */
  makeCell_: function(row, column, dataMap, getScale) {
    const cell = dataMap[row] && dataMap[row][column];

    if (cell === undefined) {
      return {
        'missing': true,
      };
    }

    const cellWeight = cell['value'];
    const scale = getScale(cellWeight);
    const backgroundClassName = 'b' + Math.round(scale * 16);

    return {
      'cell': true,
      'value': cellWeight,
      'cssClass': backgroundClassName + ' ' + (scale > 0.5 ? 'white-text' : ''),
      'rowName': row,
      'columnName': column,
      'tooltip': cell['tooltip'],
      'details': cell['details'],
    };
  },

  /**
   * Builds the cells that will be used for visualization. predictedClasses and
   * actualClasses contains class names and specifies how the rows and columns
   * will be sorted.
   * @param {!Object} data
   * @param {!Array<string>} rowNames
   * @param {!Array<string>} columnNames
   * @param {!Object<string>} rowColors
   * @param {!Object<string>} columnColors
   * @param {number} minValue
   * @param {number} maxValue
   * @param {?number} pivot
   * @param {boolean} expanded
   * @return {!Array<!Array<!Object>>}
   */
  computeMatrix_: function(
      data, rowNames, columnNames, rowColors, columnColors, minValue, maxValue,
      pivot, expanded) {
    if (this.tooMany_(rowNames, columnNames)) {
      // If too many classes to visualize, return empty array isntead.
      return [];
    }
    const makeHeader = (name, colors, cssClasses) => {
      return {
        'header': true,
        'name': name,
        'cssClass': cssClasses.concat([colors[name], 'header']).join(' '),
      };
    };

    const matrix = [];
    if (expanded) {
      matrix.push([{'label': true}]);

      // Create the column header if expanded.
      const header = columnNames.reduce((acc, column, index) => {
        acc.push(makeHeader(column, columnColors, ['col']));
        return acc;
      }, [{'widget': true}]);
      matrix.push(header);
    }

    const getScale = (value) => {
      let scale;

      if (pivot !== null) {
        if (value > pivot) {
          scale = (value - pivot) / (maxValue - pivot);
        } else {
          scale = (pivot - value) / (pivot - minValue);
        }
      } else {
        scale = (value - minValue) / (maxValue - minValue);
      }

      return Math.max(Math.min(scale || 0, 1), 0);
    };
    rowNames.forEach(rowName => {
      const row = [];
      if (expanded) {
        row.push(makeHeader(rowName, rowColors, ['row']));
      }

      columnNames.forEach(columnName => {
        row.push(this.makeCell_(rowName, columnName, data, getScale));
      });

      matrix.push(row);
    });
    return matrix;
  },

  /**
   * Creates the title for a cell.
   * @param {boolean} expanded
   * @param {!Object} cell
   * @return {string|undefined}
   * @private
   */
  getCellTitle_: function(expanded, cell) {
    return expanded ? cell['tooltip'] : undefined;
  },

  /**
   * Tap handler for a cell.
   * @param {!Event} e
   * @private
   */
  cellTapped_: function(e) {
    if (this.expanded) {
      const targetCell = e['model']['cell'];
      const event = this.fire(
          'show-details', targetCell['details'], {'cancelable': true});
      // Check if we should show details in place.
      if (!event.defaultPrevented) {
        // Update selectedCell_ if we are already expanded.
        this.selectedCell_ =
            this.selectedCell_ == targetCell ? null : targetCell;
      }
    } else {
      // Check to see if we should expand in place if not expanded.
      const event = this.fire('expand', {}, {'cancelable': true});
      if (!event.defaultPrevented) {
        // If no other component is handling the expand event, expand in place.
        this.expanded = true;
      }
    }
  },

  /**
   * Tap handler for a class header.
   * @param {!Event} e
   * @private
   */
  headerTapped_: function(e) {
    // Sort the matrix by the values on the column / row clicked.
    const target = e['model']['cell']['name'];
    const sortByColumn = e.srcElement.classList.contains('col');
    const matrix = this.data;

    const toSort = (sortByColumn ? this.rowNames_ : this.columnNames_).slice();
    toSort.sort((a, b) => {
      const cellA =
          matrix[sortByColumn ? a : target][sortByColumn ? target : a];
      const cellB =
          matrix[sortByColumn ? b : target][sortByColumn ? target : b];
      // If the cell is missing, use Infinity to put it at the end.
      const valA =
          cellA && cellA['value'] !== undefined ? cellA['value'] : Infinity;
      const valB =
          cellB && cellB['value'] !== undefined ? cellB['value'] : Infinity;
      return valA - valB;
    });

    if (sortByColumn) {
      this.rowNames_ = toSort;
    } else {
      this.columnNames_ = toSort;
    }
  },

  /**
   * @param {!Array<string>} rows
   * @param {!Array<string>} columns
   * @return {boolean} Whether there are too much data for the component to
   *     render efficidently.
   */
  tooMany_: function(rows, columns) {
    // We currently support up to 32 rows / columns.
    return rows.length > 32 || columns.length > 32;
  },
});
