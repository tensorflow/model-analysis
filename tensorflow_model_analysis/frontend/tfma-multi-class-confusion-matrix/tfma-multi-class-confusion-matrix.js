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
   * A visualization of multi-class confusion matrix. It is designed to support
   * up to 64 classes.
   */
  is: 'tfma-multi-class-confusion-matrix',

  properties: {
    /**
     * The serialized form of the data.
     * @type {string}
     */
    data: {type: String, value: '', observer: 'dataChanged_'},

    /**
     * The JSON representation of the data.
     * @type {!Object}
     */
    jsonData: {type: Object},

    /**
     * Whether the component is in expanded mode.
     * @type {boolean}
     */
    expanded: {type: Boolean, value: false, reflectToAttribute: true},

    /**
     * An array containing names of all classes.
     * @private {!Array<string>}
     */
    classNames_: {
      type: Array,
      computed: 'computeClassNames_(jsonData)',
      observer: 'classNamesChanged_'
    },

    /**
     * An object tracking the color assigned for each class.
     * @private {!Object}
     */
    classColors_: {type: Array, computed: 'computeClassColors_(classNames_)'},

    /**
     * An array of class names determining how actual class (columns in the
     * matrix)  are sorted.
     * @private {!Array<string>}
     */
    actualClasses_: {type: Array},

    /**
     * An array of class names determining how predicted class (rows in the
     * matrix) are sorted.
     * @private {!Array<string>}
     */
    predictedClasses_: {type: Array},

    /**
     * A summary of the raw data.
     * @private {!Object}
     */
    summary_:
        {type: Object, computed: ' computeSummary_(classNames_, jsonData)'},

    /**
     * A two dimension array containing the matrix that will be rendered.
     * @private {!Array<!Array<!Object>>}
     */
    matrix_: {
      type: Array,
      computed: 'computeMatrix_(summary_, predictedClasses_, actualClasses_, ' +
          'classColors_, expanded)'
    },
  },

  /**
   * Observer for the property data.
   * @param {string} serializedData
   * @private
   */
  dataChanged_: function(serializedData) {
    if (serializedData) {
      try {
        this.jsonData = /** @type {!Object} */ (JSON.parse(serializedData));
      } catch (e) {
      }
    }
  },

  /**
   * Determines the list of class names from parsed data.
   * @param {!Object} jsonData
   * @return {!Object}
   */
  computeClassNames_: function(jsonData) {
    const classes = {};
    const entries = jsonData['entries'] || [];
    entries.forEach(entry => {
      // Track all predicted and actual classes.
      classes[entry['actualClass']] = 1;
      classes[entry['predictedClass']] = 1;
    });

    // Sort the class names alphabetically.
    return Object.keys(classes).sort();
  },

  /**
   * Observer for classNames_  property.
   * @param {!Array<string>} value
   * @private
   */
  classNamesChanged_: function(value) {
    // Make a copy of value and use it to initalize row and column classes.
    this.predictedClasses_ = value.slice();
    this.actualClasses_ = value.slice();
  },

  /**
   * Determines the colors associated with all classes.
   * @param {!Array<string>} classNames
   * @return {!Object<string>}
   * @private
   */
  computeClassColors_: function(classNames) {
    const colors = {};
    classNames.forEach((className, index) => {
      colors[className] = 'c' + (index % 16);
    });
    return colors;
  },

  /**
   * Builds the summary object.
   * @param {!Array<string>} classNames
   * @param {!Object} jsonData
   * @return {!Object}
   * @private
   */
  computeSummary_: function(classNames, jsonData) {
    const rowSummary = {};
    const columnSummary = {};
    const matrix = {};

    // Initialize everything to all zeroes.
    classNames.forEach(predictedClassName => {
      const row = {};
      matrix[predictedClassName] = row;
      classNames.forEach(actualClassName => {
        row[actualClassName] = 0;
      });
      rowSummary[predictedClassName] = {truePositive: 0, totalWeight: 0};
      columnSummary[predictedClassName] = {truePositive: 0, totalWeight: 0};
    });

    let minWeight = Infinity;
    let maxWeight = -Infinity;
    let totalWeight = 0;
    const entries = jsonData['entries'] || [];
    entries.forEach(entry => {
      const weight = entry['weight'] || 0;
      const actualClass = entry['actualClass'];
      const predictedClass = entry['predictedClass'];
      // Track all predicted and actual classes.
      matrix[actualClass][predictedClass] = weight;
      if (actualClass == predictedClass) {
        columnSummary[actualClass].truePositive = weight;
        rowSummary[actualClass].truePositive = weight;
      }
      columnSummary[actualClass].totalWeight += weight;
      rowSummary[predictedClass].totalWeight += weight;

      totalWeight += weight;
      if (minWeight > weight) {
        minWeight = weight;
      }
      if (maxWeight < weight) {
        maxWeight = weight;
      }
    });

    return {
      rows: rowSummary,
      columns: columnSummary,
      matrix: matrix,
      weight: {
        min: minWeight,
        max: maxWeight,
        total: totalWeight,
      }
    };
  },

  /**
   * Creates an object containing all the information necessary to render a cell
   * in the confusion matrix and the details section.
   * @param {string} actualClass
   * @param {string} predictedClass
   * @return {!Object}
   */
  makeCell_: function(actualClass, predictedClass) {
    const summary = this.summary_;
    const cellWeight = summary.matrix[actualClass][predictedClass];
    const minWeight = summary.weight.min;
    const maxWeight = summary.weight.max;

    const blend = (a, b, scale) => a * (1 - scale) + b * scale;
    const scale = (cellWeight - minWeight) / (maxWeight - minWeight) || 0;

    // Blending from rgb(240, 240, 240) to rgb(10, 71, 164).
    const r = Math.round(blend(240, 10, scale));
    const g = Math.round(blend(240, 71, scale));
    const b = Math.round(blend(240, 164, scale));

    return {
      'cell': true,
      'value': cellWeight,
      'style': 'background-color: ' +
          'rgb(' + r + ',' + g + ',' + b + ');' +
          (scale > 0.5 ? 'color: white;' : ''),
      'actual': actualClass,
      'predicted': predictedClass,
    };
  },

  /**
   * Builds the cells that will be used for visualization. predictedClasses and
   * actualClasses contains class names and specifies how the rows and columns
   * will be sorted.
   * @param {!Object} summary
   * @param {!Array<string>} predictedClasses
   * @param {!Array<string>} actualClasses
   * @param {!Object<string>} classColors
   * @param {boolean} expanded
   * @return {!Array<!Array<!Object>>}
   */
  computeMatrix_: function(
      summary, predictedClasses, actualClasses, classColors, expanded) {
    const makeHeader = (className, cssClasses) => {
      return {
        'header': true,
        'name': className,
        'cssClass':
            cssClasses.concat([classColors[className], 'header']).join(' '),
      };
    };

    const makeLabel = (classCount) => {
      const rowOrColumnCount = classCount + 1;
      const CELL_WIDTH = 50;
      const CELL_HEIGHT = 30;
      const MATRIX_PADDING_OFFSET = -22;

      return {
        'label': true,
        'styleForActual': 'width:' + classCount * CELL_WIDTH + 'px;top:' +
            (-rowOrColumnCount * CELL_HEIGHT + MATRIX_PADDING_OFFSET) +
            'px;left:' + CELL_WIDTH + 'px;',
        'styleForPredicted': 'width:' + classCount * CELL_HEIGHT +
            'px;left:' + MATRIX_PADDING_OFFSET + 'px;'
      };
    };

    const matrix = [];
    if (expanded) {
      // Create the column header if expanded.
      const header = actualClasses.reduce((acc, column, index) => {
        acc.push(makeHeader(column, ['col']));
        return acc;
      }, [{'widget': true}]);
      matrix.push(header);
    }

    predictedClasses.forEach(rowClass => {
      const row = [];
      if (expanded) {
        row.push(makeHeader(rowClass, ['row']));
      }

      actualClasses.forEach(columnClass => {
        row.push(this.makeCell_(columnClass, rowClass));
      });

      matrix.push(row);
    });
    if (expanded) {
      matrix.push([makeLabel(predictedClasses.length)]);
    }
    return matrix;
  },

  /**
   * Creates the title for a cell.
   * @param {boolean} expanded
   * @param {string} value
   * @return {string|undefined}
   * @private
   */
  getCellTitle_: function(expanded, value) {
    return expanded ? 'Weight: ' + value + '.\nClick to get more details' :
                      undefined;
  },

  /**
   * Tap handler for a cell.
   * @param {!Event} e
   * @private
   */
  cellTapped_: function(e) {
    const targetCell = e['model']['cell'];
    if (this.expanded) {
      // Update selectedCell_ if we are already expanded.
      this.selectedCell_ = this.selectedCell_ == targetCell ? null : targetCell;
    } else {
      // Check to see if we should expand in place if not expanded.
      const event = this.fire(
          'expand-confusion-matrix', this.jsonData, {'cancelable': true});
      if (!event.defaultPrevented) {
        // If no other component is handling the expand-confusion-matrix event,
        // expand in palce.
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
    const sortByPredicted = e.srcElement.classList.contains('col');
    const classNames = this.classNames_.slice();
    // matrix[actual][predicted]
    const matrix = this.summary_.matrix;
    classNames.sort((a, b) => {
      const valA =
          matrix[sortByPredicted ? target : a][sortByPredicted ? a : target];
      const valB =
          matrix[sortByPredicted ? target : b][sortByPredicted ? b : target];
      return valA - valB;
    });
    if (sortByPredicted) {
      this.predictedClasses_ = classNames;
    } else {
      this.actualClasses_ = classNames;
    }
  },
});
