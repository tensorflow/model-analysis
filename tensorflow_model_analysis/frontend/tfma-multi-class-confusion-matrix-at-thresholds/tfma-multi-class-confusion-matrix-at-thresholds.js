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
import {template} from './tfma-multi-class-confusion-matrix-at-thresholds-template.html.js';

import '@polymer/iron-collapse/iron-collapse.js';
import '@polymer/iron-icon/iron-icon.js';
import '@polymer/iron-icons/iron-icons.js';
import '@polymer/paper-dropdown-menu/paper-dropdown-menu.js';
import '@polymer/paper-icon-button/paper-icon-button.js';
import '@polymer/paper-item/paper-item.js';
import '@polymer/paper-listbox/paper-listbox.js';
import '@polymer/paper-radio-button/paper-radio-button.js';
import '@polymer/paper-radio-group/paper-radio-group.js';
import '@polymer/paper-tooltip/paper-tooltip.js';


/**
 * @enum {string}
 */
const FieldNames = {
  ACTUAL_CLASS_ID: 'actualClassId',
  ENTRIES: 'entries',
  FALSE_NEGATIVES: 'falseNegatives',
  FALSE_POSITIVES: 'falsePositives',
  MATRICES: 'matrices',
  NUM_WEIGHTED_EXAMPLES: 'numWeightedExamples',
  POSITIVES: 'positives',
  PREDICTED_CLASS_ID: 'predictedClassId',
  THRESHOLD: 'threshold',
  TRUE_NEGATIVES: 'trueNegatives',
  TRUE_POSITIVES: 'truePositives',
};

/**
 * @enum {string}
 */
const SortBy = {
  ALPHABETICAL: 'a',
  FALSE_POSITIVES: 'fp',
  FALSE_NEGATIVES: 'fn',
  NO_PREDICTION: 'np',
  POSITIVES: 'p',
  TRUE_POSITIVES: 'tp',
};

/**
 * @typedef {{
 *   positives: number,
 *   truePositives: number,
 *   falsePositives: number,
 *   falseNegatives: number,
 * }}
 */
let SummaryCell;

/**
 * @typedef {{
 *   entries: !Object<!SummaryCell>,
 *   totalPositives: number,
 *   totalTruePositives: number,
 *   totalFalsePositives: number,
 *   totalFalseNegatives: number,
 *   totalNoPrediction: number,
 * }}
 */
let SummaryRow;

/**
 * @typedef {{
 *  matrix: !Object<!SummaryRow>,
 * }}
 */
let SummaryMatrix;


/**
 * @typedef {{
 *   'cell': boolean,
 *   'positives': (string|number),
 *   'falsePositives': (string|number),
 *   'falseNegatives': (string|number),
 *   'positiveClasses':string,
 *   'falsePositiveClasses':string,
 *   'falseNegativeClasses':string,
 * }}
 */
let DataCell;

/**
 * @typedef {{
 *   'classId': boolean,
 *   'text': (string|number),
 *   'headerClasses': (string|undefined)
 * }}
 */
let HeaderCell;

/**
 * @typedef {!Array<!Array<(!DataCell|!HeaderCell)>>}
 */
let RenderMatrix;

/**
 * @typedef {{
 *   actualClass: string,
 *   container: !Element,
 *   predictedClass: string,
 * }}
 */
let PointerEventData;

const ModeText = {
  [SortBy.POSITIVES]: 'Total prediction count',
  [SortBy.FALSE_POSITIVES]: 'Incorrect prediciton counts',
  [SortBy.FALSE_NEGATIVES]: 'False negative count',
};

const SortText = {
  [SortBy.ALPHABETICAL]: 'Alphabetical',
  [SortBy.FALSE_POSITIVES]: 'Incorrect prediction count',
  [SortBy.FALSE_NEGATIVES]: 'False negatives',
  [SortBy.NO_PREDICTION]: 'No prediction',
  [SortBy.POSITIVES]: 'Total prediction count',
  [SortBy.TRUE_POSITIVES]: 'Correct prediction count',
};

const NO_PREDICTION_CLASS_ID = -1;
const NO_PREDICTION_CLASS_ID_STRING = NO_PREDICTION_CLASS_ID + '';

const TOOLTIP_REMOVAL_TIMEOUT_MS = 100;

/**
 * tfma-multi-class-confusion-matrix-at-thresholds visualizes single-label /
 * multi-label multi-class confusion matrix at different thresholds.
 *
 * @polymer
 */
export class MultiClassConfusionMatrixAtThresholds extends PolymerElement {
  constructor() {
    super();
  }

  static get is() {
    return 'tfma-multi-class-confusion-matrix-at-thresholds';
  }

  /** @return {!HTMLTemplateElement} */
  static get template() {
    return template;
  }

  /** @return {!PolymerElementProperties} */
  static get properties() {
    return {
      /**
       * The input data. It should be of type MultiClassConfusionMatrix.
       * @type {!Object}
       */
      data: {type: Object},

      /**
       * A map where the keys are the class names in string and the values are
       * class ids in number.
       * @type {!Object<number>}
       */
      classNames: {type: Object, value: () => ({})},

      /**
       * A map where the keys are the class ids and the values are the
       * corresponding class name strings.
       * @private {!Object<string>}
       */
      displayNames_: {
        type: Object,
        computed: 'computeDisplayNames_(classNames, availableClassIds_)',
      },

      /**
       * Whether the data contains results for a multi-label multi-class model
       * or single-label multi-class model.
       */
      multiLabel: {type: Boolean, value: false},

      /**
       * An array of thresholds.
       * @private {!Array<number>}
       */
      thresholds_: {
        type: Array,
        computed: 'computeThresholds_(data)',
        observer: 'thresholdsChanged_',
      },

      /**
       * The selected threshold.
       * @private {number}
       */
      selectedThreshold_: {type: String},

      /**
       * An array of class ids.
       * @private {!Array<string>}
       */
      availableClassIds_: {
        type: Array,
        computed: 'computeAvailableClassIds_(data)',
      },

      /**
       * The key is the threshold and the value is the summary built from input
       * data.
       * @private {!Object<!SummaryMatrix>}
       */
      summary_: {
        type: Object,
        computed: 'computeSummary_(data, multiLabel, availableClassIds_)'
      },

      /**
       * The matrix with the selected threshold.
       * @private {!SummaryMatrix}
       */
      selectedMatrix_: {
        type: Object,
        computed: 'computeSelectedMatrix_(summary_, selectedThreshold_)',
      },

      /**
       * A list of class ids sorted according to the user's choice.
       * @private {!Array<string>}
       */
      sortedClassIds_: {
        type: Array,
        computed: 'computeSortedClassIds_(' +
            'selectedMatrix_, availableClassIds_, displayNames_, sort_)',
        observer: 'sortedClassIdsChanged_'
      },

      /**
       * The 2d array used for building the matrix.
       * @private {!Array<!Array<!Object>>}
       */
      matrix_: {
        type: Array,
        computed: 'computeMatrix_(' +
            'selectedMatrix_, sortedClassIds_, numberOfClassesShown_, ' +
            'displayNames_, sort_, showPercentage_)'
      },

      /**
       * The mode in which the matrix should be visualized.
       * @private {string}
       */
      mode_: {type: String, value: SortBy.POSITIVES},

      /**
       * The method in which classes in the matrix should be sorted.
       * @private {!SortBy}
       */
      sort_: {type: String, value: SortBy.ALPHABETICAL},

      /**
       * Whether to show percentage or raw value.
       * @private {boolean}
       */
      showPercentage_: {type: Boolean, value: false},

      /**
       * The number of classes to show. This helps keep the component performant
       * by ommitting some classes from being rendered.
       * @private {number}
       */
      numberOfClassesShown_: {type: Number},

      /**
       * Whether the controls are visible.
       * @private {boolean}
       */
      controlOpened_: {
        type: Boolean,
        value: true,
      },

      compact_: {
        type: Boolean,
        value: false,
      },

      sortBy_: {
        type: Object,
        value: {
          'ALPHABETICAL': SortBy.ALPHABETICAL,
          'POSITIVES': SortBy.POSITIVES,
          'FALSE_POSITIVES': SortBy.FALSE_POSITIVES,
          'FALSE_NEGATIVES': SortBy.FALSE_NEGATIVES,
          'NO_PREDICTION': SortBy.NO_PREDICTION,
          'TRUE_POSITIVES': SortBy.TRUE_POSITIVES,
        }
      },

      /**
       * The predicted class of the predicted cell.
       * @private {string}
       */
      selectedPrecitedClass_: {type: String},

      /**
       * The predicted class of the selected cell.
       * @private {string}
       */
      selectedActualClass_: {type: String},

      /**
       * The total value for sorting for the selected row.
       * @private {number}
       */
      selectedRowTotal_: {type: Number},

      /**
       * The value selected cell.
       * @private {number}
       */
      selectedCellValue_: {type: Number},

      /**
       * The percentage of the selected cell compared against the sort value in
       * the toltip.
       * @private {string}
       */
      selectedCellPercentage_: {type: String},

      /**
       * Whether the tooltip should be visible.
       * @private {boolean}
       */
      showTooltip_: {type: Boolean, value: false},

      /**
       * The id of the seTimeout that removes the tooltip from the UI.
       * @private {number}
       */
      removeTooltipTimeout_: {type: Number, value: 0},

      /**
       * The anchor element for the tooltip.
       * @private {?Element}
       */
      anchor_: {
        type: Object,
        value: null,
      },

      /**
       * The target anchor element.
       * @private {?Element}
       */
      targetAnchor_: {
        type: Object,
        value: null,
      },
    };
  }

  /**
   * Extracts all the thresholds in the data and sort it in ascending order.
   * @param {!Object} data
   * @return {!Array<number>}
   * @private
   */
  computeThresholds_(data) {
    const matrices = data[FieldNames.MATRICES] || [];
    return matrices.map((matrix) => (matrix[FieldNames.THRESHOLD] || 0))
        .sort((a, b) => a - b);
  }

  /**
   * Observer for the property thresholds.
   * @param {!Array<number>} thresholds
   * @private
   */
  thresholdsChanged_(thresholds) {
    if (thresholds && thresholds.length) {
      // Initialize selectedThreshold_ to the median threshold.
      this.selectedThreshold_ = thresholds[Math.floor(thresholds.length / 2)];
    }
  }

  /**
   * Extracts all class ids in data.
   * @param {!Object} data
   * @return {!Array<string>}
   * @private
   */
  computeAvailableClassIds_(data) {
    const matrices = data[FieldNames.MATRICES];
    const allClasses = {};
    matrices.forEach((matrix) => {
      const entries = matrix[FieldNames.ENTRIES];
      entries.forEach((entry) => {
        const predictedClassId = entry[FieldNames.PREDICTED_CLASS_ID] || 0;
        const actualClassId = entry[FieldNames.ACTUAL_CLASS_ID] || 0;
        allClasses[predictedClassId] = 1;
        allClasses[actualClassId] = 1;
      });
    });
    return Object.keys(allClasses);
  }

  /**
   * Parses the data object and constructs the summary needed for rendering.
   * @param {!Object|undefined} data
   * @param {boolean} multiLabel
   * @param {!Array<string>|undefined} classIds
   * @return {!Object<!SummaryMatrix>|undefined}
   * @private
   */
  computeSummary_(data, multiLabel, classIds) {
    if (!data || !classIds) {
      return undefined;
    }
    const matrices = data[FieldNames.MATRICES];
    const summary = {};
    for (let i = 0; i < matrices.length; i++) {
      const matrix = matrices[i];
      summary[matrix[FieldNames.THRESHOLD] || 0] = this.computeSummaryMatrix_(
          matrix[FieldNames.ENTRIES] || [], multiLabel, classIds);
    }
    return summary;
  }

  /**
   * Parses the data object and constructs the summary needed for rendering.
   * @param {!Array<!Object>|undefined} entries
   * @param {boolean} multiLabel
   * @param {!Array<string>|undefined} classIds
   * @return {!SummaryMatrix|undefined}
   * @private
   */
  computeSummaryMatrix_(entries, multiLabel, classIds) {
    if (!entries || !classIds) {
      return undefined;
    }

    const matrix = {};
    for (const entry of entries) {
      const actual = entry[FieldNames.ACTUAL_CLASS_ID] || 0;
      const predicted = entry[FieldNames.PREDICTED_CLASS_ID] || 0;
      if (!matrix[actual]) {
        matrix[actual] = {entries: {}};
      }
      const row = matrix[actual].entries;
      const isDiagonal = predicted == actual;
      const truePositives = entry[FieldNames.TRUE_POSITIVES] || 0;
      const falsePositives = entry[FieldNames.FALSE_POSITIVES] || 0;
      const numWeightedExamples = entry[FieldNames.NUM_WEIGHTED_EXAMPLES] || 0;

      const truePositivesToUse =
          multiLabel ? truePositives : (isDiagonal ? numWeightedExamples : 0);
      const falsePositivesToUse =
          multiLabel ? falsePositives : (isDiagonal ? 0 : numWeightedExamples);
      const falseNegatives = entry[FieldNames.FALSE_NEGATIVES] || 0;

      row[predicted] = {
        positives: truePositivesToUse + falsePositivesToUse,
        truePositives: truePositivesToUse,
        falsePositives: falsePositivesToUse,
        falseNegatives: falseNegatives,
      };
    }

    classIds.forEach((rowId) => {
      if (rowId == NO_PREDICTION_CLASS_ID) {
        // Do not create a row for no prediction.
        return;
      }

      // Fill in holes in the matrix.
      if (!matrix[rowId]) {
        matrix[rowId] = {entries: {}};
      }
      const currentRow = matrix[rowId].entries;
      let totalPositives = 0;
      let totalTruePositives = 0;
      let totalFalsePositives = 0;
      let totalFalseNegatives = 0;
      let noPrediction = 0;

      for (let columnId of /** @type {!Array<string>} */ (classIds)) {
        if (!currentRow[columnId]) {
          currentRow[columnId] = {
            positives: 0,
            truePositives: 0,
            falsePositives: 0,
            falseNegatives: 0,
          };
        }

        // Skip no prediciton.
        if (columnId == NO_PREDICTION_CLASS_ID) {
          noPrediction = currentRow[columnId].falsePositives;
          continue;
        }
        totalPositives += currentRow[columnId].positives;
        totalTruePositives += currentRow[columnId].truePositives;
        totalFalsePositives += currentRow[columnId].falsePositives;
        totalFalseNegatives += currentRow[columnId].falseNegatives;
      }

      matrix[rowId].totalPositives = totalPositives;
      matrix[rowId].totalTruePositives = totalTruePositives;
      matrix[rowId].totalFalsePositives = totalFalsePositives;
      matrix[rowId].totalFalseNegatives = totalFalseNegatives;
      matrix[rowId].totalNoPrediction = noPrediction;
    });
    return /** @type {!SummaryMatrix} */ (matrix);
  }

  /**
   * @param {!Object<!SummaryMatrix>|undefined} summary
   * @param {number} threshold
   * @return {!SummaryMatrix|undefined} The summary matrix for the given
   *     threshold.
   * @private
   */
  computeSelectedMatrix_(summary, threshold) {
    return summary && summary[threshold];
  }

  /**
   * Given the class id and the sort method, extracts the value that should be
   * used in sorting.
   * @param {!SummaryMatrix} summary
   * @param {string} classId
   * @param {!SortBy} sort
   * @return {number}
   * @private
   */
  getSortValue_(summary, classId, sort) {
    const row = summary[classId];
    if (sort == SortBy.TRUE_POSITIVES) {
      return row.totalTruePositives;
    } else if (sort == SortBy.FALSE_POSITIVES) {
      return row.totalFalsePositives;
    } else if (sort == SortBy.NO_PREDICTION) {
      return row.totalNoPrediction;
    } else if (sort == SortBy.FALSE_NEGATIVES) {
      return row.totalFalseNegatives;
    } else {
      // For alphabetical or any unexpected values, use the total prediction
      // count.
      return row.totalPositives;
    }
  }

  /**
   * Sorts the list of class ids based on the method specified by the user.
   * @param {!SummaryMatrix|undefined} summary
   * @param {!Array<number>|undefined} classIds
   * @param {!Object<string>} displayNames
   * @param {!SortBy} sort
   * @return {!Array<string>|undefined}
   * @private
   */
  computeSortedClassIds_(summary, classIds, displayNames, sort) {
    if (!summary || !classIds || !displayNames) {
      return undefined;
    }

    return [...classIds].sort((classA, classB) => {
      // Put no prediction to the back.
      if (classA == NO_PREDICTION_CLASS_ID) {
        return 1;
      }
      if (classB == NO_PREDICTION_CLASS_ID) {
        return -1;
      }
      if (sort == SortBy.ALPHABETICAL) {
        return displayNames[classA] > displayNames[classB] ? 1 : -1;
      }
      const countA = this.getSortValue_(
          /** @type {!SummaryMatrix} */ (summary), classA, sort);
      const countB = this.getSortValue_(
          /** @type {!SummaryMatrix} */ (summary), classB, sort);
      return countB - countA;
    });
  }

  /**
   * Creates a HeaderCell for rendering.
   * @param {string|number} text
   * @param {string=} classnames
   * @return {!HeaderCell}
   * @private
   */
  makeHeaderCell_(text, classnames) {
    return {'classId': true, 'text': text, 'headerClasses': classnames};
  }

  /**
   * Creates a DataCell for rendering.
   * @param {number|string} positives
   * @param {number|string} falsePositives
   * @param {number|string} falseNegatives
   * @param {string} positiveClassString
   * @param {string} falsePositiveClassString
   * @param {string} falseNegativeClassString
   * @return {!DataCell}
   * @private
   */
  makeDataCell_(
      positives, falsePositives, falseNegatives, positiveClassString,
      falsePositiveClassString, falseNegativeClassString) {
    return {
      'cell': true,
      'positives': positives,
      'falsePositives': falsePositives,
      'falseNegatives': falseNegatives,
      'positiveClasses': 'positive ' + positiveClassString,
      'falsePositiveClasses': 'false-positive ' + falsePositiveClassString,
      'falseNegativeClasses': 'false-negative ' + falseNegativeClassString,
    };
  }

  /**
   * Builds the matrix to display.
   * @param {!SummaryMatrix|undefined} summaryMatrix
   * @param {!Array<string>|undefined} classIds
   * @param {number} classesToShow
   * @param {!Object<string>} displayNames
   * @param {!SortBy} sort
   * @param {boolean} showPercentage
   * @return {!RenderMatrix|undefined}
   * @private
   */
  computeMatrix_(
      summaryMatrix, classIds, classesToShow, displayNames, sort,
      showPercentage) {
    if (!summaryMatrix || !classIds || !displayNames) {
      return undefined;
    }

    // No prediction counts are already visible as the last column in single
    // label multi-class so we should skip it.
    const showSortColumn =
        sort != SortBy.NO_PREDICTION && sort != SortBy.ALPHABETICAL;

    // Creates matrix row by row.
    // First, header row consists of class ids.
    const header = [
      this.makeHeaderCell_('Class Id'),
      this.makeHeaderCell_('Total', 'guide'),
    ];
    classIds.forEach((classId, index) => {
      header.push(this.makeHeaderCell_(displayNames[classId]));
    });

    if (showSortColumn) {
      header.push(this.makeHeaderCell_('Sort By', 'guide'));
    }
    const matrix = [header];
    const determineClass = (rowId, columnId, value, mode) => {
      if (value) {
        const baseline = this.getSortValue_(
            /** @type {!SummaryMatrix} */ (summaryMatrix), rowId, mode);
        const scale = Math.round(Math.min(value / baseline, 1) * 16);
        return ' b' + scale + (rowId == columnId ? ' diag' : '');
      }
      return '';
    };
    const formatNumber = (value) => {
      if (value < 1000) {
        return value;
      } else {
        const base = [1, 1e3, 1e6, 1e9, 1e12, 1e15, 1e18];
        const unit = ['', 'k', 'M', 'G', 'T', 'P', 'E'];
        const index = Math.floor(Math.log10(value) / 3);
        return (value / base[index]).toFixed(2) + unit[index];
      }
    };
    const formatValue = (value, baseline, showPercentage) => {
      baseline = baseline || 1;
      if (value && showPercentage) {
        return (value / baseline * 100).toFixed(1) + '%';
      } else {
        return formatNumber(value);
      }
    };

    // Then, fill in each row.
    classIds.forEach((rowId, index) => {
      if (index >= classesToShow || rowId == NO_PREDICTION_CLASS_ID) {
        return;
      }
      const sourceRow = summaryMatrix[rowId];
      const row = [
        this.makeHeaderCell_(displayNames[rowId]),
        this.makeDataCell_(
            formatNumber(sourceRow.totalPositives),
            formatNumber(sourceRow.totalFalsePositives),
            formatNumber(sourceRow.totalFalseNegatives), 'guide header',
            'guide header', 'guide header')
      ];
      const baseline = this.getSortValue_(summaryMatrix, rowId, sort);

      classIds.forEach((columnId, index) => {
        if (index >= classesToShow && columnId != NO_PREDICTION_CLASS_ID) {
          return;
        }
        const positives = sourceRow.entries[columnId].positives;
        const falsePositives = sourceRow.entries[columnId].falsePositives;
        const falseNegatives = sourceRow.entries[columnId].falseNegatives;
        const formattedPositives =
            formatValue(positives, baseline, showPercentage);
        const formattedFalsePositives =
            formatValue(falsePositives, baseline, showPercentage);
        const formattedFalseNegatives =
            formatValue(falseNegatives, baseline, showPercentage);
        if (columnId == NO_PREDICTION_CLASS_ID) {
          row.push(this.makeDataCell_(
              formattedPositives || '', formattedFalsePositives || '',
              formattedFalseNegatives || '', 'guide header', 'guide header',
              'guide header'));
        } else {
          row.push(this.makeDataCell_(
              formattedPositives || '', formattedFalsePositives || '',
              formattedFalseNegatives || '',
              determineClass(rowId, columnId, positives, SortBy.POSITIVES),
              determineClass(
                  rowId, columnId, falsePositives, SortBy.FALSE_POSITIVES),
              determineClass(
                  rowId, columnId, falseNegatives, SortBy.FALSE_NEGATIVES)));
        }
      });

      if (showSortColumn) {
        row.push(this.makeHeaderCell_(
            formatNumber(this.getSortValue_(summaryMatrix, rowId, sort)),
            'guide'));
      }
      matrix.push(row);
    });
    return matrix;
  }

  /**
   * @param {string} mode
   * @param {boolean} compact
   * @return {string} The class names that should be applied to the table
   *     container.
   * @private
   */
  determineTableClass_(mode, compact) {
    return [
      'container',
      mode == SortBy.POSITIVES ?
          '' :
          ('show-false-' +
           (mode == SortBy.FALSE_POSITIVES ? 'positive' : 'negative')),
      compact ? 'compact' : ''
    ].join(' ');
  }

  /**
   * Observer for the property sortedClassIds_.
   * @param {!Array<string>} classIds
   * @private
   */
  sortedClassIdsChanged_(classIds) {
    const hasNoPrediction =
        classIds.indexOf(NO_PREDICTION_CLASS_ID_STRING) >= 0;
    const count = classIds.length - (hasNoPrediction ? 1 : 0);
    this.numberOfClassesShown_ = Math.min(count, 64);
  }

  /**
   * Toggles the control UI.
   * @private
   */
  toggleControl_() {
    this.controlOpened_ = !this.controlOpened_;
  }

  /**
   * @param {boolean} open
   * @return {string} The icon for the button that collapse / expand the
   *     control UI.
   * @private
   */
  getIcon_(open) {
    return open ? 'expand-less' : 'expand-more';
  }

  /**
   * Extracts information from pointer events we are interested in. Null if we
   * are not interested in this event.
   * @param {!Event} event
   * @return {?PointerEventData}
   * @private
   */
  parsePointerEvent_(event) {
    let container = event.target;
    while (container && !container.classList.contains('cell')) {
      container = container.parentElement;
    }
    if (container) {
      // Minus one to skip header row.
      const row = parseInt(container.getAttribute('row'), 10) - 1;
      // Minus two to skip class id and total count column
      const column = parseInt(container.getAttribute('column'), 10) - 2;
      if (row >= 0 && column >= 0 && column < this.numberOfClassesShown_) {
        return {
          actualClass: this.sortedClassIds_[row],
          container: container,
          predictedClass: this.sortedClassIds_[column],
        };
      }
    }
    return null;
  }

  /**
   * Checks if teh event originates from the tooltip.
   * @param {!Event} event
   * @return {boolean}
   * @private
   */
  eventFromTooltip_(event) {
    let element = event.target;
    while (element && element != this.$.tooltip) {
      element = element.parentElement;
    }
    return !!element;
  }

  /**
   * Handler for pointerover event.
   * @param {!Event} event
   * @private
   */
  onPointerOver_(event) {
    // Do nothing if entering tooltip div.
    if (this.eventFromTooltip_(event)) {
      return;
    }

    const pointerEvent = this.parsePointerEvent_(event);
    if (pointerEvent) {
      if (this.removeTooltipTimeout_) {
        clearTimeout(this.removeTooltipTimeout_);
        this.removeTooltipTimeout_ = 0;
      }

      const getDisplayValue = (actualClass, predictedClass, mode) => {
        const entry = this.selectedMatrix_[pointerEvent.actualClass]
                          .entries[pointerEvent.predictedClass];
        if (mode == SortBy.POSITIVES) {
          return entry.positives;
        } else if (mode == SortBy.FALSE_POSITIVES) {
          return entry.falsePositives;
        } else if (mode == SortBy.FALSE_NEGATIVES) {
          return entry.falseNegatives;
        }
      };

      const sortValue = this.getSortValue_(
          this.selectedMatrix_, pointerEvent.actualClass, this.sort_);
      this.updateAnchor_(pointerEvent.container);
      this.selectedPrecitedClass_ =
          this.displayNames_[pointerEvent.predictedClass];
      this.selectedActualClass_ = this.displayNames_[pointerEvent.actualClass];
      this.selectedRowTotal_ = sortValue;
      this.selectedCellValue_ = getDisplayValue(
          pointerEvent.actualClass, pointerEvent.predictedClass, this.mode_);
      this.selectedCellPercentage_ =
          (this.selectedCellValue_ / this.selectedRowTotal_ * 100).toFixed(2);
    } else {
      this.maybeRemoveTooltip_();
    }
  }

  /**
   * Updates the anchor of the tooltip.
   * @param {?Element} targetAnchor
   * @private
   */
  updateAnchor_(targetAnchor) {
    if (targetAnchor == this.anchor_) {
      return;
    }

    if (this.anchor_) {
      this.anchor_.classList.remove('anchor');
      this.anchor_ = null;
    }

    if (targetAnchor) {
      targetAnchor.classList.add('anchor');
      targetAnchor.appendChild(this.$.tooltip);
      this.anchor_ = targetAnchor;
    }

    this.showTooltip_ = !!targetAnchor;
  }

  /**
   * Schedule a callback to remove the tooltip if one has not been set yet.
   * @private
   */
  maybeRemoveTooltip_() {
    if (!this.removeTooltipTimeout_) {
      this.removeTooltipTimeout_ = setTimeout(() => {
        this.updateAnchor_(null);
      }, TOOLTIP_REMOVAL_TIMEOUT_MS);
    }
  }

  /**
   * @param {string} mode
   * @return {string} The text for the display mode if available; empty
   *     string, otherwise.
   * @private
   */
  getModeText_(mode) {
    return ModeText[mode] || '';
  }

  /**
   * @param {string} sort
   * @return {string} The text for the sorting mode if available; empty
   *     string, otherwise.
   * @private
   */
  getSortText_(sort) {
    return SortText[sort] || '';
  }

  /**
   * Creates a map from class id to the class name that should be displayed.
   * @param {!Object<number>} classNames
   * @param {!Array<string>|undefined} classIds
   * @return {!Object<string>|undefined}
   * @private
   */
  computeDisplayNames_(classNames, classIds) {
    if (!classIds) {
      return undefined;
    }

    const displayNames = {
      [NO_PREDICTION_CLASS_ID]: SortText[SortBy.NO_PREDICTION],
    };

    for (let key in classNames) {
      displayNames[classNames[key]] = key;
    }

    classIds.forEach((classId) => {
      if (!displayNames[classId]) {
        displayNames[classId] = classId;
      }
    });
    return displayNames;
  }
}

customElements.define(
    'tfma-multi-class-confusion-matrix-at-thresholds',
    MultiClassConfusionMatrixAtThresholds);
