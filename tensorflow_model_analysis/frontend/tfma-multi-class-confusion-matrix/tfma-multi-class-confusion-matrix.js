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
import '../tfma-matrix/tfma-matrix.js';
import {template} from './tfma-multi-class-confusion-matrix-template.html.js';

/**
 * tfma-multi-class-confusion-matrix renders a matrix that can sorted by any
 * column or row.
 *
 * @polymer
 */
export class MultiClassConfusionMatrix extends PolymerElement {
  constructor() {
    super();
  }

  static get is() {
    return 'tfma-multi-class-confusion-matrix';
  }

  /** @return {!HTMLTemplateElement} */
  static get template() {
    return template;
  }

  /** @return {!PolymerElementProperties} */
  static get properties() {
    return {
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
      classNames_: {type: Array, computed: 'computeClassNames_(jsonData)'},

      /**
       * A summary of the raw data.
       * @private {!Object}
       */
      summary_:
          {type: Object, computed: ' computeSummary_(classNames_, jsonData)'},
    };
  }

  /**
   * Observer for the property data.
   * @param {string} serializedData
   * @private
   */
  dataChanged_(serializedData) {
    if (serializedData) {
      try {
        this.jsonData = /** @type {!Object} */ (JSON.parse(serializedData));
      } catch (e) {
      }
    }
  }

  /**
   * Determines the list of class names from parsed data.
   * @param {!Object} jsonData
   * @return {!Object}
   * @private
   */
  computeClassNames_(jsonData) {
    const classes = {};
    const entries = jsonData['entries'] || [];
    entries.forEach(entry => {
      // Track all predicted and actual classes.
      classes[entry['actualClass']] = 1;
      classes[entry['predictedClass']] = 1;
    });

    return Object.keys(classes);
  }

  /**
   * Determines the colors associated with all classes.
   * @param {!Array<string>} classNames
   * @return {!Object<string>}
   * @private
   */
  computeClassColors_(classNames) {
    const colors = {};
    classNames.forEach((className, index) => {
      colors[className] = 'c' + (index % 16);
    });
    return colors;
  }

  /**
   * Builds the summary object.
   * @param {!Array<string>} classNames
   * @param {!Object} jsonData
   * @return {!Object}
   * @private
   */
  computeSummary_(classNames, jsonData) {
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
      matrix[actualClass][predictedClass] = {
        'value': weight,
        'tooltip': 'Weight: ' + weight + '.\nClick to get more details',
        'details': 'Weight for ' + actualClass + ', ' + predictedClass +
            ' is ' + weight + '.',
      };
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
      'rows': rowSummary,
      'columns': columnSummary,
      'matrix': matrix,
      'weight': {
        'min': minWeight,
        'max': maxWeight,
        'total': totalWeight,
      }
    };
  }

  /**
   * Handler for expand event.
   * @param {!Event} e
   * @private
   */
  onExpand_(e) {
    e.stopPropagation();

    // Check to see if we should expand in place if not expanded.
    const event = new CustomEvent('expand-confusion-matrix', {
      'detail': this.jsonData,
      'cancelable': true,
      'bubbles': true,
      'composed': true,
    });
    this.dispatchEvent(event);
    if (event.defaultPrevented) {
      e.preventDefault();
    }
  }
}

customElements.define(
    'tfma-multi-class-confusion-matrix', MultiClassConfusionMatrix);
