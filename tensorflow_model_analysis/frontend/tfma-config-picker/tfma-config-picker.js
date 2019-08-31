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

import {template} from './tfma-config-picker-template.html.js';

import '../tfma-multi-select/tfma-multi-select.js';

/**
 * tfma-config-picker allows the user to select the desired config from a nested
 * structure.
 *
 * @polymer
 */
export class ConfigPicker extends PolymerElement {
  constructor() {
    super();
  }

  static get is() {
    return 'tfma-config-picker';
  }

  /** @return {!HTMLTemplateElement} */
  static get template() {
    return template;
  }

  /** @return {!PolymerElementProperties} */
  static get properties() {
    return {
      /**
       * A map where the keys are output names and the values are list of
       * classes for that output.
       * @type {!Object<!Array<number|string>>}
       */
      allConfigs: {type: Object},

      /**
       * The list of all outputs.
       * @private {!Array<string>}
       */
      availableOutputs_: {
        type: Array,
        computed: 'computeAvailableOutputs_(allConfigs)',
        observer: 'availableOutputsChanged_',
      },

      /**
       * The list of selected outputs.
       * @private {!Array<string>}
       */
      selectedOutputs_: {
        type: Array,
        value: () => [],
      },

      availableCombos_: {
        type: Array,
        computed: 'computeAvailableCombos_(allConfigs, selectedOutputs_)',
      },

      /**
       * The values to be rendered. Each element represents a row. If the
       * original data is not a 1d array, the values are turned into a string
       * like "[[1, 2, 3], [4,5,6]]".
       * @type {!Array<number|string>}
       */
      availableClasses_: {
        type: Array,
        computed: 'computeAvailableClasses_(availableCombos_)',
        observer: 'availableClassesChanged_',
      },

      /**
       * The list of selected classes.
       * @private {!Array<string>}
       */
      selectedClasses_: {type: Array},

      /**
       * An object representing selected config. The keys are the name of the
       * output and the values are the class ids selected for that output.
       * @private {!Object<number|string>}
       */
      selectedConfigs: {
        type: Object,
        computed: 'computeSelectedConfigs_(' +
            'availableCombos_, availableClasses_, selectedClasses_)',
        notify: true,
      }
    };
  }

  /**
   * Determines the list of available outputs.
   * @param {!Object} allConfigs
   * @return {!Array<string>}
   * @private
   */
  computeAvailableOutputs_(allConfigs) {
    return Object.keys(allConfigs).sort();
  }

  /**
   * Observer for the property data. It will update the arrayData property.
   * @param {!Array<string>} availableOutputs
   * @private
   */
  availableOutputsChanged_(availableOutputs) {
    // Clears selected classes when available outputs changed.
    this.selectedClasses_ = [];

    // If there is only one output, select it automatically.
    if (availableOutputs.length == 1) {
      this.selectedOutputs_ = [availableOutputs[0]];
    }
  }

  /**
   * Determines all available combinations of configurations.
   * @param {!Object<!Array<string|number>>} allConfigs
   * @param {!Array<string>} selectedOutputs
   * @return {!Array<!Object>}
   * @private
   */
  computeAvailableCombos_(allConfigs, selectedOutputs) {
    const availableCombo = [];
    // If there are more than one output, we should prepend output name to help
    // differentiate bewteen the same class ids from different outputs.
    const prependOutputName = selectedOutputs.length > 1;

    selectedOutputs.forEach(outputName => {
      const classes = allConfigs[outputName] || [];
      classes.forEach(classId => {
        availableCombo.push({
          outputName: outputName,
          classId: classId,
          prependOutputName: prependOutputName,
        });
      });
    });

    return availableCombo;
  }

  /**
   * Builds the list of available classes from all possible configuration.
   * @param {!Array<!Object>} availableCombos
   * @return {!Array<string>}
   * @private
   */
  computeAvailableClasses_(availableCombos) {
    const maybeAddOutputPrefix = (combo) => combo.prependOutputName ?
        (combo.outputName || 'Empty Output') + ', ' :
        '';
    const determineClassIdToDisplay = (combo) =>
        (combo.classId == '' ? 'No class' : combo.classId);

    return availableCombos.reduce((acc, combo) => {
      acc.push(maybeAddOutputPrefix(combo) + determineClassIdToDisplay(combo));
      return acc;
    }, []);
  }

  /**
   * Observer for property availableClassses.
   * @param {!Array<string>} availableClasses
   * @private
   */
  availableClassesChanged_(availableClasses) {
    // If there is only one class, select it automatically.
    if (availableClasses.length == 1) {
      this.selectedClasses_ = [availableClasses[0]];
    }
  }

  /**
   * Builds the selected config from what available and what's selected.
   * @param {!Array<!Object>} availableCombos
   * @param {!Array<string>} availableClasses
   * @param {!Array<string>} selectedClasses
   * @return {!Object<!Object<string>>}
   * @private
   */
  computeSelectedConfigs_(availableCombos, availableClasses, selectedClasses) {
    const config = {};
    if (availableCombos && availableClasses && selectedClasses) {
      selectedClasses.forEach(selectedClass => {
        // Use index lookup to avoid parsing the generated strings which might
        // contain output names.
        const index = availableClasses.indexOf(selectedClass);
        const selectedCombo = availableCombos[index];
        if (selectedCombo) {
          const outputName = selectedCombo.outputName;
          if (config[outputName]) {
            config[outputName].push(selectedCombo.classId);
          } else {
            config[outputName] = [selectedCombo.classId];
          }
        }
      });
    }
    return config;
  }

  /**
   * @param {!Array} array
   * @return {boolean} True if the given array has more than one element.
   * @private
   */
  show_(array) {
    return !!array && array.length > 1;
  }
}

customElements.define('tfma-config-picker', ConfigPicker);
