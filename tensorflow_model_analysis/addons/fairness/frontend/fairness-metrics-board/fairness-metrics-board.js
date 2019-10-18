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
import {template} from './fairness-metrics-board-template.html.js';

import '@polymer/paper-dropdown-menu/paper-dropdown-menu.js';
import '@polymer/paper-item/paper-item.js';
import '@polymer/paper-listbox/paper-listbox.js';
import '../fairness-metric-summary/fairness-metric-summary.js';
import '../fairness-privacy-container/fairness-privacy-container.js';

/**
 * Error key present for the slices which are omitted because of privacy
 * concerns.
 * @private {string}
 * @const
 */
const OMITTED_SLICE_ERROR_KEY = '__ERROR__';

/**
 * @polymer
 */
export class FairnessMetricsBoard extends PolymerElement {
  constructor() {
    super();
  }

  static get is() {
    return 'fairness-metrics-board';
  }

  /** @return {!HTMLTemplateElement} */
  static get template() {
    return template;
  }

  /** @return {!PolymerElementProperties} */
  static get properties() {
    return {
      /** @type {!Array<!Object>} */
      data: {type: Array},

      /** @type {string} */
      weightColumn: {type: String},

      /** @type {!Array<string>} */
      metrics: {type: Array},

      /** @type {!Array<string>} */
      thresholds: {type: Array, observer: 'thresholdsChanged_'},

      /** @type {string} */
      baseline_: {type: String, computed: 'computeBaseline_(slices_)'},

      /**
       * The list of all slice names.
       * @private {!Array<string>}
       */
      slices_: {type: Array, computed: 'computeSlices_(data)'},

      /**
       * The list of all slices omitted to ensure privacy.
       * @private {!Array<string>}
       */
      omittedSlices_: {type: Array, computed: 'computeOmittedSlices_(data)'},


      /**
       * A flag used to update the selected thresholds displayed in the UI.
       * @private {boolean}
       */
      thresholdsMenuOpened_:
          {type: Boolean, observer: 'thresholdsMenuOpenedChanged_'},

      /**
       * The list of seleted thresholds.
       * @private {!Array<string>}
       */
      selectedThresholds_: {type: Array},

      /**
       * A copy of selectedThresholds_ to push data binding.
       * @private {!Array<string>}
       */
      thresholdsToPlot_: {
        type: Array,
        computed: 'computeThresholdsToPlot_(selectedThresholds_.length)'
      },
    };
  }


  /**
   * Extracts the baseline slice name.
   * @param {!Array<string>} slices
   * @return {string|undefined}
   * @private
   */
  computeBaseline_(slices) {
    if (!slices) {
      return;
    }
    return slices[0];
  }

  /**
   * Extracts the names of the slices from the data.
   * @param {!Array<!Object>} data
   * @return {!Array<string>|undefined}
   * @private
   */
  computeSlices_(data) {
    if (!data) {
      return;
    }
    return data.filter(d => !d['metrics'][OMITTED_SLICE_ERROR_KEY])
        .map(d => d['slice']);
  }

  /**
   * Extracts the names of the slices omitted to ensure privacy.
   * @param {!Array<!Object>} data
   * @return {!Array<string>|undefined}
   * @private
   */
  computeOmittedSlices_(data) {
    if (!data) {
      return;
    }
    return data.filter(d => d['metrics'][OMITTED_SLICE_ERROR_KEY])
        .map(d => d['slice']);
  }

  /**
   * Determines the thresholds to use in metric summary.
   * @param {number} unused The number of thresholds selected by the user.
   * @return {!Array<string>}
   * @private
   */
  computeThresholdsToPlot_(unused) {
    return this.selectedThresholds_.slice();
  }

  /**
   * Observer for property thresholds_. Automatically selects the median
   * thresholds as default.
   * @param {!Array<string>} thresholds
   * @private
   */
  thresholdsChanged_(thresholds) {
    const sortedThresholds =
        thresholds
            .map(threshold => {
              return {
                threshold: threshold,
                numericalValue: parseFloat(threshold)
              };
            })
            .sort((a, b) => a.numericalValue - b.numericalValue);

    if (thresholds.length) {
      this.$.thresholdsList.select(
          sortedThresholds[Math.floor(sortedThresholds.length / 2)].threshold);
    }
  }

  /**
   * Observer for thresholdsMenuOpened_ flag. Updates the string for the
   * thresholds selected.
   * @param {boolean} open
   * @private
   */
  thresholdsMenuOpenedChanged_(open) {
    if (!open) {
      setTimeout(() => {
        // HACK: Fire off a fake iron-select event with fake label with multiple
        // selected thresholds so that they are displayed in the menu. In case
        // none is selected, use ' '.
        this.$.thresholdsList.fire(
            'iron-select',
            {'item': {'label': this.thresholdsToPlot_.join(', ') || ' '}});
      }, 0);
    }
  }
}

customElements.define('fairness-metrics-board', FairnessMetricsBoard);
