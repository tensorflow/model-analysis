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
const OMITTED_SLICE_ERROR_MESSAGE =
    'Example count for this slice key is lower than the minimum required value';

/**
 * Name of master slice containing all data.
 * @private {string}
 * @const
 */
const OVERALL_SLICE_KEY = 'Overall';

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
      evalName: {type: String},

      /** @type {!Array<!Object>} */
      dataCompare: {type: Array},

      /** @type {string} */
      evalNameCompare: {type: String},

      /** @type {string} */
      weightColumn: {type: String},

      /** @type {!Array<string>} */
      metrics: {type: Array},

      /** @type {string|undefined} */
      baseline_: {type: String, value: 'Overall'},

      /**
       * The list of all slice names.
       * @private {!Array<string>}
       */
      slices_: {
        type: Array,
        computed: 'computeSlices_(data)',
        observer: 'slicesChanged_'
      },

      /**
       * The list of all slices omitted to ensure privacy.
       * @private {!Array<string>}
       */
      omittedSlices_:
          {type: Array, computed: 'computeOmittedSlices_(data)', value: []},
    };
  }


  /**
   * Extracts the baseline slice name.
   * @param {!Array<string>} slices
   * @return {undefined}
   * @private
   */
  slicesChanged_(slices) {
    if (!slices || !slices.length) {
      this.baseline_ = undefined;
    } else if (slices.includes('Overall')) {
      this.baseline_ = 'Overall';
    } else {
      this.baseline_ = slices[0];
    }
  }

  /**
   * Extracts and sort the names of the slices from the data.
   * @param {!Array<!Object>} data
   * @return {!Array<string>|undefined}
   * @private
   */
  computeSlices_(data) {
    if (!data) {
      return;
    }
    return data.filter(d => !this.containsOmittedSliceError_(d))
        .map(d => d['slice'])
        .sort(function(x, y) {
          if (x.localeCompare(OVERALL_SLICE_KEY) == 0) {
            return -1;
          } else if (y.localeCompare(OVERALL_SLICE_KEY) == 0) {
            return 1;
          } else
            return x.localeCompare(y);
        });
  }

  /**
   * Extracts the names of the slices omitted to ensure privacy.
   * @param {!Array<!Object>} data
   * @return {!Array<string>|undefined}
   * @private
   */
  computeOmittedSlices_(data) {
    if (!data) {
      return [];
    }
    return data.filter(d => this.containsOmittedSliceError_(d))
        .map(d => d['slice']);
  }

  /**
   * Check if a slice contains ommitted slice error message.
   * @param {!Object} slicingMetric
   * @return {boolean}
   * @private
   */
  containsOmittedSliceError_(slicingMetric) {
    const omittedSliceMessage = slicingMetric && slicingMetric['metrics'] &&
        slicingMetric['metrics'][OMITTED_SLICE_ERROR_KEY];
    if (!omittedSliceMessage || typeof omittedSliceMessage !== 'string') {
      return false;
    }
    // The message is either in clear text.
    if (omittedSliceMessage.includes(OMITTED_SLICE_ERROR_MESSAGE)) {
      return true;
    }
    // Or base64 encoded.
    try {
      return atob(omittedSliceMessage).includes(OMITTED_SLICE_ERROR_MESSAGE);
    } catch (err) {
      console.log(err);
    }
    return false;
  }

  /**
   * Display the info dialog.
   * @param {!Object} event
   * @private
   */
  openInfoDialog_(event) {
    event.stopPropagation();
    const dialog = event.target.parentElement.querySelector('paper-dialog');
    dialog.open();
  }
}

customElements.define('fairness-metrics-board', FairnessMetricsBoard);
