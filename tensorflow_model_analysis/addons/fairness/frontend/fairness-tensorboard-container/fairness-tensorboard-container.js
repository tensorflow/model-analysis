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
import {template} from './fairness-tensorboard-container-template.html.js';
import {SelectEventMixin} from '../../../../frontend/tfma-nb-event-mixin/tfma-nb-event-mixin.js';
import '../fairness-nb-container/fairness-nb-container.js';
import * as tensorboard_util from '../../../../../../tensorboard/components/experimental/plugin_lib/google/index.js';

// Server end point to get eval results based on run selected by the user.
const GET_EVAL_RESULTS_ENDPOINT = './get_evaluation_result';

/**
 * @extends HTMLElement
 * @polymer
 */
export class FairnessTensorboardContainer extends SelectEventMixin
(PolymerElement) {
  constructor() {
    super();
    tensorboard_util.runs.getRuns().then(runs => {
      this.loadEvaluationRuns(runs);
    });

    tensorboard_util.runs.setOnRunsChanged(
        (runs) => this.loadEvaluationRuns(runs));
  }

  loadEvaluationRuns(runs) {
    this.evaluationRuns_ = runs;
    // To select first evaluation run by default.
    if (this.evaluationRuns_ === undefined || !this.evaluationRuns_.length) {
      return;
    }

    const hasSelection = runs.includes(this.selectedEvaluationRun_);
    if (!hasSelection) {
      this.selectedEvaluationRun_ = this.evaluationRuns_[0];
    }
  }

  static get is() {
    return 'fairness-tensorboard-container';
  }

  /** @return {!HTMLTemplateElement} */
  static get template() {
    return template;
  }


  /** @return {!PolymerElementProperties} */
  static get properties() {
    return {
      /**
       * List of evaluation runs containing slicing metrics which will be
       * rendered in UI.
       * @private {!Array<string>}
       */
      evaluationRuns_: {type: Array, notify: true},

      /**
       * Evaluation run selected by the user.
       * @private {string}
       */
      selectedEvaluationRun_: {type: String, observer: 'runChanged_'},

      /**
       * The slicing metrics evaluation result. It's a list of dict with key
       * "slice" and "metrics". For example:
       * [
       *   {
       *     "slice":"Overall",
       *     "sliceValue": "Overall"
       *     "metrics": {
       *       "auc": {
       *         "doubleValue": 0.6
       *       }
       *     }
       *   }, {
       *     "slice":"feature:1",
       *     "sliceValue":"1",
       *     "metrics": {
       *       "auc": {
       *         "doubleValue": 0.6
       *       }
       *     }
       *   }
       * ]
       * @private {!Array<!Object>}
       */
      slicingMetrics_: {type: Array, notify: true, value: []},
    };
  }

  runChanged_(run) {
    fetch(`${GET_EVAL_RESULTS_ENDPOINT}?run=${run}`)
        .then(res => res.json())
        .then(slicingMetrics => {
          this.slicingMetrics_ = slicingMetrics;
        });
  }
};

customElements.define(
    'fairness-tensorboard-container', FairnessTensorboardContainer);
