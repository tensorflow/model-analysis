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
import {template} from './tfma-google-chart-wrapper-template.html.js';

import '@org_googlewebcomponents_google_chart/google-chart/google-chart.js';

/**
 * tfma-google-chart-wrapper is a simple wrapper for google-chart component. It
 * adds a ResizeObserver on the google-chart component and calls redraw when
 * appropriate.
 *
 * @polymer
 */
export class GoogleChartWrapper extends PolymerElement {
  constructor() {
    super();
  }

  static get is() {
    return 'tfma-google-chart-wrapper';
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

      /**
       * Chart rendering options.
       * @type {!Object}
       */
      options: {type: Object},

      /**
       * The type of the chart to render.
       * @type {string}
       */
      type: {type: String},

      /**
       * A static instance of ResizeObserver used by all wrapper objects.
       * @private {?ResizeObserver}
       */
      resizeObserver_: {
        type: Object,
        value: () => ('ResizeObserver' in window) ?
            new ResizeObserver(entries => {
              entries.forEach(entry => {
                entry.target['redraw']();
              });
            }) :
            null
      },
    };
  }

  /** @override */
  connectedCallback() {
    super.connectedCallback();
    if (this.resizeObserver_) {
      this.resizeObserver_.observe(this.$['chart']);
    }
  }

  /** @override */
  disconnectedCallback() {
    super.disconnectedCallback();
    if (this.resizeObserver_) {
      this.resizeObserver_.unobserve(this.$['chart']);
    }
  }
}

customElements.define('tfma-google-chart-wrapper', GoogleChartWrapper);
