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
import {template} from './tfma-line-chart-template.html.js';

import '@org_googlewebcomponents_google_chart/google-chart/google-chart.js';

/**
 * tfma-line-chart renders the a time series for a single metric.
 *
 * @polymer
 */
export class LineChart extends PolymerElement {
  constructor() {
    super();
  }

  static get is() {
    return 'tfma-line-chart';
  }

  /** @return {!HTMLTemplateElement} */
  static get template() {
    return template;
  }

  /** @return {!PolymerElementProperties} */
  static get properties() {
    return {
      /**
       * Data table feeding into the chart.
       * @type {!Array}
       */
      data: {type: Array},

      /**
       * Rendering options for the chart. Note that default values will be
       * provided, if not set, for the following properties: legend, tooltip and
       * chartArea.
       * @private {!Object}
       */
      options_: {
        type: Object,
        value: {
          legend: {position: 'none'},
          tooltip: {trigger: 'focus'},
          chartArea: {top: 30, left: 45, width: '85%', height: '75%'},
          hAxis: {ticks: []},
        }
      },

      /**
       * Chart title.
       * @type {string}
       */
      title: {type: String, value: ''},

      events_: {type: Array, value: ['onmouseover', 'onmouseout']}
    };
  }

  /**
   * Selects a data point.
   * @param {!Array<{row: (number|undefined), column: (number|undefined)}>|
   *   {row: (number|undefined), column: (number|undefined)}} points
   *     Selected data point(s), represented by row/column number.
   */
  select(points) {
    if (!Array.isArray(points)) {
      points = [points];
    }
    this.$.chart.selection = points;
  }

  /**
   * Clears the selected data point.
   */
  clearSelection() {
    this.$.chart.selection = [];
  }

  /**
   * Mouse over event handler.
   * @param {!Event} event
   * @private
   */
  onMouseOver_(event) {
    const point = event.detail.data;
    this.select(point);
    this.dispatchEvent(new CustomEvent(
        tfma.Event.SELECT,
        {detail: {point: point}, bubbles: true, composed: true}));
  }

  /**
   * Mouse out event handler.
   * @private
   */
  onMouseOut_() {
    this.clearSelection();
    this.dispatchEvent(new CustomEvent(
        tfma.Event.CLEAR_SELECTION, {bubbles: true, composed: true}));
  }
}

customElements.define('tfma-line-chart', LineChart);
