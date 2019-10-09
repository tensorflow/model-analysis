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

/**
 * @fileoverview
 * Defines reusable event mixins used for communication between widgets in
 * notebook environment. All communications are done through the event
 * "tfma-event". The event's detail object will container two fields. eventType
 * is the actual type of the event while event detail is the details about the
 * event.
 */

const TFMA_EVENT = 'tfma-event';

/** @enum {string} */
const TFMA_EVENT_TYPES = {
  SLICE_SELECTED: 'slice-selected',
};

/**
 * Helper method to dispatch a tfma-event with specified values.
 * @param {!HTMLElement} element
 * @param {string} eventType
 * @param {!Object} eventDetail
 */
const dispatch = (element, eventType, eventDetail) => {
  element.dispatchEvent(new CustomEvent(TFMA_EVENT, {
    detail: {'type': eventType, 'detail': eventDetail},
    bubbles: true,
    composed: true
  }));
};

/**
 * Mixin that should handle all select events in TFMA.
 * @param {function(new:HTMLElement)} baseClass base class to extend.
 * @return {function(new:HTMLElement)} Generated class
 */
export const SelectEventMixin = (baseClass) => class extends baseClass {
  constructor() {
    super();

    this.addEventListener('select', (e) => {
      this.handleSelect_(e);
    });
  }

  /**
   * Event handler for the select event.
   * @param {!Event} selectEvent
   * @private
   */
  handleSelect_(selectEvent) {
    const eventSource = selectEvent['path'] && selectEvent['path'][0];
    if (eventSource) {
      let selectedSlice;
      if (eventSource.tagName == 'TFMA-METRICS-TABLE') {
        const selectedRow = eventSource['selection'][0]['row'];
        const tableData = eventSource['data']['getDataTable']();
        selectedSlice = tableData[selectedRow][0];
      } else if (
          eventSource.tagName == 'TFMA-SLICE-OVERVIEW' ||
          eventSource.tagName == 'FAIRNESS-BOUNDED-VALUE-BAR-CHART') {
        selectedSlice = selectEvent['detail'];
      }

      if (selectedSlice) {
        selectedSlice = selectedSlice.split(':');
        const sliceName = selectedSlice[0];
        const sliceValue = selectedSlice[1];
        dispatch(
            this, TFMA_EVENT_TYPES.SLICE_SELECTED,
            {'sliceName': sliceName, 'sliceValue': sliceValue});
      }
    }
  }
};
