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

import {template} from './tfma-multi-select-template.html.js';

import '@polymer/paper-dropdown-menu/paper-dropdown-menu.js';
import '@polymer/paper-item/paper-item.js';
import '@polymer/paper-listbox/paper-listbox.js';

/**
 * tfma-multi-select is simple variation of paper-dropdown-menu that allows
 * multi select. This is achieved through two changes:
 * - Keep paper-listbox open after an item is selected / deselected.
 *   To do this, simply stop iron-activate and iron-select events from
 *   propagating to the paper-menu-button.
 * - Manually determine the display value for paper-dropdown-menu
 *   The value is determined by joining all selected items.
 *
 * @polymer
 */
export class MultiSelect extends PolymerElement {
  constructor() {
    super();
  }

  static get is() {
    return 'tfma-multi-select';
  }

  /** @return {!HTMLTemplateElement} */
  static get template() {
    return template;
  }

  /** @return {!PolymerElementProperties} */
  static get properties() {
    return {
      /**
       * A list of items to be selected.
       * @type {!Array<string>}
       */
      items: {type: Array, observer: 'itemsChanged_'},

      /**
       * The label to use.
       * @type {string}
       */
      label: {type: String},

      /**
       * The list of indices of all selected items.
       * @private {!Array<string>}
       */
      selectedIndices_: {type: Array, value: () => []},

      /**
       * The list of all selected items.
       * @type {!Array<string>}
       */
      selectedItems: {
        type: Array,
        notify: true,
        computed: 'computeSelectedItems_(' +
            'items, selectedIndices_, selectedIndices_.length)',
      },

      /**
       * A comma sepearated string for all selected items.
       * @type {string}
       */
      selectedItemsString: {
        type: String,
        computed: 'computeSelectedItemsString_(selectedItems)'
      },
    };
  }

  /**
   * Stops the event from bubbling up.
   * @param {!Event} event
   * @private
   */
  stopEvent_(event) {
    event.stopPropagation();
  }

  /**
   * Observer for property items.
   * @private
   */
  itemsChanged_() {
    this.selectedIndices_ = [];
  }

  /**
   * Construct the list of selected items.
   * @param {!Array<string>|undefined} items
   * @param {!Array<number>} selectedIndices
   * @param {number} unusedCount Since we use the same array instance for
   *     selectedIndices throughout, we use the length of the array as a signal
   *     to determine if its content has changed. However, this value is not
   *     used directly.
   * @return {!Array<string>}
   * @private
   */
  computeSelectedItems_(items, selectedIndices, unusedCount) {
    const selectedItems = [];
    if (items) {
      selectedIndices.forEach(id => {
        selectedItems.push(items[id]);
      });
    }
    return selectedItems;
  }

  /**
   * @param {!Array<string>} selectedItems
   * @return {string} A comma seperated string of containing all selected
   *     items.
   * @private
   */
  computeSelectedItemsString_(selectedItems) {
    return selectedItems.join(', ');
  }

  /**
   * Toggles the item with the index.
   */
  selectIndex(index) {
    this.$['listbox']['selectIndex'](index);
  }
}

customElements.define('tfma-multi-select', MultiSelect);
