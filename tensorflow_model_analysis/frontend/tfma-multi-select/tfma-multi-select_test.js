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
suite('tests', () => {
  /**
   * Test component element.
   * @type {!Element}
   */
  let element;

  test('testSelectAndDeselect', done => {
    element = fixture('element');
    element.items = ['a', 'b', 'c'];
    const dropdown = element.$['dropdown'];
    const listbox = element.$['listbox'];

    const openDropdown = () => {
      assert.isFalse(dropdown.opened);
      dropdown.open();
      setTimeout(selectA, 1);
    };

    const selectA = () => {
      listbox.selectIndex(0);
      setTimeout(checkASelected, 1);
    };

    const checkASelected = () => {
      assert.equal(dropdown.value, 'a');
      assert.equal(element.selectedItemsString, 'a');
      assert.deepEqual(element.selectedItems, ['a']);
      setTimeout(checkDropdownStillOpen, 1);
    };

    const checkDropdownStillOpen = () => {
      assert.isTrue(dropdown.opened);
      setTimeout(selectC, 1);
    };

    const selectC = () => {
      listbox.selectIndex(2);

      setTimeout(checkCSelected, 1);
    };

    const checkCSelected = () => {
      assert.equal(dropdown.value, 'a, c');
      assert.equal(element.selectedItemsString, 'a, c');
      assert.deepEqual(element.selectedItems, ['a', 'c']);
      setTimeout(deselectC, 1);
    };

    const deselectC = () => {
      listbox.selectIndex(2);
      setTimeout(checkCDeselected, 1);
    };

    const checkCDeselected = () => {
      assert.equal(dropdown.value, 'a');
      assert.equal(element.selectedItemsString, 'a');
      assert.deepEqual(element.selectedItems, ['a']);
      setTimeout(deselectA, 1);
    };

    const deselectA = () => {
      listbox.selectIndex(0);
      setTimeout(checkADeselected, 1);
    };

    const checkADeselected = () => {
      assert.equal(dropdown.value, '');
      assert.equal(element.selectedItemsString, '');
      assert.deepEqual(element.selectedItems, []);
      done();
    };

    setTimeout(openDropdown, 0);
  });
});
