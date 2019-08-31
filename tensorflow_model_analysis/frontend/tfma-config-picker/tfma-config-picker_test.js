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

  test('testSingleOutputNoClass', done => {
    element = fixture('element');
    const singleOutputNioClassConfig = {'': ['']};
    element.allConfigs = singleOutputNioClassConfig;

    const verifyNoDropdown = () => {
      // No dropdown should be present since there is only one output and
      // no class id.
      assert.isNull(element.shadowRoot.querySelector('tfma-multi-select'));
      setTimeout(checkSelectedConfigs, 1);
    };

    const checkSelectedConfigs = () => {
      // The only output and no class should be selected automatically.
      assert.deepEqual(element.selectedConfigs, singleOutputNioClassConfig);
      done();
    };

    setTimeout(verifyNoDropdown, 0);
  });

  test('testSingleOutputMultiClass', done => {
    element = fixture('element');
    element.allConfigs = {'': ['classId:0', 'classId:1', 'classId:2']};
    let dropdown;

    const verifyDropdownAvailable = () => {
      // Class id dropdown should be present since we will automatically select
      // the only avialable output.
      dropdown = element.shadowRoot.querySelector('tfma-multi-select');
      assert.isNotNull(dropdown);
      assert.equal(dropdown.label, 'Class Ids');
      setTimeout(chooseClass2, 1);
    };

    const chooseClass2 = () => {
      dropdown.selectIndex(2);
      setTimeout(checkClass2Selected, 1);
    };

    const checkClass2Selected = () => {
      assert.deepEqual(element.selectedConfigs, {'': ['classId:2']});
      setTimeout(chooseClass0, 0);
    };

    const chooseClass0 = () => {
      dropdown.selectIndex(0);
      setTimeout(checkClass0Selected, 1);
    };

    const checkClass0Selected = () => {
      assert.deepEqual(
          element.selectedConfigs, {'': ['classId:2', 'classId:0']});
      done();
    };

    setTimeout(verifyDropdownAvailable, 0);
  });

  test('testMultiOutputNoClass', done => {
    element = fixture('element');
    element.allConfigs = {'output1': [''], 'output2': ['']};
    let outputDropdown;
    let classIdsDropdown;

    const checkOutputDropdownAvailable = () => {
      // Class id dropdown should not be present since we will automatically
      // select the only avialable output.
      outputDropdown = element.shadowRoot.querySelector('tfma-multi-select');
      assert.isNotNull(outputDropdown);
      assert.equal(outputDropdown.label, 'Outputs');
      setTimeout(chooseOutput1, 1);
    };

    const chooseOutput1 = () => {
      outputDropdown.selectIndex(0);
      setTimeout(checkOutput1Selected, 1);
    };

    const checkOutput1Selected = () => {
      assert.deepEqual(element.selectedConfigs, {'output1': ['']});
      setTimeout(checkClassIdsDropdownNotAvailable, 1);
    };

    const checkClassIdsDropdownNotAvailable = () => {
      const dropdowns =
          element.shadowRoot.querySelectorAll('tfma-multi-select');
      assert.equal(dropdowns.length, 1);
      assert.equal(outputDropdown, dropdowns[0]);
      setTimeout(chooseOutput2, 1);
    };

    const chooseOutput2 = () => {
      outputDropdown.selectIndex(1);
      setTimeout(checkNoConfigSelected, 1);
    };

    const checkNoConfigSelected = () => {
      // The selected config is removed since there are more than one
      // posibilities now.
      const dropdowns =
          element.shadowRoot.querySelectorAll('tfma-multi-select');
      assert.equal(dropdowns[0], outputDropdown);
      classIdsDropdown = dropdowns[1];
      assert.equal(classIdsDropdown.label, 'Class Ids');
      setTimeout(selectOutput2NoClass, 1);
    };

    const selectOutput2NoClass = () => {
      classIdsDropdown.selectIndex(1);
      setTimeout(checkOutput2Selected, 1);
    };

    const checkOutput2Selected = () => {
      assert.deepEqual(element.selectedConfigs, {'output2': ['']});
      done();
    };

    setTimeout(checkOutputDropdownAvailable, 0);
  });

  test('testMultiOutputMultiClass', done => {
    element = fixture('element');
    element.allConfigs = {
      'output1': ['classId:0', 'classId:1', 'classId:2'],
      'output2': ['classId:0', 'classId:2', 'classId:4']
    };
    let outputDropdown;
    let classIdsDropdown;

    const selectOutput1 = () => {
      outputDropdown = element.shadowRoot.querySelector('tfma-multi-select');
      outputDropdown.selectIndex(0);
      setTimeout(selectOutput2, 1);
    };

    const selectOutput2 = () => {
      outputDropdown.selectIndex(1);
      setTimeout(selectOutput1Class0, 1);
    };

    const selectOutput1Class0 = () => {
      const dropdowns =
          element.shadowRoot.querySelectorAll('tfma-multi-select');
      assert.equal(dropdowns.length, 2);
      assert.equal(outputDropdown, dropdowns[0]);
      classIdsDropdown = dropdowns[1];
      classIdsDropdown.selectIndex(0);
      setTimeout(checkOutput1Class0Selected, 1);
    };

    const checkOutput1Class0Selected = () => {
      assert.deepEqual(element.selectedConfigs, {'output1': ['classId:0']});
      setTimeout(selectOutput2Class2And4, 1);
    };

    const selectOutput2Class2And4 = () => {
      classIdsDropdown.selectIndex(4);
      classIdsDropdown.selectIndex(5);
      setTimeout(checkOutput2Class2And4Selected, 1);
    };

    const checkOutput2Class2And4Selected = () => {
      assert.deepEqual(
          element.selectedConfigs,
          {'output1': ['classId:0'], 'output2': ['classId:2', 'classId:4']});
      setTimeout(unselectOutput1, 1);
    };

    const unselectOutput1 = () => {
      outputDropdown.selectIndex(0);
      setTimeout(allSelectionCleared, 1);
    };

    const allSelectionCleared = () => {
      assert.deepEqual(element.selectedConfigs, {});
      done();
    };

    setTimeout(selectOutput1, 0);
  });
});
