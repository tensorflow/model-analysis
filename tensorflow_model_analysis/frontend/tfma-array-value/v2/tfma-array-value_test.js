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

  test('parseData', () => {
    element = fixture('element');
    element.data = JSON.stringify({
      'shape': [3, 2, 2],
      'dataType': 'INT32',
      'int32Values': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    });
    assert.deepEqual(
        element.arrayData,
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]);
  });

  test('parseDataChecksNumberOfElementMatches', () => {
    element = fixture('element');
    element.data = JSON.stringify({
      'shape': [3, 2, 2],
      'dataType': 'INT32',
      'int32Values': [1, 2, 3, 4, 5, 6]
    });
    assert.deepEqual(element.arrayData, []);
  });

  test('showUpToThreeRowsIfNotExpanded', () => {
    element = fixture('element');
    element.arrayData = [1, 2, 3, 4];
    assert.isFalse(element.expanded);
    assert.deepEqual(element.values_, ['1', '2', '3']);
  });

  test('clickToExpandShowsAllData', done => {
    element = fixture('element');
    element.arrayData = [[1, 0], [2, 0], [3, 0], [4, 0]];

    const clickElement = () => {
      element.$.root.click();
      setTimeout(checkAllRowsShown, 0);
    };

    const checkAllRowsShown = () => {
      assert.isTrue(element.expanded);
      assert.deepEqual(
          element.values_, ['[1, 0]', '[2, 0]', '[3, 0]', '[4, 0]']);
      done();
    };

    setTimeout(clickElement, 0);
  });
});
