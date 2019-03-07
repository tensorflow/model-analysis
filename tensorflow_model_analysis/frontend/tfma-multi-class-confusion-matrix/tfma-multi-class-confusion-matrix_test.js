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
  const DEFAULT_DATA = {
    'entries': [
      {'actualClass': 'A', 'predictedClass': 'A', 'weight': 100},
      {'actualClass': 'A', 'predictedClass': 'B', 'weight': 10},
      {'actualClass': 'B', 'predictedClass': 'A', 'weight': 50},
      {'actualClass': 'B', 'predictedClass': 'B', 'weight': 30}
    ],
  };

  /**
   * Test component element.
   * @type {!Element}
   */
  let element;

  const run = (cb, data) => {
    element = fixture('matrix');
    element.expanded = true;
    element.jsonData = data;
    setTimeout(cb, 1);
  };

  test('parseData', done => {
    const checkMatrixContent = () => {
      const matrix = element.shadowRoot.querySelector('tfma-matrix');
      const headerRow =
          matrix.shadowRoot.querySelector('.matrix-row:nth-child(2)');
      assert.deepEqual(
          headerRow.textContent.trim().split(/[\s\n]+/), ['A', 'B']);

      const contentRow1 =
          matrix.shadowRoot.querySelector('.matrix-row:nth-child(3)');
      assert.deepEqual(
          contentRow1.textContent.trim().split(/[\s\n]+/), ['A', '100', '10']);

      const contentRow2 =
          matrix.shadowRoot.querySelector('.matrix-row:nth-child(4)');
      assert.deepEqual(
          contentRow2.textContent.trim().split(/[\s\n]+/), ['B', '50', '30']);

      done();
    };

    run(checkMatrixContent, DEFAULT_DATA);
  });
});
