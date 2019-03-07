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

  test('parseDataCorrectly', done => {
    const count = 5;
    insertValueAtCutoffs(createDataString(count), () => {
      assert.equal(element.shadowRoot.querySelectorAll('div.table').length, 1);
      assert.equal(element.shadowRoot.querySelectorAll('div.tr').length, count);
      assert.equal(
          element.shadowRoot.querySelectorAll('div.td').length, count * 2);
      done();
    });
  });

  test('parseCorruptedDataWithoutException', done => {
    const count = 5;
    const dataString = createDataString(count);
    // Corrupt the data string by taking its substring.
    insertValueAtCutoffs(dataString.substring(5), () => {
      assert.equal(element.shadowRoot.querySelectorAll('.tr').length, 0);
      done();
    });
  });

  test('KIs0RepresentsAll', done => {
    const data = JSON.stringify({'values': [{'cutoff': 0, 'value': 1}]});
    insertValueAtCutoffs(data, () => {
      assert.equal(
          element.shadowRoot.querySelector('div.td').textContent.trim(), 'All');
      done();
    });
  });

  /**
   * @param {number} count
   * @return {string} Stringified array containing json representation of value at cutoffs data.
   */
  function createDataString(count) {
    const precision = [];
    for (let i = 1; i <= count; i++) {
      precision.push(
          {'cutoff': i, 'value': 1 - 0.1 * i});
    }
    return JSON.stringify({'values': precision});
  }

  /**
   * Instantiates a tfma-value-at-cutoffs element using provided data and calls the call back
   * asynchronously.
   * @param {string} data
   * @param {function()} cb
   */
  function insertValueAtCutoffs(data, cb) {
    element = fixture('plain-fixture');
    element.data = data;
    setTimeout(cb, 0);
  }
});
