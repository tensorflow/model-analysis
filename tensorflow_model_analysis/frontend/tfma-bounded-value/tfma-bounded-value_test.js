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
   * @type {?Element}
   */
  let element;

  test('setData', () => {
    element = fixture('plain-fixture');
    element.data = '{"lowerBound": 1, "upperBound": 2, "value": 1.5}';
    checkText(element, '1.50000 (1.00000, 2.00000)');
  });

  test('createBoundedValueWithAttributes', () => {
    element = fixture('attributes-inlined-fixture');
    checkText(element, '1.50000 (1.00000, 2.00000)');
  });

  test('valuesNaN', () => {
    element = fixture('plain-fixture');
    element.data = '{"lowerBound": "NaN", "upperBound": "NaN", "value": "NaN"}';
    checkText(element, 'NaN (NaN, NaN)');
  });
});

/**
 * Checks the given element contains the provided text.
 * @param {!Element} element
 * @param {string} expectedText
 */
function checkText(element, expectedText) {
  assert.isTrue(flatten(getTextContent(element)).indexOf(expectedText) >= 0);
}

/**
 * Extracts the text content of the element.
 * @param {!Element} element
 * @return {string}
 */
function getTextContent(element) {
  return element.root.textContent.trim();
}

/**
 * For string comparison purpose, "flattens" the given html by removing all
 * unnecessary newline and white space and replace them with a single white
 * space.
 * @param {string} html
 * @return {string}
 */
function flatten(html) {
  return html.replace(/(\s+)/gm, ' ').trim();
}
