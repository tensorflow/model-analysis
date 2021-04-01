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

suite('tests', () => {
  const TEST_STEP_TIMEOUT_MS = 50;
  const AVAILABLE_METRICS = ['a', 'b', 'c', 'd', 'e'];

  // Returns a promise that resolves after a given amount of time.
  const delay = (delayInMs) => {
    return new Promise((resolve) => setTimeout(resolve, delayInMs));
  };

  test('TestBehaviorOfMetrics', async () => {
    const element = fixture('test-fixture');
    element.availableMetrics = AVAILABLE_METRICS;
    await delay(TEST_STEP_TIMEOUT_MS);

    // Check if default metric is selected.
    assert.deepEqual(
        element.selectedMetrics, [AVAILABLE_METRICS[0]],
        'First metric will be selected by default.');
    let items = element.shadowRoot.querySelector('paper-listbox').items;
    items.forEach((item, index) => {
      // Only the first metrics will be chosen by default.
      if (index == 0) {
        assert.isTrue(
            item.querySelector('paper-checkbox').checked,
            'First metric will be selected by default.');
      } else {
        assert.isFalse(
            item.querySelector('paper-checkbox').checked,
            'Rest metrics should not be selected by default.');
      }
    });

    // Tap last metric.
    items[items.length - 1].fire('tap');
    await delay(TEST_STEP_TIMEOUT_MS);

    // Check if last metric is selected.
    assert.deepEqual(
        element.selectedMetrics,
        [AVAILABLE_METRICS[0], AVAILABLE_METRICS[AVAILABLE_METRICS.length - 1]],
        'First and last metric will be selected.');
    items.forEach((item, index) => {
      if (index == 0 || index == items.length - 1) {
        assert.isTrue(
            item.querySelector('paper-checkbox').checked,
            'First and last metric will be selected.');
      } else {
        assert.isFalse(
            item.querySelector('paper-checkbox').checked,
            'Rest metrics should not be selected.');
      }
    });

    // Tap "Select all".
    element.$['selectAll'].fire('tap');
    await delay(TEST_STEP_TIMEOUT_MS);

    // Check if all metrics are selected.
    assert.deepEqual(
        element.selectedMetrics, ['a', 'e', 'b', 'c', 'd'],
        'All metrics should be selected.');
    items.forEach((item, index) => {
      assert.isTrue(
          item.querySelector('paper-checkbox').checked,
          'All metrics should be selected.');
    });

    // Tap "Select all" again.
    element.$['selectAll'].fire('tap');
    await delay(TEST_STEP_TIMEOUT_MS);

    // Check if all metrics are unselected.
    assert.deepEqual(
        element.selectedMetrics,
        [undefined, undefined, undefined, undefined, undefined],
        'All metrics should be unselected.');
    items.forEach((item, index) => {
      assert.isFalse(
          item.querySelector('paper-checkbox').checked,
          'All metrics should be unselected.');
    });
  });

  test('ListsAllAvailableMetrics, always shows in expected order and format', async () => {
    const element = fixture('test-fixture');
    element.availableMetrics = shuffledCopy([
      'post_export_metrics/false_discovery_rate',
      'aaa_unexpected_unknown',
      'accuracy',
      'lift@3',
      'post_export_metrics/false_positive_rate',
      'auc'
    ]);
    await delay(TEST_STEP_TIMEOUT_MS);

    const items = element.shadowRoot.querySelector('paper-listbox').items;
    const uiTexts = Array.from(items).map(item => item.textContent.trim());
    assert.sameOrderedMembers(uiTexts, [
      'accuracy',
      'auc',
      'false_positive_rate',
      'aaa_unexpected_unknown',
      'false_discovery_rate',
      'lift@3',
    ], 'metrics name in the UI list should be in the expected order');
  });
});


// from https://stackoverflow.com/a/2450976
/**
 * Copies an array then shuffles it randomly.  Non-deterministic.
 * @param {!Array} original
 * @return {!Array}
 */
function shuffledCopy(original) {
  const array = original.slice(0);
  let currentIndex = array.length;
  let temporaryValue;
  let randomIndex;

  // While there remain elements to shuffle...
  while (0 !== currentIndex) {

    // Pick a remaining element...
    randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex -= 1;

    // And swap it with the current element.
    temporaryValue = array[currentIndex];
    array[currentIndex] = array[randomIndex];
    array[randomIndex] = temporaryValue;
  }

  return array;
}
