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

  test('TestBehaviorOfMetrics', async (done) => {
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

    done();
  });

  test('ListsAllAvailableMetrics', async (done) => {
    const element = fixture('test-fixture');
    element.availableMetrics = AVAILABLE_METRICS;
    await delay(TEST_STEP_TIMEOUT_MS);

    let items = element.shadowRoot.querySelector('paper-listbox').items;
    assert.equal(items.length, AVAILABLE_METRICS.length);
    items.forEach((item, index) => {
      assert.equal(
          item.textContent.trim(), AVAILABLE_METRICS[index],
          'metrics name in the UI should match the "availableMetrics".');
    });
    done();
  });
});
