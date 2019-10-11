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
  let element;
  const AVAILABLE_METRICS = ['a', 'b', 'c', 'd', 'e'];

  setup(() => {
    element = fixture('test-fixture');
    element.availableMetrics = AVAILABLE_METRICS;
  });

  test('ListsAllAvailableMetrics', done => {
    const check = () => {
      let items = element.shadowRoot.querySelector('paper-listbox').items;
      assert.equal(items.length, AVAILABLE_METRICS.length);
      items.forEach((item, index) => {
        assert.equal(item.textContent.trim(), AVAILABLE_METRICS[index]);
      });
      done();
    };
    setTimeout(check, 0);
  });


  test('ChecksDefaultMetrics', done => {
    const check = () => {
      assert.deepEqual(element.selectedMetrics, [AVAILABLE_METRICS[0]]);

      let items = element.shadowRoot.querySelector('paper-listbox').items;
      items.forEach((item, index) => {
        // Only the first metrics will be chosen by default.
        if (index == 0) {
          assert.isTrue(item.querySelector('paper-checkbox').checked);
        } else {
          assert.isFalse(item.querySelector('paper-checkbox').checked);
        }
      });
      done();
    };
    setTimeout(check, 0);
  });

  test('TapMetrics', done => {
    const tapMetrics = () => {
      // Tap last metrics
      let items = element.shadowRoot.querySelector('paper-listbox').items;
      items[items.length - 1].fire('tap');
      setTimeout(check, 100);
    };
    const check = () => {
      assert.deepEqual(element.selectedMetrics, [
        AVAILABLE_METRICS[0], AVAILABLE_METRICS[AVAILABLE_METRICS.length - 1]
      ]);

      let items = element.shadowRoot.querySelector('paper-listbox').items;
      items.forEach((item, index) => {
        if (index == 0 || index == items.length - 1) {
          assert.isTrue(item.querySelector('paper-checkbox').checked);
        } else {
          assert.isFalse(item.querySelector('paper-checkbox').checked);
        }
      });
      done();
    };
    setTimeout(tapMetrics, 0);
  });
});
