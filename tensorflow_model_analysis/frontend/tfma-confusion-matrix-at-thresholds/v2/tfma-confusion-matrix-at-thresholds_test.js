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
   * @private {!Element}
   */
  let element;

  setup(() => {
    element = fixture('my-fixture');
  });

  function next(cb, opt_timeout) {
    setTimeout(cb, opt_timeout || 0);
  }

  test('parsesValuesCorrectly', () => {
    const data = {
      'matrices': [createConfusionMatrixAtThresholds(
          0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87)]
    };
    element.data = JSON.stringify(data);

    const displayedData = element.displayedData_;
    assert.equal(displayedData.length, 1);
    const item = displayedData[0];
    assert.equal(item['threshold'], '0.81000');
    assert.equal(item['precision'], '0.82000');
    assert.equal(item['recall'], '0.83000');
    assert.equal(item['truePositives'], '0.84000');
    assert.equal(item['trueNegatives'], '0.85000');
    assert.equal(item['falsePositives'], '0.86000');
    assert.equal(item['falseNegatives'], '0.87000');
  });

  test('expandAndCollapse', done => {
    element.data = JSON.stringify(createDefaultData());

    const showThreeRows = () => {
      const html = element.root.innerHTML;
      assert.isTrue(html.indexOf('0.51000') >= 0);
      assert.isTrue(html.indexOf('0.52000') >= 0);
      assert.isTrue(html.indexOf('0.53000') >= 0);
      assert.equal(-1, html.indexOf('0.54000'));

      next(expand);
    };

    const toggle = () => {
      element.shadowRoot.querySelector('div.table')
          .parentNode.dispatchEvent(new CustomEvent('click'));
    };

    const expand = () => {
      toggle();
      next(showFourRows);
    };

    const showFourRows = () => {
      const html = element.root.innerHTML;
      assert.isTrue(html.indexOf('0.5100000') >= 0);
      assert.isTrue(html.indexOf('0.5200000') >= 0);
      assert.isTrue(html.indexOf('0.5300000') >= 0);
      assert.isTrue(html.indexOf('0.5400000') >= 0);

      next(collapse);
    };

    const collapse = () => {
      toggle();
      next(showThreeRowsAgain);
    };

    const showThreeRowsAgain = () => {
      const html = element.root.innerHTML;
      assert.isTrue(html.indexOf('0.51000') >= 0);
      assert.isTrue(html.indexOf('0.52000') >= 0);
      assert.isTrue(html.indexOf('0.53000') >= 0);
      assert.equal(-1, html.indexOf('0.54000'));

      done();
    };

    next(showThreeRows);
  });

  test('showExpanded', done => {
    element.expanded = true;
    element.data = JSON.stringify(createDefaultData());


    const showFourRows = () => {
      const html = element.root.innerHTML;
      assert.isTrue(html.indexOf('0.5100000') >= 0);
      assert.isTrue(html.indexOf('0.5200000') >= 0);
      assert.isTrue(html.indexOf('0.5300000') >= 0);
      assert.isTrue(html.indexOf('0.5400000') >= 0);

      done();
    };

    next(showFourRows);
  });

  test('nonInteraciveDoesNotExpand', done => {
    element.interactive = false;
    element.data = JSON.stringify(createDefaultData());

    const showThreeRows = () => {
      const html = element.root.innerHTML;
      assert.isTrue(html.indexOf('0.51000') >= 0);
      assert.isTrue(html.indexOf('0.52000') >= 0);
      assert.isTrue(html.indexOf('0.53000') >= 0);
      assert.equal(-1, html.indexOf('0.54000'));

      next(click);
    };

    const click = () => {
      element.shadowRoot.querySelector('div.table')
          .parentNode.dispatchEvent(new CustomEvent('click'));
      next(stillShowThreeRows);
    };

    const stillShowThreeRows = () => {
      const html = element.root.innerHTML;
      assert.isTrue(html.indexOf('0.51000') >= 0);
      assert.isTrue(html.indexOf('0.52000') >= 0);
      assert.isTrue(html.indexOf('0.53000') >= 0);
      assert.equal(-1, html.indexOf('0.54000'));

      done();
    };

    next(showThreeRows);
  });

  test('nonInteraciveDoesNotCollapse', done => {
    element.interactive = false;
    element.expanded = true;
    element.data = JSON.stringify(createDefaultData());

    const showFourRows = () => {
      const html = element.root.innerHTML;
      assert.isTrue(html.indexOf('0.5100000') >= 0);
      assert.isTrue(html.indexOf('0.5200000') >= 0);
      assert.isTrue(html.indexOf('0.5300000') >= 0);
      assert.isTrue(html.indexOf('0.5400000') >= 0);

      next(click);
    };

    const click = () => {
      element.shadowRoot.querySelector('div.table')
          .parentNode.dispatchEvent(new CustomEvent('click'));
      next(stillShowFourRows);
    };

    const stillShowFourRows = () => {
      const html = element.root.innerHTML;
      assert.isTrue(html.indexOf('0.5100000') >= 0);
      assert.isTrue(html.indexOf('0.5200000') >= 0);
      assert.isTrue(html.indexOf('0.5300000') >= 0);
      assert.isTrue(html.indexOf('0.5400000') >= 0);

      done();
    };

    next(showFourRows);
  });

  test('doNotExpandIfEventHandled', done => {
    element.data = JSON.stringify(createDefaultData());

    const showThreeRows = () => {
      const html = element.root.innerHTML;
      assert.isTrue(html.indexOf('0.51000') >= 0);
      assert.isTrue(html.indexOf('0.52000') >= 0);
      assert.isTrue(html.indexOf('0.53000') >= 0);
      assert.equal(-1, html.indexOf('0.54000'));

      next(addListener);
    };

    const addListener = () => {
      element.addEventListener('expand-metric', e => {
        e.preventDefault();
      });
      next(expand);
    };

    const expand = () => {
      element.shadowRoot.querySelector('div.table')
          .parentNode.dispatchEvent(new CustomEvent('click'));
      next(stillShowThreeRows);
    };

    const stillShowThreeRows = () => {
      const html = element.root.innerHTML;
      assert.isTrue(html.indexOf('0.51000') >= 0);
      assert.isTrue(html.indexOf('0.52000') >= 0);
      assert.isTrue(html.indexOf('0.53000') >= 0);
      assert.equal(-1, html.indexOf('0.54000'));

      done();
    };

    next(showThreeRows);
  });
});

/**
 * @param {number} threshold
 * @param {number} precision
 * @param {number} recall
 * @param {number} truePositive
 * @param {number} trueNegative
 * @param {number} falsePositive
 * @param {number} falseNegative
 * @return {!Object} A ConfusionMatrixAtThreshold in JSON format.
 */
function createConfusionMatrixAtThresholds(
    threshold, precision, recall, truePositive, trueNegative, falsePositive,
    falseNegative) {
  return {
    'threshold': threshold,
    'precision': precision,
    'recall': recall,
    'truePositives': truePositive,
    'trueNegatives': trueNegative,
    'falsePositives': falsePositive,
    'falseNegatives': falseNegative,
  };
}

/**
 * @return {!Object} The default test data.
 */
function createDefaultData() {
  return {
    'matrices': [
      createConfusionMatrixAtThresholds(
          0.51, 0.91, 0.81, 0.71, 0.61, 0.125, 0.375),
      createConfusionMatrixAtThresholds(0.52, 0.92, 0.82, 0.72, 0.62, 0.1, 0.5),
      createConfusionMatrixAtThresholds(0.53, 0.93, 0.83, 0.73, 0.63, 0.1, 0.5),
      createConfusionMatrixAtThresholds(0.54, 0.94, 0.84, 0.74, 0.64, 0.1, 0.5)
    ],
  };
}
