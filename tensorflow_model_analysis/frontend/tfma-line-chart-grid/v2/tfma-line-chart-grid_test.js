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
   * Linechart grid element.
   * @private {!Element}
   */
  let element;

  const METRICS = ['metricA', 'metricB', 'metricC'];
  const MODELS = [1, 2, 3];
  const DATA = ['a', 'b', 'c'];
  const MODEL_HEADER = 'Model';
  const DATA_HEADER = 'Data';
  const getMetric = (model, metric) => {
    switch (metric) {
      case 'metricA':
        return model + 1;
        break;
      case 'metricB':
        return model * 3;
        break;
      case 'metricC':
        return model * 10;
        break;
    }
  };

  const provider = {
    getLineChartData: (metric) => {
      return MODELS.map((model, index) => {
        const data = DATA[index];
        return [
          data, model, 'Model: ' + model + ', Data: ' + data,
          getMetric(model, metric)
        ];
      });
    },

    getModelIds: () => MODELS,

    /**
     * Get eval config of the run with the given index.
     * @param {number} index
     * @return {?{
     *   model: (string|number),
     *   data: (string|number)
     * }}
     */
    getEvalConfig: (index) => {
      return {
        'model': MODELS[index],
        'data': DATA[index],
      };
    },

    getModelColumnName: () => MODEL_HEADER,

    getDataColumnName: () => DATA_HEADER,
  };

  /**
   * Sets up the fixture and calls the provided callback asynchronously.
   * @param {function()} cb
   */
  const run = (cb) => {
    element = fixture('grid-fixture');
    element.metrics = METRICS;
    element.provider = provider;
    setTimeout(cb, 0);
  };

  test('ComponentSetup', done => {
    const checkDefault = () => {
      const charts = element.shadowRoot.querySelectorAll('tfma-line-chart');
      assert.equal(charts.length, 1);
      assert.equal(
          charts[0].shadowRoot.querySelector('#title').innerText, METRICS[0]);
      setTimeout(checkDropdown, 0);
    };
    const checkDropdown = () => {
      const items = element.shadowRoot.querySelectorAll('paper-item');
      assert.equal(items.length, 2);
      assert.equal(items[0].value, 'metricB');
      assert.equal(items[1].value, 'metricC');
      done();
    };
    run(checkDefault);
  });

  test('SkipBlackListedMetric', done => {
    const setBlackList = () => {
      element.blacklist = 'metricB';
      setTimeout(skipMetricB, 0);
    };
    const skipMetricB = () => {
      const items = element.shadowRoot.querySelectorAll('paper-item');
      assert.equal(items.length, 1);
      assert.equal(items[0].value, 'metricC');
      done();
    };
    run(setBlackList);
  });

  test('addAndRemoveChart', done => {
    const addChartForMetricB = () => {
      element.shadowRoot.querySelector('paper-listbox').select('metricB');
      setTimeout(checkChartAdded, 0);
    };
    const checkChartAdded = () => {
      const charts = element.shadowRoot.querySelectorAll('tfma-line-chart');
      assert.equal(charts.length, 2);
      assert.equal(
          charts[1].shadowRoot.querySelector('#title').innerText, METRICS[1]);
      setTimeout(removeChartForMetricA, 0);
    };
    const removeChartForMetricA = () => {
      const closeButton = element.shadowRoot.querySelector(
          'paper-icon-button[metric="metricA"]');
      closeButton.click();
      setTimeout(checkMetricAChartRemoved, 0);
    };
    const checkMetricAChartRemoved = () => {
      const charts = element.shadowRoot.querySelectorAll('tfma-line-chart');
      assert.equal(charts.length, 1);
      assert.equal(
          charts[0].shadowRoot.querySelector('#title').innerText, METRICS[1]);
      setTimeout(checkDropdown, 0);
    };
    const checkDropdown = () => {
      const items = element.shadowRoot.querySelectorAll('paper-item');
      assert.equal(items.length, 2);
      assert.equal(items[0].value, 'metricA');
      assert.equal(items[1].value, 'metricC');
      setTimeout(removeChartForMetricB, 0);
    };
    const removeChartForMetricB = () => {
      const closeButton = element.shadowRoot.querySelector(
          'paper-icon-button[metric="metricB"]');
      closeButton.click();
      setTimeout(checkMetricBChartRemoved, 10);
    };
    const checkMetricBChartRemoved = () => {
      assert.isNull(element.shadowRoot.querySelector('tfma-line-chart'));
      setTimeout(checkDropdownAgain, 0);
    };
    const checkDropdownAgain = () => {
      const items = element.shadowRoot.querySelectorAll('paper-item');
      assert.equal(items.length, 3);
      assert.equal(items[0].value, 'metricA');
      assert.equal(items[1].value, 'metricB');
      assert.equal(items[2].value, 'metricC');
      done();
    };

    run(addChartForMetricB);
  });

  test('setLineChartData', done => {
    const checkChartDataForMetricA = () => {
      const chartA =
          element.shadowRoot.querySelector('tfma-line-chart[metric="metricA"]');
      assert.deepEqual(chartA.data, [
        [
          {'label': DATA_HEADER, 'type': 'number'},
          {'role': 'annotation', 'type': 'string'},
          {'role': 'annotationText', 'type': 'string'},
          {'label': 'metricA', 'type': 'number'}
        ],
        ['a', 1, 'Model: 1, Data: a', 2],
        ['b', 2, 'Model: 2, Data: b', 3],
        ['c', 3, 'Model: 3, Data: c', 4],
      ]);
      done();
    };
    run(checkChartDataForMetricA);
  });

  test('highlight', done => {
    let selectEventFired = 0;
    const setUpEventListener = () => {
      element.addEventListener(tfma.Event.SELECT, () => {
        selectEventFired++;
      });
      setTimeout(highlightDataB, 0);
    };
    const highlightDataB = () => {
      const selection = {};
      selection[DATA_HEADER] = 'b';
      selection[MODEL_HEADER] = 2;
      element.highlight(selection);
      setTimeout(dataBHighlighted, 0);
    };
    const dataBHighlighted = () => {
      const chartA =
          element.shadowRoot.querySelector('tfma-line-chart[metric="metricA"]');
      assert.deepEqual(chartA.$['chart'].selection, [{'row': 1}]);
      setTimeout(highlightDataA, 0);
    };
    const highlightDataA = () => {
      const selection = {};
      selection[DATA_HEADER] = 'a';
      selection[MODEL_HEADER] = 1;
      element.highlight(selection);
      setTimeout(dataAHighlighted, 0);
    };
    const dataAHighlighted = () => {
      const chartA =
          element.shadowRoot.querySelector('tfma-line-chart[metric="metricA"]');
      assert.deepEqual(chartA.$['chart'].selection, [{'row': 0}]);
      assert.equal(selectEventFired, 0);
      done();
    };
    run(setUpEventListener);
  });

  test('highlightFromChart', done => {
    let selection = null;
    let chartA;
    const setUp = () => {
      element.addEventListener(tfma.Event.SELECT, (e) => {
        selection = e.detail;
      });
      chartA =
          element.shadowRoot.querySelector('tfma-line-chart[metric="metricA"]');
      setTimeout(selectRow2, 0);
    };
    const selectRow2 = () => {
      chartA.dispatchEvent(new CustomEvent(tfma.Event.SELECT, {
        'bubbles': true,
        'composed': true,
        'detail': {'point': {'row': 2}}
      }));
      setTimeout(dataCHighlighted, 0);
    };
    const dataCHighlighted = () => {
      const expectedSelection = {};
      expectedSelection[MODEL_HEADER] = 3;
      expectedSelection[DATA_HEADER] = 'c';
      assert.deepEqual(selection, expectedSelection);
      setTimeout(selectRow0, 0);
    };
    const selectRow0 = () => {
      chartA.dispatchEvent(new CustomEvent(
          tfma.Event.SELECT,
          {'bubbles': true, 'coposed': true, 'detail': {'point': {'row': 0}}}));
      setTimeout(dataAHighlighted, 0);
    };
    const dataAHighlighted = () => {
      const expectedSelection = {};
      expectedSelection[MODEL_HEADER] = 1;
      expectedSelection[DATA_HEADER] = 'a';
      assert.deepEqual(selection, expectedSelection);
      done();
    };
    run(setUp);
  });
});
