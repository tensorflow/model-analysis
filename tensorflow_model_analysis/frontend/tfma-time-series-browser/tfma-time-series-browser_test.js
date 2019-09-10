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

  const makeData = index => tfma.Data.build(
      ['a', 'b', 'c'],
      [{'metrics': {'a': index + 1, 'b': index * -3, 'c': index * index}}]);

  const run = cb => {
    element = fixture('time-series-fixture');
    element.seriesData = new tfma.SeriesData(
        [
          {
            'config': {'dataIdentifier': '1', 'modelIdentifier': '0'},
            'data': makeData(0),
          },
          {
            'config': {'dataIdentifier': '2', 'modelIdentifier': '1'},
            'data': makeData(1),
          },
          {
            'config': {'dataIdentifier': '1', 'modelIdentifier': '2'},
            'data': makeData(2),
          }
        ],
        true);

    let cbPending = 2;
    element.addEventListener('google-chart-ready', () => {
      cbPending--;
      if (cbPending === 0) cb();
    });
  };

  test('mouseOverDataPointInChartSelectsRowInTable', done => {
    const mouseOverDataPointInChart = () => {
      const chart = element.$['grid']
                        .shadowRoot.querySelector('tfma-line-chart')
                        .$['chart'];
      chart.dispatchEvent(new CustomEvent('google-chart-onmouseover', {
        'bubbles': true,
        'composed': true,
        'detail': {'data': {'row': 2, 'column': 0}}
      }));
      setTimeout(rowInTableSelected, 0);
    };

    const rowInTableSelected = () => {
      const table = element.$.table.shadowRoot.querySelector('google-chart');
      assert.deepEqual(table.selection, [{'row': 2}]);
      done();
    };

    run(mouseOverDataPointInChart);
  });

  test('selectRowInTableSelectsPointInChart', done => {
    const selectRowInTable = () => {
      const table = element.$.table.shadowRoot.querySelector('google-chart');
      table.selection = [{'row': 2}];
      setTimeout(dataPointIonChartSelected, 0);
    };

    const dataPointIonChartSelected = () => {
      const chart = element.$['grid']
                        .shadowRoot.querySelector('tfma-line-chart')
                        .$['chart'];
      assert.deepEqual(chart.selection, [{'row': 2}]);
      done();
    };

    run(selectRowInTable);
  });
});
