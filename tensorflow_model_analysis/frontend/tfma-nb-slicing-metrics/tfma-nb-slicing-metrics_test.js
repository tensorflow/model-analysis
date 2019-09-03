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
  let element;

  /**
   * Sets up the test fixture with given data and runs the test by calling the
   * provided callback.
   * @param {function()} cb
   * @param {!Array<!Object>} data
   */
  function runWithData(cb, data) {
    element = fixture('slicing-metrics');
    element.data = data;
    element.config = {'weightedExamplesColumn': 'weightedExamples'};
    setTimeout(cb, 0);
  }

  /**
   * Sets up the test fixture and runs the test by calling the provided
   * callback.
   * @param {function()} cb
   */
  function run(cb) {
    runWithData(cb, [
      {
        'slice': 'col:1',
        'metrics': {
          '': {
            '': {
              'averageRefinedPrediction': {'doubleValue': 0.51},
              'averageLabel': {'doubleValue': 0.61},
              'weightedExamples': {'doubleValue': 10},
            },
          },
        },
      },
      {
        'slice': 'col:2',
        'metrics': {
          '': {
            '': {
              'averageRefinedPrediction': {'doubleValue': 0.52},
              'averageLabel': {'doubleValue': 0.62},
              'weightedExamples': {'doubleValue': 20},
            },
          },
        }
      },
      {
        'slice': 'col:3',
        'metrics': {
          '': {
            '': {
              'averageRefinedPrediction': {'doubleValue': 0.53},
              'averageLabel': {'doubleValue': 0.63},
              'weightedExamples': {'doubleValue': 30},
            },
          },
        },
      }
    ]);
  }

  test('ParseData', done => {
    const checkDataParsed = () => {
      const browser =
          element.shadowRoot.querySelector('tfma-slicing-metrics-browser');
      assert.deepEqual(browser.metrics, [
        'averageLabel', 'averageRefinedPrediction', 'calibration',
        'weightedExamples'
      ]);

      assert.deepEqual(browser.data, [
        {
          'slice': 'col:1',
          'metrics': {
            'averageRefinedPrediction': 0.51,
            'averageLabel': 0.61,
            'weightedExamples': 10
          }
        },
        {
          'slice': 'col:2',
          'metrics': {
            'averageRefinedPrediction': 0.52,
            'averageLabel': 0.62,
            'weightedExamples': 20,
          }
        },
        {
          'slice': 'col:3',
          'metrics': {
            'averageRefinedPrediction': 0.53,
            'averageLabel': 0.63,
            'weightedExamples': 30,
          }
        }
      ]);
      done();
    };
    run(checkDataParsed);
  });

  test('ParseConfig', done => {
    const checkConfigParsed = () => {
      const browser =
          element.shadowRoot.querySelector('tfma-slicing-metrics-browser');
      assert.equal(browser.weightedExamplesColumn, 'weightedExamples');
      done();
    };
    run(checkConfigParsed);
  });

  test('setsWeightColumnTo1IfAbsent', done => {
    const newWeightColumn = 'newWeightColumn';

    const changeWeightColumn = () => {
      element.config = {'weightedExamplesColumn': newWeightColumn};
      setTimeout(checkBrowser, 0);
    };
    const checkBrowser = () => {
      const browser =
          element.shadowRoot.querySelector('tfma-slicing-metrics-browser');
      assert.equal(browser.weightedExamplesColumn, newWeightColumn);
      const data = browser.data;
      assert.equal(data[0]['metrics'][newWeightColumn], 1);
      done();
    };
    run(changeWeightColumn);
  });

  test('SelectRowTriggersTfmaEvent', done => {
    let table;
    let eventType;
    let eventDetail;
    const waitTillReady = () => {
      const browser =
          element.shadowRoot.querySelector('tfma-slicing-metrics-browser');
      table = browser.shadowRoot.querySelector('tfma-metrics-table');
      table.addEventListener('google-chart-ready', readyCallback);
    };
    const readyCallback = () => {
      table.removeEventListener('google-chart-ready', readyCallback);
      setTimeout(selectRow, 1);
    };
    const selectRow = () => {
      element.addEventListener('tfma-event', e => {
        eventType = e.detail['type'];
        eventDetail = e.detail['detail'];
      });

      table.selection = [{'row': 2}];
      setTimeout(verify, 1);
    };
    const verify = () => {
      assert.equal(eventType, 'slice-selected');
      assert.equal(eventDetail['sliceName'], 'col');
      assert.equal(eventDetail['sliceValue'], 3);

      done();
    };
    run(waitTillReady);
  });

  test('choosesWeightColumnWhenMultipleConfigsAreSelected', done => {
    const expectedWeightColumn = 'output1/0/weightedExamples';

    const pickClassIds = () => {
      const picker = element.shadowRoot.querySelector('tfma-config-picker');
      const selector = picker.shadowRoot.querySelector('tfma-multi-select');
      selector.selectIndex(1);
      selector.selectIndex(0);
      setTimeout(checkBrowser, 0);
    };
    const checkBrowser = () => {
      const browser =
          element.shadowRoot.querySelector('tfma-slicing-metrics-browser');
      assert.equal(browser.weightedExamplesColumn, expectedWeightColumn);
      done();
    };

    runWithData(pickClassIds, [{
                  'slice': 'col:1',
                  'metrics': {
                    'output1': {
                      'classId:0': {
                        'averageRefinedPrediction': {'doubleValue': 0.51},
                        'averageLabel': {'doubleValue': 0.61},
                        'weightedExamples': {'doubleValue': 10},
                      },
                      'classId:1': {
                        'averageRefinedPrediction': {'doubleValue': 0.52},
                        'averageLabel': {'doubleValue': 0.62},
                        'weightedExamples': {'doubleValue': 20},
                      },
                    },
                  },
                }]);
  });
});
