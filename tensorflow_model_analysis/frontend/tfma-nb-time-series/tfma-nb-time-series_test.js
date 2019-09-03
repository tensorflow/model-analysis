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
   * The component element to test.
   */
  let element;

  /**
   * Sets up the test fixture and runs the test by calling the provided
   * callback.
   * @param {function()} cb
   */
  function run(cb) {
    element = fixture('slicing-metrics');
    element.data = [
      {
        'config': {
          'dataIdentifier': 'a.data',
          'modelIdentifier': 'a.model',
        },
        'metrics': {
          '': {
            '': {
              'averageRefinedPrediction': {'doubleValue': 0.51},
              'averageLabel': {'doubleValue': 0.61},
              'weightedExamples': {'doubleValue': 10},
            },
          },
        }
      },
      {
        'config': {
          'dataIdentifier': 'b.data',
          'modelIdentifier': 'b.model',
        },
        'metrics': {
          '': {
            '': {
              'averageRefinedPrediction': {'doubleValue': 0.52},
              'averageLabel': {'doubleValue': 0.62},
              'weightedExamples': {'doubleValue': 20},
            },
          },
        },
      },
      {
        'config': {
          'dataIdentifier': 'c.data',
          'modelIdentifier': 'c.model',
        },
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
    ];
    element.config = {'isDataCentric': true};
    setTimeout(cb, 0);
  }

  test('ParseData', done => {
    const checkDataParsed = () => {
      const seriesData =
          element.shadowRoot.querySelector('tfma-time-series-browser')
              .seriesData;
      assert.deepEqual(seriesData.getMetrics(), [
        'averageLabel', 'averageRefinedPrediction', 'calibration',
        'weightedExamples'
      ]);
      assert.deepEqual(
          seriesData.getModelIds(), ['c.model', 'b.model', 'a.model']);
      assert.deepEqual(seriesData.getDataTable(), [
        ['c.data', 'c.model', 0.63, 0.53, 0.8412698412698413, 30],
        ['b.data', 'b.model', 0.62, 0.52, 0.8387096774193549, 20],
        ['a.data', 'a.model', 0.61, 0.51, 0.8360655737704918, 10]
      ]);
      done();
    };
    run(checkDataParsed);
  });
});
