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
  const CALIBRATION_PLOT = 'matrices';
  const CALIBRATION_PLOT_BUCKETS = 'boundaries';
  const AUC_PLOT = 'auc_matrices';
  const AUC_MATRICES = 'thresholds';
  const BUCKETS = [{'v': 1}, {'v': 2}];
  const MATRICES = [{'a': 1}, {'b': 2}, {'c': 3}];
  const DEFAULT_DATA = {
    [CALIBRATION_PLOT]: {
      [CALIBRATION_PLOT_BUCKETS]: BUCKETS,
    },
    [AUC_PLOT]: {
      [AUC_MATRICES]: MATRICES,
    }
  };
  const DEFAULT_KEYS = {
    'calibrationPlot': {
      'metricName': CALIBRATION_PLOT,
      'dataSeries': CALIBRATION_PLOT_BUCKETS,
    },
    'aucPlot': {
      'metricName': AUC_PLOT,
      'dataSeries': AUC_MATRICES,
    }
  };

  let element;

  /**
   * Sets up the test fixture with the given data and metric keys and runs the
   * test by calling the provided callback.
   * @param {!Object} data
   * @param {!Object} keys
   * @param {function()} cb
   */
  function runWith(data, keys, cb) {
    element = fixture('plot');
    element.data = data;
    element.config = {'sliceName': 'Test', 'metricKeys': keys};
    setTimeout(cb, 0);
  }

  /**
   * Sets up the test fixture with the default data and metric keys and runs the
   * test by calling the provided callback.
   * @param {function()} cb
   */
  function run(cb) {
    runWith(DEFAULT_DATA, DEFAULT_KEYS, cb);
  }

  test('ParseData', done => {
    const checkDataParsed = () => {
      assert.deepEqual(element.plotData_, {
        'plotData': {
          [tfma.PlotDataFieldNames.CALIBRATION_DATA]:
              {[CALIBRATION_PLOT_BUCKETS]: BUCKETS},
          [tfma.PlotDataFieldNames.PRECISION_RECALL_CURVE_DATA]:
              {[AUC_MATRICES]: MATRICES}
        }
      });
      assert.deepEqual(element.availableTypes_, [
        tfma.PlotTypes.CALIBRATION_PLOT,
        tfma.PlotTypes.RESIDUAL_PLOT,
        tfma.PlotTypes.PREDICTION_DISTRIBUTION,
        tfma.PlotTypes.PRECISION_RECALL_CURVE,
        tfma.PlotTypes.ROC_CURVE,
        tfma.PlotTypes.ACCURACY_CHARTS,
        tfma.PlotTypes.GAIN_CHART,
      ]);
      done();
    };
    run(checkDataParsed);
  });

  test('ParseHeading', done => {
    const checkHeading = () => {
      assert.deepEqual(element.heading_, 'Plot for Test');
      done();
    };
    run(checkHeading);
  });

  test('KeyMismatch', done => {
    const checkNoCalibrationPlot = () => {
      assert.isUndefined(
          element
              .plotData_['plotData'][tfma.PlotDataFieldNames.CALIBRATION_DATA]);
      assert.equal(element.availableTypes_.length, 0);
      assert.equal(element.heading_, 'Plot data not available');
      done();
    };
    runWith(
        DEFAULT_DATA, {'no keys': {'for': 'calibration plot'}},
        checkNoCalibrationPlot);
  });
});
