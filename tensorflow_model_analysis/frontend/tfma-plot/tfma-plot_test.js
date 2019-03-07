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

  setup(() => {
    element = fixture('plot-fixture');
  });

  test('ShowSupportedTypesAsAvailable', done => {
    element.availableTypes = [
      tfma.PlotTypes.CALIBRATION_PLOT, tfma.PlotTypes.PREDICTION_DISTRIBUTION,
      tfma.PlotTypes.MICRO_PRECISION_RECALL_CURVE, 'unsupported'
    ];

    setTimeout(() => {
      const tabs = element.shadowRoot.querySelectorAll('paper-tab');
      assert.equal(tabs.length, 3);
      let tab = tabs[0];
      assert.equal(tab.name, element.tabNames_['Calibration']);
      assert.equal(tab.textContent.trim(), 'Calibration Plot');


      tab = tabs[1];
      assert.equal(tab.name, element.tabNames_['Prediction']);
      assert.equal(tab.textContent.trim(), 'Prediction Distribution');

      tab = tabs[2];
      assert.equal(tab.name, element.tabNames_['Micro']);
      assert.equal(tab.textContent.trim(), 'Micro PR Curve');

      done();
    }, 0);
  });

  test('InitialTypeSetsSelectedTab', done => {
    element.availableTypes = [
      tfma.PlotTypes.CALIBRATION_PLOT,
      tfma.PlotTypes.PREDICTION_DISTRIBUTION,
      tfma.PlotTypes.PRECISION_RECALL_CURVE,
    ];
    element.initialType = tfma.PlotTypes.PREDICTION_DISTRIBUTION;

    setTimeout(() => {
      const tabs = element.shadowRoot.querySelector('paper-tabs');
      assert.equal(tabs.selected, element.tabNames_['Prediction']);

      const pages = element.shadowRoot.querySelector('iron-pages');
      assert.equal(pages.selected, element.tabNames_['Prediction']);

      done();
    }, 0);
  });

  test('SwitchingTabResetsInitialType', done => {
    element.availableTypes = [
      tfma.PlotTypes.CALIBRATION_PLOT,
      tfma.PlotTypes.PREDICTION_DISTRIBUTION,
      tfma.PlotTypes.PRECISION_RECALL_CURVE,
    ];
    element.initialType = tfma.PlotTypes.PREDICTION_DISTRIBUTION;

    setTimeout(() => {
      const tabs = element.shadowRoot.querySelector('paper-tabs');
      const unselectedTab = tabs.querySelector('paper-tab:not(.iron-selected)');
      unselectedTab.dispatchEvent(
          new CustomEvent('tap', {bubbles: true, composed: true}));

      assert.isNull(element.initialType);
      done();
    }, 0);
  });

  test('ShowSpinnerWhenLoading', done => {
    element.availableTypes = [
      tfma.PlotTypes.CALIBRATION_PLOT,
    ];
    element.loading = true;

    setTimeout(() => {
      const spinner = element.shadowRoot.querySelector('paper-spinner');
      assert.isTrue(spinner.active);

      element.loading = false;
      assert.isFalse(spinner.active);

      done();
    }, 0);
  });

  test('ShowReloadScreenOnError', done => {
    element.availableTypes = [
      tfma.PlotTypes.CALIBRATION_PLOT,
    ];
    element.data = null;
    element.loading = false;

    setTimeout(() => {
      const reloadScreen = element.$.reload;
      assert.isTrue(reloadScreen.offsetHeight > 0);
      done();
    }, 0);
  });

  test('TapReloadButtonFiresReloadEvent', done => {
    let reloadEventFired = false;
    element.availableTypes = [
      tfma.PlotTypes.CALIBRATION_PLOT,
    ];
    element.data = null;
    element.loading = false;
    element.addEventListener('reload-plot-data', () => {
      reloadEventFired = true;
    });

    setTimeout(() => {
      const button =
          element.shadowRoot.querySelector('#button-container paper-button');
      button.fire('tap');

      assert.isTrue(reloadEventFired);
      done();
    }, 0);
  });

  test('HideReloadScreenOnSuccess', done => {
    element.availableTypes = [
      tfma.PlotTypes.CALIBRATION_PLOT,
    ];
    const calibrationPlotData = [];
    element.data = {
      'plotData': {'bucketByRefinedPrediction': calibrationPlotData}
    };
    element.loading = false;

    setTimeout(() => {
      const reloadScreen = element.$.reload;
      assert.equal(0, reloadScreen.offsetHeight);
      done();
    }, 0);
  });

  test('ExtractCalibrationData', done => {
    element.availableTypes = [
      tfma.PlotTypes.CALIBRATION_PLOT,
    ];
    const calibrationPlotData = [];
    element.data = {
      'plotData':
          {'bucketByRefinedPrediction': {'buckets': calibrationPlotData}}
    };
    element.loading = false;

    setTimeout(() => {
      const calibrationPlot =
          element.shadowRoot.querySelector('tfma-calibration-plot');
      assert.equal(calibrationPlotData, calibrationPlot.buckets);
      done();
    }, 0);
  });

  test('ExtractPrecisionRecallCurveData', done => {
    element.availableTypes = [
      tfma.PlotTypes.CALIBRATION_PLOT,
    ];
    const precisionRecallCurveData = [];
    element.data = {
      'plotData': {
        'binaryClassificationByThreshold':
            {'matrices': precisionRecallCurveData}
      }
    };
    element.loading = false;

    setTimeout(() => {
      const precisionRecallCurve = element.$[element.tabNames_['Precision']];
      assert.equal(precisionRecallCurveData, precisionRecallCurve.data);
      done();
    }, 0);
  });

  test('ExtractMacroPrecisionRecallCurveData', done => {
    element.availableTypes = [
      tfma.PlotTypes.MACRO_PRECISION_RECALL_CURVE,
    ];
    const macroPrecisionRecallCurveData = [];
    element.data = {
      'plotData': {
        'macroValuesByThreshold': {'matrices': macroPrecisionRecallCurveData}
      }
    };
    element.loading = false;

    setTimeout(() => {
      const precisionRecallCurve = element.$[element.tabNames_['Macro']];
      assert.equal(macroPrecisionRecallCurveData, precisionRecallCurve.data);
      done();
    }, 0);
  });

  test('ExtractMicroPrecisionRecallCurveData', done => {
    element.availableTypes = [
      tfma.PlotTypes.MICRO_PRECISION_RECALL_CURVE,
    ];
    const microPrecisionRecallCurveData = [];
    element.data = {
      'plotData': {
        'microValuesByThreshold': {'matrices': microPrecisionRecallCurveData}
      }
    };
    element.loading = false;

    setTimeout(() => {
      const precisionRecallCurve = element.$[element.tabNames_['Micro']];
      assert.equal(microPrecisionRecallCurveData, precisionRecallCurve.data);
      done();
    }, 0);
  });

  test('ExtractWeightedPrecisionRecallCurveData', done => {
    element.availableTypes = [
      tfma.PlotTypes.WEIGHTED_PRECISION_RECALL_CURVE,
    ];
    const weightedPrecisionRecallCurveData = [];
    element.data = {
      'plotData': {
        'weightedValuesByThreshold':
            {'matrices': weightedPrecisionRecallCurveData}
      }
    };
    element.loading = false;

    setTimeout(() => {
      const precisionRecallCurve = element.$[element.tabNames_['Weighted']];
      assert.equal(weightedPrecisionRecallCurveData, precisionRecallCurve.data);
      done();
    }, 0);
  });
});
