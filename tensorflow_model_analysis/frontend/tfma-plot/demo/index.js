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
(() => {
  const input = [];
  const count = 1000;
  const step = 1 / count;
  let threshold = 0;

  for (var i = 0; i < count; i++) {
    const weight = Math.random() * Math.pow(10, Math.random() * 2);
    const prediction = (threshold + Math.random() * step) * weight;
    const label = (0.975 + 0.5 * Math.random()) * prediction;

    input.push({
      'lowerThresholdInclusive': threshold - step,
      'upperThresholdExclusive': threshold,
      'numWeightedExamples': weight,
      'totalWeightedLabel': label,
      'totalWeightedRefinedPrediction': prediction,
    });
    threshold += step;
  }

  const calibrationPlot = document.getElementById('calibration');
  calibrationPlot.availableTypes = [
    tfma.PlotTypes.CALIBRATION_PLOT, tfma.PlotTypes.PRECISION_RECALL_CURVE,
    tfma.PlotTypes.PREDICTION_DISTRIBUTION
  ];
  calibrationPlot.initialType = tfma.PlotTypes.CALIBRATION_PLOT;
  calibrationPlot.loading = false;
  calibrationPlot.data = {
    'plotData': {'bucketByRefinedPrediction': {'buckets': input}}
  };

  const precisionRecallCurveInput = [];
  let precision = 1;
  for (let i = 0; i <= 128; i++) {
    precisionRecallCurveInput.push({
      'precision': precision,
      'recall': i / 128,
      'threshold': (128 - i) / 128,
    });
    precision -= Math.random() / 128;
  }
  const precisionRecallCurve = document.getElementById('pr-curve');
  precisionRecallCurve.availableTypes = [
    tfma.PlotTypes.CALIBRATION_PLOT, tfma.PlotTypes.PRECISION_RECALL_CURVE,
    tfma.PlotTypes.PREDICTION_DISTRIBUTION
  ];
  precisionRecallCurve.initialType = tfma.PlotTypes.PRECISION_RECALL_CURVE;
  precisionRecallCurve.loading = false;
  precisionRecallCurve.data = {
    'plotData': {
      'binaryClassificationByThreshold': {'matrices': precisionRecallCurveInput}
    }
  };

  const loading = document.getElementById('loading');
  loading.availableTypes = [
    tfma.PlotTypes.CALIBRATION_PLOT, tfma.PlotTypes.PRECISION_RECALL_CURVE,
    tfma.PlotTypes.PREDICTION_DISTRIBUTION
  ];
  loading.initialType = tfma.PlotTypes.CALIBRATION_PLOT;
  loading.loading = true;

  const error = document.getElementById('error');
  error.addEventListener('reload-plot-data', () => {
    error.loading = true;
    const prompt = document.getElementById('reload-prompt');
    prompt.textContent = 'Reloeading... Current time is ' + new Date();
    window.setTimeout(() => {
      prompt.textContent = 'Reloead completed at ' + new Date();
      error.loading = false;
    }, 1000);
  });
  error.loading = false;
  error.data = null;
  error.availableTypes = [
    tfma.PlotTypes.CALIBRATION_PLOT, tfma.PlotTypes.PRECISION_RECALL_CURVE,
    tfma.PlotTypes.PREDICTION_DISTRIBUTION
  ];
  error.initialType = tfma.PlotTypes.CALIBRATION_PLOT;
})();
