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
(function() {
const calibrationPlot = document.getElementById('calibration-plot');
calibrationPlot.data = JSON.stringify({
  'types': [tfma.PlotTypes.CALIBRATION_PLOT],
  'slice': 'foo:bar',
});

const precisionRecall = document.getElementById('precision-recall');
precisionRecall.data = JSON.stringify({
  'types': [tfma.PlotTypes.PRECISION_RECALL_CURVE],
  'slice': 'bar:foo',
});
const predictionDistribution =
    document.getElementById('prediction-distribution');
predictionDistribution.data = JSON.stringify({
  'types': [tfma.PlotTypes.PREDICTION_DISTRIBUTION],
  'span': 123,
});
const multiple = document.getElementById('multiple');
multiple.data = JSON.stringify({
  'types':
      [tfma.PlotTypes.CALIBRATION_PLOT, tfma.PlotTypes.PRECISION_RECALL_CURVE],
  'span': 456,
});

const display = document.getElementById('display');
const container = document.getElementById('container');
container.addEventListener('show-plot', (e) => {
  const detail = e.detail;
  const message =
      detail.dataSpan ? 'Span: ' + detail.dataSpan : 'Slice: ' + detail.slice;
  display.textContent = message;
});
}());
