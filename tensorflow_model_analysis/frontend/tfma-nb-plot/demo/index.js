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
  const METRIC_NAME = 'confusionMatrixAtThresholds';
  const DATA_SERIES_NAME = 'matrices';
  const COUNT = 512;
  const view = document.getElementById('plot');

  function makeDataSeries(count) {
    const series = [];
    for (let i = 0; i < count; i++) {
      let val = 1 - (i - count) * (i - count) / count / count;
      const tp = Math.round(count * val);
      const fn = count - tp;
      const fp = i;
      const tn = count - i;

      series.push({
        'truePositives': tp,
        'trueNegatives': tn,
        'falsePositives': fp,
        'falseNegatives': fn,
        'precision': tp / (tp + fp),
        'recall': tp / (tp + fn),
        'threshold': (128 - i) / 128,
      });
    }

    return series;
  }

  view.config = {
    'sliceName': 'Demo',
    'metricKeys':
        {'aucPlot': {'metricName': METRIC_NAME, 'dataSeries': DATA_SERIES_NAME}}
  };
  view.data = {
    [METRIC_NAME]: {
      [DATA_SERIES_NAME]: makeDataSeries(COUNT),
    }
  };
})();
