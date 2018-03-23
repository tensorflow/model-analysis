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
  const all = document.getElementById('all');
  all.data = JSON.stringify([
    {
      'binaryClassificationThreshold': {'predictionThreshold': 0.5},
      'macroPrecision': 0.91,
      'macroRecall': 0.81,
      'macroF1Score': 0.71,
      'macroAccuracy': 0.61,
      'microPrecision': 0.92,
      'microRecall': 0.82,
      'microF1Score': 0.72,
      'microAccuracy': 0.62,
      'weightedPrecision': 0.93,
      'weightedRecall': 0.83,
      'weightedF1Score': 0.73,
      'weightedAccuracy': 0.63
    },
    {
      'binaryClassificationThreshold': {'predictionThreshold': 0.4},
      'macroPrecision': 0.913,
      'macroRecall': 0.813,
      'macroF1Score': 0.713,
      'macroAccuracy': 0.613,
      'microPrecision': 0.924,
      'microRecall': 0.824,
      'microF1Score': 0.724,
      'microAccuracy': 0.624,
      'weightedPrecision': 0.935,
      'weightedRecall': 0.835,
      'weightedF1Score': 0.735,
      'weightedAccuracy': 0.635
    }
  ]);
  const weightedOnly = document.getElementById('weighted-only');
  weightedOnly.data = JSON.stringify([{
    'binaryClassificationThreshold': {'predictionThreshold': 0.6},
    'weightedPrecision': 0.944,
    'weightedRecall': 0.844,
    'weightedF1Score': 0.744,
    'weightedAccuracy': 0.644,
  }]);
  const microOnly = document.getElementById('micro-only');
  microOnly.data = JSON.stringify([{
    'binaryClassificationThreshold': {'predictionThreshold': 0.89},
    'microPrecision': 0.3,
    'microRecall': 0.2,
    'microF1Score': 0.1,
    'microAccuracy': 0,
  }]);
  const bad = document.getElementById('bad');
  bad.data = '{';
})();
