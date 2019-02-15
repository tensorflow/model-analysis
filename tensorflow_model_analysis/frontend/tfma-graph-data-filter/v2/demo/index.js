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
  const filter = document.getElementById('data-filter');
  const metrics = ['weight', 'a', 'b', 'c'];
  filter.weightedExamplesColumn = 'weight';
  filter.data = new tfma.SingleSeriesGraphData(metrics, [
    {
      'slice': 'col:1',
      'metrics': {
        'weight': 100,
        'a': 0.6,
        'b': {'value': 0.61, 'lowerBound': 0.60, 'upperBound': 0.62},
        'c': 0.51,
      }
    },
    {
      'slice': 'col:2',
      'metrics': {
        'weight': 200,
        'a': 0.62,
        'b': {'value': 0.51, 'lowerBound': 0.50, 'upperBound': 0.52},
        'c': 0.52,
      }
    }
  ]);
})();
