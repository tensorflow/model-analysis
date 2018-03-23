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
  const MATRICES_KEY = 'matrices';
  const BOUNDARIES_KEY = 'boundaries';
  const BUCKET_COUNT = 512;
  const view = document.getElementById('plot');

  function makeMatrices(count) {
    const buckets = [];
    for (let i = 0; i < count; i++) {
      const weight = Math.random() * 64;
      buckets.push([Math.random() * weight, Math.random() * weight, weight]);
    }
    return buckets;
  }

  function makeBoundaries(count) {
    const increment = 1 / count;
    let currentBoundary = increment;
    const boundaries = [];
    for (let i = 1; i < count; i++) {
      boundaries.push(currentBoundary);
      currentBoundary += increment;
    }
    return boundaries;
  }

  view.config = {
    'sliceName': 'Demo',
    'metricKeys': {
      'calibrationPlot': {
        'matrices': MATRICES_KEY,
        'boundaries': BOUNDARIES_KEY,
      }
    }
  };
  view.data = {
    [MATRICES_KEY]: makeMatrices(BUCKET_COUNT),
    [BOUNDARIES_KEY]: makeBoundaries(BUCKET_COUNT)
  };
})();
