/**
 * Copyright 2019 Google LLC
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
  const matrix = document.getElementById('matrix');
  const input = document.getElementById('class-count');
  const checkbox = document.getElementById('multi-label');

  const setFakeData = () => {
    const count = parseFloat(input.value);
    const multiLabel = checkbox.checked;

    let min = 0;
    let max = 10000;
    const fakeCount = (s) => Math.round(s * Math.random() * (max - min) + min);

    let errorThreshold = 0.7;

    const matrices = [];
    for (let k = 0; k <= 100; k++) {
      const entries = [];
      for (let i = 0; i < count; i++) {
        for (let j = 0; j < count; j++) {
          const hasError = i != j && Math.random() > errorThreshold;
          const hasCorrect =
              i == j || multiLabel && Math.random() <= errorThreshold;

          if (!hasCorrect && !hasError) {
            continue;
          }

          const entry = {
            'actualClassId': i,
            'predictedClassId': j,
          };

          if (multiLabel) {
            if (hasCorrect) {
              entry['truePositives'] = fakeCount(1);
              entry['trueNegatives'] = fakeCount(1);
            }
            if (hasError) {
              entry['falsePositives'] = fakeCount(1);
              entry['falseNegatives'] = fakeCount(1);
            }
          } else {
            entry['numWeightedExamples'] = fakeCount(1);
          }
          entries.push(entry);
        }

        if (!multiLabel) {
          entries.push({
            'actualClassId': i,
            'predictedClassId': -1,
            'numWeightedExamples': fakeCount(count / 4),
          });
        }
      }
      matrices.push({
        'threshold': k / 100,
        'entries': entries,
      });
    }

    const classNames = {};
    for (let i = 0; i < count; i++) {
      const prefix = (i % 3) ? (i % 2 ? '' : 'b') : 'a';
      if (prefix) {
        classNames[prefix + ':' + i] = i;
      }
    }

    matrix.data = {'matrices': matrices};
    matrix.classNames = classNames;
    matrix.multiLabel = multiLabel;
  };

  setFakeData();

  input.addEventListener('change', setFakeData);
  checkbox.addEventListener('change', setFakeData);
})();
