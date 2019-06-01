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
  const plot = document.body.querySelector('tfma-gain-chart');
  const input = [];
  const count = 1000;

  let totalPositive = 3 * count;
  let totalNegative = count;
  let truePositives = 0;
  let falsePositives = 0;

  for (let i = 0; i <= count; i++) {
    input.push({
      'truePositives': truePositives,
      'falsePositives': falsePositives,
      'trueNegatives': totalNegative - falsePositives,
      'falseNegatives': totalPositive - truePositives,
      'threshold': (count - i) / count,
    });

    truePositives = Math.min(
        totalPositive, truePositives + Math.round((1 + Math.random()) * 30));
    falsePositives = Math.min(
        totalNegative, 1 + falsePositives + Math.round(Math.random() * 5));
  }

  plot.data = input;
})();
