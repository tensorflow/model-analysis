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
const count = 50;
const step = 0.02;

const makeOneBucket = (threshold, weight, prediction) => {
  const bucket = {
    'lowerThresholdInclusive': threshold - step,
    'upperThresholdExclusive': threshold,
  };
  bucket['numWeightedExamples'] = weight;
  bucket['totalWeightedLabel'] = Math.ceil(weight * Math.random());
  bucket['totalWeightedRefinedPrediction'] = prediction * weight;
  return bucket;
};

const input = [];
let threshold = 0;
let weight;
let prediction;

for (let i = 0; i < count; i++) {
  weight = Math.ceil(Math.random() * 300);
  prediction = threshold - step * Math.random();
  input.push(makeOneBucket(threshold, weight, prediction));
  threshold += step;
}

let plot = document.getElementById('plot');
plot.data = input;
})();
