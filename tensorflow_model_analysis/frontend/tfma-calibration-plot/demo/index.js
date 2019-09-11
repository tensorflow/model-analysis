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
  const step = 0.00005;
  const minThreshold = -1;
  const maxThreshold = 3 + step;
  const bucketCutOff = 0.15;

  const makeOneBucket = (threshold, weight, label, prediction) => {
    const bucket = {
      'lowerThresholdInclusive': threshold - step,
      'upperThresholdExclusive': threshold,
    };

    // Drop some buckets to make sure we can handle it.
    if (Math.random() > bucketCutOff) {
      bucket['numWeightedExamples'] = weight;
      bucket['totalWeightedLabel'] = label;
      bucket['totalWeightedRefinedPrediction'] = prediction;
    }

    return bucket;
  };

  const input = [];
  let threshold = minThreshold;
  let weight;
  let prediction;
  let label;

  for (let i = 0; threshold <= maxThreshold; i++) {
    weight = Math.random() * Math.pow(10, Math.random() * 2);
    prediction = (threshold + Math.random() * step) * weight;
    label = (0.975 + 0.5 * Math.random()) * prediction;
    input.push(makeOneBucket(threshold, weight, label, prediction));
    threshold += step;
  }

  let plot = document.getElementById('plot1');
  plot.numberOfBuckets = 128;
  plot.buckets = input;

  plot = document.getElementById('plot2');
  plot.numberOfBuckets = 16;
  plot.buckets = input;
  plot.overrides = {
    'title': 'Color:' + plot['color'] + ', size:' + plot['size'] + ', scale:' +
        plot['scale'] + ', fit:' + plot['fit'],
    'colorHighValue': 'yellow',
    'colorMaxValue': 3,
  };
})();
