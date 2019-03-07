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
  const data = [
    {'numWeightedExamples': 0, 'totalWeightedRefinedPrediction': 0},
    {'numWeightedExamples': 16, 'totalWeightedRefinedPrediction': 1.5},
    {'numWeightedExamples': 1, 'totalWeightedRefinedPrediction': 0.126},
    {'numWeightedExamples': 2, 'totalWeightedRefinedPrediction': 0.44},
    {'numWeightedExamples': 3, 'totalWeightedRefinedPrediction': 2.1},
    {'numWeightedExamples': 1, 'totalWeightedRefinedPrediction': 1}
  ];

  const plot = document.getElementById('plot');
  plot.data = data;
})();
