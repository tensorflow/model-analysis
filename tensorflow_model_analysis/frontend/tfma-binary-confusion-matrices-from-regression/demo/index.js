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
  const widgets = document.querySelectorAll(
      '#container tfma-binary-confusion-matrices-from-regression');
  const makeMatrix_ = (count) => {
    return {
      'binaryClassificationThreshold': {'predictionThreshold': 0.5},
      'matrix': {
        'f1Score': 0.8 / (100 + count) * 100,
        'accuracy': 0.7 / (100 + count) * 100,
        'precision': 0.6 / (100 + count) * 100,
        'recall': 0.9 / (100 + count) * 100,
      }
    };
  };
  let recordCount = 0;
  for (let i = widgets.length - 1, widget; widget = widgets[i]; i--) {
    const data = [makeMatrix_(recordCount++), makeMatrix_(recordCount++)];
    widget.data = JSON.stringify(data);
  }
  const defaultValuesOmitted =
      document.querySelector('#default-values-omitted');
  defaultValuesOmitted.data =
      JSON.stringify([{'binaryClassificationThreshold': {}, 'matrix': {}}]);
})();
