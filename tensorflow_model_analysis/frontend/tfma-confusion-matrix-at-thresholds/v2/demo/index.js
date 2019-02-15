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
      '#container tfma-confusion-matrix-at-thresholds');
  const makeMatrix_ = (count) => {
    return {
      'threshold': 0.5,
      'precision': 0.6 / (100 + count) * 100,
      'recall': 0.9 / (100 + count) * 100,
      'truePositives': 0.91 / (100 + count) * 100,
      'trueNegatives': 0.92 / (100 + count) * 100,
      'falsePositives': 0.81 / (100 + count) * 100,
      'falseNegatives': 0.82 / (100 + count) * 100,
    };
  };
  const setData_ = (element, data) => {
    element.data = JSON.stringify({'matrices': data});
  };
  let recordCount = 0;
  for (let i = widgets.length - 1, widget; widget = widgets[i]; i--) {
    const data = {
      'matrices': [makeMatrix_(recordCount++), makeMatrix_(recordCount++)]
    };
    widget.data = JSON.stringify(data);
  }

  const collapsed = document.getElementById('collapsed');
  const collapsedData = [];
  for (let i = 0; i < 10; i++) {
    collapsedData.push(makeMatrix_(i), makeMatrix_(i));
  }
  setData_(collapsed, collapsedData);

  const expanded = document.getElementById('expanded');
  const expandedData = [];
  for (let i = 0; i < 10; i++) {
    expandedData.push(makeMatrix_(i), makeMatrix_(i));
  }
  setData_(expanded, expandedData);

  const nonInteractive = document.getElementById('nonInteractive');
  const nonInteractiveData = [];
  for (let i = 0; i < 10; i++) {
    nonInteractiveData.push(makeMatrix_(i), makeMatrix_(i));
  }
  nonInteractive.interactive = false;
  setData_(nonInteractive, nonInteractiveData);
})();
