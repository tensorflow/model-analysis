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
  const grid = document.getElementById('grid');
  const models = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  const fakeDataFromModel = (model) => model + 10;
  const getMetric = (model, metric) => {
    switch (metric) {
      case 'a':
        return Math.sin(model / 10);
        break;
      case 'b':
        return model * model;
        break;
      case 'c':
        return 1 / model;
        break;
    }
  };
  grid.provider = {
    'getLineChartData': (metric) => {
      return models.map((model) => {
        const data = fakeDataFromModel(model);
        return [
          data, model, 'Model: ' + model + ', Data: ' + data,
          getMetric(model, metric)
        ];
      });
    },

    'getModelIds': () => models,

    /**
     * Get eval config of the run with the given index.
     * @param {number} index
     * @return {?{
     *   model: (string|number),
     *   data: (string|number)
     * }}
     */
    'getEvalConfig': (index) => {
      return {
        'model': models[index],
        'data': fakeDataFromModel(models[index]),
      };
    },

    'getModelColumnName': () => 'Model',

    'getDataColumnName': () => 'Data',
  };
  grid.metrics = ['a', 'b', 'c'];

  let highlighted = 0;
  const highlight = document.getElementById('highlight-data-point');
  highlight.addEventListener('click', () => {
    const model = models[highlighted];
    const data = fakeDataFromModel(model);
    highlighted = (highlighted + 11) % models.length;
    grid.highlight({'Model': model, 'Data': data});
  });

  const clear = document.getElementById('clear-selection');
  clear.addEventListener('click', () => {
    grid.highlight(null);
  });

  const text = document.getElementById('text');
  grid.addEventListener('select', (e) => {
    text.innerText = 'Selected Model: ' + e.detail['Model'] +
        ' and Data: ' + e.detail['Data'];
  });
  grid.addEventListener('clear-selection', () => {
    text.innerText = '';
  });
})();
