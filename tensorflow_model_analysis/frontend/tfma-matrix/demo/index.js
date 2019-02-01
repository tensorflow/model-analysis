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
  const simple = document.getElementById('simple');
  simple.data = {
    'R1': {
      'C1': {
        'value': 50,
        'tooltip': 'Click',
        'details': 'R1, C1',
      },
      'C2': {
        'value': 15,
        'tooltip': 'Click',
        'details': 'R1, C2',
      },
      'C3': {
        'value': 80,
        'tooltip': 'Click',
        'details': 'R1, C3',
      },
    },
    'R2': {
      'C1': {
        'value': 100,
        'tooltip': 'Click',
        'details': 'R2, C1',
      },
      'C3': {
        'value': 0,
        'tooltip': 'Click',
        'details': 'R2, C3',
      },
    },
  };

  const pivot = document.getElementById('pivot');
  pivot.data = {
    'R1': {
      'C1': {
        'value': 0.5,
        'tooltip': 'Click',
        'details': 'R1, C1',
      },
      'C2': {
        'value': 0.7,
        'tooltip': 'Click',
        'details': 'R1, C2',
      },
      'C3': {
        'value': -0.5,
        'tooltip': 'Click',
        'details': 'R1, C3',
      },
    },
    'R2': {
      'C1': {
        'value': -0.12,
        'tooltip': 'Click',
        'details': 'R2, C1',
      },
      'C2': {
        'value': 0.3,
        'tooltip': 'Click',
        'details': 'R2, C2',
      },
      'C3': {
        'value': 1,
        'tooltip': 'Click',
        'details': 'R2, C3',
      },
    },
    'R3': {
      'C1': {
        'value': -0.33,
        'tooltip': 'Click',
        'details': 'R2, C1',
      },
      'C2': {
        'value': -1,
        'tooltip': 'Click',
        'details': 'R2, C2',
      },
      'C3': {
        'value': 0,
        'tooltip': 'Click',
        'details': 'R2, C3',
      },
    },
  };

  const makeData = (row, column) => {
    const data = {};
    for (let i = 1; i <= row; i++) {
      const currentRow = {};
      for (let j = 1; j <= column; j++) {
        currentRow['C' + j] = {
          'value': Math.round(Math.random() * 100),
          'tooltip': 'Click',
          'details': 'R' + i + ', C' + j,
        };
      }
      data['R' + i] = currentRow;
    }
    return data;
  };

  const expandMe = document.getElementById('expand-me');
  expandMe.data = makeData(10, 8);

  const unexpandable = document.getElementById('unexpandable');
  unexpandable.data = makeData(10, 8);
  unexpandable.addEventListener('expand', (e) => {
    e.preventDefault();
  });

  const scale = document.getElementById('scale');
  scale.data = simple.data;
})();
