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
const makeData = (count) => {
  const classes = [];
  // Craete some random class naems consisting of digits.
  for (let i = 0; i < count; i++) {
    classes.push(Math.random().toFixed(10).substring(2, 9));
  }

  const entries = [];
  classes.forEach(actual => {
    classes.forEach(predicted => {
      entries.push({
        'actualClass': actual,
        'predictedClass': predicted,
        'weight': Math.round(Math.random() * 1000),
      });
    });
  });
  return entries;
};

const matrix1 = document.getElementById('json');
matrix1.jsonData = {
  'entries': [
    {'actualClass': 'A', 'predictedClass': 'A', 'weight': 100},
    {'actualClass': 'A', 'predictedClass': 'B', 'weight': 10},
    {'actualClass': 'B', 'predictedClass': 'A', 'weight': 50},
    {'actualClass': 'B', 'predictedClass': 'B', 'weight': 30}
  ],
};
const matrix2 = document.getElementById('serialized');
matrix2.data = JSON.stringify({'entries': makeData(16)});
const matrix3 = document.getElementById('unexpandable');
matrix3.jsonData = {
  'entries': makeData(16)
};
matrix3.addEventListener('expand-confusion-matrix', (e) => {
  e.preventDefault();
});
const matrix4 = document.getElementById('scale');
matrix4.jsonData = {
  'entries': makeData(3)
};
const matrix5 = document.getElementById('omit');
matrix5.jsonData = {
  'entries': makeData(1024)
};
})();
