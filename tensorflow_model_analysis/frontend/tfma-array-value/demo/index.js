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
const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
const data1 = {
  'shape': [12],
  'dataType': 'INT32',
  'int32Values': data
};
const ex1 = document.getElementById('ex1');
ex1.data = JSON.stringify(data1);

const ex2 = document.getElementById('ex2');
ex2.arrayData = [[1,2,3,4], [5,6,7,8], [9,10,11,12]];

const data3 = {
  'shape': [4, 1, 3, 1],
  'dataType': 'INT32',
  'int32Values': data
};
const ex3 = document.getElementById('ex3');
ex3.data = JSON.stringify(data3);

const ex4 = document.getElementById('ex4');
ex4.arrayData = [[1,2], [3,4], [5,6], [7,8], [9,10], [11,12]];
})();
