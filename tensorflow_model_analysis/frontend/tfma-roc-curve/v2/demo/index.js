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
  const plot = document.getElementById('plot');
  const input = [];
  let tp;
  let fn;

  let fp;
  let tn;

  for (let i = 0; i <= 128; i++) {
    const val = 1 - (i - 128) * (i - 128) / 128 / 128;
    tp = Math.round(128 * val);
    fn = 128 - tp;
    fp = i;
    tn = 128 - i;

    input.push({
      'truePositives': tp,
      'trueNegatives': tn,
      'falsePositives': fp,
      'falseNegatives': fn,
      'threshold': (128 - i) / 128,
    });
  }
  plot.data = input;
})();
