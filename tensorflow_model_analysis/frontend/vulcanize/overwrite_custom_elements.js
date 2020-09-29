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

/**
 * @fileoverview This overwrites customElements.define by a function that does
 * the same thing, except it also catches exceptions, in case a component has
 * already been defined.
 * Without this function, the script will stop loading TFMA on the first
 * conflict.
 */

const originalDefine = customElements.define.bind(customElements);
/**
 * Overrides customElements.define.
 * @param {string} name
 * @param {!Function} constructor
 */
customElements.define = function(name, constructor) {
  try {
    originalDefine(name, constructor);
  } catch(e) {
    console.log(e);
  }
};
