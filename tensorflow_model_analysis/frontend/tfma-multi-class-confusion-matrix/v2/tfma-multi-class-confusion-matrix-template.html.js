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
const template = /** @type {!HTMLTemplateElement} */(document.createElement('template'));
template.innerHTML = `
<tfma-matrix row-label="Actual" column-label="Predcited" expanded="{{expanded}}"
             data="[[summary_.matrix]]" on-expand="onExpand_"
             min-value="[[summary_.weight.min]]" max-value="[[summary_.weight.max]]">
</tfma-matrix>
`;
export {template};
