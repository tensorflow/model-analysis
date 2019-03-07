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
<style>
  #root div:nth-child(even) {
    background: #e4e4e4;
  }
  #root div {
    text-align: right;
  }
</style>
<div id="root" on-tap="toggleExpanded_">
  <template is="dom-repeat" items="[[values_]]">
    <div>[[item]]</div>
  </template>
</div>
<template is="dom-if" if="[[expandable_]]">
  <paper-tooltip for="root">
    More data available. Click anywhere to expand / collapse.
  </paper-tooltip>
</template>
`;
export {template};
