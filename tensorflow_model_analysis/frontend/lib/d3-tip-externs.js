/**
 * Copyright 2019 Google LLC
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
 * A partial externs file for d3-tip for classes / methods used in this project.
 * @externs
 */


/**
 * @return {?}
 */
d3.tip = function() {};

/**
 * @constructor
 */
d3.tip.x = function() {};

/**
 * @param {!Object|string} el
 * @param {Array<number>=} offsets
 * @param {?=} target
 * @return {?}
 */
d3.tip.x.prototype.show;

/**
 * @param {!Object} el
 * @return {?}
 */
d3.tip.x.prototype.hide;

/**
 * @param {string} property
 * @param {string} value
 * @return {?}
 */
d3.tip.x.prototype.attr;

/**
 * @param {string} property
 * @param {string} value
 * @return {?}
 */
d3.tip.x.prototype.style;

/**
 * @param {string} dir
 * @return {?}
 */
d3.tip.x.prototype.direction;

/**
 * @param {!Array<number>} dimensions
 * @return {?}
 */
d3.tip.x.prototype.offset;

/**
 * @param {function(?) : ?} func
 * @return {?}
 */
d3.tip.x.prototype.html;
