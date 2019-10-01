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
 * A partial externs file for d3 for classes / methods used in this project.
 * @externs
 */

/**
 * @const
 * @suppress {checkTypes}
 */
var d3 = {};


// Local Variables

/**
 * @return {!d3.local}
 * @constructor
 * @template T
 */
d3.local = function() {};

// Selecting Elements

/**
 * @return {!d3.selection}
 * @constructor
 */
d3.selection = function() {};

/**
 * @param {?string | Element | Window} selector
 *     WARNING: d3.select(window) doesn't return a fully functional selection.
 *     Registering event listeners with on() and setting properties with
 *     property() work, but most methods throw an Error or silently fail.
 * @return {!d3.selection}
 */
d3.select = function(selector) {};

/**
 * @param {?string | Array<!Element> | NodeList<!Element>} selector
 * @return {!d3.selection}
 */
d3.selectAll = function(selector) {};

/**
 * @param {?string | function(this:Element): ?Element} selector
 * @return {!d3.selection}
 */
d3.selection.prototype.select = function(selector) {};

/**
 * @param {?string | function(this:Element): !IArrayLike<!Element>} selector
 * @return {!d3.selection}
 */
d3.selection.prototype.selectAll = function(selector) {};

/**
 * @param {string |
 *     function(this:Element, ?, number, !IArrayLike<!Element>): boolean} filter
 * @return {!d3.selection}
 */
d3.selection.prototype.filter = function(filter) {};

/**
 * @param {!d3.selection} other
 * @return {!d3.selection}
 */
d3.selection.prototype.merge = function(other) {};

/**
 * @param {string} selector
 * @return {function(this:Element): boolean}
 */
d3.matcher = function(selector) {};

/**
 * @param {string} selector
 * @return {function(this:Element): ?Element}
 */
d3.selector = function(selector) {};

/**
 * @param {string} selector
 * @return {function(this:Element): !IArrayLike<!Element>}
 */
d3.selectorAll = function(selector) {};

/**
 * @param {!(Node | Document | Window)} node
 * @return {!Window}
 */
d3.window = function(node) {};

/**
 * @param {!Element} node
 * @param {string} name
 * @return {string}
 */
d3.style = function(node, name) {};

/**
 * @param {?string | d3.transition=} nameOrTransition
 * @return {!d3.transition}
 * @constructor
 */
d3.transition = function(nameOrTransition) {};


// Modifying Elements

/**
 * @param {string} name
 * @param {?string | number | boolean | d3.local |
 *     function(this:Element, ?, number, !IArrayLike<!Element>):
 *         ?(string | number | boolean)=} value
 */
d3.selection.prototype.attr = function(name, value) {};

/**
 * @param {string} names Space separated CSS class names.
 * @param {boolean |
 *     function(this:Element, ?, number, !IArrayLike<!Element>): boolean=}
 *     value
 */
d3.selection.prototype.classed = function(names, value) {};

/**
 * @param {string} name
 * @param {?string | number |
 *    function(this:Element, ?, number, !IArrayLike<!Element>):
 *        ?(string | number)=} value
 * @param {?string=} priority
 * @return {?} Style value (1 argument) or this (2+ arguments)
 */
d3.selection.prototype.style = function(name, value, priority) {};

/**
 * @param {string | !d3.local} name
 * @param {* | function(this:Element, ?, number, !IArrayLike<!Element>)=}
 *     value
 */
d3.selection.prototype.property = function(name, value) {};

/**
 * @param {?string |
 *     function(this:Element, ?, number, !IArrayLike<!Element>): ?string=}
 *     value
 */
d3.selection.prototype.text = function(value) {};

/**
 * @param {?string |
 *     function(this:Element, ?, number, !IArrayLike<!Element>): ?string=}
 *     value
 */
d3.selection.prototype.html = function(value) {};

/**
 * @param {string |
 *     function(this:Element, ?, number, !IArrayLike<!Element>): !Element} type
 * @return {!d3.selection}
 */
d3.selection.prototype.append = function(type) {};

/**
 * @param {string |
 *     function(this:Element, ?, number, !IArrayLike<!Element>): !Element} type
 * @param {?string |
 *     function(this:Element, ?, number, !IArrayLike<!Element>): ?Element=}
 *     before
 * @return {!d3.selection}
 */
d3.selection.prototype.insert = function(type, before) {};

/**
 * @return {!d3.selection}
 */
d3.selection.prototype.remove = function() {};

/**
 * @param {boolean=} deep
 * @return {!d3.selection}
 */
d3.selection.prototype.clone = function(deep) {};

/**
 * @param {function(?, ?): number} compare
 * @return {!d3.selection}
 */
d3.selection.prototype.sort = function(compare) {};

/**
 * @return {!d3.selection}
 */
d3.selection.prototype.order = function() {};

/**
 * @return {!d3.selection}
 */
d3.selection.prototype.raise = function() {};

/**
 * @return {!d3.selection}
 */
d3.selection.prototype.lower = function() {};

/**
 * @param {string} name
 * @return {!d3.selection}
 */
d3.create = function(name) {};

/**
 * @param {string} name
 * @return {function(this:Element): !Element}
 */
d3.creator = function(name) {};

// Control Flow

/**
 * @param {function(this:Element, ?, number, !IArrayLike<!Element>)} callback
 * @return {!d3.selection}
 */
d3.selection.prototype.each = function(callback) {};


/**
 * Adding the d3.tip.x into the param type annotation. Then the js compiler will
 * not complain that the object passed in is not a function.
 * @param {(!Function|!d3.tip.x)} callback
 * @param {...?} var_args
 */
d3.selection.prototype.call = function(callback, var_args) {};

/**
 * @return {boolean}
 */
d3.selection.prototype.empty = function() {};

/**
 * @return {!Array<!Element>}
 */
d3.selection.prototype.nodes = function() {};

/**
 * @return {?Element}
 */
d3.selection.prototype.node = function() {};

/**
 * @return {number}
 */
d3.selection.prototype.size = function() {};


// Joining Data

/**
 * @param {!Array |
 *     function(this:Element, ?, number, !IArrayLike<!Element>): !Array=}
 *     data
 * @param {function(this:Element, ?, number, !IArrayLike)=} key
 */
d3.selection.prototype.data = function(data, key) {};

/**
 * @return {!d3.selection}
 */
d3.selection.prototype.enter = function() {};

/**
 * @return {!d3.selection}
 */
d3.selection.prototype.exit = function() {};

/**
 * @param {* | function(this:Element, ?, number, !IArrayLike<!Element>)=}
 *     value
 * @return {?}
 */
d3.selection.prototype.datum = function(value) {};


/**
 * @param {!Element} container
 * @return {!Array<number>}
 */
d3.mouse = function(container) {};

// Handling Events

/**
 * @param {string} typenames
 * @param {?function(this:Element, ?, number, !IArrayLike<!Element>)=}
 *     listener
 * @param {boolean=} capture
 * @return {?} d3.selection (2+ arguments), listener function (1 argument) or
 *     undefined (1 argument).
 */
d3.selection.prototype.on = function(typenames, listener, capture) {};


// Band Scales

/**
 * @return {!d3.BandScale}
 */
d3.scaleBand = function() {};

/**
 * @typedef {function(string): number}
 */
d3.BandScale;

/**
 * @private {!d3.BandScale}
 */
d3.BandScale_;

/**
 * @param {!(Array<string> | Array<number>)=} domain
 */
d3.BandScale_.domain = function(domain) {};

/**
 * @param {!Array<number>=} range
 */
d3.BandScale_.range = function(range) {};

/**
 * @param {!Array<number>=} range
 */
d3.BandScale_.rangeRound = function(range) {};

/**
 * @param {boolean=} round
 */
d3.BandScale_.round = function(round) {};

/**
 * @param {number=} padding
 */
d3.BandScale_.paddingInner = function(padding) {};

/**
 * @param {number=} padding
 */
d3.BandScale_.paddingOuter = function(padding) {};

/**
 * @param {number=} padding
 */
d3.BandScale_.padding = function(padding) {};

/**
 * @param {number=} align
 */
d3.BandScale_.align = function(align) {};

/**
 * @return {number}
 */
d3.BandScale_.bandwidth = function() {};

/**
 * @return {number}
 */
d3.BandScale_.step = function() {};

/**
 * @return {!d3.BandScale}
 */
d3.BandScale_.copy = function() {};


// Linear Scales

/**
 * @return {!d3.LinearScale}
 */
d3.scaleLinear = function() {};

/**
 * Besides numbers, continuous scales also support RGB string ranges.
 * @typedef {function(number): ?}
 */
d3.LinearScale;

/**
 * @private {!d3.LinearScale}
 */
d3.LinearScale_;

/**
 * @param {number} value
 * @return {number}
 */
d3.LinearScale_.invert = function(value) {};

/**
 * @param {!Array<number>=} domain
 */
d3.LinearScale_.domain = function(domain) {};

/**
 * @param {!(Array<number> | Array<string>)=} range
 */
d3.LinearScale_.range = function(range) {};

/**
 * @param {!Array<number>=} range
 */
d3.LinearScale_.rangeRound = function(range) {};

/**
 * @param {boolean=} clamp
 */
d3.LinearScale_.clamp = function(clamp) {};

/**
 * @param {function(?, ?): function(number)=} interpolate
 */
d3.LinearScale_.interpolate = function(interpolate) {};

/**
 * @param {number=} count
 * @return {!Array<number>}
 */
d3.LinearScale_.ticks = function(count) {};

/**
 * @param {number=} count
 * @param {string=} specifier
 * @return {function(number): string}
 */
d3.LinearScale_.tickFormat = function(count, specifier) {};

/**
 * @param {number=} count
 * @return {!d3.LinearScale}
 */
d3.LinearScale_.nice = function(count) {};

/**
 * @return {!d3.LinearScale}
 */
d3.LinearScale_.copy = function() {};

// Ordinal Scales

/**
 * @param {!Array<?>=} range
 * @return {!d3.OrdinalScale}
 */
d3.scaleOrdinal = function(range) {};

/**
 * @typedef {function((string | number)): ?}
 */
d3.OrdinalScale;

/**
 * @private {!d3.OrdinalScale}
 */
d3.OrdinalScale_;

/**
 * @param {!(Array<string> | Array<number>)=} domain
 */
d3.OrdinalScale_.domain = function(domain) {};

/**
 * @param {!Array=} range
 */
d3.OrdinalScale_.range = function(range) {};

/**
 * @param {?=} value
 */
d3.OrdinalScale_.unknown = function(value) {};

/**
 * @return {!d3.OrdinalScale}
 */
d3.OrdinalScale_.copy = function() {};

/**
 * @type {{name: string}}
 */
d3.scaleImplicit;

// Curves

/**
 * @param {!CanvasPathMethods} context
 * @return {!d3.Curve2d}
 */
d3.curveBasis = function(context) {};

/**
 * @param {!CanvasPathMethods} context
 * @return {!d3.Curve2d}
 */
d3.curveBasisClosed = function(context) {};

/**
 * @param {!CanvasPathMethods} context
 * @return {!d3.Curve2d}
 */
d3.curveBasisOpen = function(context) {};

/**
 * @param {!CanvasPathMethods} context
 * @return {!d3.Curve}
 */
d3.curveBundle = function(context) {};

/**
 * @param {number} beta
 * @return {function(!CanvasPathMethods): !d3.Curve}
 */
d3.curveBundle.beta = function(beta) {};

/**
 * @param {!CanvasPathMethods} context
 * @return {!d3.Curve2d}
 */
d3.curveCardinal = function(context) {};

/**
 * @param {number} tension
 * @return {function(!CanvasPathMethods): !d3.Curve2d}
 */
d3.curveCardinal.tension = function(tension) {};

/**
 * @param {!CanvasPathMethods} context
 * @return {!d3.Curve2d}
 */
d3.curveCardinalClosed = function(context) {};

/**
 * @param {number} tension
 * @return {function(!Object): !d3.Curve2d}
 */
d3.curveCardinalClosed.tension = function(tension) {};

/**
 * @param {!CanvasPathMethods} context
 * @return {!d3.Curve2d}
 */
d3.curveCardinalOpen = function(context) {};

/**
 * @param {number} tension
 * @return {function(!CanvasPathMethods): !d3.Curve2d}
 */
d3.curveCardinalOpen.tension = function(tension) {};

/**
 * @param {!CanvasPathMethods} context
 * @return {!d3.Curve2d}
 */
d3.curveCatmullRom = function(context) {};

/**
 * @param {number} alpha
 * @return {function(!CanvasPathMethods): !d3.Curve2d}
 */
d3.curveCatmullRom.alpha = function(alpha) {};

/**
 * @param {!CanvasPathMethods} context
 * @return {!d3.Curve2d}
 */
d3.curveCatmullRomClosed = function(context) {};

/**
 * @param {number} alpha
 * @return {function(!Object): !d3.Curve2d}
 */
d3.curveCatmullRomClosed.alpha = function(alpha) {};

/**
 * @param {!CanvasPathMethods} context
 * @return {!d3.Curve2d}
 */
d3.curveCatmullRomOpen = function(context) {};

/**
 * @param {number} alpha
 * @return {function(!CanvasPathMethods): !d3.Curve2d}
 */
d3.curveCatmullRomOpen.alpha = function(alpha) {};

/**
 * @param {!CanvasPathMethods} context
 * @return {!d3.Curve2d}
 */
d3.curveLinear = function(context) {};

/**
 * @param {!CanvasPathMethods} context
 * @return {!d3.Curve2d}
 */
d3.curveLinearClosed = function(context) {};

/**
 * @param {!CanvasPathMethods} context
 * @return {!d3.Curve2d}
 */
d3.curveMonotoneX = function(context) {};

/**
 * @param {!CanvasPathMethods} context
 * @return {!d3.Curve2d}
 */
d3.curveMonotoneY = function(context) {};

/**
 * @param {!CanvasPathMethods} context
 * @return {!d3.Curve2d}
 */
d3.curveNatural = function(context) {};

/**
 * @param {!CanvasPathMethods} context
 * @return {!d3.Curve2d}
 */
d3.curveStep = function(context) {};

/**
 * @param {!CanvasPathMethods} context
 * @return {!d3.Curve2d}
 */
d3.curveStepAfter = function(context) {};

/**
 * @param {!CanvasPathMethods} context
 * @return {!d3.Curve2d}
 */
d3.curveStepBefore = function(context) {};

// Custom Curves

/**
 * @interface
 */
d3.Curve = function() {};

/**
 * @return {void}
 */
d3.Curve.prototype.lineStart = function() {};

/**
 * @return {void}
 */
d3.Curve.prototype.lineEnd = function() {};

/**
 * @param {number} x
 * @param {number} y
 * @return {void}
 */
d3.Curve.prototype.point = function(x, y) {};

/**
 * @interface
 * @extends {d3.Curve}
 */
d3.Curve2d = function() {};

/**
 * @return {void}
 */
d3.Curve2d.prototype.areaStart = function() {};

/**
 * @return {void}
 */
d3.Curve2d.prototype.areaEnd = function() {};

// Lines

/**
 * @return {!d3.Line}
 */
d3.line = function() {};

/**
 * @typedef {function(!Array)}
 */
d3.Line;

/**
 * @private {!d3.Line}
 */
d3.Line_;

/**
 * @param {number | function(T, number, !Array<T>): number=} x
 * @template T
 */
d3.Line_.x = function(x) {};

/**
 * @param {number | function(T, number, !Array<T>): number=} y
 * @template T
 */
d3.Line_.y = function(y) {};

/**
 * @param {boolean | function(T, number, !Array<T>): boolean=} defined
 * @template T
 */
d3.Line_.defined = function(defined) {};

/**
 * @param {function(!CanvasPathMethods): !d3.Curve=} curve
 */
d3.Line_.curve = function(curve) {};

/**
 * @param {?CanvasPathMethods=} context
 */
d3.Line_.context = function(context) {};

/**
 * @return {!d3.RadialLine}
 * @deprecated Use d3.lineRadial
 */
d3.radialLine = function() {};

/**
 * @return {!d3.RadialLine}
 */
d3.lineRadial = function() {};

/**
 * @typedef {function(!Array)}
 */
d3.RadialLine;

/**
 * @private {!d3.RadialLine}
 */
d3.RadialLine_;

/**
 * @param {number | function(T, number, !Array<T>): number=} angle
 * @template T
 */
d3.RadialLine_.angle = function(angle) {};

/**
 * @param {number | function(T, number, !Array<T>): number=} radius
 * @template T
 */
d3.RadialLine_.radius = function(radius) {};

/**
 * @param {boolean | function(T, number, !Array<T>): boolean=} defined
 * @template T
 */
d3.RadialLine_.defined = function(defined) {};

/**
 * @param {function(!CanvasPathMethods): !d3.Curve=} curve
 */
d3.RadialLine_.curve = function(curve) {};

/**
 * @param {?CanvasPathMethods=} context
 */
d3.RadialLine_.context = function(context) {};


// Axes

/**
 * @param {function(?): ?} scale
 * @return {!d3.Axis}
 */
d3.axisTop = function(scale) {};

/**
 * @param {function(?): ?} scale
 * @return {!d3.Axis}
 */
d3.axisRight = function(scale) {};

/**
 * @param {function(?): ?} scale
 * @return {!d3.Axis}
 */
d3.axisBottom = function(scale) {};

/**
 * @param {function(?): ?} scale
 * @return {!d3.Axis}
 */
d3.axisLeft = function(scale) {};

/**
 * @typedef {function(!(d3.selection | d3.transition))}
 */
d3.Axis;

/**
 * @private {!d3.Axis}
 */
d3.Axis_;

/**
 * @param {function(?): ?=} scale
 */
d3.Axis_.scale = function(scale) {};

/**
 * @param {?} countOrIntervalOrAny
 * @param {...?} var_args
 * @return {!d3.Axis}
 */
d3.Axis_.ticks = function(countOrIntervalOrAny, var_args) {};

/**
 * @param {!Array=} args
 */
d3.Axis_.tickArguments = function(args) {};

/**
 * @param {?Array=} values
 */
d3.Axis_.tickValues = function(values) {};

/**
 * @param {?function(?): string=} format
 */
d3.Axis_.tickFormat = function(format) {};

/**
 * @param {number=} size
 */
d3.Axis_.tickSize = function(size) {};

/**
 * @param {number=} size
 */
d3.Axis_.tickSizeInner = function(size) {};

/**
 * @param {number=} size
 */
d3.Axis_.tickSizeOuter = function(size) {};

/**
 * @param {number=} padding
 */
d3.Axis_.tickPadding = function(padding) {};

// Dragging

/**
 * @return {!d3.Drag}
 */
d3.drag = function() {};

/**
 * @typedef {function(!d3.selection)}
 */
d3.Drag;

/**
 * @private {!d3.Drag}
 */
d3.Drag_;

/**
 * @param {!Element | function(this:Element, T, !Array<T>): !Element=}
 *     container
 * @template T
 */
d3.Drag_.container = function(container) {};

/**
 * @param {function(this:Element, T, !Array<T>): boolean=} filter
 * @template T
 */
d3.Drag_.filter = function(filter) {};

/**
 * @param {function(this:Element): boolean=} touchable
 * @return {!Function}
 */
d3.Drag_.touchable = function(touchable) {};

/**
 * @param {function(this:Element, T, !Array<T>)=} subject
 * @template T
 */
d3.Drag_.subject = function(subject) {};

/**
 * @param {number=} distance
 * @return {?} Distance (0 arguments) or this (1 argument).
 */
d3.Drag_.clickDistance = function(distance) {};

/**
 * @param {?function(this:Element, T, number, !Array<T>): void=}
 *     listener
 * @template T
 */
d3.Drag_.on = function(typenames, listener) {};

/**
 * @param {!Window} window
 * @return {void}
 */
d3.dragDisable = function(window) {};

/**
 * @param {!Window} window
 * @param {boolean=} noclick
 * @return {void}
 */
d3.dragEnable = function(window, noclick) {};

// Drag Events

/**
 * @interface
 */
d3.DragEvent = function() {};

/**
 * @type {!d3.Drag}
 */
d3.DragEvent.prototype.target;

/**
 * @type {string}
 */
d3.DragEvent.prototype.type;

/**
 * @type {?}
 */
d3.DragEvent.prototype.subject;

/**
 * @type {number}
 */
d3.DragEvent.prototype.x;

/**
 * @type {number}
 */
d3.DragEvent.prototype.y;

/**
 * @type {number}
 */
d3.DragEvent.prototype.dx;

/**
 * @type {number}
 */
d3.DragEvent.prototype.dy;

/**
 * @type {number | string}
 */
d3.DragEvent.prototype.identifier;

/**
 * @type {number}
 */
d3.DragEvent.prototype.active;

/**
 * @type {!Event}
 */
d3.DragEvent.prototype.sourceEvent;

/**
 * @param {string} typenames
 * @param {?function(this:Element, ?, number, !IArrayLike<!Element>)=}
 *     listener
 */
d3.DragEvent.prototype.on = function(typenames, listener) {};

// Statistics

/**
 * @param {!Array<T>} array
 * @param {?function(T, number, !Array<T>): U=} accessor
 * @return {U | undefined}
 * @template T, U
 */
d3.min = function(array, accessor) {};

/**
 * @param {!Array<T>} array
 * @param {?function(T, number, !Array<T>): U=} accessor
 * @return {U | undefined}
 * @template T, U
 */
d3.max = function(array, accessor) {};
