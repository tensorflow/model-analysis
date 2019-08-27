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
 * @externs
 * @suppress {duplicate}
 */
var Plottable = {};

/** @constructor */
Plottable.Axes = function() {};
/** @constructor */
Plottable.Axes.Numeric = function(scale, position) {};

Plottable.Components.prototype.root = function() {};
Plottable.Components.prototype.rootElement = function() {};
/** @constructor */
Plottable.Components.Group = function(list) {};
Plottable.Components.Group.prototype.entityNearest = function(pointer) {};
Plottable.Components.Group.prototype.componentAt = function(
    rowIndex, columnIndex) {};
Plottable.Components.Group.prototype.components = function() {};
/** @constructor */
Plottable.Components.prototype.GuideLineLayer = function(orientation) {};
Plottable.Components.prototype.GuideLineLayer.prototype.scale = function(
    scale) {};
Plottable.Components.prototype.GuideLineLayer.prototype.value = function(
    value) {};
/** @constructor */
Plottable.Components.Table = function(list) {};
Plottable.Components.Table.prototype.renderTo = function(target) {};
Plottable.Components.Table.prototype.componentAt = function(
    rowIndex, columnIndex) {};


/** @constructor */
Plottable.Components.AxisLabel = function(text) {};
Plottable.Components.AxisLabel.prototype.padding = function(num) {};


/** @constructor */
Plottable.Dataset = function(data) {};

/** @constructor */
Plottable.Interactions = function() {};
Plottable.Interactions.prototype.detach = function() {};
/** @constructor */
Plottable.Interactions.Pointer = function() {};
Plottable.Interactions.Pointer.prototype.attachTo = function(component) {};
Plottable.Interactions.Pointer.prototype.onPointerExit = function(component) {};
Plottable.Interactions.Pointer.prototype.onPointerMove = function(component) {};

/** @constructor */
Plottable.Plots = function() {};
Plottable.Plots.prototype.addDataset = function(dataset) {};
/** @constructor */
Plottable.Plots.Bar = function() {};
Plottable.Plots.Bar.prototype.addDataset = function(dataset) {};
/** @constructor */
Plottable.Plots.Segment = function() {};
Plottable.Plots.Segment.prototype.addDataset = function(dataset) {};
Plottable.Plots.Segment.prototype.attr = function(attr, func) {};
Plottable.Plots.Segment.prototype.x = function(func, scale) {};
Plottable.Plots.Segment.prototype.x2 = function(func, scale) {};
Plottable.Plots.Segment.prototype.y = function(func, scale) {};
Plottable.Plots.Segment.prototype.y2 = function(func, scale) {};

/** @constructor */
Plottable.Scales = function() {};
/** @constructor */
Plottable.Scales.Category = function() {};
/** @constructor */
Plottable.Scales.Color = function() {};
Plottable.Scales.Color.prototype.range = function(scheme) {};
/** @constructor */
Plottable.Scales.Linear = function() {};


/** @constructor */
Plottable.Utils = function() {};
/** @constructor */
Plottable.Utils.DOM = function() {};
Plottable.Utils.DOM.contains = function(parent, child) {};
Plottable.Utils.DOM.prototype.getHtmlElementAncestors = function(func) {};
/** @constructor */
Plottable.Utils.Translator = function(func) {};
Plottable.Utils.Translator.prototype.isEventInside = function(func) {};
