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
const widgets = require('@jupyter-widgets/base');
const _ = require('lodash');
const version = require('../package.json').version;

/**
 * Helper method to load the vulcanized templates.
 */
function loadVulcanizedTemplate() {
  let templatePath;
  const dataBaseUrl =
      document.querySelector('body').getAttribute('data-base-url');
  // Jupyter Classic
  if (dataBaseUrl) {
    templatePath = dataBaseUrl + 'nbextensions/tensorflow_model_analysis/';
  }
  // Jupyter Lab
  else if (window['isJupyterLab']) {
    let baseUrl = '/';
    const jupyterConfigData = document.getElementById('jupyter-config-data');
    if (jupyterConfigData) {
      const configData = JSON.parse(jupyterConfigData.textContent || '');
      if (configData) {
        baseUrl = configMap['baseUrl'] || '/';
      }
    }

    templatePath = baseUrl + 'nbextensions/tensorflow_model_analysis/';
  }
  // Kubeflow
  else {
    templatePath = __webpack_public_path__;
  }
  // templatePath ends with a slash.
  const templateLocation = `${templatePath}vulcanized_tfma.js`;

  // If the vulcanizes tempalets are not loaded yet, load it now.
  if (!document.querySelector('script[src="' + templateLocation + '"]')) {
    const script = document.createElement('script');
    script.setAttribute('src', templateLocation);
    document.head.appendChild(script);
  }
}

/**
 * HACK: Calls the render callback in a setTimeout. This delay avoids some
 * rendering artifacts.
 * @param {!Function} cb
 */
function delayedRender(cb) {
  setTimeout(cb, 0);
}

const MODULE_NAME = 'tensorflow_model_analysis';
const MODEL_VERSION = version;
const VIEW_VERSION = version;
const SLICING_METRICS_MODEL_NAME = 'SlicingMetricsModel';
const SLICING_METRICS_VIEW_NAME = 'SlicingMetricsView';
const SLICING_METRICS_ELEMENT_NAME = 'tfma-nb-slicing-metrics';
const TIME_SERIES_MODEL_NAME = 'TimeSeriesModel';
const TIME_SERIES_VIEW_NAME = 'TimeSeriesView';
const TIME_SERIES_ELEMENT_NAME = 'tfma-nb-time-series';
const PLOT_MODEL_NAME = 'PlotModel';
const PLOT_VIEW_NAME = 'PlotView';
const PLOT_ELEMENT_NAME = 'tfma-nb-plot';
const FAIRNESS_INDICATOR_MODEL_NAME = 'FairnessIndicatorModel';
const FAIRNESS_INDICATOR_VIEW_NAME = 'FairnessIndicatorView';
const FAIRNESS_INDICATOR_ELEMENT_NAME = 'fairness-nb-container';

const SlicingMetricsModel = widgets.DOMWidgetModel.extend({
  defaults: _.extend(widgets.DOMWidgetModel.prototype.defaults(), {
    _model_name: SLICING_METRICS_MODEL_NAME,
    _view_name: SLICING_METRICS_VIEW_NAME,
    _model_module: MODULE_NAME,
    _view_module: MODULE_NAME,
    _model_module_version: MODEL_VERSION,
    _view_module_version: VIEW_VERSION,
    config: {},
    data: [],
    js_events: [],
  })
});

const SlicingMetricsView = widgets.DOMWidgetView.extend({
  render: function() {
    loadVulcanizedTemplate();

    this.view_ = document.createElement(SLICING_METRICS_ELEMENT_NAME);
    this.el.appendChild(this.view_);

    this.view_.addEventListener('tfma-event', (e) => {
      handleTfmaEvent(e, this);
    });

    delayedRender(() => {
      this.configChanged_();
      this.dataChanged_();
      this.model.on('change:config', this.configChanged_, this);
      this.model.on('change:data', this.dataChanged_, this);
    });
  },
  dataChanged_: function() {
    this.view_.data = this.model.get('data');
  },
  configChanged_: function() {
    this.view_.config = this.model.get('config');
  },
});

const TimeSeriesModel = widgets.DOMWidgetModel.extend({
  defaults: _.extend(widgets.DOMWidgetModel.prototype.defaults(), {
    _model_name: TIME_SERIES_MODEL_NAME,
    _view_name: TIME_SERIES_VIEW_NAME,
    _model_module: MODULE_NAME,
    _view_module: MODULE_NAME,
    _model_module_version: MODEL_VERSION,
    _view_module_version: VIEW_VERSION,
    config: {},
    data: [],
  })
});

const TimeSeriesView = widgets.DOMWidgetView.extend({
  render: function() {
    loadVulcanizedTemplate();

    this.view_ = document.createElement(TIME_SERIES_ELEMENT_NAME);
    this.el.appendChild(this.view_);

    delayedRender(() => {
      this.configChanged_();
      this.dataChanged_();
      this.model.on('change:config', this.configChanged_, this);
      this.model.on('change:data', this.dataChanged_, this);
    });
  },
  dataChanged_: function() {
    this.view_.data = this.model.get('data');
  },
  configChanged_: function() {
    this.view_.config = this.model.get('config');
  },
});

const PlotModel = widgets.DOMWidgetModel.extend({
  defaults: _.extend(widgets.DOMWidgetModel.prototype.defaults(), {
    _model_name: PLOT_MODEL_NAME,
    _view_name: PLOT_VIEW_NAME,
    _model_module: MODULE_NAME,
    _view_module: MODULE_NAME,
    _model_module_version: MODEL_VERSION,
    _view_module_version: VIEW_VERSION,
    config: {},
    data: [],
  })
});

const PlotView = widgets.DOMWidgetView.extend({
  render: function() {
    loadVulcanizedTemplate();

    this.view_ = document.createElement(PLOT_ELEMENT_NAME);
    this.el.appendChild(this.view_);

    delayedRender(() => {
      this.configChanged_();
      this.dataChanged_();
      this.model.on('change:config', this.configChanged_, this);
      this.model.on('change:data', this.dataChanged_, this);
    });
  },
  dataChanged_: function() {
    this.view_.data = this.model.get('data');
  },
  configChanged_: function() {
    this.view_.config = this.model.get('config');
  },
});

const FairnessIndicatorModel = widgets.DOMWidgetModel.extend({
  defaults: _.extend(widgets.DOMWidgetModel.prototype.defaults(), {
    _model_name: FAIRNESS_INDICATOR_MODEL_NAME,
    _view_name: FAIRNESS_INDICATOR_VIEW_NAME,
    _model_module: MODULE_NAME,
    _view_module: MODULE_NAME,
    _model_module_version: MODEL_VERSION,
    _view_module_version: VIEW_VERSION,
    slicingMetrics: [],
    slicingMetricsCompare: [],
    evalName: '',
    evalNameCompare: '',
    js_events: [],
  })
});

const FairnessIndicatorView = widgets.DOMWidgetView.extend({
  render: function() {
    loadVulcanizedTemplate();

    this.view_ = document.createElement(FAIRNESS_INDICATOR_ELEMENT_NAME);
    this.el.appendChild(this.view_);

    this.view_.addEventListener('tfma-event', (e) => {
      handleTfmaEvent(e, this);
    });

    delayedRender(() => {
      this.slicingMetricsChanged_();
      this.slicingMetricsCompareChanged_();
      this.evalNameChanged_();
      this.evalNameCompareChanged_();
      this.model.on('change:slicingMetrics', this.slicingMetricsChanged_, this);
      this.model.on(
          'change:slicingMetricsCompare', this.slicingMetricsCompareChanged_,
          this);
      this.model.on('change:evalName', this.evalNameChanged_, this);
      this.model.on(
          'change:evalNameCompare', this.evalNameCompareChanged_, this);
    });
  },
  slicingMetricsChanged_: function() {
    this.view_.slicingMetrics = this.model.get('slicingMetrics');
  },
  slicingMetricsCompareChanged_: function() {
    this.view_.slicingMetricsCompare = this.model.get('slicingMetricsCompare');
  },
  evalNameChanged_: function() {
    this.view_.evalName = this.model.get('evalName');
  },
  evalNameCompareChanged_: function() {
    this.view_.evalNameCompare = this.model.get('evalNameCompare');
  },
});

/**
 * Handler for events of type "tfma-event" for the given view element.
 * @param {!Event} tfmaEvent
 * @param {!Element} view
 */
const handleTfmaEvent = (tfmaEvent, view) => {
  const model = view.model;
  const jsEvents = model.get('js_events').slice();
  const detail = tfmaEvent.detail;
  jsEvents.push({'name': detail.type, 'detail': detail.detail});
  model.set('js_events', jsEvents);
  view.touch();
};

module.exports = {
  [PLOT_MODEL_NAME]: PlotModel,
  [PLOT_VIEW_NAME]: PlotView,
  [SLICING_METRICS_MODEL_NAME]: SlicingMetricsModel,
  [SLICING_METRICS_VIEW_NAME]: SlicingMetricsView,
  [TIME_SERIES_MODEL_NAME]: TimeSeriesModel,
  [TIME_SERIES_VIEW_NAME]: TimeSeriesView,
  [FAIRNESS_INDICATOR_MODEL_NAME]: FairnessIndicatorModel,
  [FAIRNESS_INDICATOR_VIEW_NAME]: FairnessIndicatorView,
};
