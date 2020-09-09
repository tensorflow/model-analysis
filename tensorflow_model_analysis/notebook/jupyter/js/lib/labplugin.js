var plugin = require('./index');
var base = require('@jupyter-widgets/base');

module.exports = {
  id: 'tensorflow_model_analysis',
  requires: [base.IJupyterWidgetRegistry],
  activate: function(app, widgets) {
      widgets.registerWidget({
          name: 'tensorflow_model_analysis',
          version: plugin.version,
          exports: plugin
      });
  },
  autoStart: true
};
