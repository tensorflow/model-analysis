# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Defines TFMA's Jupyter notebook widgets."""
import ipywidgets as widgets
from tensorflow_model_analysis.version import VERSION
import traitlets


@widgets.register
class SlicingMetricsViewer(widgets.DOMWidget):
  """The slicing metrics visualization widget."""
  _view_name = traitlets.Unicode('SlicingMetricsView').tag(sync=True)
  _model_name = traitlets.Unicode('SlicingMetricsModel').tag(sync=True)
  _view_module = traitlets.Unicode('tensorflow_model_analysis').tag(sync=True)
  _model_module = traitlets.Unicode('tensorflow_model_analysis').tag(sync=True)
  _view_module_version = traitlets.Unicode(VERSION).tag(sync=True)
  _model_module_version = traitlets.Unicode(VERSION).tag(sync=True)
  data = traitlets.List([]).tag(sync=True)
  config = traitlets.Dict(dict()).tag(sync=True)

  # Used for handling on the js side.
  event_handlers = {}
  js_events = traitlets.List([]).tag(sync=True)

  @traitlets.observe('js_events')
  def _handle_js_events(self, change):
    if self.js_events:
      if self.event_handlers:
        for event in self.js_events:
          event_name = event['name']
          if event_name in self.event_handlers:
            self.event_handlers[event_name](event['detail'])

      # clears the event queue.
      self.js_events = []


@widgets.register
class TimeSeriesViewer(widgets.DOMWidget):
  """The time series visualization widget."""
  _view_name = traitlets.Unicode('TimeSeriesView').tag(sync=True)
  _model_name = traitlets.Unicode('TimeSeriesModel').tag(sync=True)
  _view_module = traitlets.Unicode('tensorflow_model_analysis').tag(sync=True)
  _model_module = traitlets.Unicode('tensorflow_model_analysis').tag(sync=True)
  _view_module_version = traitlets.Unicode(VERSION).tag(sync=True)
  _model_module_version = traitlets.Unicode(VERSION).tag(sync=True)
  data = traitlets.List([]).tag(sync=True)
  config = traitlets.Dict(dict()).tag(sync=True)


@widgets.register
class PlotViewer(widgets.DOMWidget):
  """The time series visualization widget."""
  _view_name = traitlets.Unicode('PlotView').tag(sync=True)
  _model_name = traitlets.Unicode('PlotModel').tag(sync=True)
  _view_module = traitlets.Unicode('tensorflow_model_analysis').tag(sync=True)
  _model_module = traitlets.Unicode('tensorflow_model_analysis').tag(sync=True)
  _view_module_version = traitlets.Unicode(VERSION).tag(sync=True)
  _model_module_version = traitlets.Unicode(VERSION).tag(sync=True)
  data = traitlets.Dict([]).tag(sync=True)
  config = traitlets.Dict(dict()).tag(sync=True)
