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
from traitlets import Dict
from traitlets import List
from traitlets import observe
from traitlets import Unicode


@widgets.register
class SlicingMetricsViewer(widgets.DOMWidget):
  """The slicing metrics visualization widget."""
  _view_name = Unicode('SlicingMetricsView').tag(sync=True)
  _model_name = Unicode('SlicingMetricsModel').tag(sync=True)
  _view_module = Unicode('tfma_widget_js').tag(sync=True)
  _model_module = Unicode('tfma_widget_js').tag(sync=True)
  _view_module_version = Unicode('^0.1.0').tag(sync=True)
  _model_module_version = Unicode('^0.1.0').tag(sync=True)
  data = List([]).tag(sync=True)
  config = Dict(dict()).tag(sync=True)

  # Used for handling on the js side.
  event_handlers = {}
  js_events = List([]).tag(sync=True)

  @observe('js_events')
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
  _view_name = Unicode('TimeSeriesView').tag(sync=True)
  _model_name = Unicode('TimeSeriesModel').tag(sync=True)
  _view_module = Unicode('tfma_widget_js').tag(sync=True)
  _model_module = Unicode('tfma_widget_js').tag(sync=True)
  _view_module_version = Unicode('^0.1.0').tag(sync=True)
  _model_module_version = Unicode('^0.1.0').tag(sync=True)
  data = List([]).tag(sync=True)
  config = Dict(dict()).tag(sync=True)


@widgets.register
class PlotViewer(widgets.DOMWidget):
  """The time series visualization widget."""
  _view_name = Unicode('PlotView').tag(sync=True)
  _model_name = Unicode('PlotModel').tag(sync=True)
  _view_module = Unicode('tfma_widget_js').tag(sync=True)
  _model_module = Unicode('tfma_widget_js').tag(sync=True)
  _view_module_version = Unicode('^0.1.0').tag(sync=True)
  _model_module_version = Unicode('^0.1.0').tag(sync=True)
  data = Dict([]).tag(sync=True)
  config = Dict(dict()).tag(sync=True)
