# Copyright 2019 Google LLC
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
"""Defines Fairness Indicator's Jupyter notebook widgets."""
import ipywidgets as widgets
from tensorflow_model_analysis.version import VERSION
import traitlets
from traitlets import List
from traitlets import Unicode


@widgets.register
class FairnessIndicatorViewer(widgets.DOMWidget):
  """The fairness indicator visualization widget."""
  _view_name = Unicode('FairnessIndicatorView').tag(sync=True)
  _view_module = Unicode('tensorflow_model_analysis').tag(sync=True)
  _view_module_version = Unicode(VERSION).tag(sync=True)
  _model_name = Unicode('FairnessIndicatorModel').tag(sync=True)
  _model_module = Unicode('tensorflow_model_analysis').tag(sync=True)
  _model_module_version = Unicode(VERSION).tag(sync=True)
  slicingMetrics = List().tag(sync=True)
  slicingMetricsCompare = List().tag(sync=True)
  evalName = Unicode().tag(sync=True)
  evalNameCompare = Unicode().tag(sync=True)

  # Used for handling on the js side.
  eventHandlers = {}
  js_events = List([]).tag(sync=True)

  @traitlets.observe('js_events')
  def _handle_js_events(self, change):
    if self.js_events:
      if self.eventHandlers:
        for event in self.js_events:
          event_name = event['name']
          if event_name in self.eventHandlers:
            self.eventHandlers[event_name](event['detail'])

      # clears the event queue.
      self.js_events = []
