# Copyright 2020 Google LLC
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
"""Jupyter renderer API."""

# Standard __future__ imports
from tensorflow_model_analysis.addons.fairness.notebook.jupyter import widget


# TODO(b/147495615): add slicing_metrics arg, which will get passed in to
# FairnessIndicatorViewer, which will pass it to fairness-nb-container
def render_fairness_indicator(event_handlers=None) -> None:
  """Renders the fairness indicator view in Colab.

  Args:
    event_handlers: The event handler callback.

  Returns:
    FairnessIndicatorViewer object
  """
  view = widget.FairnessIndicatorViewer()
  view.event_handlers = event_handlers
  return view
