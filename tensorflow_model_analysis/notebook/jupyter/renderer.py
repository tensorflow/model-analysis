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
"""Jupyter renderer API."""

import tensorflow_model_analysis.notebook.jupyter.tfma_widget as tfma_widget


def render_slicing_metrics(data, config, event_handlers=None):
  """Renders the slicing metrics view in Jupyter.

  Args:
    data: A list of dictionary containing metrics for correpsonding slices.
    config: A dictionary of the configuration.
    event_handlers: A dictionary of where keys are event types and values are
      event handlers.

  Returns:
    A SlicingMetricsViewer.
  """
  view = tfma_widget.SlicingMetricsViewer()
  view.data = data
  view.config = config
  view.event_handlers = event_handlers

  return view


def render_time_series(data, config):
  """Renders the time series view in Jupyter.

  Args:
    data: A list of dictionary containing metrics for different evaluation runs.
    config: A dictionary of the configuration.

  Returns:
    A TimeSeriesViewer.
  """
  view = tfma_widget.TimeSeriesViewer()
  view.data = data
  view.config = config

  return view


def render_plot(data, config):
  """Renders the plot view in Jupyter.

  Args:
    data: A dictionary containing plot data.
    config: A dictionary of the configuration.

  Returns:
    A PlotViewer.
  """
  view = tfma_widget.PlotViewer()
  view.data = data
  view.config = config

  return view
