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
"""View API for Tensorflow Model Analysis."""
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import model_eval_lib
import tensorflow_model_analysis.notebook.jupyter.tfma_widget as tfma_widget
from tensorflow_model_analysis.slicer.slicer import SingleSliceSpec
from tensorflow_model_analysis.view import util
from tensorflow_model_analysis.types_compat import Optional


def render_slicing_metrics(
    result,
    slicing_column = None,
    slicing_spec = None):
  """Renders the slicing metrics view as widget.

  Args:
    result: An tfma.EvalResult.
    slicing_column: The column to slice on.
    slicing_spec: The slicing spec to filter results. If neither column nor spec
    is set, show overall.

  Returns:
    A SlicingMetricsViewer object.
  """
  view = tfma_widget.SlicingMetricsViewer()

  view.data = util.get_slicing_metrics(result.slicing_metrics, slicing_column,
                                       slicing_spec)
  view.config = {
      'weightedExamplesColumn': result.config.example_weight_metric_key
  }
  return view


def render_time_series(results,
                       slice_spec = None,
                       display_full_path = False):
  """Renders the time series view as widget.

  Args:
    results: An tfma.EvalResults.
    slice_spec: A slicing spec determining the slice to show time series on.
    Show overall if not set.
    display_full_path: Whether to display the full path to model / data in the
    visualization or just show file name.

  Returns:
    A TimeSeriesViewer object
  """
  slice_spec_to_use = slice_spec if slice_spec else SingleSliceSpec()
  view = tfma_widget.TimeSeriesViewer()
  view.data = util.get_time_series(results, slice_spec_to_use,
                                   display_full_path)
  view.config = {
      'isModelCentric': results.get_mode() == constants.MODEL_CENTRIC_MODE
  }
  return view


def render_plot(result,
                slicing_spec = None):
  """Renders the plot view as widget.

  Args:
    result: An tfma.EvalResult.
    slicing_spec: The slicing spec to identify the slice. Show overall if unset.

  Returns:
    A PlotViewer object.
  """
  slice_spec_to_use = slicing_spec if slicing_spec else SingleSliceSpec()
  view = tfma_widget.PlotViewer()
  view.data, view.config = util.get_plot_data_and_config(
      result.plots, slice_spec_to_use)

  return view
