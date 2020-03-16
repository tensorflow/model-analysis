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

# Standard __future__ imports
from tensorflow_model_analysis.addons.fairness.notebook.jupyter import widget


def render_fairness_indicator(slicing_metrics=None,
                              multi_slicing_metrics=None,
                              event_handlers=None):
  """Renders the fairness indicator view in Colab.

  Args:
    slicing_metrics: A dictionary containing data for visualization.
    multi_slicing_metrics: A dictionary that maps eval name to the
      slicing_metrics. Only two eval results will be accepted.
    event_handlers: The event handler callback.

  Returns:
    FairnessIndicatorViewer object
  """
  if ((slicing_metrics and multi_slicing_metrics) or
      (not slicing_metrics and not multi_slicing_metrics)):
    raise ValueError(
        'Exactly one of the "slicing_metrics" and "multi_slicing_metrics" '
        'parameters must be set.')

  view = widget.FairnessIndicatorViewer()
  if slicing_metrics:
    view.slicingMetrics = slicing_metrics
  else:  # multi_slicing_metrics
    eval_name, eval_name_compare = multi_slicing_metrics.keys()
    slicing_metrics = multi_slicing_metrics[eval_name]
    slicing_metrics_compare = multi_slicing_metrics[eval_name_compare]
    view.slicingMetrics = slicing_metrics
    view.slicingMetricsCompare = slicing_metrics_compare
    view.evalName = eval_name
    view.evalNameCompare = eval_name_compare

  view.eventHandlers = event_handlers
  return view
