# Lint as: python2, python3
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
"""Colab renderer API."""

# Standard __future__ imports

import json
from IPython import display

from tensorflow_model_analysis.notebook.colab import util


def render_fairness_indicator(slicing_metrics=None,
                              multi_slicing_metrics=None,
                              event_handlers=None) -> None:
  """Renders the fairness inidcator view in Colab.

  Colab requires custom visualization to be rendered in a sandbox so we cannot
  use Jupyter widget.

  Args:
    slicing_metrics: A dictionary containing data for visualization.
    multi_slicing_metrics: A dictionary that maps eval name to the
      slicing_metrics. Only two eval results will be accepted.
    event_handlers: The event handler callback.
  """
  if ((slicing_metrics and multi_slicing_metrics) or
      (not slicing_metrics and not multi_slicing_metrics)):
    raise ValueError(
        'Exactly one of the "slicing_metrics" and "multi_slicing_metrics" '
        'parameters must be set.')
  event_handler_code = util.maybe_generate_event_handler_code(event_handlers)

  if slicing_metrics:
    display.display(
        display.HTML("""
            <script src="/nbextensions/tensorflow_model_analysis/vulcanized_tfma.js">
            </script>
            <fairness-nb-container id="component"></fairness-nb-container>
            <script>
              const element = document.getElementById('component');

              {event_handler_code}

              element.slicingMetrics = JSON.parse('{slicingMetrics}');
            </script>
            """.format(
                event_handler_code=event_handler_code,
                slicingMetrics=json.dumps(slicing_metrics))))
  else:
    eval_name, eval_name_compare = tuple(multi_slicing_metrics.keys())
    slicing_metrics = multi_slicing_metrics[eval_name]
    slicing_metrics_compare = multi_slicing_metrics[eval_name_compare]
    display.display(
        display.HTML("""
            <script src="/nbextensions/tensorflow_model_analysis/vulcanized_tfma.js">
            </script>
            <fairness-nb-container id="component"></fairness-nb-container>
            <script>
              const element = document.getElementById('component');

              {event_handler_code}

              element.slicingMetrics = JSON.parse('{slicingMetrics}');
              element.slicingMetricsCompare = JSON.parse('{slicingMetricsCompare}');
              element.evalName = JSON.parse('{evalName}');
              element.evalNameCompare = JSON.parse('{evalNameCompare}');
            </script>
            """.format(
                event_handler_code=event_handler_code,
                slicingMetrics=json.dumps(slicing_metrics),
                slicingMetricsCompare=json.dumps(slicing_metrics_compare),
                evalName=json.dumps(eval_name),
                evalNameCompare=json.dumps(eval_name_compare))))
