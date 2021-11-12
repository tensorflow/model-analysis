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
"""Utility for Colab Fairness Indicators renderer API."""

from typing import Optional, Dict, Any

from IPython import display
from tensorflow_model_analysis.notebook.colab.util import make_trusted_event_handler_js
from tensorflow_model_analysis.notebook.colab.util import PythonEventHandlersMap
from tensorflow_model_analysis.notebook.colab.util import to_base64_encoded_json


def render_fairness_indicators_html(
    ui_payload,
    trusted_html_for_vulcanized_tfma_js: str,
    event_handlers: Optional[PythonEventHandlersMap] = None,
) -> None:
  """Renders an HTML stub for the Fairness Indicators UI and display it.

  Args:
    ui_payload: A Python dictionary of data provided as attributes to the
      <fairness-nb-container /> JS web component.
    trusted_html_for_vulcanized_tfma_js: A trusted string of HTML which contains
      a script tag inlining the vulcanized_tfma.js dependency, or a reference to
      that script.  Callers are responsible for ensuring this HTML is trusted.
    event_handlers: Dict of event keys and handler functions, for responding in
      Python to events fired in the JS web component.  Used for WIT integration
      in notebooks (see `wit.create_selection_callback`).
  """
  template = """
    {trusted_html_for_vulcanized_tfma_js}
    <fairness-nb-container id="component"></fairness-nb-container>
    <script>
      const element = document.getElementById('component');

      {trusted_event_handler_js}

      const json = JSON.parse(atob('{base64_encoded_json_payload}'));
      element.slicingMetrics = json.slicingMetrics;
      element.slicingMetricsCompare = json.slicingMetricsCompare;
      element.evalName = json.evalName;
      element.evalNameCompare = json.evalNameCompare;
    </script>
    """
  html = template.format(
      base64_encoded_json_payload=to_base64_encoded_json(ui_payload),
      trusted_html_for_vulcanized_tfma_js=trusted_html_for_vulcanized_tfma_js,
      trusted_event_handler_js=make_trusted_event_handler_js(event_handlers))
  display.display(display.HTML(html))


def make_fi_ui_payload(slicing_metrics=None,
                       multi_slicing_metrics=None) -> Dict[str, Any]:
  """Creates a Python Dict that can be passed into the Fairness Indicators JS UI.

  Args:
    slicing_metrics: A dictionary containing data for visualization.
    multi_slicing_metrics: A dictionary that maps eval name to the
      slicing_metrics. Only two eval results will be accepted.

  Returns:
    Python Dict that can be passed to JS UI
  """
  if ((slicing_metrics and multi_slicing_metrics) or
      (not slicing_metrics and not multi_slicing_metrics)):
    raise ValueError(
        'Exactly one of the "slicing_metrics" and "multi_slicing_metrics" '
        'parameters must be set.')

  if slicing_metrics:
    return {'slicingMetrics': slicing_metrics}

  eval_name, eval_name_compare = multi_slicing_metrics.keys()
  slicing_metrics = multi_slicing_metrics[eval_name]
  slicing_metrics_compare = multi_slicing_metrics[eval_name_compare]
  return {
      'slicingMetrics': slicing_metrics,
      'slicingMetricsCompare': slicing_metrics_compare,
      'evalName': eval_name,
      'evalNameCompare': eval_name_compare
  }
