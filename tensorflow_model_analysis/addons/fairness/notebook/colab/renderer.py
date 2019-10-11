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


def render_fairness_indicator(slicing_metrics, event_handlers=None) -> None:
  """Renders the fairness inidcator view in Colab.

  Colab requires custom visualization to be rendered in a sandbox so we cannot
  use Jupyter widget.

  Args:
    slicing_metrics: A dictionary containing data for visualization.
    event_handlers: The event handler callback.
  """
  event_handler_code = util.maybe_generate_event_handler_code(event_handlers)

  display.display(
      display.HTML("""
          <script src="/nbextensions/tfma_widget_js/vulcanized_tfma.js">
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
