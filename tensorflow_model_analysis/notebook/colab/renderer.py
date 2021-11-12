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
"""TFMA API for OSS Colab renderer."""

from typing import Any, Callable, Dict, List, Optional, Union
from tensorflow_model_analysis.notebook.colab import util


# See also `loadVulcanizedTemplate` in JS, which adjusts the script path
# further at runtime, depending on the environment.
def get_trusted_html_for_vulcanized_js():
  """Returns a trusted string of HTML that will load vulcanized_tfma.js."""
  return """
  <script src="/nbextensions/tensorflow_model_analysis/vulcanized_tfma.js"></script>
  """


def render_slicing_metrics(
    data: List[Dict[str, Union[Dict[str, Any], str]]],
    config: Dict[str, str],
    event_handlers: Optional[Callable[[Dict[str, Union[str, float]]],
                                      None]] = None
) -> None:
  """Renders the slicing metrics view in Colab.

  Args:
    data: A list of dictionary containing metrics for correpsonding slices.
    config: A dictionary of the configuration.
    event_handlers: event handlers
  """
  util.render_tfma_component(
      'tfma-nb-slicing-metrics',
      data,
      config,
      event_handlers=event_handlers,
      trusted_html_for_vulcanized_tfma_js=get_trusted_html_for_vulcanized_js())


def render_time_series(data: List[Dict[str, Union[Dict[Union[float, str], Any],
                                                  str]]],
                       config: Dict[str, bool]) -> None:
  """Renders the time series view in Colab.

  Args:
    data: A list of dictionary containing metrics for different evaluation runs.
    config: A dictionary of the configuration.
  """
  util.render_tfma_component(
      'tfma-nb-time-series',
      data,
      config,
      trusted_html_for_vulcanized_tfma_js=get_trusted_html_for_vulcanized_js())


def render_plot(
    data: Dict[str, List[Union[str, float, List[float]]]],
    config: Dict[str, Union[Dict[str, Dict[str, str]], str]]) -> None:
  """Renders the plot view in Colab.

  Args:
    data: A dictionary containing plot data.
    config: A dictionary of the configuration.
  """
  util.render_tfma_component(
      'tfma-nb-plot',
      data,
      config,
      trusted_html_for_vulcanized_tfma_js=get_trusted_html_for_vulcanized_js())
