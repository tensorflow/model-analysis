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

from tensorflow_model_analysis.notebook.colab import util
from typing import Any, Callable, Dict, List, Optional, Text, Union


def render_slicing_metrics(
    data: List[Dict[Text, Union[Dict[Text, Any], Text]]],
    config: Dict[Text, Text],
    event_handlers: Optional[
        Callable[[Dict[Text, Union[Text, float]]], None]] = None) -> None:
  """Renders the slicing metrics view in Colab.

  Args:
    data: A list of dictionary containing metrics for correpsonding slices.
    config: A dictionary of the configuration.
    event_handlers: event handlers
  """
  util.render_component(
      'tfma-nb-slicing-metrics', data, config, event_handlers=event_handlers)


def render_time_series(
    data: List[Dict[Text, Union[Dict[Union[float, Text], Any], Text]]],
    config: Dict[Text, bool]) -> None:
  """Renders the time series view in Colab.

  Args:
    data: A list of dictionary containing metrics for different evaluation runs.
    config: A dictionary of the configuration.
  """
  util.render_component('tfma-nb-time-series', data, config)


def render_plot(data: Dict[Text, List[Union[Text, float, List[float]]]],
                config: Dict[Text, Union[Dict[Text, Dict[Text, Text]], Text]]
               ) -> None:
  """Renders the plot view in Colab.

  Args:
    data: A dictionary containing plot data.
    config: A dictionary of the configuration.
  """
  util.render_component('tfma-nb-plot', data, config)
