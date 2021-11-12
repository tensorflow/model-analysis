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
"""Fairness Indicators API for OSS Colab renderer."""

from tensorflow_model_analysis.addons.fairness.notebook.colab import util
from tensorflow_model_analysis.notebook.colab.renderer import get_trusted_html_for_vulcanized_js


def render_fairness_indicator(slicing_metrics=None,
                              multi_slicing_metrics=None,
                              event_handlers=None) -> None:
  """Renders the fairness indicators view in Colab.

  Colab requires custom visualization to be rendered in a sandbox so we cannot
  use Jupyter widget.

  Args:
    slicing_metrics: A dictionary containing data for visualization.
    multi_slicing_metrics: A dictionary that maps eval name to the
      slicing_metrics. Only two eval results will be accepted.
    event_handlers: The event handler callback.
  """
  ui_payload = util.make_fi_ui_payload(slicing_metrics, multi_slicing_metrics)
  util.render_fairness_indicators_html(
      ui_payload=ui_payload,
      trusted_html_for_vulcanized_tfma_js=get_trusted_html_for_vulcanized_js(),
      event_handlers=event_handlers)
