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
"""Utility for Colab TFMA renderer API."""

import base64
import json
from typing import Any, Callable, Dict, List, Optional, Union
from google.colab import output
from IPython import display

# For expressing the set of JS events to listen and respond to in Python
PythonEventHandlersMap = Callable[[Dict[str, Union[str, float]]], None]

# Safelist the web component names that can be rendered.
_TRUSTED_TFMA_COMPONENT_NAMES = frozenset(
    ['tfma-nb-slicing-metrics', 'tfma-nb-time-series', 'tfma-nb-plot'])


def _create_handler_wrapper(event_handlers):
  """Wraps the event handler and registers it as a callback for the js side.

  Wraps the event handler and registers it as a callback for js. Keep count and
  use it as aprt of the callback name to ensure uniqueness.

  Args:
    event_handlers: The hadnler for the js events.

  Returns:
    The name of the js callback, safe to render as HTML or JS.
  """
  trusted_name = 'tfma_eventCallback' + str(int(_create_handler_wrapper.count))
  _create_handler_wrapper.count += 1

  def wrapped_function(name='', detail=None):
    if event_handlers and name in event_handlers:
      event_handlers[name](detail)

  output.register_callback(trusted_name, wrapped_function)
  return trusted_name


_create_handler_wrapper.count = 0


def make_trusted_event_handler_js(event_handlers):
  """Generates event handler code in js if provided.

  If python event_handlers are provided, generate corresponding js callback and
  trigger it when applicable.  See tfma-nb-event-mixin.js

  Args:
   event_handlers: The hadnler for the hs events.

  Returns:
    Trusted js code that will call the event handler in python.
  """
  if event_handlers:
    trusted_callback_name = _create_handler_wrapper(event_handlers)
    return """
          element.addEventListener('tfma-event', (event) => {{
            google.colab.kernel.invokeFunction(
                '{trusted_callback_name}', [], {{
                name: event.detail.type,
                detail: event.detail.detail
            }});
          }});
     """.format(trusted_callback_name=trusted_callback_name)
  else:
    return '/** No event handlers needed. */'


def to_base64_encoded_json(obj) -> str:
  """Encode a Python object as a base64-endoded JSON string.

  When embedding JSON inline inside HTML, serialize it to a JSON string in
  Python and base64 encode that string to escape it so that it's safe to render
  inside HTML.  Then on the JS side, base64 decode it and parse it as JSON.

  Args:
    obj: any Python object serializable to JSON

  Returns:
    base64-encoded string of JSON
  """
  json_string = json.dumps(obj)
  return base64.b64encode(json_string.encode('utf-8')).decode('utf-8')


def render_tfma_component(
    component_name: str,
    data: Union[List[Dict[str, Union[Dict[str, Any], str]]],
                Dict[str, List[Union[str, float, List[float]]]]],
    config: Dict[str, Union[Dict[str, Dict[str, str]], str, bool]],
    trusted_html_for_vulcanized_tfma_js: str,
    event_handlers: Optional[PythonEventHandlersMap] = None,
) -> None:
  """Renders the specified TFMA component in Colab.

  Colab requires custom visualization to be rendered in a sandbox so we cannot
  use Jupyter widget.

  Args:
    component_name: The name of the TFMA web component to render.
    data: A dictionary containing data for visualization.
    config: A dictionary containing the configuration.
    trusted_html_for_vulcanized_tfma_js: Optional string of trusted HTML that is
      rendered unescaped.  This can be a script tag referencing a trusted
      external JS file or a script tag with trusted JS inline.
    event_handlers: Handlers for events on the js side.
  """

  if component_name not in _TRUSTED_TFMA_COMPONENT_NAMES:
    raise ValueError('component_name must be one of: ' +
                     ','.join(_TRUSTED_TFMA_COMPONENT_NAMES))

  ui_payload = {'config': config, 'data': data}
  template = """
    {trusted_html_for_vulcanized_tfma_js}
    <{trusted_tfma_component_name} id="component"></{trusted_tfma_component_name}>
    <script>
    const element = document.getElementById('component');

    {trusted_event_handler_js}

    const json = JSON.parse(atob('{base64_encoded_json_payload}'));
    element.config = json.config;
    element.data = json.data;
    </script>
    """
  html = template.format(
      trusted_tfma_component_name=component_name,
      trusted_html_for_vulcanized_tfma_js=trusted_html_for_vulcanized_tfma_js,
      trusted_event_handler_js=make_trusted_event_handler_js(event_handlers),
      base64_encoded_json_payload=to_base64_encoded_json(ui_payload))
  display.display(display.HTML(html))
