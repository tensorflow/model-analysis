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
"""Utility for Colab renderer API."""

# Standard __future__ imports

import json
from IPython import display

from typing import Any, Callable, Dict, List, Optional, Text, Union
from google.colab import output


def create_handler_wrapper(event_handlers):
  """Wraps the event handler and registers it as a callback for the js side.

  Wraps the event handler and registers it as a callback for js. Keep count and
  use it as aprt of the callback name to ensure uniqueness.

  Args:
    event_handlers: The hadnler for the js events.

  Returns:
    The name of the js callback.
  """
  name = 'tfma_eventCallback' + str(create_handler_wrapper.count)
  create_handler_wrapper.count += 1

  def wrapped_function(name='', detail=None):
    if event_handlers and name in event_handlers:
      event_handlers[name](detail)

  output.register_callback(name, wrapped_function)
  return name


create_handler_wrapper.count = 0


def maybe_generate_event_handler_code(event_handlers):
  """Generates event handler code in js if provided.

  If python event_handlers are provided, generate corresponding js callback and
  trigger it when applicable.

  Args:
   event_handlers: The hadnler for the hs events.

  Returns:
    The js code that will call the event handler in python.
  """
  if event_handlers:
    return """
          element.addEventListener('tfma-event', (event) => {{
            google.colab.kernel.invokeFunction(
                '{callback_name}', [], {{
                name: event.detail.type,
                detail: event.detail.detail
            }});
          }});
     """.format(callback_name=create_handler_wrapper(event_handlers))
  else:
    return '/** No event handlers needed. */'


def render_component(
    component_name: Text,
    data: Union[List[Dict[Text, Union[Dict[Text, Any], Text]]],
                Dict[Text, List[Union[Text, float, List[float]]]]],
    config: Dict[Text, Union[Dict[Text, Dict[Text, Text]], Text, bool]],
    event_handlers: Optional[
        Callable[[Dict[Text, Union[Text, float]]], None]] = None) -> None:
  """Renders the specified component in Colab.

  Colab requires custom visualization to be rendered in a sandbox so we cannot
  use Jupyter widget.

  Args:
    component_name: The name of the component to render.
    data: A dictionary containing data for visualization.
    config: A dictionary containing the configuration.
    event_handlers: Handlers for evetns on the js side.
  """
  event_handler_code = maybe_generate_event_handler_code(event_handlers)

  display.display(
      display.HTML("""
          <script src="/nbextensions/tensorflow_model_analysis/vulcanized_tfma.js">
          </script>
          <{component_name} id="component"></{component_name}>
          <script>
          const element = document.getElementById('component');

          {event_handler_code}

          element.config = JSON.parse('{config}');
          element.data = JSON.parse('{data}');
          </script>
          """.format(
              event_handler_code=event_handler_code,
              component_name=component_name,
              config=json.dumps(config),
              data=json.dumps(data))))
