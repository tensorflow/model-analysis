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
"""Init module for TensorFlow Model Analysis on notebook."""


from tensorflow_model_analysis import view
from tensorflow_model_analysis.api import tfma_unit as test
from tensorflow_model_analysis.api.model_eval_lib import *  # pylint: disable=wildcard-import
from tensorflow_model_analysis.constants import DATA_CENTRIC_MODE
from tensorflow_model_analysis.constants import MODEL_CENTRIC_MODE
from tensorflow_model_analysis.eval_saved_model import export
from tensorflow_model_analysis.eval_saved_model import exporter
from tensorflow_model_analysis.eval_saved_model import post_export_metrics
from tensorflow_model_analysis.slicer.slicer import SingleSliceSpec
from tensorflow_model_analysis.version import VERSION_STRING


def _jupyter_nbextension_paths():
  return [{
      'section': 'notebook',
      'src': 'static',
      'dest': 'tfma_widget_js',
      'require': 'tfma_widget_js/extension'
  }]
