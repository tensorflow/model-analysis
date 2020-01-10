# Copyright 2019 Google LLC
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
"""Fairness Visualization API."""
import sys


def _is_colab():
  return "google.colab" in sys.modules


if _is_colab():
  from tensorflow_model_analysis.addons.fairness.notebook.colab.renderer import *  # pylint: disable=wildcard-import,g-import-not-at-top
  from tensorflow_model_analysis.addons.fairness.notebook.colab.widget import *  # pylint: disable=wildcard-import,g-import-not-at-top
else:
  from tensorflow_model_analysis.addons.fairness.notebook.jupyter.widget import *  # pylint: disable=wildcard-import,g-import-not-at-top
  from tensorflow_model_analysis.addons.fairness.notebook.jupyter.renderer import *  # pylint: disable=wildcard-import,g-import-not-at-top
