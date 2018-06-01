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
"""Constants used in TensorFlow Model Analysis."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function




# Mode for multiple model analysis runs
UNKNOWN_EVAL_MODE = 'unknown_eval_mode'
MODEL_CENTRIC_MODE = 'model_centric_mode'
DATA_CENTRIC_MODE = 'data_centric_mode'

# Keys to ExampleAndExtract extraction dictionary.
FEATURES_PREDICTIONS_LABELS_KEY = 'fpl'
