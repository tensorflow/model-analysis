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
"""Constants for the EvalSavedModel."""

EVAL_SAVED_MODEL_EXPORT_NAME = 'TFMA'
EVAL_SAVED_MODEL_TAG = 'eval_saved_model'

SIGNATURE_DEF_INPUTS_PREFIX = 'inputs'
SIGNATURE_DEF_INPUT_REFS_KEY = 'input_refs'
SIGNATURE_DEF_LABELS_PREFIX = 'labels'
SIGNATURE_DEF_FEATURES_PREFIX = 'features'
SIGNATURE_DEF_TFMA_VERSION_KEY = 'tfma/version'

# If features/predictions/labels was not a dictionary, we internally wrap them
# in a dictionary with the respective default keys.
#
# Note that the key names start with two underscores to avoid collisions
# in the rare case that there are actually keys named 'features', 'predictions'
# or 'labels' in the respective dictionaries.
DEFAULT_FEATURES_DICT_KEY = '__features'
DEFAULT_PREDICTIONS_DICT_KEY = '__predictions'
DEFAULT_LABELS_DICT_KEY = '__labels'

EVAL_TAG = 'eval'
DEFAULT_EVAL_SIGNATURE_DEF_KEY = 'eval'
PREDICTIONS_NAME = 'predictions'
METRICS_NAME = 'metrics'
METRIC_VALUE_SUFFIX = 'value'
METRIC_UPDATE_SUFFIX = 'update_op'
