# Copyright 2021 Google LLC
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
"""Init module for TensorFlow Model Analysis utils."""

from tensorflow_model_analysis.utils import keras_lib
from tensorflow_model_analysis.utils.config_util import has_change_threshold
from tensorflow_model_analysis.utils.config_util import update_eval_config_with_defaults
from tensorflow_model_analysis.utils.config_util import verify_eval_config
from tensorflow_model_analysis.utils.math_util import calculate_confidence_interval
from tensorflow_model_analysis.utils.model_util import CombineFnWithModels
from tensorflow_model_analysis.utils.model_util import DoFnWithModels
from tensorflow_model_analysis.utils.model_util import get_baseline_model_spec
from tensorflow_model_analysis.utils.model_util import get_model_spec
from tensorflow_model_analysis.utils.model_util import get_model_type
from tensorflow_model_analysis.utils.model_util import get_non_baseline_model_specs
from tensorflow_model_analysis.utils.model_util import model_construct_fn
from tensorflow_model_analysis.utils.model_util import verify_and_update_eval_shared_models
from tensorflow_model_analysis.utils.util import compound_key
from tensorflow_model_analysis.utils.util import create_keys_key
from tensorflow_model_analysis.utils.util import create_values_key
from tensorflow_model_analysis.utils.util import get_by_keys
from tensorflow_model_analysis.utils.util import merge_extracts
from tensorflow_model_analysis.utils.util import unique_key

__all__ = [
  "calculate_confidence_interval",
  "CombineFnWithModels",
  "compound_key",
  "create_keys_key",
  "create_values_key",
  "DoFnWithModels",
  "get_baseline_model_spec",
  "get_by_keys",
  "get_model_spec",
  "get_model_type",
  "get_non_baseline_model_specs",
  "has_change_threshold",
  "merge_extracts",
  "model_construct_fn",
  "unique_key",
  "update_eval_config_with_defaults",
  "verify_and_update_eval_shared_models",
  "verify_eval_config",
]
