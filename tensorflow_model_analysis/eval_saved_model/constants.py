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
SIGNATURE_DEF_ITERATOR_INITIALIZER_KEY = 'iterator_initializer'
SIGNATURE_DEF_TFMA_VERSION_KEY = 'tfma/version'

FEATURES_NAME = 'features'
LABELS_NAME = 'labels'

# TODO(b/79777718): Really tf.saved_model.tag_constants.EVAL
EVAL_TAG = 'eval'
# TODO(b/79777718): Really model_fn.EXPORT_TAG_MAP[ModeKeys.EVAL]
DEFAULT_EVAL_SIGNATURE_DEF_KEY = 'eval'
# TODO(b/79777718): Really tf.estimator.export.EvalOutput.PREDICTIONS_NAME
PREDICTIONS_NAME = 'predictions'
# TODO(b/79777718): Really tf.estimator.export.EvalOutput.METRICS_NAME
METRICS_NAME = 'metrics'
# TODO(b/79777718): Really tf.estimator.export.EvalOutput.METRIC_VALUE_SUFFIX
METRIC_VALUE_SUFFIX = 'value'
# TODO(b/79777718): Really tf.estimator.export.EvalOutput.METRIC_UPDATE_SUFFIX
METRIC_UPDATE_SUFFIX = 'update_op'
