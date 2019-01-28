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

# Types of placeholders
PLACEHOLDER = 'placeholder'
SPARSE_PLACEHOLDER = 'sparse_placeholder'

METRICS_NAMESPACE = 'tensorflow_model_analysis'

# Keys for Extracts dictionary (keys starting with _ will not be materialized).

# Input key. Could be a serialised tf.train.Example, a CSV row, JSON data, etc
# depending on what the EvalInputReceiver was configured to accept as input.
INPUT_KEY = 'input'
# Features, predictions, and labels key.
FEATURES_PREDICTIONS_LABELS_KEY = '_fpl'
# Contains SliceKeyTypes that are used to fanout and aggregate.
SLICE_KEY_TYPES_KEY = '_slice_key_types'
# Human-readable slice strings that are written to the diagnostic table for
# analysis.
SLICE_KEYS_KEY = 'slice_keys'

# Separator used when combining multiple layers of Extracts keys into a single
# string. Normally we would like to use '.' or '/' as a separator, but the
# output gets written to a table backed by a proto based schema which limits the
# characters that can be used to [a-zA-Z_].
KEY_SEPARATOR = '__'
# Suffix used for Extracts keys when the extract represents the keys for the top
# k of a set of possible values.
KEYS_SUFFIX = 'keys'
# Suffix used for Extracts keys when the extract represents the values for the
# top k of a set of possible values.
VALUES_SUFFIX = 'values'
# Suffix used for Extracts keys when the extract represents an attribution
# score.
SCORE_SUFFIX = 'score'

# Keys for Evaluation/Validation dictionaries

# Metrics output key.
METRICS_KEY = 'metrics'
# Plots output key.
PLOTS_KEY = 'plots'
# Analysis output key.
ANALYSIS_KEY = 'analysis'

# Keys for validation alternatives
BASELINE_KEY = 'baseline'
CANDIDATE_KEY = 'candidate'
