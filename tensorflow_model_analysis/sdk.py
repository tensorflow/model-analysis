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
"""SDK for TensorFlow Model Analysis."""

# In contrast to __init__.py, this file includes only imports required for
# defining / configuring TFMA components, not imports for actually running or
# executing TFMA pipelines.
#
# This is useful for example, when defining a TFX pipeline, when users need
# access to the protos or constants to configure the TFMA components,
# but do not need or want to import the modules required for actually executing
# a TFMA pipeline (such as Beam and TensorFlow, which introduce a lot of
# dependencies).

# pylint: disable=unused-import

# Allow constants to be imported at the top-level since they live in root dir.
from tensorflow_model_analysis.constants import ANALYSIS_KEY
from tensorflow_model_analysis.constants import ARROW_INPUT_COLUMN
from tensorflow_model_analysis.constants import ARROW_RECORD_BATCH_KEY
from tensorflow_model_analysis.constants import ATTRIBUTIONS_KEY
from tensorflow_model_analysis.constants import BASELINE_KEY
from tensorflow_model_analysis.constants import BASELINE_SCORE_KEY
from tensorflow_model_analysis.constants import CANDIDATE_KEY
from tensorflow_model_analysis.constants import DATA_CENTRIC_MODE
from tensorflow_model_analysis.constants import EXAMPLE_SCORE_KEY
from tensorflow_model_analysis.constants import EXAMPLE_WEIGHTS_KEY
# TODO(b/120222218): Remove after passing of native FPL supported.
from tensorflow_model_analysis.constants import FEATURES_KEY
from tensorflow_model_analysis.constants import FEATURES_PREDICTIONS_LABELS_KEY
from tensorflow_model_analysis.constants import INPUT_KEY
from tensorflow_model_analysis.constants import LABELS_KEY
from tensorflow_model_analysis.constants import METRICS_KEY
from tensorflow_model_analysis.constants import MODEL_CENTRIC_MODE
from tensorflow_model_analysis.constants import PLOTS_KEY
from tensorflow_model_analysis.constants import PREDICTIONS_KEY
from tensorflow_model_analysis.constants import SLICE_KEY_TYPES_KEY
from tensorflow_model_analysis.constants import TF_ESTIMATOR
from tensorflow_model_analysis.constants import TF_GENERIC
from tensorflow_model_analysis.constants import TF_JS
from tensorflow_model_analysis.constants import TF_KERAS
from tensorflow_model_analysis.constants import TF_LITE
from tensorflow_model_analysis.constants import TFMA_EVAL
from tensorflow_model_analysis.constants import VALIDATIONS_KEY

# Allow proto types to be imported at the top-level since proto's live in
# the tensorflow_model_analysis namespace.
# pylint: disable=g-importing-member
from tensorflow_model_analysis.proto.config_pb2 import AggregationOptions
from tensorflow_model_analysis.proto.config_pb2 import BinarizationOptions
from tensorflow_model_analysis.proto.config_pb2 import ConfidenceIntervalOptions
from tensorflow_model_analysis.proto.config_pb2 import CrossSliceMetricThreshold
from tensorflow_model_analysis.proto.config_pb2 import CrossSliceMetricThresholds
from tensorflow_model_analysis.proto.config_pb2 import CrossSlicingSpec
from tensorflow_model_analysis.proto.config_pb2 import EvalConfig
from tensorflow_model_analysis.proto.config_pb2 import ExampleWeightOptions
from tensorflow_model_analysis.proto.config_pb2 import GenericChangeThreshold
from tensorflow_model_analysis.proto.config_pb2 import GenericValueThreshold
from tensorflow_model_analysis.proto.config_pb2 import MetricConfig
from tensorflow_model_analysis.proto.config_pb2 import MetricDirection
from tensorflow_model_analysis.proto.config_pb2 import MetricsSpec
from tensorflow_model_analysis.proto.config_pb2 import MetricThreshold
from tensorflow_model_analysis.proto.config_pb2 import ModelSpec
from tensorflow_model_analysis.proto.config_pb2 import Options
from tensorflow_model_analysis.proto.config_pb2 import PaddingOptions
from tensorflow_model_analysis.proto.config_pb2 import PerSliceMetricThreshold
from tensorflow_model_analysis.proto.config_pb2 import PerSliceMetricThresholds
from tensorflow_model_analysis.proto.config_pb2 import RepeatedInt32Value
from tensorflow_model_analysis.proto.config_pb2 import RepeatedStringValue
from tensorflow_model_analysis.proto.config_pb2 import SlicingSpec
# pylint: enable=g-importing-member

# Import VERSION as VERSION_STRING for backwards compatibility.
from tensorflow_model_analysis.version import VERSION as VERSION_STRING

__all__ = [
  "ANALYSIS_KEY",
  "ARROW_INPUT_COLUMN",
  "ARROW_RECORD_BATCH_KEY",
  "ATTRIBUTIONS_KEY",
  "BASELINE_KEY",
  "BASELINE_SCORE_KEY",
  "CANDIDATE_KEY",
  "DATA_CENTRIC_MODE",
  "EXAMPLE_SCORE_KEY",
  "EXAMPLE_WEIGHTS_KEY",
  "FEATURES_KEY",
  "FEATURES_PREDICTIONS_LABELS_KEY",
  "INPUT_KEY",
  "LABELS_KEY",
  "METRICS_KEY",
  "MODEL_CENTRIC_MODE",
  "MetricDirection",
  "PLOTS_KEY",
  "PREDICTIONS_KEY",
  "SLICE_KEY_TYPES_KEY",
  "TFMA_EVAL",
  "TF_ESTIMATOR",
  "TF_GENERIC",
  "TF_JS",
  "TF_KERAS",
  "TF_LITE",
  "VALIDATIONS_KEY",
  "VERSION_STRING",
]
