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
"""Init module for TensorFlow Model Analysis."""

################################################################################
# This file acts as the public API for root-level interfaces. It should only
# include imports that are either in the root directory itself or under the api/
# or proto/ subdirectories. All other directories should have a single import
# for the entire directory in this file and have their own __init__.py for
# exposing their own public interfaces.
################################################################################

# pylint: disable=unused-import
# pylint: disable=g-bad-import-order
# pylint: disable=g-import-not-at-top
# pylint: disable=g-statement-before-imports
# See b/148667210 for why the ImportError is ignored.
try:
  # Allow api module types to be imported at the top-level since they are the
  # main public interface to using TFMA.
  from tensorflow_model_analysis.api import tfma_unit as test
  from tensorflow_model_analysis.api.model_eval_lib import AttributionsForSlice
  from tensorflow_model_analysis.api.model_eval_lib import analyze_raw_data
  from tensorflow_model_analysis.api.model_eval_lib import BatchedInputsToExtracts
  from tensorflow_model_analysis.api.model_eval_lib import default_eval_shared_model
  from tensorflow_model_analysis.api.model_eval_lib import default_evaluators
  from tensorflow_model_analysis.api.model_eval_lib import default_extractors
  from tensorflow_model_analysis.api.model_eval_lib import default_writers
  from tensorflow_model_analysis.api.model_eval_lib import ExtractAndEvaluate
  from tensorflow_model_analysis.api.model_eval_lib import ExtractEvaluateAndWriteResults
  from tensorflow_model_analysis.api.model_eval_lib import InputsToExtracts
  from tensorflow_model_analysis.api.model_eval_lib import is_batched_input
  from tensorflow_model_analysis.api.model_eval_lib import is_legacy_estimator
  from tensorflow_model_analysis.api.model_eval_lib import load_attributions
  from tensorflow_model_analysis.api.model_eval_lib import load_eval_result
  from tensorflow_model_analysis.api.model_eval_lib import load_eval_results
  from tensorflow_model_analysis.api.model_eval_lib import load_metrics
  from tensorflow_model_analysis.api.model_eval_lib import load_plots
  from tensorflow_model_analysis.api.model_eval_lib import load_validation_result
  from tensorflow_model_analysis.api.model_eval_lib import make_eval_results
  from tensorflow_model_analysis.api.model_eval_lib import MetricsForSlice
  from tensorflow_model_analysis.api.model_eval_lib import multiple_data_analysis
  from tensorflow_model_analysis.api.model_eval_lib import multiple_model_analysis
  from tensorflow_model_analysis.api.model_eval_lib import PlotsForSlice
  from tensorflow_model_analysis.api.model_eval_lib import run_model_analysis
  from tensorflow_model_analysis.api.model_eval_lib import WriteResults
  from tensorflow_model_analysis.api.model_eval_lib import ValidationResult
  from tensorflow_model_analysis.api.verifier_lib import Validate

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
  from tensorflow_model_analysis.proto.config_pb2 import SlicingSpec
  # pylint: enable=g-importing-member

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
  from tensorflow_model_analysis.constants import FEATURES_PREDICTIONS_LABELS_KEY
  from tensorflow_model_analysis.constants import FEATURES_KEY
  from tensorflow_model_analysis.constants import INPUT_KEY
  from tensorflow_model_analysis.constants import LABELS_KEY
  from tensorflow_model_analysis.constants import METRICS_KEY
  from tensorflow_model_analysis.constants import MODEL_CENTRIC_MODE
  from tensorflow_model_analysis.constants import PLOTS_KEY
  from tensorflow_model_analysis.constants import PREDICTIONS_KEY
  from tensorflow_model_analysis.constants import SLICE_KEY_TYPES_KEY
  from tensorflow_model_analysis.constants import TF_GENERIC
  from tensorflow_model_analysis.constants import TF_ESTIMATOR
  from tensorflow_model_analysis.constants import TF_JS
  from tensorflow_model_analysis.constants import TF_LITE
  from tensorflow_model_analysis.constants import TF_KERAS
  from tensorflow_model_analysis.constants import VALIDATIONS_KEY

  # TODO(b/171992041): Remove these imports in the future.
  # For backwards compatibility allow eval_metrics_graph and exporter to be
  # accessed from top-level model. These will be deprecated in the future.
  from tensorflow_model_analysis.eval_metrics_graph import eval_metrics_graph
  from tensorflow_model_analysis.eval_saved_model import export
  from tensorflow_model_analysis.eval_saved_model import exporter
  from tensorflow_model_analysis.post_export_metrics import post_export_metrics

  # Allow types to be imported at the top-level since they live in root dir.
  from tensorflow_model_analysis.types import AddMetricsCallbackType
  from tensorflow_model_analysis.types import EvalSharedModel
  from tensorflow_model_analysis.types import Extracts
  # TODO(b/120222218): Remove after passing of native FPL supported.
  from tensorflow_model_analysis.types import FeaturesPredictionsLabels
  # TODO(b/120222218): Remove after passing of native FPL supported.
  from tensorflow_model_analysis.types import MaterializedColumn
  from tensorflow_model_analysis.types import MaybeMultipleEvalSharedModels
  from tensorflow_model_analysis.types import ModelLoader
  from tensorflow_model_analysis.types import RaggedTensorValue
  from tensorflow_model_analysis.types import SparseTensorValue
  from tensorflow_model_analysis.types import TensorType
  from tensorflow_model_analysis.types import TensorTypeMaybeDict
  from tensorflow_model_analysis.types import TensorValue
  from tensorflow_model_analysis.types import VarLenTensorValue

  # Import VERSION as __version__ for compatibility with other TFX components.
  from tensorflow_model_analysis.version import VERSION as __version__
  # Import VERSION as VERSION_STRING for backwards compatibility.
  from tensorflow_model_analysis.version import VERSION as VERSION_STRING  # pylint: disable=reimported

  # TODO(b/73882264): The orders should be kept in order to make benchmark on
  # DataFlow work. We need to look into why the import orders matters for the
  # DataFlow benchmark.
  from tensorflow_model_analysis import addons
  from tensorflow_model_analysis import experimental
  from tensorflow_model_analysis import extractors
  from tensorflow_model_analysis import slicer
  from tensorflow_model_analysis import validators
  from tensorflow_model_analysis import evaluators
  from tensorflow_model_analysis import metrics
  from tensorflow_model_analysis import utils
  from tensorflow_model_analysis import writers
  from tensorflow_model_analysis import view
  from tensorflow_model_analysis import model_agnostic_eval

  # TODO(b/171992041): Deprecate use of EvalResult in the future.
  from tensorflow_model_analysis.view.view_types import EvalResult

except ImportError as err:
  import sys
  sys.stderr.write('Error importing: {}'.format(err))
# pylint: enable=g-statement-before-imports
# pylint: enable=g-import-not-at-top

def _jupyter_nbextension_paths():
  return [{
    'section': 'notebook',
    'src': 'static',
    'dest': 'tensorflow_model_analysis',
    'require': 'tensorflow_model_analysis/extension'
  }]
