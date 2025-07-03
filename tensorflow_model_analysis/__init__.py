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
  from tensorflow_model_analysis.sdk import *

  # Allow api module types to be imported at the top-level since they are the
  # main public interface to using TFMA.
  from tensorflow_model_analysis.api import dataframe
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

  # TODO(b/73882264): The orders should be kept in order to make benchmark on
  # DataFlow work. We need to look into why the import orders matters for the
  # DataFlow benchmark.
  from tensorflow_model_analysis import extractors
  from tensorflow_model_analysis import slicer
  from tensorflow_model_analysis import validators
  from tensorflow_model_analysis import evaluators
  from tensorflow_model_analysis import metrics
  from tensorflow_model_analysis import utils
  from tensorflow_model_analysis import writers
  from tensorflow_model_analysis import view
  # TODO(b/228406044): Stop exposing tfma.types and migrate all internal users
  # to use the top-level symbols exported below (e.g. tfma.Extracts).
  from tensorflow_model_analysis.api import types

  # TODO(b/171992041): Deprecate use of EvalResult in the future.
  from tensorflow_model_analysis.view.view_types import EvalResult

  # Allow types to be imported at the top-level since they live in root dir.
  from tensorflow_model_analysis.api.types import AddMetricsCallbackType
  from tensorflow_model_analysis.api.types import EvalSharedModel
  from tensorflow_model_analysis.api.types import Extracts
  # TODO(b/120222218): Remove after passing of native FPL supported.
  from tensorflow_model_analysis.api.types import FeaturesPredictionsLabels
  # TODO(b/120222218): Remove after passing of native FPL supported.
  from tensorflow_model_analysis.api.types import MaterializedColumn
  from tensorflow_model_analysis.api.types import MaybeMultipleEvalSharedModels
  from tensorflow_model_analysis.api.types import ModelLoader
  from tensorflow_model_analysis.api.types import RaggedTensorValue
  from tensorflow_model_analysis.api.types import SparseTensorValue
  from tensorflow_model_analysis.api.types import TensorType
  from tensorflow_model_analysis.api.types import TensorTypeMaybeDict
  from tensorflow_model_analysis.api.types import TensorValue
  from tensorflow_model_analysis.api.types import VarLenTensorValue

  # Import VERSION as __version__ for compatibility with other TFX components.
  from tensorflow_model_analysis.version import VERSION as __version__

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

__all__ = [
  'AddMetricsCallbackType',
  'AggregationOptions',
  'analyze_raw_data',
  'AttributionsForSlice',
  'BatchedInputsToExtracts',
  'BinarizationOptions',
  'ConfidenceIntervalOptions',
  'CrossSliceMetricThreshold',
  'CrossSliceMetricThresholds',
  'CrossSlicingSpec',
  'default_eval_shared_model',
  'default_evaluators',
  'default_extractors',
  'default_writers',
  'EvalConfig',
  'EvalResult',
  'EvalSharedModel',
  'ExampleWeightOptions',
  'ExtractAndEvaluate',
  'ExtractEvaluateAndWriteResults',
  'Extracts',
  'FeaturesPredictionsLabels',
  'GenericChangeThreshold',
  'GenericValueThreshold',
  'InputsToExtracts',
  'is_batched_input',
  'is_legacy_estimator',
  'load_attributions',
  'load_eval_result',
  'load_eval_results',
  'load_metrics',
  'load_plots',
  'load_validation_result',
  'make_eval_results',
  'MaterializedColumn',
  'MaybeMultipleEvalSharedModels',
  'MetricConfig',
  'MetricsForSlice',
  'MetricsSpec',
  'MetricThreshold',
  'ModelLoader',
  'ModelSpec',
  'multiple_data_analysis',
  'multiple_model_analysis',
  'Options',
  'PaddingOptions',
  'PerSliceMetricThreshold',
  'PerSliceMetricThresholds',
  'PlotsForSlice',
  'RaggedTensorValue',
  'RepeatedInt32Value',
  'RepeatedStringValue',
  'run_model_analysis',
  'SlicingSpec',
  'SparseTensorValue',
  'TensorType',
  'TensorTypeMaybeDict',
  'TensorValue',
  'Validate',
  'ValidationResult',
  'VarLenTensorValue',
  'WriteResults'
]
