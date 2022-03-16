# Copyright 2020 Google LLC
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
"""Transformed features extractor."""

from typing import Dict

import apache_beam as beam
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.extractors import extractor
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.utils import model_util

_TRANSFORMED_FEATURES_EXTRACTOR_STAGE_NAME = 'ExtractTransformedFeatures'

# TODO(b/173029091): Re-add tft_layer.
_DEFAULT_SIGNATURE_NAMES = ('transformed_features', 'transformed_labels')


def TransformedFeaturesExtractor(
    eval_config: config_pb2.EvalConfig,
    eval_shared_model: types.MaybeMultipleEvalSharedModels,
) -> extractor.Extractor:
  """Creates an extractor for extracting transformed features.

  The extractor's PTransform loads the saved_model(s) invoking the preprocessing
  functions against every extract yielding a copy of the incoming extracts with
  a tfma.TRANSFORMED_FEATURES_KEY containing the output from the preprocessing
  functions.

  Args:
    eval_config: Eval config.
    eval_shared_model: Shared model (single-model evaluation) or list of shared
      models (multi-model evaluation).

  Returns:
    Extractor for extracting preprocessed features.
  """
  eval_shared_models = model_util.verify_and_update_eval_shared_models(
      eval_shared_model)

  # pylint: disable=no-value-for-parameter
  return extractor.Extractor(
      stage_name=_TRANSFORMED_FEATURES_EXTRACTOR_STAGE_NAME,
      ptransform=_ExtractTransformedFeatures(
          eval_config=eval_config,
          eval_shared_models={m.model_name: m for m in eval_shared_models}))


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(types.Extracts)
def _ExtractTransformedFeatures(  # pylint: disable=invalid-name
    extracts: beam.pvalue.PCollection,
    eval_config: config_pb2.EvalConfig,
    eval_shared_models: Dict[str, types.EvalSharedModel],
) -> beam.pvalue.PCollection:
  """A PTransform that updates extracts to include transformed features.

  Args:
    extracts: PCollection of extracts containing raw inputs keyed by
      tfma.FEATURES_KEY (if preprocessing function inputs are named) or
      tfma.INPUTS_KEY (if preprocessing functions take raw tf.Examples as input)
    eval_config: Eval config.
    eval_shared_models: Shared model parameters keyed by model name.

  Returns:
    PCollection of Extracts updated with the to include transformed features
    stored under the key tfma.TRANSFORMED_FEATURES_KEY.
  """
  signature_names = {}
  for spec in eval_config.model_specs:
    model_name = '' if len(eval_config.model_specs) == 1 else spec.name
    signature_names[model_name] = list(spec.preprocessing_function_names)

  return (extracts
          | 'Predict' >> beam.ParDo(
              model_util.ModelSignaturesDoFn(
                  eval_config=eval_config,
                  eval_shared_models=eval_shared_models,
                  signature_names={
                      constants.TRANSFORMED_FEATURES_KEY: signature_names
                  },
                  default_signature_names=list(_DEFAULT_SIGNATURE_NAMES),
                  prefer_dict_outputs=True)))
