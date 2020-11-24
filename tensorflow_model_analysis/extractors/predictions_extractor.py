# Lint as: python3
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
"""Batched predict extractor."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import copy

from typing import Dict, Optional, Text

import apache_beam as beam
from tensorflow_model_analysis import config
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import model_util
from tensorflow_model_analysis import types
from tensorflow_model_analysis.extractors import extractor
from tfx_bsl.tfxio import tensor_adapter

_PREDICTIONS_EXTRACTOR_STAGE_NAME = 'ExtractPredictions'


def PredictionsExtractor(
    eval_config: config.EvalConfig,
    eval_shared_model: Optional[types.MaybeMultipleEvalSharedModels] = None,
    tensor_adapter_config: Optional[tensor_adapter.TensorAdapterConfig] = None,
) -> extractor.Extractor:
  """Creates an extractor for performing predictions over a batch.

  The extractor runs in two modes:

  1) If one or more EvalSharedModels are provided

  The extractor's PTransform loads and runs the serving saved_model(s) against
  every extract yielding a copy of the incoming extracts with an additional
  extract added for the predictions keyed by tfma.PREDICTIONS_KEY. The model
  inputs are searched for under tfma.FEATURES_KEY (keras only) or tfma.INPUT_KEY
  (if tfma.FEATURES_KEY is not set or the model is non-keras). If multiple
  models are used the predictions will be stored in a dict keyed by model name.

  2) If no EvalSharedModels are provided

  The extractor's PTransform uses the config's ModelSpec.prediction_key(s)
  to lookup the associated prediction values stored as features under the
  tfma.FEATURES_KEY in extracts. The resulting values are then added to the
  extracts under the key tfma.PREDICTIONS_KEY.

  Note that the use of a prediction_key in the ModelSpecs serve two use cases:
    (a) as a key into the dict of predictions output (option 1)
    (b) as the key for a pre-computed prediction stored as a feature (option 2)

  Args:
    eval_config: Eval config.
    eval_shared_model: Shared model (single-model evaluation) or list of shared
      models (multi-model evaluation) or None (predictions obtained from
      features).
    tensor_adapter_config: Tensor adapter config which specifies how to obtain
      tensors from the Arrow RecordBatch. If None, we feed the raw examples to
      the model.

  Returns:
    Extractor for extracting predictions.
  """
  eval_shared_models = model_util.verify_and_update_eval_shared_models(
      eval_shared_model)
  if eval_shared_models:
    eval_shared_models = {m.model_name: m for m in eval_shared_models}

  # pylint: disable=no-value-for-parameter
  return extractor.Extractor(
      stage_name=_PREDICTIONS_EXTRACTOR_STAGE_NAME,
      ptransform=_ExtractPredictions(
          eval_config=eval_config,
          eval_shared_models=eval_shared_models,
          tensor_adapter_config=tensor_adapter_config))


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(types.Extracts)
def _ExtractPredictions(  # pylint: disable=invalid-name
    extracts: beam.pvalue.PCollection,
    eval_config: config.EvalConfig,
    eval_shared_models: Optional[Dict[Text, types.EvalSharedModel]],
    tensor_adapter_config: Optional[tensor_adapter.TensorAdapterConfig] = None,
) -> beam.pvalue.PCollection:
  """A PTransform that adds predictions and possibly other tensors to extracts.

  Args:
    extracts: PCollection of extracts containing model inputs keyed by
      tfma.FEATURES_KEY (if model inputs are named) or tfma.INPUTS_KEY (if model
      takes raw tf.Examples as input).
    eval_config: Eval config.
    eval_shared_models: Shared model parameters keyed by model name or None.
    tensor_adapter_config: Tensor adapter config which specifies how to obtain
      tensors from the Arrow RecordBatch.

  Returns:
    PCollection of Extracts updated with the predictions.
  """

  if eval_shared_models:
    signature_names = {}
    for spec in eval_config.model_specs:
      model_name = '' if len(eval_config.model_specs) == 1 else spec.name
      signature_names[model_name] = [spec.signature_name]

    return (
        extracts
        | 'Predict' >> beam.ParDo(
            model_util.ModelSignaturesDoFn(
                eval_config=eval_config,
                eval_shared_models=eval_shared_models,
                signature_names={constants.PREDICTIONS_KEY: signature_names},
                prefer_dict_outputs=False,
                tensor_adapter_config=tensor_adapter_config)))
  else:

    def extract_predictions(  # pylint: disable=invalid-name
        batched_extracts: types.Extracts) -> types.Extracts:
      """Extract predictions from extracts containing features."""
      result = copy.copy(batched_extracts)
      predictions = model_util.get_feature_values_for_model_spec_field(
          list(eval_config.model_specs), 'prediction_key', 'prediction_keys',
          result)
      if predictions is not None:
        result[constants.PREDICTIONS_KEY] = predictions
      return result

    return extracts | 'ExtractPredictions' >> beam.Map(extract_predictions)
