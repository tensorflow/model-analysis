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

from typing import Dict, Optional, Text

import apache_beam as beam
from tensorflow_model_analysis import config
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import model_util
from tensorflow_model_analysis import types
from tensorflow_model_analysis.extractors import extractor
from tfx_bsl.tfxio import tensor_adapter

BATCHED_PREDICT_EXTRACTOR_STAGE_NAME = 'ExtractBatchPredictions'


def BatchedPredictExtractor(
    eval_config: config.EvalConfig,
    eval_shared_model: types.MaybeMultipleEvalSharedModels,
    tensor_adapter_config: Optional[tensor_adapter.TensorAdapterConfig] = None,
) -> extractor.Extractor:
  """Creates an extractor for performing predictions over a batch.

  The extractor's PTransform loads and runs the serving saved_model(s) against
  every extract yielding a copy of the incoming extracts with an additional
  extract added for the predictions keyed by tfma.PREDICTIONS_KEY. The model
  inputs are searched for under tfma.FEATURES_KEY (keras only) or tfma.INPUT_KEY
  (if tfma.FEATURES_KEY is not set or the model is non-keras). If multiple
  models are used the predictions will be stored in a dict keyed by model name.

  Args:
    eval_config: Eval config.
    eval_shared_model: Shared model (single-model evaluation) or list of shared
      models (multi-model evaluation).
    tensor_adapter_config: Tensor adapter config which specifies how to obtain
      tensors from the Arrow RecordBatch. If None, we feed the raw examples to
      the model.

  Returns:
    Extractor for extracting predictions.
  """
  eval_shared_models = model_util.verify_and_update_eval_shared_models(
      eval_shared_model)

  # pylint: disable=no-value-for-parameter
  return extractor.Extractor(
      stage_name=BATCHED_PREDICT_EXTRACTOR_STAGE_NAME,
      ptransform=_ExtractBatchedPredictions(
          eval_config=eval_config,
          eval_shared_models={m.model_name: m for m in eval_shared_models},
          tensor_adapter_config=tensor_adapter_config))


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(types.Extracts)
def _ExtractBatchedPredictions(  # pylint: disable=invalid-name
    extracts: beam.pvalue.PCollection,
    eval_config: config.EvalConfig,
    eval_shared_models: Dict[Text, types.EvalSharedModel],
    tensor_adapter_config: Optional[tensor_adapter.TensorAdapterConfig] = None,
) -> beam.pvalue.PCollection:
  """A PTransform that adds predictions and possibly other tensors to extracts.

  Args:
    extracts: PCollection of extracts containing model inputs keyed by
      tfma.FEATURES_KEY (if model inputs are named) or tfma.INPUTS_KEY (if model
      takes raw tf.Examples as input).
    eval_config: Eval config.
    eval_shared_models: Shared model parameters keyed by model name.
    tensor_adapter_config: Tensor adapter config which specifies how to obtain
      tensors from the Arrow RecordBatch.

  Returns:
    PCollection of Extracts updated with the predictions.
  """

  signature_names = {}
  for spec in eval_config.model_specs:
    model_name = '' if len(eval_config.model_specs) == 1 else spec.name
    signature_names[model_name] = [spec.signature_name]

  return (extracts
          | 'Predict' >> beam.ParDo(
              model_util.ModelSignaturesDoFn(
                  eval_config=eval_config,
                  eval_shared_models=eval_shared_models,
                  signature_names={constants.PREDICTIONS_KEY: signature_names},
                  prefer_dict_outputs=False,
                  tensor_adapter_config=tensor_adapter_config)))
