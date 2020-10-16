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

from typing import Dict, List, Optional, Text

import apache_beam as beam
import numpy as np
import tensorflow as tf
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


@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(types.Extracts)
class _BatchedPredictionDoFn(model_util.BatchReducibleBatchedDoFnWithModels):
  """A DoFn that loads the models and predicts."""

  def __init__(
      self,
      eval_config: config.EvalConfig,
      eval_shared_models: Dict[Text, types.EvalSharedModel],
      tensor_adapter_config: Optional[
          tensor_adapter.TensorAdapterConfig] = None,
  ) -> None:
    super(_BatchedPredictionDoFn, self).__init__(
        {k: v.model_loader for k, v in eval_shared_models.items()})
    self._eval_config = eval_config
    self._tensor_adapter_config = tensor_adapter_config
    self._tensor_adapter = None

  def setup(self):
    super(_BatchedPredictionDoFn, self).setup()
    if self._tensor_adapter_config is not None:
      self._tensor_adapter = tensor_adapter.TensorAdapter(
          self._tensor_adapter_config)

  def _batch_reducible_process(
      self, batched_extract: types.Extracts) -> List[types.Extracts]:
    result = copy.copy(batched_extract)
    record_batch = batched_extract[constants.ARROW_RECORD_BATCH_KEY]
    serialized_examples = batched_extract[constants.INPUT_KEY]
    predictions = [None] * record_batch.num_rows
    for spec in self._eval_config.model_specs:
      # To maintain consistency between settings where single models are used,
      # always use '' as the model name regardless of whether a name is passed.
      model_name = spec.name if len(self._eval_config.model_specs) > 1 else ''
      if model_name not in self._loaded_models:
        raise ValueError(
            'loaded model for "{}" not found: eval_config={}'.format(
                spec.name, self._eval_config))
      model = self._loaded_models[model_name]
      signature_name = spec.signature_name
      input_specs = model_util.get_input_specs(model, signature_name) or {}
      # If tensor_adaptor and input_specs exist then filter the inputs by input
      # names (unlike estimators, keras does not accept unknown inputs).
      # However, avoid getting the tensors if we appear to be feeding serialized
      # examples to the model.
      if (self._tensor_adapter and input_specs and
          not (len(input_specs) == 1 and
               next(iter(input_specs.values())).dtype == tf.string and
               model_util.find_input_name_in_features(
                   set(self._tensor_adapter.TypeSpecs().keys()),
                   next(iter(input_specs.keys()))) is None)):
        inputs = model_util.filter_tensors_by_input_names(
            self._tensor_adapter.ToBatchTensors(record_batch),
            list(input_specs.keys()))
      else:
        inputs = None
      if not inputs:
        # Assume serialized examples
        assert serialized_examples is not None, 'Raw examples not found.'
        inputs = serialized_examples
        # If a signature name was not provided, default to using the serving
        # signature since parsing normally will be done outside model.
        if not signature_name:
          signature_name = model_util.get_default_signature_name(model)
      signature = model_util.get_callable(model, signature_name)
      if signature is None:
        raise ValueError(
            'PredictExtractor V2 requires a keras model or a serving model. '
            'If using EvalSavedModel then you must use PredictExtractor V1.')
      if isinstance(inputs, dict):
        if signature is model:
          outputs = signature(inputs)
        else:
          outputs = signature(**inputs)
      else:
        outputs = signature(tf.constant(inputs, dtype=tf.string))
      for i in range(record_batch.num_rows):
        if isinstance(outputs, dict):
          output = {k: v[i].numpy() for k, v in outputs.items()}
          # Keras and regression serving models return a dict of predictions
          # even for single-outputs. Convert these to a single tensor for
          # compatibility with the labels (and model.predict API).
          if len(output) == 1:
            output = list(output.values())[0]
        else:
          output = np.asarray(outputs)[i]
        # If only one model, the predictions are stored without using a dict
        if len(self._eval_config.model_specs) == 1:
          predictions[i] = output
        else:
          if predictions[i] is None:
            predictions[i] = {}
          predictions[i][spec.name] = output  # pytype: disable=unsupported-operands
    result[constants.PREDICTIONS_KEY] = predictions
    return [result]


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

  return (extracts
          | 'Predict' >> beam.ParDo(
              _BatchedPredictionDoFn(
                  eval_config=eval_config,
                  eval_shared_models=eval_shared_models,
                  tensor_adapter_config=tensor_adapter_config)))
