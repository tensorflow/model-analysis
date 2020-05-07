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
import tensorflow as tf
from tensorflow_model_analysis import config
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import model_util
from tensorflow_model_analysis import types
from tensorflow_model_analysis.extractors import extractor
from tfx_bsl.tfxio import tensor_adapter

BATCHED_PREDICT_EXTRACTOR_STAGE_NAME = 'ExtractBatchPredictions'

PREDICT_SIGNATURE_DEF_KEY = 'predict'


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
      self, batched_extract: types.Extracts) -> types.Extracts:
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
      loaded_model = self._loaded_models[model_name]
      if not hasattr(loaded_model, 'signatures'):
        raise ValueError(
            'PredictExtractor V2 requires a keras model or a serving model. '
            'If using EvalSavedModel then you must use PredictExtractor V1.')
      signatures = loaded_model.signatures

      signature_key = spec.signature_name
      # TODO(mdreves): Add support for multiple signatures per output.
      if not signature_key:
        # First try 'predict' then try 'serving_default'. The estimator output
        # for the 'serving_default' key does not include all the heads in a
        # multi-head model. However, keras only uses the 'serving_default' for
        # its outputs. Note that the 'predict' key only exists for estimators
        # for multi-head models, for single-head models only 'serving_default'
        # is used.
        signature_key = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        if PREDICT_SIGNATURE_DEF_KEY in signatures:
          signature_key = PREDICT_SIGNATURE_DEF_KEY
      if signature_key not in signatures:
        raise ValueError('{} not found in model signatures: {}'.format(
            signature_key, signatures))
      signature = signatures[signature_key]

      # If input names exist then filter the inputs by these names (unlike
      # estimators, keras does not accept unknown inputs).
      input_names = None
      # First arg of structured_input_signature tuple is shape, second is dtype
      # (we currently only support named params passed as a dict)
      if (signature.structured_input_signature and
          len(signature.structured_input_signature) == 2 and
          isinstance(signature.structured_input_signature[1], dict)):
        input_names = [name for name in signature.structured_input_signature[1]]
      elif hasattr(loaded_model, 'input_names'):
        # Calling keras_model.input_names does not work properly in TF 1.15.0.
        # As a work around, make sure the signature.structured_input_signature
        # check is before this check (see b/142807137).
        input_names = loaded_model.input_names
      inputs = None
      if input_names:
        get_tensors = True
        if len(input_names) == 1:
          # Avoid getting the tensors in case we are feeding serialized examples
          # to the model.
          if model_util.find_input_name_in_features(
              set(record_batch.schema.names), input_names[0]) is None:
            get_tensors = False
        if get_tensors:
          inputs = model_util.filter_tensors_by_input_names(
              self._tensor_adapter.ToBatchTensors(record_batch), input_names)
      if not inputs:
        # Assume serialized examples
        assert serialized_examples is not None, 'Raw examples not found.'
        inputs = serialized_examples
      if isinstance(inputs, dict):
        outputs = signature(**inputs)
      else:
        outputs = signature(tf.constant(inputs, dtype=tf.string))
      for i in range(record_batch.num_rows):
        output = {k: v[i].numpy() for k, v in outputs.items()}
        # Keras and regression serving models return a dict of predictions even
        # for single-outputs. Convert these to a single tensor for compatibility
        # with the labels (and model.predict API).
        if len(output) == 1:
          output = list(output.values())[0]
        # If only one model, the predictions are stored without using a dict
        if len(self._eval_config.model_specs) == 1:
          predictions[i] = output
        else:
          if predictions[i] is None:
            predictions[i] = {}
          predictions[i][spec.name] = output  # pytype: disable=unsupported-operands
    result[constants.PREDICTIONS_KEY] = predictions
    return [result]  # pytype: disable=bad-return-type


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
