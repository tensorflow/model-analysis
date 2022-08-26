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
"""Batched predictions extractor."""

import copy
from typing import Dict, Optional, Tuple

from absl import logging
import apache_beam as beam
import numpy as np
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.extractors import extractor
from tensorflow_model_analysis.extractors import materialized_predictions_extractor
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.utils import model_util
from tensorflow_model_analysis.utils import util
from tfx_bsl.public.beam import run_inference
from tfx_bsl.public.proto import model_spec_pb2

from tensorflow_serving.apis import prediction_log_pb2


_PREDICTIONS_EXTRACTOR_STAGE_NAME = 'ExtractPredictions'


def PredictionsExtractor(
    eval_config: config_pb2.EvalConfig,
    eval_shared_model: Optional[types.MaybeMultipleEvalSharedModels] = None,
    experimental_bulk_inference: bool = False,
    batch_size: Optional[int] = None) -> extractor.Extractor:
  """Creates an extractor for performing predictions over a batch.

  The extractor's PTransform loads and runs the serving saved_model(s) against
  every Extracts yielding a copy of the incoming Extracts with an additional
  Extracts added for the predictions keyed by tfma.PREDICTIONS_KEY. The model
  inputs are searched for under tfma.FEATURES_KEY (keras only) or tfma.INPUT_KEY
  (if tfma.FEATURES_KEY is not set or the model is non-keras). If multiple
  models are used the predictions will be stored in a dict keyed by model name.

  Note that the prediction_key in the ModelSpecs also serves as a key into the
  dict of the prediction's output.

  Args:
    eval_config: Eval config.
    eval_shared_model: Shared model (single-model evaluation) or list of shared
      models (multi-model evaluation) or None (predictions obtained from
      features).
    experimental_bulk_inference: Controls which inference implementation will
      be used. If True, will use the experimental TFX-BSL Bulk Inference
      implementation.
    batch_size: For testing only. Allows users to set a static batch size in
      unit tests.

  Returns:
    Extractor for extracting predictions.
  """
  # TODO(b/239975835): Remove this Optional support for version 1.0.
  if eval_shared_model is None:
    logging.warning(
        'Calling the PredictionsExtractor with eval_shared_model=None is '
        'deprecated and no longer supported. This will break in version 1.0. '
        'Please update your implementation to call '
        'MaterializedPredictionsExtractor directly.')
    _, ptransform = materialized_predictions_extractor.MaterializedPredictionsExtractor(
        eval_config)
    # Note we are changing the stage name here for backwards compatibility. Old
    # clients expect these code paths to have the same stage name. New clients
    # should never reference the private stage name.
    return extractor.Extractor(
        stage_name=_PREDICTIONS_EXTRACTOR_STAGE_NAME, ptransform=ptransform)
  eval_shared_models = model_util.verify_and_update_eval_shared_models(
      eval_shared_model)
  # This should never happen, but verify_and_update_eval_shared_models can
  # theoretically return None or empty iterables.
  if not eval_shared_models:
    raise ValueError('No valid model(s) were provided. Please ensure that '
                     'EvalConfig.ModelSpec is correctly configured to enable '
                     'using the PredictionsExtractor.')

  if experimental_bulk_inference:
    # pylint: disable=no-value-for-parameter
    ptransform = _ExtractPredictionsOSS(
        eval_config=eval_config,
        eval_shared_models={m.model_name: m for m in eval_shared_models},
        batch_size=batch_size)
  else:
    ptransform = _ExtractPredictions(  # pylint: disable=no-value-for-parameter
        eval_config=eval_config,
        eval_shared_models={m.model_name: m for m in eval_shared_models})
  return extractor.Extractor(
      stage_name=_PREDICTIONS_EXTRACTOR_STAGE_NAME, ptransform=ptransform)


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(types.Extracts)
def _ExtractPredictions(
    extracts: beam.pvalue.PCollection, eval_config: config_pb2.EvalConfig,
    eval_shared_models: Dict[str, types.EvalSharedModel]
) -> beam.pvalue.PCollection:
  """A PTransform that adds predictions and possibly other tensors to Extracts.

  Args:
    extracts: PCollection of Extracts containing model inputs keyed by
      tfma.FEATURES_KEY (if model inputs are named) or tfma.INPUTS_KEY (if model
      takes raw tf.Examples as input).
    eval_config: Eval config.
    eval_shared_models: Shared model parameters keyed by model name.

  Returns:
    PCollection of Extracts updated with the predictions.
  """
  signature_names = {}
  for model_spec in eval_config.model_specs:
    model_name = '' if len(eval_config.model_specs) == 1 else model_spec.name
    signature_names[model_name] = [model_spec.signature_name]

  return (extracts
          | 'Inference' >> beam.ParDo(
              model_util.ModelSignaturesDoFn(
                  eval_config=eval_config,
                  eval_shared_models=eval_shared_models,
                  signature_names={constants.PREDICTIONS_KEY: signature_names},
                  prefer_dict_outputs=False)))


def _create_inference_input_tuple(  # pylint: disable=invalid-name
    extracts: types.Extracts) -> Tuple[types.Extracts, bytes]:
  """Creates a tuple containing the Extracts and input to the model."""
  try:
    # Note after split_extracts splits the Extracts batch, INPUT_KEY has a value
    # that is a 0 dimensional np.array. These arrays are indexed using the [()]
    # syntax.
    model_input = extracts[constants.INPUT_KEY][()]
  except KeyError as e:
    raise ValueError(
        f'Extracts must contain the input keyed by "{constants.INPUT_KEY}" for '
        'inference.') from e
  if not isinstance(model_input, bytes):
    raise ValueError(
        f'Extracts value at key: "{constants.INPUT_KEY}" is not of '
        'type bytes. Only serialized tf.Examples and serialized '
        'tf.SequenceExamples are currently supported. The value '
        f'is {model_input} and type {type(model_input)}.')
  return (extracts, model_input)


def _parse_prediction_log_to_tensor_value(  # pylint: disable=invalid-name
    prediction_log: prediction_log_pb2.PredictionLog) -> np.ndarray:
  """Parses the model inference values from a PredictionLog.

  Args:
    prediction_log: Prediction_log_pb2.PredictionLog containing inference
      results.

  Returns:
    Values parsed from the PredictionLog inference result. These values are
    formated in the format expected in TFMA PREDICTION_KEY Extracts value.
  """
  log_type = prediction_log.WhichOneof('log_type')
  if log_type == 'classify_log':
    raise NotImplementedError('ClassifyLog processing not implemented yet.')
  elif log_type == 'regress_log':
    return np.array([
        regression.value
        for regression in prediction_log.regress_log.response.result.regressions
    ],
                    dtype=float)
  elif log_type == 'predict_log':
    raise NotImplementedError('PredictLog processing not implemented yet.')
  elif log_type == 'multi_inference_log':
    raise NotImplementedError(
        'MultiInferenceLog processing not implemented yet.')
  elif log_type == 'session_log':
    raise ValueError('SessionLog processing is not supported.')
  else:
    raise NotImplementedError(f'Unsupported log_type: {log_type}')


def _insert_predictions_into_extracts(  # pylint: disable=invalid-name
    inference_tuple: Tuple[types.Extracts,
                           Tuple[prediction_log_pb2.PredictionLog, ...]],
    output_keys: Tuple[str]) -> types.Extracts:
  """Inserts tensor values from PredictionLogs into the Extracts.

  Args:
    inference_tuple: Tuple consisting of the Extracts and a nested tuple of
      predicition logs.
    output_keys: List of strings that will be used to create out output tensor
      dict value in PREDICTIONS_KEY.

  Returns:
    Extracts with the PREDICTIONS_KEY populated. Note: By convention,
    PREDICTIONS_KEY will point to a dictionary if there are multiple prediction
    logs and a single value if there is only one prediction log.
  """
  extracts = copy.copy(inference_tuple[0])
  prediction_logs = inference_tuple[1]
  tensor_values = [
      _parse_prediction_log_to_tensor_value(log) for log in prediction_logs
  ]
  if len(tensor_values) == 1:
    extracts[constants.PREDICTIONS_KEY] = tensor_values[0]
  else:
    if len(output_keys) != len(tensor_values):
      raise ValueError('Each key must correspond to a tensor value. Length of '
                       f'output_keys: {len(output_keys)}. Length of '
                       f'tensor_values: {len(tensor_values)}.')
    extracts[constants.PREDICTIONS_KEY] = dict(zip(output_keys, tensor_values))
  return extracts


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(types.Extracts)
def _ExtractPredictionsOSS(
    extracts: beam.pvalue.PCollection,
    eval_config: config_pb2.EvalConfig,
    eval_shared_models: Dict[str, types.EvalSharedModel],
    batch_size: Optional[int] = None) -> beam.pvalue.PCollection:
  """A PTransform that adds predictions and possibly other tensors to Extracts.

  Args:
    extracts: PCollection of Extracts containing model inputs keyed by
      tfma.FEATURES_KEY (if model inputs are named) or tfma.INPUTS_KEY (if model
      takes raw tf.Examples as input).
    eval_config: Eval config.
    eval_shared_models: Shared model parameters keyed by model name.
    batch_size: Allows overriding dynamic batch size.

  Returns:
    PCollection of Extracts updated with the predictions.
  """
  model_name_and_inference_spec_type_tuples = []
  for model_spec in eval_config.model_specs:
    try:
      eval_shared_model = eval_shared_models[model_spec.name]
    except KeyError as e:
      raise ValueError(
          'ModelSpec.name should match EvalSharedModel.model_name.') from e
    inference_spec_type = model_spec_pb2.InferenceSpecType()
    inference_spec_type.saved_model_spec.model_path = eval_shared_model.model_path
    inference_spec_type.saved_model_spec.tag[:] = eval_shared_model.model_loader.tags
    if model_spec.signature_name:
      inference_spec_type.saved_model_spec.signature_name[:] = [
          model_spec.signature_name
      ]
    model_name_and_inference_spec_type_tuples.append(
        (model_spec.name, inference_spec_type))

  # The output of RunInferencePerModel will align with this ordering. Output
  # order is determined by the ordering of the inference_spec_types parameter.
  aligned_model_names, aligned_inference_spec_types = zip(
      *model_name_and_inference_spec_type_tuples)
  extracts = (
      extracts
      # Extracts are fed in pre-batched, but BulkInference has specific
      # batch handling and batching requirements. To accomodate the API and
      # encapsulate the inference batching logic, we unbatch here. This function
      # returns new Extracts dicts and will not modify the input Extracts.
      | 'SplitExtracts' >> beam.FlatMap(
          util.split_extracts, expand_zero_dims=False)
      # The BulkInference API allows for key forwarding. To avoid a join
      # after running inference, we forward the unbatched Extracts as a key.
      | 'CreateInferenceInputTuple' >> beam.Map(_create_inference_input_tuple)
      # TODO(b/241022420): Set load_override_fn here to avoid loading the
      # model twice.
      | 'RunInferencePerModel' >> run_inference.RunInferencePerModel(
          inference_spec_types=aligned_inference_spec_types)
      # Combine predictions back into the original Extracts.
      | 'InsertPredictionsIntoExtracts' >> beam.Map(
          _insert_predictions_into_extracts, output_keys=aligned_model_names))
  # Beam batch will group single Extracts into a batch. Then
  # merge_extracts will flatten the batch into a single "batched"
  # extract.
  batch_extracts_stage_name = 'BatchSingleExampleExtracts'
  if batch_size is not None:
    extracts |= batch_extracts_stage_name >> beam.BatchElements(
        min_batch_size=batch_size, max_batch_size=batch_size)
  else:
    extracts |= batch_extracts_stage_name >> beam.BatchElements()
  return extracts | 'MergeExtracts' >> beam.Map(
      util.merge_extracts, squeeze_two_dim_vector=False)
