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
"""Base inference implementation updates extracts with inference results."""

from typing import Dict, Optional, Sequence, Tuple, Union

from absl import logging
import apache_beam as beam
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import types
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.utils import model_util
from tensorflow_model_analysis.utils import util

from tensorflow.python.saved_model import loader_impl  # pylint: disable=g-direct-tensorflow-import
from tensorflow_serving.apis import prediction_log_pb2


def is_valid_config_for_bulk_inference(
    eval_config: config_pb2.EvalConfig,
    eval_shared_model: Optional[types.MaybeMultipleEvalSharedModels] = None
) -> bool:
  """Validates config for use with Tfx-Bsl and ServoBeam Bulk Inference."""
  eval_shared_models = model_util.verify_and_update_eval_shared_models(
      eval_shared_model)
  if eval_shared_models is None:
    logging.warning('Invalid Bulk Inference Config: There must be at least one '
                    'eval_shared_model to run servo/tfx-bsl bulk inference.')
    return False
  for eval_shared_model in eval_shared_models:
    if eval_shared_model.model_type not in (constants.TF_GENERIC,
                                            constants.TF_ESTIMATOR):
      logging.warning('Invalid Bulk Inference Config: Only TF2 and TF '
                      'Estimator models are supported for servo/tfx-bsl bulk '
                      'inference')
      return False
  name_to_eval_shared_model = {m.model_name: m for m in eval_shared_models}
  for model_spec in eval_config.model_specs:
    eval_shared_model = model_util.get_eval_shared_model(
        model_spec.name, name_to_eval_shared_model)
    saved_model = loader_impl.parse_saved_model(eval_shared_model.model_path)
    if model_spec.signature_name:
      signature_name = model_spec.signature_name
    else:
      signature_name = (
          model_util.get_default_signature_name_from_saved_model_proto(
              saved_model))
    try:
      signature_def = model_util.get_signature_def_from_saved_model_proto(
          signature_name, saved_model)
    except ValueError:
      logging.warning('Invalid Bulk Inference Config: models must have a '
                      'signature to run servo/tfx-bsl bulk inference. Consider '
                      'setting the signature explicitly in the ModelSpec.')
      return False
    if len(signature_def.inputs) != 1:
      logging.warning('Invalid Bulk Inference Config: signature must accept '
                      'only one input for servo/tfx-bsl bulk inference.')
      return False
    if list(signature_def.inputs.values())[0].dtype != tf.string:
      logging.warning('Invalid Bulk Inference Config: signature must accept '
                      'string input to run servo/tfx-bsl bulk inference.')
      return False
  return True


def _create_inference_input_tuple(
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


def _parse_prediction_log_to_tensor_value(
    prediction_log: prediction_log_pb2.PredictionLog
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
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
    assert len(
        prediction_log.classify_log.response.result.classifications) == 1, (
            'We expecth the number of classifications per PredictionLog to be '
            'one because TFX-BSL RunInference expects single input/output and '
            'handles batching entirely internally.')
    classes = np.array([
        c.label for c in
        prediction_log.classify_log.response.result.classifications[0].classes
    ],
                       dtype=object)
    scores = np.array([
        c.score for c in
        prediction_log.classify_log.response.result.classifications[0].classes
    ],
                      dtype=np.float32)
    return {'classes': classes, 'scores': scores}
  elif log_type == 'regress_log':
    return np.array([
        regression.value
        for regression in prediction_log.regress_log.response.result.regressions
    ],
                    dtype=float)
  elif log_type == 'predict_log':
    output_tensor_name_to_tensor = {
        k: np.squeeze(tf.make_ndarray(v), axis=0)
        for k, v in prediction_log.predict_log.response.outputs.items()
    }
    # If there is only one tensor (i.e. one dictionary item), we remove the
    # tensor from the dict and return it directly. Generally, TFMA will not
    # return a dictionary with a single value.
    if len(output_tensor_name_to_tensor) == 1:
      return list(output_tensor_name_to_tensor.values())[0]
    return output_tensor_name_to_tensor
  elif log_type == 'multi_inference_log':
    raise NotImplementedError(
        'MultiInferenceLog processing not implemented yet.')
  elif log_type == 'session_log':
    raise ValueError('SessionLog processing is not supported.')
  else:
    raise NotImplementedError(f'Unsupported log_type: {log_type}')


def _insert_predictions_into_extracts(
    inference_tuple: Tuple[
        types.Extracts, Dict[str, prediction_log_pb2.PredictionLog]
    ],
    output_keypath: Sequence[str],
) -> types.Extracts:
  """Inserts tensor values from PredictionLogs into the Extracts.

  Args:
    inference_tuple: This is the output of inference. It includes the key
      forwarded extracts and a dict of model name to predicition logs.
    output_keypath: A sequence of keys to be used as the path to traverse and
      insert the outputs in the extract.

  Returns:
    Extracts with the PREDICTIONS_KEY populated. Note: By convention,
    PREDICTIONS_KEY will point to a dictionary if there are multiple
    prediction logs and a single value if there is only one prediction log.
  """
  extracts, model_names_to_prediction_logs = inference_tuple
  model_name_to_tensors = {
      name: _parse_prediction_log_to_tensor_value(log)
      for name, log in model_names_to_prediction_logs.items()
  }
  # If there is only one model (i.e. one dictionary item), we remove the model
  # output from the dict and store it directly under the PREDICTIONS_KEY. This
  # is in line with the general TFMA pattern of not storing one-item
  # dictionaries.
  if len(model_name_to_tensors) == 1:
    return util.copy_and_set_by_keys(
        extracts, output_keypath, next(iter(model_name_to_tensors.values()))
    )
  else:
    return util.copy_and_set_by_keys(
        extracts, output_keypath, model_name_to_tensors
    )


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(types.Extracts)
def RunInference(  # pylint: disable=invalid-name
    extracts: beam.pvalue.PCollection,
    inference_ptransform: beam.PTransform,
    output_batch_size: Optional[int] = None,
    output_keypath: Sequence[str] = (constants.PREDICTIONS_KEY,),
) -> beam.pvalue.PCollection:
  """A PTransform that adds predictions and possibly other tensors to Extracts.

  Args:
    extracts: PCollection of Extracts containing model inputs keyed by
      tfma.FEATURES_KEY (if model inputs are named) or tfma.INPUTS_KEY (if model
      takes raw tf.Examples as input).
    inference_ptransform: Bulk inference ptransform used to generate
      predictions. This allows users to use different implementations depending
      on evironment or Beam runner (e.g. a cloud-friendly OSS implementation or
      an internal-specific implementation). These implementations should accept
      a pcollection consisting of tuples containing a key and a single example.
      The key may be anything and the example may be a tf.Example or serialized
      tf.Example.
    output_batch_size: Sets a static output batch size.
    output_keypath: A sequence of keys to be used as the path to traverse and
      insert the outputs in the extract.

  Returns:
    PCollection of Extracts updated with the predictions.
  """
  extracts = (
      extracts
      # Extracts are fed in pre-batched, but BulkInference has specific
      # batch handling and batching requirements. To accomodate the API and
      # encapsulate the inference batching logic, we unbatch here. This function
      # returns new Extracts dicts and will not modify the input Extracts.
      | 'SplitExtracts'
      >> beam.FlatMap(util.split_extracts, expand_zero_dims=False)
      # The BulkInference API allows for key forwarding. To avoid a join
      # after running inference, we forward the unbatched Extracts as a key.
      | 'CreateInferenceInputTuple' >> beam.Map(_create_inference_input_tuple)
      | 'RunInferencePerModel' >> inference_ptransform
      # Combine predictions back into the original Extracts.
      | 'InsertPredictionsIntoExtracts'
      >> beam.Map(_insert_predictions_into_extracts, output_keypath)
  )
  # Beam batch will group single Extracts into a batch. Then
  # merge_extracts will flatten the batch into a single "batched"
  # extract.
  if output_batch_size is not None:
    batch_kwargs = {
        'min_batch_size': output_batch_size,
        'max_batch_size': output_batch_size
    }
  else:
    # Default batch parameters.
    batch_kwargs = {}
  return (extracts
          | 'BatchSingleExampleExtracts' >> beam.BatchElements(**batch_kwargs)
          | 'MergeExtracts' >> beam.Map(
              util.merge_extracts, squeeze_two_dim_vector=False))
