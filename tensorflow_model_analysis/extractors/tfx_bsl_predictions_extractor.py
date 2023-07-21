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
"""Predictions extractor for using TFX-BSL Bulk Inference."""

import copy
from typing import Dict, Iterable, List, Optional, Tuple, TypeVar, Union

import apache_beam as beam
import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import types
from tensorflow_model_analysis.extractors import extractor
from tensorflow_model_analysis.extractors import inference_base
from tensorflow_model_analysis.extractors import predictions_extractor
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.utils import model_util
from tfx_bsl.public.beam import run_inference
from tfx_bsl.public.proto import model_spec_pb2

from tensorflow_serving.apis import prediction_log_pb2


_K = TypeVar('_K')
PossibleInputTypes = Union[tf.train.Example, bytes]
KeyAndOutput = Tuple[_K, PossibleInputTypes]
MapModelNameToOutput = Dict[str, prediction_log_pb2.PredictionLog]
KeyAndOutputMap = Tuple[_K, MapModelNameToOutput]


class TfxBslInferenceWrapper(beam.PTransform):
  """Wrapper for TFX-BSL bulk inference implementation."""

  def __init__(self,
               model_specs: List[config_pb2.ModelSpec],
               name_to_eval_shared_model: Dict[str, types.EvalSharedModel]):
    """Converts TFMA config into library-specific configuration.

    Args:
      model_specs: TFMA ModelSpec config to be translated to TFX-BSL Config.
      name_to_eval_shared_model: Map of model name to associated EvalSharedModel
        object.
    """
    super().__init__()
    model_names = []
    inference_specs = []
    for model_spec in model_specs:
      eval_shared_model = model_util.get_eval_shared_model(
          model_spec.name, name_to_eval_shared_model)
      inference_spec_type = model_spec_pb2.InferenceSpecType(
          saved_model_spec=model_spec_pb2.SavedModelSpec(
              model_path=eval_shared_model.model_path,
              tag=eval_shared_model.model_loader.tags,
              signature_name=[model_spec.signature_name],
          ),
          batch_parameters=model_spec_pb2.BatchParameters(
              min_batch_size=model_spec.inference_batch_size,
              max_batch_size=model_spec.inference_batch_size,
          ),
      )
      model_names.append(model_spec.name)
      inference_specs.append(inference_spec_type)
    self._aligned_model_names = tuple(model_names)
    self._aligned_inference_specs = tuple(inference_specs)

  def expand(
      self, pcoll: beam.PCollection[KeyAndOutput]
  ) -> beam.PCollection[KeyAndOutputMap]:
    # TODO(b/241022420): Set load_override_fn here to avoid loading the model
    # twice.
    return (
        pcoll
        | 'TfxBslBulkInference' >> run_inference.RunInferencePerModel(
            inference_spec_types=self._aligned_inference_specs)
        | 'CreateModelNameToPredictionLog' >>
        beam.MapTuple(lambda extracts, logs:  # pylint: disable=g-long-lambda
                      (extracts, dict(zip(self._aligned_model_names, logs)))))


def TfxBslPredictionsExtractor(
    eval_config: config_pb2.EvalConfig,
    eval_shared_model: types.MaybeMultipleEvalSharedModels,
    output_batch_size: Optional[int] = None,
    output_keypath: Iterable[str] = (constants.PREDICTIONS_KEY,),
) -> extractor.Extractor:
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
      models (multi-model evaluation).
    output_batch_size: Sets a static output batch size for bulk inference. Note:
      this only affects the rebatched output batch size to set inference batch
      size set ModelSpec.inference_batch_size.
    output_keypath: A sequence of keys to be used as the path to traverse and
      insert the outputs in the extract.

  Returns:
    Extractor for extracting predictions.
  """
  eval_shared_models = model_util.verify_and_update_eval_shared_models(
      eval_shared_model)
  # This should never happen, but verify_and_update_eval_shared_models can
  # theoretically return None or empty iterables.
  if not eval_shared_models:
    raise ValueError('No valid model(s) were provided. Please ensure that '
                     'EvalConfig.ModelSpec is correctly configured to enable '
                     'using the PredictionsExtractor.')

  name_to_eval_shared_model = {m.model_name: m for m in eval_shared_models}
  model_specs = []
  for model_spec in eval_config.model_specs:
    if not model_spec.signature_name:
      eval_shared_model = model_util.get_eval_shared_model(
          model_spec.name, name_to_eval_shared_model)
      model_spec = copy.copy(model_spec)
      # Select a default signature. Note that this may differ from the
      # 'serving_default' signature.
      model_spec.signature_name = model_util.get_default_signature_name_from_model_path(
          eval_shared_model.model_path)
    model_specs.append(model_spec)

  tfx_bsl_inference_ptransform = inference_base.RunInference(
      inference_ptransform=TfxBslInferenceWrapper(
          model_specs, name_to_eval_shared_model
      ),
      output_batch_size=output_batch_size,
      output_keypath=output_keypath,
  )
  # pylint: disable=no-value-for-parameter
  return extractor.Extractor(
      stage_name=predictions_extractor.PREDICTIONS_EXTRACTOR_STAGE_NAME,
      ptransform=tfx_bsl_inference_ptransform)
