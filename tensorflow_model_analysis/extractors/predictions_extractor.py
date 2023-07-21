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
"""Batched predictions extractor for ModelSignaturesDoFn."""

from typing import List, Optional, Sequence

from absl import logging
import apache_beam as beam
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import types
from tensorflow_model_analysis.extractors import extractor
from tensorflow_model_analysis.extractors import materialized_predictions_extractor
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.utils import model_util


PREDICTIONS_EXTRACTOR_STAGE_NAME = 'ExtractPredictions'


def PredictionsExtractor(
    eval_config: config_pb2.EvalConfig,
    eval_shared_model: Optional[types.MaybeMultipleEvalSharedModels] = None,
    output_keypath: Sequence[str] = (constants.PREDICTIONS_KEY,),
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
      models (multi-model evaluation) or None (predictions obtained from
      features).
    output_keypath: A sequence of keys to be used as the path to traverse and
      insert the outputs in the extract.

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
    _, ptransform = (
        materialized_predictions_extractor.MaterializedPredictionsExtractor(
            eval_config, output_keypath=output_keypath
        )
    )
    # Note we are changing the stage name here for backwards compatibility. Old
    # clients expect these code paths to have the same stage name. New clients
    # should never reference the private stage name.
    return extractor.Extractor(
        stage_name=PREDICTIONS_EXTRACTOR_STAGE_NAME, ptransform=ptransform)

  return extractor.Extractor(
      stage_name=PREDICTIONS_EXTRACTOR_STAGE_NAME,
      ptransform=_ModelSignaturesInferenceWrapper(  # pylint: disable=no-value-for-parameter
          model_specs=list(eval_config.model_specs),
          eval_shared_model=eval_shared_model,
          output_keypath=output_keypath,
      ),
  )


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(types.Extracts)
def _ModelSignaturesInferenceWrapper(
    extracts: beam.pvalue.PCollection,
    model_specs: List[config_pb2.ModelSpec],
    eval_shared_model: types.MaybeMultipleEvalSharedModels,
    output_keypath: Sequence[str],
) -> beam.pvalue.PCollection:
  """A PTransform that adds predictions and possibly other tensors to Extracts.

  Args:
    extracts: PCollection of Extracts containing model inputs keyed by
      tfma.FEATURES_KEY (if model inputs are named) or tfma.INPUTS_KEY (if model
      takes raw tf.Examples as input).
    model_specs: Model specs each of which corresponds to each of the
      eval_shared_models.
    eval_shared_model: Shared model parameters keyed by model name.
    output_keypath: A sequence of keys to be used as the path to traverse and
      insert the outputs in the extract.

  Returns:
    PCollection of Extracts updated with the predictions.
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
  signature_names = {}
  for model_spec in model_specs:
    model_name = '' if len(model_specs) == 1 else model_spec.name
    signature_names[model_name] = [model_spec.signature_name]

  return extracts | 'Inference' >> beam.ParDo(
      model_util.ModelSignaturesDoFn(
          model_specs=model_specs,
          eval_shared_models=name_to_eval_shared_model,
          output_keypath=output_keypath,
          signature_names=signature_names,
          prefer_dict_outputs=False,
      )
  )
