# Copyright 2022 Google LLC
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
"""Counterfactual predictions extractor."""

from typing import Dict, Iterable, Mapping, Optional

import apache_beam as beam
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import types
from tensorflow_model_analysis.extractors import extractor
from tensorflow_model_analysis.extractors import predictions_extractor
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.utils import model_util

_SUPPORTED_MODEL_TYPES = frozenset([constants.TF_KERAS, constants.TF_GENERIC])
_COUNTERFACTUAL_PREDICTIONS_EXTRACTOR_NAME = 'CounterfactualPredictionsExtractor'
# The extracts key under which the non-CF INPUT_KEY value is temporarily stored,
# when invoking one or more PredictionsExtractors on modified inputs.
_TEMP_ORIG_INPUT_KEY = 'non_counterfactual_input'
CounterfactualConfig = Dict[str, str]


def CounterfactualPredictionsExtractor(  # pylint: disable=invalid-name
    eval_shared_models: types.MaybeMultipleEvalSharedModels,
    eval_config: config_pb2.EvalConfig,
    cf_configs: Mapping[str, CounterfactualConfig]) -> extractor.Extractor:
  """Creates a CF predictions extractor by wrapping the PredictionsExtractor.

  Example usage:

    eval_config = tfma.EvalConfig(model_specs=[
       tfma.ModelSpec(name='orig', is_baseline=True),
       tfma.ModelSpec(name='cf')])
    eval_shared_models = {
      'orig': eval_shared_model,
      'cf' eval_shared_model}
    cf_configs = {'cf': {'x_cf': 'x'}}
    extractors = tfma.default_extractors(eval_shared_models, eval_config,
        custom_predictions_extractor=CounterfactualPredictionsExtractor(
            eval_shared_models,cf_configs))
    tfma.run_model_analysis(eval_shared_models, eval_config,
        extractors=etractors)

  Args:
    eval_shared_models: The set of eval_shared_models for which to generate
      predictions. If a model is to be computed with original inputs and CF
      inputs, it should be provided twice, with distinct names. The name of the
      model to be computed with CF inputs should match the name provided in
      cf_configs as well as the ModelSpec.name in the provided EvalConfig.
    eval_config: The EvalConfig for this evaluation. If a model is to be
      computed with original inputs and CF inputs, it should correspond to two
      ModelSpecs with distinct names. The CF model name should match the name
      provided in cf_configs as well as the EvalSharedModel.model_name.
    cf_configs: A mapping from a model name to the CF config which should be
      used to preprocess its inputs. Any models in eval_shared_models not
      specified will have their predictions computed on the original input

  Returns:
    A tfma.Extractor which performs counterfactual inference along with non-
    counterfactual inference.

  Raises:
    ValueError if eval_shared_models is empty.
  """
  eval_shared_models, cf_configs = _validate_and_update_models_and_configs(
      eval_shared_models, cf_configs)
  cf_ptransforms = {}
  non_cf_models = []
  for model in eval_shared_models:
    cf_config = cf_configs.get(model.model_name, None)
    if cf_config:
      # filter EvalConfig so that it matches single EvalSavedModel
      cf_eval_config = _filter_model_specs(eval_config, [model])
      # TODO(b/258850519): Refactor default_extractors logic to expose new api
      # for constructing the default predictions extractor and call it here.
      predictions_ptransform = predictions_extractor.PredictionsExtractor(
          eval_shared_model=model,
          eval_config=cf_eval_config,
          output_keypath=(constants.PREDICTIONS_KEY, model.model_name),
      ).ptransform
      cf_ptransforms[model.model_name] = _ExtractCounterfactualPredictions(  # pylint: disable=no-value-for-parameter
          config=cf_config,
          predictions_ptransform=predictions_ptransform)
    else:
      non_cf_models.append(model)
  non_cf_eval_config = _filter_model_specs(eval_config, non_cf_models)
  if non_cf_models:
    output_keypath = (constants.PREDICTIONS_KEY,)
    if len(non_cf_models) == 1:
      output_keypath = output_keypath + (non_cf_models[0].model_name,)
    non_cf_ptransform = predictions_extractor.PredictionsExtractor(
        eval_shared_model=non_cf_models,
        eval_config=non_cf_eval_config,
        output_keypath=output_keypath,
    ).ptransform
  else:
    non_cf_ptransform = None
  return extractor.Extractor(
      stage_name=_COUNTERFACTUAL_PREDICTIONS_EXTRACTOR_NAME,
      ptransform=_ExtractPredictions(  # pylint: disable=no-value-for-parameter
          cf_ptransforms=cf_ptransforms, non_cf_ptransform=non_cf_ptransform
      ),
  )


def _validate_and_update_models_and_configs(
    eval_shared_models: types.MaybeMultipleEvalSharedModels,
    cf_configs: Mapping[str, CounterfactualConfig]):
  """Validates and updates the EvalSharedModels and CF configs.

  Args:
    eval_shared_models: The set of EvalSharedModels to validate and update.
    cf_configs: The CF configs to validate and update.

  Returns:
    A tuple of updated eval_shared_models and cf_configs.

  Raises:
    ValueError if:
      - eval_shared_models is empty
      - eval_shared_models are not all _SUPPORTED_MODEL_TYPES
      - cf_configs is empty
      - The model names in cf_configs do not match eval_shared_models
  """
  eval_shared_models = model_util.verify_and_update_eval_shared_models(
      eval_shared_models)
  if not eval_shared_models:
    raise ValueError(
        'The CounterfactualPredictionsExtractor requires at least one '
        f'EvalSharedModel, but got normalized models: {eval_shared_models}.')
  model_types = {m.model_type for m in eval_shared_models}
  if not model_types.issubset(_SUPPORTED_MODEL_TYPES):
    raise ValueError(
        f'Only {_SUPPORTED_MODEL_TYPES} model types are supported, but found '
        f'model types: {model_types}.')
  if not cf_configs:
    raise ValueError('The CounterfactualPredictionsExtractor requires at least '
                     'one cf_configs, but got 0.')

  if len(eval_shared_models) == 1:
    if len(cf_configs) == 1:
      # Follow the normalization logic in verify_and_update_eval_shared_models
      # and rekey cf_config in single model case under the empty key, "".
      cf_configs = {'': next(iter(cf_configs.values()))}
    else:
      raise ValueError(
          'The CounterfactualPredictionsExtractor was provided only one '
          'EvalSharedModel, in which case exactly one config is expected, but '
          f'got {len(cf_configs)}: {cf_configs}')

  configured_model_names = set(cf_configs)
  eval_shared_model_names = {model.model_name for model in eval_shared_models}
  unmatched_config_names = configured_model_names - eval_shared_model_names
  if unmatched_config_names:
    raise ValueError(
        'model_name_to_config contains model names which do not match the '
        'eval_shared_model model_names. Configured names: '
        f'{configured_model_names}, eval_shared_models names: '
        f'{eval_shared_model_names}. Unmatched configured model names: '
        f'{unmatched_config_names}.')
  return eval_shared_models, cf_configs


def _filter_model_specs(
    eval_config: config_pb2.EvalConfig,
    eval_shared_models: Iterable[types.EvalSharedModel]
) -> config_pb2.EvalConfig:
  """Filters EvalConfig.model_specs to match the set of EvalSharedModels."""
  result = config_pb2.EvalConfig()
  result.CopyFrom(eval_config)
  del result.model_specs[:]
  model_names = [model.model_name for model in eval_shared_models]
  result.model_specs.extend(
      [spec for spec in eval_config.model_specs if spec.name in model_names])
  return result


def _cf_preprocess(
    extracts: types.Extracts,
    config: CounterfactualConfig,
) -> types.Extracts:
  """Preprocesses extracts for counterfactual prediction.

  This method is to be called on each Extracts object prior to applying the
  wrapped prediction PTransform.

  Args:
    extracts: An Extracts instance which is suitable for feeding to a non-CF
      prediction extractor.
    config: The counterfactual config which determines how the inputs should be
      counterfactually modified.

  Returns:
    An Extracts instance which, when fed to a predictions PTransform, will
    produce counterfactual predictions.
  """
  result = extracts.copy()
  result[_TEMP_ORIG_INPUT_KEY] = result[constants.INPUT_KEY]
  cf_inputs = []
  for serialized_input in result[constants.INPUT_KEY]:
    cf_example = tf.train.Example.FromString(serialized_input)
    for dst_key, src_key in config.items():
      cf_example.features.feature[dst_key].CopyFrom(
          cf_example.features.feature[src_key])
    cf_inputs.append(cf_example.SerializeToString())
  cf_inputs = np.array(cf_inputs, dtype=object)
  result[constants.INPUT_KEY] = cf_inputs
  return result


def _cf_postprocess(extracts: types.Extracts) -> types.Extracts:
  """Postprocesses the result of applying a CF prediction ptransform.

  This method takes in an Extracts instance that has been prepocessed by
  _preprocess_cf and has had a prediction PTransform applied, and makes it look
  as if just a non-CF prediction PTransform had been applied.

  Args:
    extracts: An Extracts instance which has been preprocessed by _preprocess_cf
      and gone through a prediction PTransform.

  Returns:
    An Extracts instance which appears to have been produced by a standard
    predictions PTransform.
  """
  extracts = extracts.copy()
  extracts[constants.INPUT_KEY] = extracts[_TEMP_ORIG_INPUT_KEY]
  del extracts[_TEMP_ORIG_INPUT_KEY]
  return extracts


@beam.ptransform_fn
def _ExtractCounterfactualPredictions(  # pylint: disable=invalid-name
    extracts: beam.PCollection[types.Extracts],
    config: CounterfactualConfig,
    predictions_ptransform: beam.PTransform,
) -> beam.PCollection[types.Extracts]:
  """Computes counterfactual predictions for a single model."""
  return (
      extracts
      | 'PreprocessInputs' >> beam.Map(_cf_preprocess, config=config)
      | 'Predict' >> predictions_ptransform
      | 'PostProcessPredictions' >> beam.Map(_cf_postprocess)
  )


@beam.ptransform_fn
def _ExtractPredictions(  # pylint: disable=invalid-name
    extracts: beam.PCollection[types.Extracts],
    cf_ptransforms: Dict[str, beam.PTransform],
    non_cf_ptransform: Optional[beam.PTransform],
) -> beam.PCollection[types.Extracts]:
  """Applies both CF and non-CF prediction ptransforms and merges results.

  Args:
    extracts: Incoming TFMA extracts.
    cf_ptransforms: A mapping from model name to
      _ExtractCounterfactualPredictions ptransforms
    non_cf_ptransform: Optionally, a ptransform responsible for computing the
      non-counterfactual predictions.

  Returns:
    A PCollection of extracts containing merged predictions from both
    counterfactual and non-counterfactual models.
  """
  if non_cf_ptransform:
    extracts = extracts | 'PredictNonCF' >> non_cf_ptransform
  for model_name, cf_ptransform in cf_ptransforms.items():
    extracts = extracts | f'PredictCF[{model_name}]' >> cf_ptransform
  return extracts
