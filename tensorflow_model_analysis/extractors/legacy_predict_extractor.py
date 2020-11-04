# Lint as: python3
# Copyright 2018 Google LLC
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
"""Public API for performing evaluations using the EvalSavedModel."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import copy

from typing import Any, Dict, List, Optional, Sequence, Text

import apache_beam as beam
import numpy as np

from tensorflow_model_analysis import config
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import model_util
from tensorflow_model_analysis import types
from tensorflow_model_analysis.eval_saved_model import constants as eval_saved_model_constants
from tensorflow_model_analysis.extractors import extractor
from tensorflow_model_analysis.extractors import legacy_feature_extractor

_PREDICT_EXTRACTOR_STAGE_NAME = 'Predict'

_FEATURES_PREDICTIONS_LABELS_KEY_MAP = {
    eval_saved_model_constants.FEATURES_NAME: constants.FEATURES_KEY,
    eval_saved_model_constants.PREDICTIONS_NAME: constants.PREDICTIONS_KEY,
    eval_saved_model_constants.LABELS_NAME: constants.LABELS_KEY,
}


def PredictExtractor(
    eval_shared_model: types.MaybeMultipleEvalSharedModels,
    desired_batch_size: Optional[int] = None,
    materialize: Optional[bool] = True,
    eval_config: Optional[config.EvalConfig] = None) -> extractor.Extractor:
  """Creates an Extractor for TFMAPredict.

  The extractor's PTransform loads and runs the eval_saved_model against every
  example yielding a copy of the Extracts input with an additional extract
  of type FeaturesPredictionsLabels keyed by
  tfma.FEATURES_PREDICTIONS_LABELS_KEY unless eval_config is not None in which
  case the features, predictions, and labels will be stored separately under
  tfma.FEATURES_KEY, tfma.PREDICTIONS_KEY, and tfma.LABELS_KEY respectively.

  Args:
    eval_shared_model: Shared model (single-model evaluation) or list of shared
      models (multi-model evaluation).
    desired_batch_size: Optional batch size for batching in Aggregate.
    materialize: True to call the FeatureExtractor to add MaterializedColumn
      entries for the features, predictions, and labels.
    eval_config: Eval config.

  Returns:
    Extractor for extracting features, predictions, labels, and other tensors
    during predict.
  """
  eval_shared_models = model_util.verify_and_update_eval_shared_models(
      eval_shared_model)

  # pylint: disable=no-value-for-parameter
  return extractor.Extractor(
      stage_name=_PREDICT_EXTRACTOR_STAGE_NAME,
      ptransform=_TFMAPredict(
          eval_shared_models={m.model_name: m for m in eval_shared_models},
          desired_batch_size=desired_batch_size,
          materialize=materialize,
          eval_config=eval_config))


@beam.typehints.with_input_types(beam.typehints.List[types.Extracts])
@beam.typehints.with_output_types(types.Extracts)
class _TFMAPredictionDoFn(model_util.BatchReducibleDoFnWithModels):
  """A DoFn that loads the model and predicts."""

  def __init__(self, eval_shared_models: Dict[Text, types.EvalSharedModel],
               eval_config):
    super(_TFMAPredictionDoFn, self).__init__(
        {k: v.model_loader for k, v in eval_shared_models.items()})
    self._eval_config = eval_config

  def _get_example_weights(self, model_name: Text, features: Dict[Text,
                                                                  Any]) -> Any:
    spec = model_util.get_model_spec(self._eval_config, model_name)
    if not spec:
      raise ValueError(
          'Missing model_spec for model_name "{}"'.format(model_name))
    if spec.example_weight_key:
      if spec.example_weight_key not in features:
        raise ValueError(
            'Missing feature for example_weight_key "{}": features={}'.format(
                spec.example_weight_key, features))
      return features[spec.example_weight_key]
    elif spec.example_weight_keys:
      example_weights = {}
      for k, v in spec.example_weight_keys.items():
        if v not in features:
          raise ValueError(
              'Missing feature for example_weight_key "{}": features={}'.format(
                  k, features))
        example_weights[k] = features[v]
      return example_weights
    else:
      return np.array([1.0])

  def _batch_reducible_process(
      self, elements: List[types.Extracts]) -> Sequence[types.Extracts]:
    serialized_examples = [x[constants.INPUT_KEY] for x in elements]

    # Compute features, predictions, and labels for each serialized_example
    result = []
    for model_name, loaded_model in self._loaded_models.items():
      for i, fetched in enumerate(
          loaded_model.predict_list(serialized_examples)):
        if i >= len(result):
          element_copy = copy.copy(elements[fetched.input_ref])
          for key in fetched.values:
            if key in _FEATURES_PREDICTIONS_LABELS_KEY_MAP:
              if self._eval_config:
                element_copy[_FEATURES_PREDICTIONS_LABELS_KEY_MAP[key]] = (
                    fetched.values[key])
              continue
            element_copy[key] = fetched.values[key]
          if self._eval_config:
            element_copy[constants.EXAMPLE_WEIGHTS_KEY] = (
                self._get_example_weights(model_name,
                                          element_copy[constants.FEATURES_KEY]))
          if len(self._loaded_models) == 1:
            if not self._eval_config:
              element_copy[constants.FEATURES_PREDICTIONS_LABELS_KEY] = (
                  loaded_model.as_features_predictions_labels([fetched])[0])
          else:
            if not self._eval_config:
              raise ValueError(
                  'PredictExtractor can only be used with multi-output models '
                  'if eval_config is passed.')
            # If only one model, the predictions are stored without using a dict
            element_copy[constants.PREDICTIONS_KEY] = {
                model_name: element_copy[constants.PREDICTIONS_KEY]
            }
          result.append(element_copy)
        else:
          element_copy = result[i]
          # Assume values except for predictions are same for all models.
          element_copy[constants.PREDICTIONS_KEY][model_name] = fetched.values[
              eval_saved_model_constants.PREDICTIONS_NAME]
    return result


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(types.Extracts)
def _TFMAPredict(  # pylint: disable=invalid-name
    extracts: beam.pvalue.PCollection,
    eval_shared_models: Dict[Text, types.EvalSharedModel],
    desired_batch_size: Optional[int] = None,
    materialize: Optional[bool] = True,
    eval_config: Optional[config.EvalConfig] = None) -> beam.pvalue.PCollection:
  """A PTransform that adds predictions to Extracts.

  Args:
    extracts: PCollection of Extracts containing a serialized example to be fed
      to the model.
    eval_shared_models: Shared model parameters keyed by model name.
    desired_batch_size: Optional. Desired batch size for prediction.
    materialize: True to call the FeatureExtractor to add MaterializedColumn
      entries for the features, predictions, and labels.
    eval_config: Eval config.

  Returns:
    PCollection of Extracts, where the extracts contains the features,
    predictions, labels retrieved.
  """
  batch_args = {}

  # TODO(b/143484017): Consider removing this option if autotuning is better
  # able to handle batch size selection.
  if desired_batch_size:
    batch_args = dict(
        min_batch_size=desired_batch_size, max_batch_size=desired_batch_size)

  # We don't actually need to add the add_metrics_callbacks to do Predict,
  # but because if we want to share the model between Predict and subsequent
  # stages (i.e. we use same shared handle for this and subsequent stages),
  # then if we don't add the metrics callbacks here, they won't be present
  # in the model in the later stages if we reuse the model from this stage.
  extracts = (
      extracts
      | 'Batch' >> beam.BatchElements(**batch_args)
      | 'Predict' >> beam.ParDo(
          _TFMAPredictionDoFn(
              eval_shared_models=eval_shared_models, eval_config=eval_config)))

  if materialize and not eval_config:
    additional_fetches = []
    for m in eval_shared_models.values():
      if m.additional_fetches:
        additional_fetches.extend(m.additional_fetches)
    return extracts | 'ExtractFeatures' >> legacy_feature_extractor._ExtractFeatures(  # pylint: disable=protected-access
        additional_extracts=additional_fetches or None)

  return extracts
