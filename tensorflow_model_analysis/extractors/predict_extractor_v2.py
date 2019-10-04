# Copyright 2019 Google LLC
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
"""Predict extractor."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import copy

import apache_beam as beam
import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import model_util
from tensorflow_model_analysis import types
from tensorflow_model_analysis.extractors import extractor
from typing import Generator, List, Optional

PREDICT_EXTRACTOR_STAGE_NAME = 'ExtractPredictions'

PREDICT_SIGNATURE_DEF_KEY = 'predict'


def PredictExtractor(
    eval_shared_model: types.EvalSharedModel,
    desired_batch_size: Optional[int] = None) -> extractor.Extractor:
  """Creates an extractor for performing predictions.

  The extractor's PTransform loads and runs the serving saved_model against
  every extact yielding a copy of the incoming extracts with an additional
  extract added for the predictions keyed by tfma.PREDICTIONS_KEY. The model
  inputs are searched for under tfma.FEATURES_KEY (keras only) or tfma.INPUT_KEY
  (if tfma.FEATURES_KEY is not set or the model is non-keras).

  Args:
    eval_shared_model: Shared model parameters.
    desired_batch_size: Optional batch size for batching.

  Returns:
    Extractor for extracting predictions.
  """
  # pylint: disable=no-value-for-parameter
  return extractor.Extractor(
      stage_name=PREDICT_EXTRACTOR_STAGE_NAME,
      ptransform=_ExtractPredictions(
          eval_shared_model=eval_shared_model,
          desired_batch_size=desired_batch_size))


@beam.typehints.with_input_types(beam.typehints.List[types.Extracts])
@beam.typehints.with_output_types(types.Extracts)
class _PredictionDoFn(model_util.DoFnWithModels):
  """A DoFn that loads the models and predicts."""

  def __init__(self, eval_shared_model: types.EvalSharedModel) -> None:
    super(_PredictionDoFn, self).__init__({'': eval_shared_model.model_loader})
    self._predict_batch_size = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE, 'predict_batch_size')
    self._predict_num_instances = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, 'predict_num_instances')

  def process(
      self, batch_of_extracts: List[types.Extracts]
  ) -> Generator[types.Extracts, None, None]:
    batch_size = len(batch_of_extracts)
    self._predict_batch_size.update(batch_size)
    self._predict_num_instances.inc(batch_size)

    # TODO(b/141016373): Add support for multiple models.
    loaded_model = self._loaded_models['']
    signatures = None
    if loaded_model.keras_model:
      signatures = loaded_model.keras_model.signatures
    elif loaded_model.saved_model:
      signatures = loaded_model.saved_model.signatures
    if not signatures:
      raise ValueError(
          'PredictExtractor V2 requires a keras model or a serving model. '
          'If using EvalSavedModel then you must use PredictExtractor V1.')

    # First try 'predict' then try 'serving_default'. The estimator output for
    # the 'serving_default' key does not include all the heads in a multi-head
    # model. However, keras only uses the 'serving_default' for its outputs.
    # Note that the 'predict' key only exists for estimators for multi-head
    # models, for single-head models only 'serving_default' is used.
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
    if self._loaded_models[''].keras_model is not None:
      input_names = self._loaded_models[''].keras_model.input_names
    # First arg of structured_input_signature tuple is shape, second is dtype
    # (we currently only support named params passed as a dict)
    elif (signature.structured_input_signature and
          len(signature.structured_input_signature) == 2 and
          isinstance(signature.structured_input_signature[1], dict)):
      input_names = [name for name in signature.structured_input_signature[1]]
    if input_names is not None:
      # TODO(b/138474171): Make this code more efficient.
      inputs = {}
      found = []
      for name in input_names:
        inputs[name] = []
        for extract in batch_of_extracts:
          # If features key exist, use that for features, else use input_key
          if constants.FEATURES_KEY in extract:
            input_features = extract[constants.FEATURES_KEY]
          else:
            input_features = extract[constants.INPUT_KEY]
          if isinstance(input_features, dict):
            if name in input_features:
              found.append(name)
              inputs[name].append(input_features[name])
          else:
            if len(inputs) > 1:
              raise ValueError('PredictExtractor passed single input but keras '
                               'model expects multiple: keras.input_names = %s,'
                               ' extracts = %s' % (input_names, extract))
            found.append(name)
            inputs[name].append(input_features)
      if len(found) != len(input_names):
        tf.compat.v1.logging.warning(
            'PredictExtractor inputs do not match those expected by the '
            'model: input_names = %s, found in extracts = %s' %
            (input_names, found))
    else:
      # Assume serialized examples
      inputs = [extract[constants.INPUT_KEY] for extract in batch_of_extracts]

    if isinstance(inputs, dict):
      predictions = signature(**{k: tf.constant(v) for k, v in inputs.items()})
    else:
      predictions = signature(tf.constant(inputs, dtype=tf.string))

    for i in range(batch_size):
      prediction = {k: v[i].numpy() for k, v in predictions.items()}
      # Keras and regression serving models return a dict of predictions even
      # for single-outputs. Convert these to a single tensor for compatibility
      # with the labels (and model.predict API).
      if len(prediction) == 1:
        prediction = list(prediction.values())[0]
      extracts = copy.copy(batch_of_extracts[i])
      extracts[constants.PREDICTIONS_KEY] = prediction
      yield extracts


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(types.Extracts)
def _ExtractPredictions(  # pylint: disable=invalid-name
    extracts: beam.pvalue.PCollection,
    eval_shared_model: types.EvalSharedModel,
    desired_batch_size: Optional[int] = None) -> beam.pvalue.PCollection:
  """A PTransform that adds predictions and possibly other tensors to extracts.

  Args:
    extracts: PCollection of extracts containing model inputs keyed by
      tfma.FEATURES_KEY (if model inputs are named) or tfma.INPUTS_KEY (if model
      takes raw tf.Examples as input).
    eval_shared_model: Shared model parameters.
    desired_batch_size: Optional batch size for prediction.

  Returns:
    PCollection of Extracts updated with the predictions.
  """
  batch_args = {}
  if desired_batch_size:
    batch_args = dict(
        min_batch_size=desired_batch_size, max_batch_size=desired_batch_size)

  extracts = (
      extracts
      | 'Batch' >> beam.BatchElements(**batch_args)
      | 'Predict' >> beam.ParDo(
          _PredictionDoFn(eval_shared_model=eval_shared_model)))

  return extracts
