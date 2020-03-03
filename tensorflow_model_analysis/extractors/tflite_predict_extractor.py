# Lint as: python3
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
"""Predict extractor for TFLite models."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import collections
import copy
import os
from typing import Dict, List, Optional, Union, Sequence, Text

import apache_beam as beam
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tensorflow_model_analysis import config
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import model_util
from tensorflow_model_analysis import types
from tensorflow_model_analysis.extractors import extractor

TFLITE_PREDICT_EXTRACTOR_STAGE_NAME = 'ExtractTFLitePredictions'

# TODO(dzats): Determine whether we should standardize on model filename for
# tflite models.
_TFLITE_FILE_NAME = 'tflite'


# TODO(b/149947174) Once this is addressed, subclass
# BatchReducibleDoFnWithModels instead so we obtain telemetry data and load
# a single model using a shared handle.
#
# TODO(b/149981535) Determine if we should merge with RunInference.
@beam.typehints.with_input_types(beam.typehints.List[types.Extracts])
@beam.typehints.with_output_types(types.Extracts)
class _TFLitePredictionDoFn(beam.DoFn):
  """A DoFn that loads tflite models and predicts."""

  def __init__(self, eval_config: config.EvalConfig,
               eval_shared_models: Dict[Text, types.EvalSharedModel]) -> None:
    self._eval_config = eval_config
    self._model_paths = {
        k: os.path.join(v.model_path, _TFLITE_FILE_NAME)
        for k, v in eval_shared_models.items()
    }

  def setup(self):
    self._interpreters = {}
    for model_name, model_path in self._model_paths.items():
      with tf.io.gfile.GFile(model_path, 'rb') as model_file:
        self._interpreters[model_name] = tf.lite.Interpreter(
            model_content=model_file.read())

  def process(self, elements: List[types.Extracts]) -> Sequence[types.Extracts]:
    """Invokes the tflite model on the provided inputs and stores the result."""
    # TODO(dzats): See if we can switch to a shallow copy.
    result = copy.deepcopy(elements)

    batched_features = collections.defaultdict(list)
    for e in elements:
      features = e[constants.FEATURES_KEY]
      for key, value in features.items():
        batched_features[key].append(value)

    for spec in self._eval_config.model_specs:
      model_name = spec.name if len(self._eval_config.model_specs) > 1 else ''
      if model_name not in self._model_paths:
        raise ValueError('model path for "{}" not found: eval_config={}'.format(
            spec.name, self._eval_config))

      interpreter = self._interpreters[model_name]

      input_details = interpreter.get_input_details()
      output_details = interpreter.get_output_details()

      input_features = {}
      for i in input_details:
        input_name = i['name']
        if input_name not in batched_features:
          raise ValueError(
              'feature "{}" not found in input data'.format(input_name))
        if len(np.shape(batched_features[input_name][0])) < len(i['shape']):
          input_features[input_name] = [
              np.resize(b, i['shape']) for b in batched_features[input_name]
          ]
        else:
          input_features[input_name] = batched_features[input_name]
        input_features[input_name] = tf.concat(
            input_features[input_name], axis=0)
        if np.shape(input_features[input_name]) != tuple(i['shape']):
          interpreter.resize_tensor_input(i['index'],
                                          np.shape(input_features[input_name]))
      interpreter.allocate_tensors()

      for i in input_details:
        interpreter.set_tensor(i['index'], input_features[i['name']])
      interpreter.invoke()

      outputs = {
          o['name']: interpreter.get_tensor(o['index']) for o in output_details
      }
      for i in range(len(result)):
        output = {k: v[i] for k, v in outputs.items()}

        if len(output) == 1:
          output = list(output.values())[0]

        if len(self._eval_config.model_specs) == 1:
          result[i][constants.PREDICTIONS_KEY] = output
        else:
          result[i].setdefault(constants.PREDICTIONS_KEY, {})[spec.name] = (
              output)
    return result


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(types.Extracts)
def _ExtractTFLitePredictions(  # pylint: disable=invalid-name
    extracts: beam.pvalue.PCollection, eval_config: config.EvalConfig,
    eval_shared_models: Dict[Text, types.EvalSharedModel],
    desired_batch_size: Optional[int]) -> beam.pvalue.PCollection:
  """A PTransform that adds predictions and possibly other tensors to extracts.

  Args:
    extracts: PCollection of extracts containing model inputs keyed by
      tfma.FEATURES_KEY.
    eval_config: Eval config.
    eval_shared_models: Shared model parameters keyed by model name.
    desired_batch_size: Optional batch size.

  Returns:
    PCollection of Extracts updated with the predictions.
  """
  batch_args = {}
  # TODO(b/143484017): Consider removing this option if autotuning is better
  # able to handle batch size selection.
  if desired_batch_size is not None:
    batch_args = dict(
        min_batch_size=desired_batch_size, max_batch_size=desired_batch_size)
  else:
    # TODO(b/150635972): Remove the following and allow dynamic batch sizing
    # once the bug is addressed.
    batch_args = dict(min_batch_size=1, max_batch_size=1)

  return (
      extracts
      | 'Batch' >> beam.BatchElements(**batch_args)
      | 'Predict' >> beam.ParDo(
          _TFLitePredictionDoFn(
              eval_config=eval_config, eval_shared_models=eval_shared_models)))


def TFLitePredictExtractor(
    eval_config: config.EvalConfig,
    eval_shared_model: Union[types.EvalSharedModel,
                             Dict[Text, types.EvalSharedModel]],
    desired_batch_size: Optional[int] = None) -> extractor.Extractor:
  """Creates an extractor for performing predictions on tflite models.

  The extractor's PTransform loads and interprets the tflite flatbuffer against
  every extract yielding a copy of the incoming extracts with an additional
  extract added for the predictions keyed by tfma.PREDICTIONS_KEY. The model
  inputs are searched for under tfma.FEATURES_KEY. If multiple
  models are used the predictions will be stored in a dict keyed by model name.

  Args:
    eval_config: Eval config.
    eval_shared_model: Shared model (single-model evaluation) or dict of shared
      models keyed by model name (multi-model evaluation).
    desired_batch_size: Optional batch size.

  Returns:
    Extractor for extracting predictions.
  """
  eval_shared_models = model_util.verify_and_update_eval_shared_models(
      eval_shared_model)

  # pylint: disable=no-value-for-parameter
  return extractor.Extractor(
      stage_name=TFLITE_PREDICT_EXTRACTOR_STAGE_NAME,
      ptransform=_ExtractTFLitePredictions(
          eval_config=eval_config,
          eval_shared_models={m.model_name: m for m in eval_shared_models},
          desired_batch_size=desired_batch_size))
