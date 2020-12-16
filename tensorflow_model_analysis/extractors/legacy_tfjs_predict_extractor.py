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
"""Predict extractor for TFJS models."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import collections
import copy
import json
import os
import subprocess
import tempfile
from typing import Dict, List, Optional, Union, Sequence, Text

import uuid

import apache_beam as beam
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import config
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import model_util
from tensorflow_model_analysis import types
from tensorflow_model_analysis.extractors import extractor
from tensorflow_model_analysis.extractors.tfjs_predict_extractor_util import get_tfjs_binary

_TFJS_PREDICT_EXTRACTOR_STAGE_NAME = 'ExtractTFJSPredictions'

_MODELS_SUBDIR = 'Models'
_EXAMPLES_SUBDIR = 'Input_Examples'
_OUTPUTS_SUBDIR = 'Inference_Results'

_MODEL_JSON = 'model.json'
_DATA_JSON = 'data.json'
_DTYPE_JSON = 'dtype.json'
_SHAPE_JSON = 'shape.json'
_TF_INPUT_NAME_JSON = 'tf_input_name.json'


# TODO(b/149981535) Determine if we should merge with RunInference.
@beam.typehints.with_input_types(beam.typehints.List[types.Extracts])
@beam.typehints.with_output_types(types.Extracts)
class _TFJSPredictionDoFn(model_util.BatchReducibleDoFnWithModels):
  """A DoFn that loads tfjs models and predicts."""

  def __init__(self, eval_config: config.EvalConfig,
               eval_shared_models: Dict[Text, types.EvalSharedModel]) -> None:
    super(_TFJSPredictionDoFn, self).__init__(
        {k: v.model_loader for k, v in eval_shared_models.items()})
    self._eval_config = eval_config
    self._src_model_paths = {
        k: v.model_path for k, v in eval_shared_models.items()
    }

  def setup(self):
    super(_TFJSPredictionDoFn, self).setup()
    self._binary_path = get_tfjs_binary()

    base_path = tempfile.mkdtemp()
    base_model_path = os.path.join(base_path, _MODELS_SUBDIR)

    self._model_properties = {}
    for model_name, model_path in self._src_model_paths.items():
      with tf.io.gfile.GFile(os.path.join(model_path, _MODEL_JSON)) as f:
        model_json = json.load(f)
        if ('userDefinedMetadata' in model_json and
            'signature' in model_json['userDefinedMetadata']):
          model_signature = model_json['userDefinedMetadata']['signature']
        else:
          model_signature = model_json['signature']
        model_inputs = {}
        for k, v in model_signature['inputs'].items():
          model_inputs[k] = [int(i['size']) for i in v['tensorShape']['dim']]

        model_outputs = {}
        for k, v in model_signature['outputs'].items():
          model_outputs[k] = [int(i['size']) for i in v['tensorShape']['dim']]

      cur_model_path = os.path.join(base_model_path, model_name)
      self._model_properties[model_name] = {
          'inputs': model_inputs,
          'outputs': model_outputs,
          'path': cur_model_path
      }

      # We copy models to local tmp storage so that the tfjs binary can
      # access them.
      tf.io.gfile.makedirs(cur_model_path)
      for directory, _, files in tf.io.gfile.walk(model_path):
        cur_path = os.path.join(cur_model_path,
                                os.path.relpath(directory, model_path))
        tf.io.gfile.makedirs(cur_path)
        for f in files:
          src_path = os.path.join(directory, f)
          tf.io.gfile.copy(src_path, os.path.join(cur_path, f))

  def _batch_reducible_process(
      self, elements: List[types.Extracts]) -> Sequence[types.Extracts]:
    """Invokes the tfjs model on the provided inputs and stores the result."""
    # This will be same size as elements, but we rebuild results dynamically
    # to avoid a deepcopy.
    result = []

    batched_features = collections.defaultdict(list)
    for e in elements:
      features = e[constants.FEATURES_KEY]
      for key, value in features.items():
        if value.dtype == np.int64:
          value = value.astype(np.int32)
        batched_features[key].append(value)

    for spec in self._eval_config.model_specs:
      model_name = spec.name if len(self._eval_config.model_specs) > 1 else ''
      if model_name not in self._loaded_models:
        raise ValueError('model for "{}" not found: eval_config={}'.format(
            spec.name, self._eval_config))

      model_features = {}
      for k in self._model_properties[model_name]['inputs']:
        k_name = k.split(':')[0]
        if k_name not in batched_features:
          raise ValueError('model requires feature "{}" not available in '
                           'input.'.format(k_name))
        dim = self._model_properties[model_name]['inputs'][k]
        elems = []
        for i in batched_features[k_name]:
          if np.ndim(i) > len(dim):
            raise ValueError('ranks for input "{}" are not compatible '
                             'with the model.'.format(k_name))
          # TODO(dzats): See if we can support case where multiple dimensions
          # are not defined.
          elems.append(np.reshape(i, dim))
        model_features[k] = elems

      model_features = {k: np.concatenate(v) for k, v in model_features.items()}

      batched_entries = collections.defaultdict(list)
      for feature, value in model_features.items():
        batched_entries[_DATA_JSON].append(value.tolist())
        batched_entries[_DTYPE_JSON].append(str(value.dtype))
        batched_entries[_SHAPE_JSON].append(value.shape)
        batched_entries[_TF_INPUT_NAME_JSON].append(feature)

      cur_subdir = str(uuid.uuid4())
      cur_input_path = os.path.join(self._model_properties[model_name]['path'],
                                    _EXAMPLES_SUBDIR, cur_subdir)
      tf.io.gfile.makedirs(cur_input_path)
      for entry, value in batched_entries.items():
        with tf.io.gfile.GFile(os.path.join(cur_input_path, entry), 'w') as f:
          f.write(json.dumps(value))

      cur_output_path = os.path.join(self._model_properties[model_name]['path'],
                                     _OUTPUTS_SUBDIR, cur_subdir)
      tf.io.gfile.makedirs(cur_output_path)
      inference_command = [
          self._binary_path, '--model_path=' +
          os.path.join(self._model_properties[model_name]['path'], _MODEL_JSON),
          '--inputs_dir=' + cur_input_path, '--outputs_dir=' + cur_output_path
      ]

      popen = subprocess.Popen(
          inference_command,
          stdin=subprocess.PIPE,
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE)
      stdout, stderr = popen.communicate()
      if popen.returncode != 0:
        raise ValueError(
            'Inference failed with status {}\nstdout:\n{}\nstderr:\n{}'.format(
                popen.returncode, stdout, stderr))

      try:
        with tf.io.gfile.GFile(os.path.join(cur_output_path, _DATA_JSON)) as f:
          data = json.load(f)
        with tf.io.gfile.GFile(os.path.join(cur_output_path, _DTYPE_JSON)) as f:
          dtype = json.load(f)
        with tf.io.gfile.GFile(os.path.join(cur_output_path, _SHAPE_JSON)) as f:
          shape = json.load(f)
      except FileNotFoundError as e:
        raise FileNotFoundError(
            'Unable to find files containing inference result. This likely '
            'means that inference did not succeed. Error {}'.format(e))

      name = [
          n.split(':')[0]
          for n in self._model_properties[model_name]['outputs'].keys()
      ]

      tf.io.gfile.rmtree(cur_input_path)
      tf.io.gfile.rmtree(cur_output_path)

      outputs = {}
      for n, s, t, d in zip(name, shape, dtype, data):
        d_val = [d[str(i)] for i in range(len(d))]
        outputs[n] = np.reshape(np.array(d_val, t), s)

      for v in outputs.values():
        if len(v) != len(elements):
          raise ValueError('Did not get the expected number of results.')

      for i in range(len(elements)):
        output = {k: v[i] for k, v in outputs.items()}

        if len(output) == 1:
          output = list(output.values())[0]

        if i >= len(result):
          result.append(copy.copy(elements[i]))

        if len(self._eval_config.model_specs) == 1:
          result[i][constants.PREDICTIONS_KEY] = output
        else:
          result[i].setdefault(constants.PREDICTIONS_KEY, {})[spec.name] = (
              output)
    return result


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(types.Extracts)
def _ExtractTFJSPredictions(  # pylint: disable=invalid-name
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
  return (
      extracts
      | 'Batch' >> beam.BatchElements(**batch_args)
      | 'Predict' >> beam.ParDo(
          _TFJSPredictionDoFn(
              eval_config=eval_config, eval_shared_models=eval_shared_models)))


def TFJSPredictExtractor(  # pylint: disable=invalid-name
    eval_config: config.EvalConfig,
    eval_shared_model: Union[types.EvalSharedModel,
                             Dict[Text, types.EvalSharedModel]],
    desired_batch_size: Optional[int] = None) -> extractor.Extractor:
  """Creates an extractor for performing predictions on tfjs models.

  The extractor's PTransform loads and interprets the tfjs model against
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
      stage_name=_TFJS_PREDICT_EXTRACTOR_STAGE_NAME,
      ptransform=_ExtractTFJSPredictions(
          eval_config=eval_config,
          eval_shared_models={m.model_name: m for m in eval_shared_models},
          desired_batch_size=desired_batch_size))
