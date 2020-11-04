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
"""Input extractor for extracting features, labels, weights."""
from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import copy

from typing import Any, Dict, List, Optional, Text, Tuple, Union

import apache_beam as beam
import numpy as np
from tensorflow_model_analysis import config
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.extractors import extractor
from tfx_bsl.coders import example_coder

_INPUT_EXTRACTOR_STAGE_NAME = 'ExtractInputs'


def InputExtractor(eval_config: config.EvalConfig) -> extractor.Extractor:
  """Creates an extractor for extracting features, labels, and example weights.

  The extractor's PTransform parses tf.train.Example protos stored under the
  tfma.INPUT_KEY in the incoming extracts and adds the resulting features,
  labels, and example weights to the extracts under the keys tfma.FEATURES_KEY,
  tfma.LABELS_KEY, and tfma.EXAMPLE_WEIGHTS_KEY. If the eval_config contains a
  prediction_key and a corresponding key is found in the parse example, then
  predictions will also be extracted and stored under the tfma.PREDICTIONS_KEY.
  Any extracts that already exist will be merged with the values parsed by this
  extractor with this extractor's values taking precedence when duplicate keys
  are detected.

  Note that the use of a prediction_key in an eval_config serves two use cases:
    (1) as a key into the dict of predictions output by predict extractor
    (2) as the key for a pre-computed prediction stored as a feature.
  The InputExtractor can be used to handle case (2). These cases are meant to be
  exclusive (i.e. if approach (2) is used then a predict extractor would not be
  configured and if (1) is used then a key matching the predictons would not be
  stored in the features). However, if a feature key happens to match the same
  name as the prediction output key then both paths may be executed. In this
  case, the value stored here will be replaced by the predict extractor (though
  it will still be popped from the features).

  Args:
    eval_config: Eval config.

  Returns:
    Extractor for extracting features, labels, and example weights inputs.
  """
  # pylint: disable=no-value-for-parameter
  return extractor.Extractor(
      stage_name=_INPUT_EXTRACTOR_STAGE_NAME,
      ptransform=_ExtractInputs(eval_config=eval_config))


def _keys_and_values(  # pylint: disable=invalid-name
    key_maybe_dict: Union[Text, Dict[Text, Text]],
    features: Dict[Text,
                   np.ndarray]) -> Tuple[Optional[List[Text]], Optional[Union[
                       np.ndarray, Dict[Text, np.ndarray]]]]:
  """Returns keys and values in dict given key (or dict of keys)."""
  if isinstance(key_maybe_dict, dict):
    values = {}
    keys = set()
    for output_name, key in key_maybe_dict.items():
      if key in features:
        values[output_name] = features[key]
        if key not in keys:
          keys.add(key)
    return (list(keys), values)
  elif key_maybe_dict in features:
    return ([key_maybe_dict], features[key_maybe_dict])
  else:
    return ([], None)


def _ParseExample(extracts: types.Extracts, eval_config: config.EvalConfig):
  """Parses serialized tf.train.Example to create additional extracts.

  Args:
    extracts: PCollection containing serialized examples under tfma.INPUT_KEY.
    eval_config: Eval config.

  Returns:
    Extracts with additional keys added for features, labels, and example
    weights.
  """

  features = example_coder.ExampleToNumpyDict(extracts[constants.INPUT_KEY])
  extracts = copy.copy(extracts)

  def add_to_extracts(  # pylint: disable=invalid-name
      key: Text, model_name: Text, feature_values: Any):
    """Adds features_values to extracts and feature_keys to keys_to_pop."""
    # Only key by model name if multiple models.
    if len(eval_config.model_specs) > 1:
      if key not in extracts:
        extracts[key] = {}
      extracts[key][model_name] = feature_values
    else:
      extracts[key] = feature_values

  for spec in eval_config.model_specs:
    if spec.label_key or spec.label_keys:
      _, values = _keys_and_values(
          spec.label_key or dict(spec.label_keys), features)
      add_to_extracts(constants.LABELS_KEY, spec.name, values)
    if spec.example_weight_key or spec.example_weight_keys:
      _, values = _keys_and_values(
          spec.example_weight_key or dict(spec.example_weight_keys), features)
      add_to_extracts(constants.EXAMPLE_WEIGHTS_KEY, spec.name, values)
    if spec.prediction_key or spec.prediction_keys:
      _, values = _keys_and_values(
          spec.prediction_key or dict(spec.prediction_keys), features)
      add_to_extracts(constants.PREDICTIONS_KEY, spec.name, values)
  extracts[constants.FEATURES_KEY] = features
  return extracts


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(types.Extracts)
def _ExtractInputs(extracts: beam.pvalue.PCollection,
                   eval_config: config.EvalConfig) -> beam.pvalue.PCollection:
  """Extracts inputs from serialized tf.train.Example protos.

  Args:
    extracts: PCollection containing serialized examples under tfma.INPUT_KEY.
    eval_config: Eval config.

  Returns:
    PCollection of extracts with additional features, labels, and weights added
    under the keys tfma.FEATURES_KEY, tfma.LABELS_KEY, and
    tfma.EXAMPLE_WEIGHTS_KEY.
  """
  return extracts | 'ParseExample' >> beam.Map(_ParseExample, eval_config)
