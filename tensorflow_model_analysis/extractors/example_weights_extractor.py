# Lint as: python3
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
"""Example weights extractor."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import copy

import apache_beam as beam
from tensorflow_model_analysis import config
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import model_util
from tensorflow_model_analysis import types
from tensorflow_model_analysis.extractors import extractor

_EXAMPLE_WEIGHTS_EXTRACTOR_STAGE_NAME = 'ExtractExampleWeights'


def ExampleWeightsExtractor(
    eval_config: config.EvalConfig) -> extractor.Extractor:
  """Creates an extractor for extracting example weights.

  The extractor's PTransform uses the config's ModelSpec.example_weight_key(s)
  to lookup the associated example weight values stored as features under the
  tfma.FEATURES_KEY (and optionally tfma.TRANSFORMED_FEATURES_KEY) in extracts.
  The resulting values are then added to the extracts under the key
  tfma.EXAMPLE_WEIGHTS_KEY.

  Args:
    eval_config: Eval config.

  Returns:
    Extractor for extracting example weights.
  """
  # pylint: disable=no-value-for-parameter
  return extractor.Extractor(
      stage_name=_EXAMPLE_WEIGHTS_EXTRACTOR_STAGE_NAME,
      ptransform=_ExtractExampleWeights(eval_config=eval_config))


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(types.Extracts)
def _ExtractExampleWeights(
    extracts: beam.pvalue.PCollection,
    eval_config: config.EvalConfig) -> beam.pvalue.PCollection:
  """Extracts example weights from features extracts.

  Args:
    extracts: PCollection containing features under tfma.FEATURES_KEY.
    eval_config: Eval config.

  Returns:
    PCollection of extracts with additional example weights added under the key
    tfma.EXAMPLE_WEIGHTS_KEY.
  """

  def extract_example_weights(  # pylint: disable=invalid-name
      batched_extracts: types.Extracts) -> types.Extracts:
    """Extract example weights from extracts containing features."""
    result = copy.copy(batched_extracts)
    example_weights = model_util.get_feature_values_for_model_spec_field(
        list(eval_config.model_specs), 'example_weight_key',
        'example_weight_keys', result)
    if example_weights is not None:
      result[constants.EXAMPLE_WEIGHTS_KEY] = example_weights
    return result

  return extracts | 'ExtractExampleWeights' >> beam.Map(extract_example_weights)
