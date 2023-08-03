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
"""Batched materialized predictions extractor."""

from typing import Sequence

import apache_beam as beam
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import types
from tensorflow_model_analysis.extractors import extractor
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.utils import model_util
from tensorflow_model_analysis.utils import util

_MATERIALIZED_PREDICTIONS_EXTRACTOR_STAGE_NAME = 'ExtractMaterializedPredictions'


def MaterializedPredictionsExtractor(
    eval_config: config_pb2.EvalConfig,
    output_keypath: Sequence[str] = (constants.PREDICTIONS_KEY,),
) -> extractor.Extractor:
  """Creates an extractor for rekeying preexisting predictions.

  The extractor's PTransform uses the config's ModelSpec.prediction_key(s)
  to lookup the associated prediction values stored as features under the
  tfma.FEATURES_KEY in extracts. The resulting values are then added to the
  extracts under the key tfma.PREDICTIONS_KEY.

  Args:
    eval_config: Eval config.
    output_keypath: A list of keys to be used as the path to traverse and insert
      the outputs in the extract.

  Returns:
    Extractor for rekeying preexisting predictions.
  """
  # pylint: disable=no-value-for-parameter
  return extractor.Extractor(
      stage_name=_MATERIALIZED_PREDICTIONS_EXTRACTOR_STAGE_NAME,
      ptransform=_ExtractMaterializedPredictions(
          eval_config=eval_config, output_keypath=output_keypath
      ),
  )


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(types.Extracts)
def _ExtractMaterializedPredictions(  # pylint: disable=invalid-name
    extracts: beam.pvalue.PCollection,
    eval_config: config_pb2.EvalConfig,
    output_keypath: Sequence[str],
) -> beam.pvalue.PCollection:
  """A PTransform that populates the predictions key in the extracts.

  Args:
    extracts: PCollection of extracts containing model inputs keyed by
      tfma.FEATURES_KEY (if model inputs are named) or tfma.INPUTS_KEY (if model
      takes raw tf.Examples as input).
    eval_config: Eval config.
    output_keypath: A list of keys that indicates the location to which the
      predictions are inserted.

  Returns:
    PCollection of Extracts updated with the predictions.
  """

  def rekey_predictions(  # pylint: disable=invalid-name
      batched_extracts: types.Extracts) -> types.Extracts:
    """Extract predictions from extracts containing features."""
    predictions = model_util.get_feature_values_for_model_spec_field(
        list(eval_config.model_specs),
        'prediction_key',
        'prediction_keys',
        batched_extracts,
    )
    if predictions is not None:
      return util.copy_and_set_by_keys(
          batched_extracts, list(output_keypath), predictions
      )
    else:
      return batched_extracts

  return extracts | 'RekeyPredictions' >> beam.Map(rekey_predictions)
