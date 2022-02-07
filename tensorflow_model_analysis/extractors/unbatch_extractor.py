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
"""Unbatch extractor."""

from typing import Sequence

import apache_beam as beam
import pandas as pd
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.extractors import extractor
from tensorflow_model_analysis.utils import util

UNBATCH_EXTRACTOR_STAGE_NAME = 'ExtractUnbatchedInputs'


def UnbatchExtractor() -> extractor.Extractor:
  """Creates an extractor for unbatching batched extracts.

  This extractor removes Arrow RecordBatch from the batched extract and outputs
  per-example extracts with the remaining keys. We assume that the remaining
  keys in the input extract contain list of objects (one per example).

  Returns:
    Extractor for unbatching batched extracts.
  """
  # pylint: disable=no-value-for-parameter
  return extractor.Extractor(
      stage_name=UNBATCH_EXTRACTOR_STAGE_NAME, ptransform=_UnbatchInputs())


def _extract_unbatched_inputs(  # pylint: disable=invalid-name
    mixed_legacy_batched_extract: types.Extracts) -> Sequence[types.Extracts]:
  """Extract features, predictions, labels and weights from batched extract."""
  batched_extract = {}
  # TODO(mdreves): Remove record batch
  keys_to_retain = set(mixed_legacy_batched_extract.keys())
  if constants.ARROW_RECORD_BATCH_KEY in keys_to_retain:
    keys_to_retain.remove(constants.ARROW_RECORD_BATCH_KEY)
  dataframe = pd.DataFrame()
  for key in keys_to_retain:
    # Previously a batch of transformed features were stored as a list of dicts
    # instead of a dict of np.arrays with batch dimensions. These legacy
    # conversions are done using dataframes instead.
    if isinstance(mixed_legacy_batched_extract[key], list):
      try:
        dataframe[key] = mixed_legacy_batched_extract[key]
      except Exception as e:
        raise RuntimeError(
            f'Exception encountered while adding key {key} with '
            f'batched length {len(mixed_legacy_batched_extract[key])}') from e
    else:
      batched_extract[key] = mixed_legacy_batched_extract[key]
  unbatched_extracts = util.split_extracts(batched_extract)
  legacy_unbatched_extracts = dataframe.to_dict(orient='records')
  if unbatched_extracts and legacy_unbatched_extracts:
    if len(unbatched_extracts) != len(legacy_unbatched_extracts):
      raise ValueError(
          f'Batch sizes have differing values: {len(unbatched_extracts)} != '
          f'{len(legacy_unbatched_extracts)}, '
          f'unbatched_extracts={unbatched_extracts}, '
          f'legacy_unbatched_extracts={legacy_unbatched_extracts}')
    result = []
    for unbatched_extract, legacy_unbatched_extract in zip(
        unbatched_extracts, legacy_unbatched_extracts):
      legacy_unbatched_extract.update(unbatched_extract)
      result.append(legacy_unbatched_extract)
    return result
  elif legacy_unbatched_extracts:
    return legacy_unbatched_extracts
  else:
    return unbatched_extracts


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(types.Extracts)
def _UnbatchInputs(
    extracts: beam.pvalue.PCollection) -> beam.pvalue.PCollection:
  """Extracts unbatched inputs from batched extracts.

  Args:
    extracts: PCollection containing batched extracts.

  Returns:
    PCollection of per-example extracts.
  """
  return extracts | 'UnbatchInputs' >> beam.FlatMap(_extract_unbatched_inputs)
