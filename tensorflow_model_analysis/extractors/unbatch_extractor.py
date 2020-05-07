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
"""Unbatch extractor."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

from typing import Sequence

import apache_beam as beam
import pandas as pd
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.extractors import extractor

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


def _ExtractUnbatchedInputs(
    batched_extract: types.Extracts) -> Sequence[types.Extracts]:
  """Extract features, predictions, labels and weights from batched extract."""
  keys_to_retain = set(batched_extract.keys())
  keys_to_retain.remove(constants.ARROW_RECORD_BATCH_KEY)
  dataframe = pd.DataFrame()
  for key in keys_to_retain:
    dataframe[key] = batched_extract[key]
  return dataframe.to_dict(orient='records')


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
  return extracts | 'UnbatchInputs' >> beam.FlatMap(_ExtractUnbatchedInputs)
