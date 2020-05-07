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
"""Batched input extractor."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import copy
from typing import Optional, Tuple

import apache_beam as beam
import numpy as np
import pandas as pd
import pyarrow as pa
from tensorflow_model_analysis import config
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.extractors import extractor

# TODO(b/150029186): Avoid referring to extractors by stage names.
INPUT_EXTRACTOR_STAGE_NAME = 'ExtractBatchedInputs'


def BatchedInputExtractor(
    eval_config: config.EvalConfig) -> extractor.Extractor:
  """Creates an extractor for extracting features, labels, and example weights.

  The extractor's PTransform extracts features, labels, and example weights from
  the batched features (i.e., Arrow RecordBatch) stored under
  tfma.ARROW_RECORD_BATCH_KEY in the incoming extract and adds it to the output
  extract under the keys tfma.FEATURES_KEY, tfma.LABELS_KEY, and
  tfma.EXAMPLE_WEIGHTS_KEY. If the eval_config contains a
  prediction_key and a corresponding key is found in the parse example, then
  predictions will also be extracted and stored under the
  tfma.PREDICTIONS_KEY. Any extracts that already exist will be merged
  with the values parsed by this extractor with this extractor's values taking
  precedence when duplicate keys are detected.

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
    Batched extractor for extracting features, labels, and example weights.
  """
  # pylint: disable=no-value-for-parameter
  return extractor.Extractor(
      stage_name=INPUT_EXTRACTOR_STAGE_NAME,
      ptransform=_ExtractBatchedInputs(eval_config=eval_config))


def _IsListLike(arrow_type: pa.DataType) -> bool:
  return pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type)


def _IsBinaryLike(arrow_type: pa.DataType) -> bool:
  return (pa.types.is_binary(arrow_type) or
          pa.types.is_large_binary(arrow_type) or
          pa.types.is_string(arrow_type) or
          pa.types.is_large_string(arrow_type))


def _IsSupportedArrowValueType(arrow_type: pa.DataType) -> bool:
  return (pa.types.is_integer(arrow_type) or pa.types.is_floating(arrow_type) or
          _IsBinaryLike(arrow_type))


def _DropUnsupportedColumnsAndFetchRawDataColumn(
    record_batch: pa.RecordBatch
) -> Tuple[pa.RecordBatch, Optional[np.ndarray]]:
  """Drops unsupported columns and fetches the raw data column.

  Args:
    record_batch: An Arrow RecordBatch.

  Returns:
    Arrow RecordBatch with only supported columns.
  """
  column_names, column_arrays = [], []
  serialized_examples = None
  for column_name, column_array in zip(record_batch.schema.names,
                                       record_batch.columns):
    column_type = column_array.type
    if column_name == constants.ARROW_INPUT_COLUMN:
      assert (_IsListLike(column_type) and
              _IsBinaryLike(column_type.value_type)), (
                  'Invalid type for batched input key: {}. '
                  'Expected binary like.'.format(column_type))
      serialized_examples = np.asarray(column_array.flatten())
    # Currently we only handle columns of type list<primitive|binary_like>.
    # We ignore other columns as we cannot efficiently convert them into an
    # instance dict format.
    elif (_IsListLike(column_type) and
          _IsSupportedArrowValueType(column_type.value_type)):
      column_names.append(column_name)
      column_arrays.append(column_array)
  return (pa.RecordBatch.from_arrays(column_arrays,
                                     column_names), serialized_examples)


def _ExtractInputs(batched_extract: types.Extracts,
                   eval_config: config.EvalConfig) -> types.Extracts:
  """Extract features, predictions, labels and weights from batched extract."""
  result = copy.copy(batched_extract)
  (record_batch, serialized_examples) = (
      _DropUnsupportedColumnsAndFetchRawDataColumn(
          batched_extract[constants.ARROW_RECORD_BATCH_KEY]))
  dataframe = record_batch.to_pandas()
  original_keys = set(dataframe.columns)

  # In multi-output model, the keys (labels, predictions, weights) are
  # keyed by output name. In this case, we will have a nested dict in the
  # extracts keyed by the output names.
  def _get_proj_df_dict(original, keys_dict):  # pylint: disable=invalid-name
    df_proj = pd.DataFrame()
    for output_name, key in keys_dict.items():
      if key in original_keys:
        df_proj[output_name] = original[key]
    return df_proj.to_dict(orient='records')

  def _add_proj_df(proj_df, result, key):  # pylint: disable=invalid-name
    if proj_df.shape[1] == 0:
      return
    elif proj_df.shape[1] == 1:
      result[key] = proj_df[proj_df.columns[0]]
    else:
      result[key] = proj_df.to_dict(orient='records')

  keys_to_remove = set()
  labels_df = pd.DataFrame()
  example_weights_df = pd.DataFrame()
  predictions_df = pd.DataFrame()
  for spec in eval_config.model_specs:
    if spec.label_key:
      labels_df[spec.name] = dataframe[spec.label_key]
      keys_to_remove.add(spec.label_key)
    elif spec.label_keys:
      labels_df[spec.name] = _get_proj_df_dict(dataframe, spec.label_keys)
      keys_to_remove.update(set(spec.label_keys.values()))

    if spec.example_weight_key:
      example_weights_df[spec.name] = dataframe[spec.example_weight_key]
      keys_to_remove.add(spec.example_weight_key)
    elif spec.example_weight_keys:
      example_weights_df[spec.name] = _get_proj_df_dict(
          dataframe, spec.example_weight_keys)
      keys_to_remove.update(set(spec.example_weight_keys.values()))

    if spec.prediction_key and spec.prediction_key in original_keys:
      predictions_df[spec.name] = dataframe[spec.prediction_key]
      keys_to_remove.add(spec.prediction_key)
    elif spec.prediction_keys:
      proj_df_dict = _get_proj_df_dict(dataframe, spec.prediction_keys)
      if proj_df_dict:
        predictions_df[spec.name] = _get_proj_df_dict(dataframe,
                                                      spec.prediction_keys)
        keys_to_remove.update(set(spec.prediction_keys.values()))

  _add_proj_df(labels_df, result, constants.LABELS_KEY)
  _add_proj_df(example_weights_df, result, constants.EXAMPLE_WEIGHTS_KEY)
  _add_proj_df(predictions_df, result, constants.PREDICTIONS_KEY)

  # Add a separate column with the features dict.
  feature_keys = original_keys.difference(keys_to_remove)
  result[constants.FEATURES_KEY] = dataframe[feature_keys].to_dict(
      orient='records')

  # TODO(pachristopher): Consider avoiding setting this key if we don't need
  # this any further in the pipeline. This can avoid a potentially costly copy.
  result[constants.INPUT_KEY] = serialized_examples
  return result


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(types.Extracts)
def _ExtractBatchedInputs(
    extracts: beam.pvalue.PCollection,
    eval_config: config.EvalConfig) -> beam.pvalue.PCollection:
  """Extracts features, labels and weights from batched extracts.

  Args:
    extracts: PCollection containing batched features under tfma.FEATURES_KEY.
    eval_config: Eval config.

  Returns:
    PCollection of extracts with additional features, labels, and weights added
    under the keys tfma.FEATURES_KEY, tfma.LABELS_KEY, and
    tfma.EXAMPLE_WEIGHTS_KEY.
  """
  return extracts | 'ExtractInputs' >> beam.Map(_ExtractInputs, eval_config)
