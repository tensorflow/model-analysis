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
"""Features extractor."""

import copy
from typing import Mapping, Optional, Text, Tuple

import apache_beam as beam
import numpy as np
import pyarrow as pa

from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.extractors import extractor
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.utils import util

from tensorflow_metadata.proto.v0 import schema_pb2

_FEATURES_EXTRACTOR_STAGE_NAME = 'ExtractFeatures'


def FeaturesExtractor(  # pylint: disable=invalid-name
    eval_config: config_pb2.EvalConfig,
    tensor_representations: Optional[Mapping[
        Text, schema_pb2.TensorRepresentation]] = None) -> extractor.Extractor:
  """Creates an extractor for extracting features.

  The extractor acts as follows depending on the existence of certain keys
  within the incoming extracts:

    1) Extracts contains tfma.ARROW_RECORD_BATCH_KEY

    The features stored in the RecordBatch will be extracted and added to the
    output extract under the key tfma.FEATURES_KEY and the raw serialized inputs
    will be added under the tfma.INPUT_KEY. Any extracts that already exist will
    be merged with the values from the RecordBatch with the RecordBatch values
    taking precedence when duplicate keys are detected. The
    tfma.ARROW_RECORD_BATCH_KEY key will be removed from the output extracts.

    2) Extracts contains tfma.FEATURES_KEY (but not tfma.ARROW_RECORD_BATCH_KEY)

    The operation will be a no-op and the incoming extracts will be passed as is
    to the output.

    3) Extracts contains neither tfma.FEATURES_KEY | tfma.ARROW_RECORD_BATCH_KEY

    An exception will be raised.

  Args:
    eval_config: Eval config.
    tensor_representations: Optional tensor representations to use when parsing
      the data. If tensor_representations are not passed or a representation is
      not found for a given feature name a default representation will be used
      where possible, otherwise an exception will be raised.

  Returns:
    Extractor for extracting features.
  """
  del eval_config
  # pylint: disable=no-value-for-parameter
  return extractor.Extractor(
      stage_name=_FEATURES_EXTRACTOR_STAGE_NAME,
      ptransform=_ExtractFeatures(tensor_representations or {}))


# TODO(b/214273030): Move to tfx-bsl.
def _is_list_like(arrow_type: pa.DataType) -> bool:
  return pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type)


# TODO(b/214273030): Move to tfx-bsl.
def _is_binary_like(arrow_type: pa.DataType) -> bool:
  return (pa.types.is_binary(arrow_type) or
          pa.types.is_large_binary(arrow_type) or
          pa.types.is_string(arrow_type) or
          pa.types.is_large_string(arrow_type))


# TODO(b/214273030): Move to tfx-bsl.
def _is_supported_arrow_value_type(arrow_type: pa.DataType) -> bool:
  return (pa.types.is_integer(arrow_type) or pa.types.is_floating(arrow_type) or
          _is_binary_like(arrow_type))


def _drop_unsupported_columns_and_fetch_raw_data_column(
    record_batch: pa.RecordBatch
) -> Tuple[pa.RecordBatch, Optional[np.ndarray]]:
  """Drops unsupported columns and fetches the raw data column.

  Currently, types that are not binary_like or ListArray[primitive types] are
  dropped.

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
      assert (_is_list_like(column_type) and
              _is_binary_like(column_type.value_type)), (
                  'Invalid type for batched input key: {}. '
                  'Expected binary like.'.format(column_type))
      serialized_examples = np.asarray(column_array.flatten())
    # Currently we only handle columns of type list<primitive|binary_like>.
    # We ignore other columns as we cannot efficiently convert them into an
    # instance dict format.
    elif (_is_list_like(column_type) and
          _is_supported_arrow_value_type(column_type.value_type)):
      column_names.append(column_name)
      column_arrays.append(column_array)
  return (pa.RecordBatch.from_arrays(column_arrays,
                                     column_names), serialized_examples)


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(types.Extracts)
def _ExtractFeatures(  # pylint: disable=invalid-name
    extracts: beam.pvalue.PCollection,
    tensor_representations: Mapping[str, schema_pb2.TensorRepresentation]
) -> beam.pvalue.PCollection:
  """Extracts features from extracts.

  Args:
    extracts: PCollection containing features under tfma.ARROW_RECORD_BATCH_KEY
      or tfma.FEATURES_KEY.
    tensor_representations: Tensor representations.

  Returns:
    PCollection of extracts with additional features added under the key
    tfma.FEATURES_KEY and optionally inputs added under the tfma.INPUTS_KEY.

  Raises:
    ValueError: If incoming extracts contains neither tfma.FEATURES_KEY nor
      tfma.ARROW_RECORD_BATCH_KEY.
  """

  def extract_features(extracts: types.Extracts) -> types.Extracts:
    """Extract features from extracts containing arrow table."""
    result = copy.copy(extracts)
    if constants.ARROW_RECORD_BATCH_KEY in extracts:
      (record_batch, serialized_examples) = (
          _drop_unsupported_columns_and_fetch_raw_data_column(
              extracts[constants.ARROW_RECORD_BATCH_KEY]))
      del result[constants.ARROW_RECORD_BATCH_KEY]
      features = result[
          constants.FEATURES_KEY] if constants.FEATURES_KEY in result else {}
      features.update(
          util.record_batch_to_tensor_values(record_batch,
                                             tensor_representations))
      result[constants.FEATURES_KEY] = features
      result[constants.INPUT_KEY] = serialized_examples
    elif constants.FEATURES_KEY not in extracts:
      raise ValueError(
          'Incoming extracts must contain either tfma.ARROW_RECORD_BATCH_KEY '
          f'or tfma.FEATURES_KEY, but extracts={extracts}')
    return result

  return extracts | 'ExtractFeatures' >> beam.Map(extract_features)
