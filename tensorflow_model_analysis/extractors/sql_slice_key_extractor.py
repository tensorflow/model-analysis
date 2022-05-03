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
"""Sql Slice Key Extractor."""

import copy
import functools
from typing import List, Mapping

import apache_beam as beam
import pyarrow as pa
import tensorflow as tf

from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.extractors import extractor
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.slicer import slicer_lib
from tensorflow_model_analysis.utils import util

from tfx_bsl.arrow import sql_util
from tfx_bsl.tfxio import tensor_to_arrow

_TF_MAJOR_VERSION = int(tf.version.VERSION.split('.')[0])

_SQL_SLICE_KEY_EXTRACTOR_STAGE_NAME = 'ExtractSqlSliceKeys'


def SqlSliceKeyExtractor(
    eval_config: config_pb2.EvalConfig) -> extractor.Extractor:
  """Creates an extractor for sql slice keys.

  This extractor extracts slices keys in a batch based on the SQL statement in
  the eval config.

  Args:
    eval_config: EvalConfig containing slicing_specs specifying the slices to
      slice the data into.

  Returns:
    Extractor for extracting slice keys in batch.
  """
  # pylint: disable=no-value-for-parameter
  return extractor.Extractor(
      stage_name=_SQL_SLICE_KEY_EXTRACTOR_STAGE_NAME,
      ptransform=_ExtractSqlSliceKey(eval_config))


@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(types.Extracts)
class ExtractSqlSliceKeyFn(beam.DoFn):
  """A DoFn that extracts slice keys in batch."""

  def __init__(self, eval_config: config_pb2.EvalConfig):
    self._eval_config = eval_config
    self._sqls = [
        """
        SELECT
          ARRAY(
            {}
          ) as slice_key
        FROM Examples as example;""".format(spec.slice_keys_sql)
        for spec in eval_config.slicing_specs
        if spec.slice_keys_sql
    ]
    self._sql_slicer_num_record_batch_schemas = (
        beam.metrics.Metrics.distribution(
            constants.METRICS_NAMESPACE, 'sql_slicer_num_record_batch_schemas'))

  def setup(self):

    def _GenerateQueries(
        schema: pa.Schema) -> List[sql_util.RecordBatchSQLSliceQuery]:
      result = []
      for sql in self._sqls:
        try:
          result.append(sql_util.RecordBatchSQLSliceQuery(sql, schema))
        except Exception as e:
          raise RuntimeError(f'Failed to parse sql:\n\n{sql}') from e
      return result

    # A cache for compiled sql queries, keyed by record batch schemas.
    # This way the extractor can work with record batches of different schemas,
    # which is legit but uncommon.
    self._cached_queries = functools.lru_cache(maxsize=3)(_GenerateQueries)

  def process(self, batched_extract: types.Extracts) -> List[types.Extracts]:
    features = batched_extract[constants.FEATURES_KEY]
    # Slice on transformed features if available.
    if (constants.TRANSFORMED_FEATURES_KEY in batched_extract and
        batched_extract[constants.TRANSFORMED_FEATURES_KEY] is not None):
      transformed_features = batched_extract[constants.TRANSFORMED_FEATURES_KEY]
      # If only one model, the output is stored without keying on model name.
      if not self._eval_config or len(self._eval_config.model_specs) == 1:
        features.update(transformed_features)
      else:
        # Models listed earlier have precedence in feature lookup.
        for spec in reversed(self._eval_config.model_specs):
          if spec.name in transformed_features:
            features.update(transformed_features[spec.name])

    tensors = util.to_tensorflow_tensors(features)
    tensor_specs = util.infer_tensor_specs(tensors)

    if _TF_MAJOR_VERSION < 2:
      # TODO(b/228456048): TFX-BSL doesn't support passing tensorflow tensors
      # for non-sparse/ragged values in TF 1.x (i.e. it only accepts np.ndarray
      # for dense) so we need to convert dense tensors to numpy.
      sess = tf.compat.v1.Session()

      def _convert_dense_to_numpy(values):  # pylint: disable=invalid-name
        if isinstance(values, Mapping):
          for k, v in values.items():
            if isinstance(v, Mapping):
              values[k] = _convert_dense_to_numpy(v)
            elif isinstance(v, tf.Tensor):
              values[k] = v.eval(session=sess)
        return values

      tensors = _convert_dense_to_numpy(tensors)

    converter = tensor_to_arrow.TensorsToRecordBatchConverter(tensor_specs)
    record_batch = converter.convert(tensors)
    sql_slice_keys = [[] for _ in range(record_batch.num_rows)]

    for query in self._cached_queries(record_batch.schema):
      # Example of result with batch size = 3:
      # result = [[[('feature', 'value_1')]],
      #           [[('feature', 'value_2')]],
      #           []
      #          ]
      result = query.Execute(record_batch)
      for row_index, row_result in enumerate(result):
        sql_slice_keys[row_index].extend([tuple(s) for s in row_result])

    # convert sql_slice_keys into a VarLenTensorValue where each row has dtype
    # object.
    dense_rows = []
    for row_slice_keys in sql_slice_keys:
      dense_rows.append(slicer_lib.slice_keys_to_numpy_array(row_slice_keys))
    varlen_sql_slice_keys = types.VarLenTensorValue.from_dense_rows(dense_rows)

    # Make a a shallow copy, so we don't mutate the original.
    batched_extract_copy = copy.copy(batched_extract)
    batched_extract_copy[constants.SLICE_KEY_TYPES_KEY] = varlen_sql_slice_keys

    self._sql_slicer_num_record_batch_schemas.update(
        self._cached_queries.cache_info().currsize)

    return [batched_extract_copy]


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(types.Extracts)
def _ExtractSqlSliceKey(
    extracts: beam.pvalue.PCollection,
    eval_config: config_pb2.EvalConfig) -> beam.pvalue.PCollection:
  return extracts | beam.ParDo(ExtractSqlSliceKeyFn(eval_config))
