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
from typing import List

import apache_beam as beam
import pyarrow as pa

from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.extractors import extractor
from tensorflow_model_analysis.proto import config_pb2

from tfx_bsl.arrow import sql_util

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
      return [
          sql_util.RecordBatchSQLSliceQuery(sql, schema) for sql in self._sqls
      ]

    # A cache for compiled sql queries, keyed by record batch schemas.
    # This way the extractor can work with record batches of different schemas,
    # which is legit but uncommon.
    self._cached_queries = functools.lru_cache(maxsize=3)(_GenerateQueries)

  def process(self, batched_extract: types.Extracts) -> List[types.Extracts]:
    record_batch = batched_extract[constants.ARROW_RECORD_BATCH_KEY]
    sql_slice_keys = [[] for _ in range(record_batch.num_rows)]

    for query in self._cached_queries(record_batch.schema):
      # Example of result with batch size = 3:
      # result = [[[('feature', 'value_1')]],
      #           [[('feature', 'value_2')]],
      #           []
      #          ]
      result = query.Execute(record_batch)
      for row_index in range(len(result)):
        sql_slice_keys[row_index].extend([tuple(s) for s in result[row_index]])

    # Make a a shallow copy, so we don't mutate the original.
    batched_extract_copy = copy.copy(batched_extract)
    batched_extract_copy[constants.SLICE_KEY_TYPES_KEY] = sql_slice_keys

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
