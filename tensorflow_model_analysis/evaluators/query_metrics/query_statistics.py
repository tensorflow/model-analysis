# Copyright 2018 Google LLC
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
"""Simple statistics about queries, e.g. number of queries, documents, etc."""

import dataclasses
from typing import Any, Dict, Iterable

import apache_beam as beam
from tensorflow_model_analysis.evaluators.query_metrics import query_types
from tensorflow_model_analysis.post_export_metrics import metric_keys


@dataclasses.dataclass
class _State:
  """QueryStatisticsCombineFn accumulator type."""
  total_queries: int
  total_documents: int
  min_documents: int
  max_documents: int

  def merge(self, other: '_State') -> None:
    self.total_queries += other.total_queries
    self.total_documents += other.total_documents
    self.min_documents = min(self.min_documents, other.min_documents)
    self.max_documents = max(self.max_documents, other.max_documents)

  def add(self, query_fpl: query_types.QueryFPL) -> None:
    self.total_queries += 1
    self.total_documents += len(query_fpl.fpls)
    self.min_documents = min(self.min_documents, len(query_fpl.fpls))
    self.max_documents = max(self.max_documents, len(query_fpl.fpls))


class QueryStatisticsCombineFn(beam.CombineFn):
  """Computes simple statistics about queries."""

  LARGE_INT = 1000000000

  def create_accumulator(self):
    return _State(
        total_queries=0,
        total_documents=0,
        min_documents=self.LARGE_INT,
        max_documents=0)

  def add_input(self, accumulator: _State,
                query_fpl: query_types.QueryFPL) -> _State:
    accumulator.add(query_fpl)
    return accumulator

  def merge_accumulators(self, accumulators: Iterable[_State]) -> _State:
    it = iter(accumulators)
    result = next(it)
    for acc in it:
      result.merge(acc)
    return result

  def extract_output(self, accumulator: _State) -> Dict[str, Any]:
    return {
        metric_keys.base_key('total_queries'): accumulator.total_queries,
        metric_keys.base_key('total_documents'): accumulator.total_documents,
        metric_keys.base_key('min_documents'): accumulator.min_documents,
        metric_keys.base_key('max_documents'): accumulator.max_documents
    }
