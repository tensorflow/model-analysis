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

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import apache_beam as beam
from tensorflow_model_analysis.evaluators.query_metrics import query_types
from tensorflow_model_analysis.post_export_metrics import metric_keys

from typing import Any, Dict, List, NamedTuple, Text

_State = NamedTuple('_State', [('total_queries', int), ('total_documents', int),
                               ('min_documents', int), ('max_documents', int)])


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
    return _State(
        total_queries=accumulator.total_queries + 1,
        total_documents=accumulator.total_documents + len(query_fpl.fpls),
        min_documents=min(accumulator.min_documents, len(query_fpl.fpls)),
        max_documents=max(accumulator.max_documents, len(query_fpl.fpls)))

  def merge_accumulators(self, accumulators: List[_State]) -> _State:
    total_queries = 0
    total_documents = 0
    min_documents = self.LARGE_INT
    max_documents = 0
    for acc in accumulators:
      total_queries += acc.total_queries
      total_documents += acc.total_documents
      min_documents = min(min_documents, acc.min_documents)
      max_documents = max(max_documents, acc.max_documents)
    return _State(
        total_queries=total_queries,
        total_documents=total_documents,
        min_documents=min_documents,
        max_documents=max_documents)

  def extract_output(self, accumulator: _State) -> Dict[Text, Any]:
    return {
        metric_keys.base_key('total_queries'): accumulator.total_queries,
        metric_keys.base_key('total_documents'): accumulator.total_documents,
        metric_keys.base_key('min_documents'): accumulator.min_documents,
        metric_keys.base_key('max_documents'): accumulator.max_documents
    }
