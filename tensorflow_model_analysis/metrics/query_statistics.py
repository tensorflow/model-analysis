# Lint as: python3
# Copyright 2019 Google LLC
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
"""Query statistics metrics."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

from typing import Dict, List, Text

import apache_beam as beam
from tensorflow_model_analysis.metrics import metric_types

TOTAL_QUERIES_NAME = 'total_queries'
TOTAL_DOCUMENTS_NAME = 'total_documents'
MIN_DOCUMENTS_NAME = 'min_documents'
MAX_DOCUMENTS_NAME = 'max_documents'


class QueryStatistics(metric_types.Metric):
  """Query statistic metrics.

  These metrics are query/ranking based so a query_key must also be provided in
  the associated metrics spec.
  """

  def __init__(self,
               total_queries_name=TOTAL_QUERIES_NAME,
               total_documents_name=TOTAL_DOCUMENTS_NAME,
               min_documents_name=MIN_DOCUMENTS_NAME,
               max_documents_name=MAX_DOCUMENTS_NAME):
    """Initializes query statistics metrics.

    Args:
      total_queries_name: Total queries metric name.
      total_documents_name: Total documents metric name.
      min_documents_name: Min documents name.
      max_documents_name: Max documents name.
    """
    super(QueryStatistics, self).__init__(
        _query_statistics,
        total_queries_name=total_queries_name,
        total_documents_name=total_documents_name,
        min_documents_name=min_documents_name,
        max_documents_name=max_documents_name)


metric_types.register_metric(QueryStatistics)


def _query_statistics(total_queries_name=TOTAL_QUERIES_NAME,
                      total_documents_name=TOTAL_DOCUMENTS_NAME,
                      min_documents_name=MIN_DOCUMENTS_NAME,
                      max_documents_name=MAX_DOCUMENTS_NAME,
                      query_key: Text = '') -> metric_types.MetricComputations:
  """Returns metric computations for query statistics."""
  if not query_key:
    raise ValueError('a query_key is required to use QueryStatistics metrics')

  total_queries_key = metric_types.MetricKey(name=total_queries_name)
  total_documents_key = metric_types.MetricKey(name=total_documents_name)
  min_documents_key = metric_types.MetricKey(name=min_documents_name)
  max_documents_key = metric_types.MetricKey(name=max_documents_name)

  return [
      metric_types.MetricComputation(
          keys=[
              total_queries_key, total_documents_key, min_documents_key,
              max_documents_key
          ],
          preprocessor=None,
          combiner=_QueryStatisticsCombiner(total_queries_key,
                                            total_documents_key,
                                            min_documents_key,
                                            max_documents_key))
  ]


class _QueryStatisticsAccumulator(object):
  """Query statistics accumulator."""
  __slots__ = [
      'total_queries', 'total_documents', 'min_documents', 'max_documents'
  ]

  LARGE_INT = 1000000000

  def __init__(self):
    self.total_queries = 0
    self.total_documents = 0
    self.min_documents = self.LARGE_INT
    self.max_documents = 0


class _QueryStatisticsCombiner(beam.CombineFn):
  """Computes query statistics metrics."""

  def __init__(self, total_queries_key: metric_types.MetricKey,
               total_documents_key: metric_types.MetricKey,
               min_documents_key: metric_types.MetricKey,
               max_documents_key: metric_types.MetricKey):
    self._total_queries_key = total_queries_key
    self._total_documents_key = total_documents_key
    self._min_documents_key = min_documents_key
    self._max_documents_key = max_documents_key

  def create_accumulator(self) -> _QueryStatisticsAccumulator:
    return _QueryStatisticsAccumulator()

  def add_input(
      self, accumulator: _QueryStatisticsAccumulator,
      element: metric_types.StandardMetricInputs
  ) -> _QueryStatisticsAccumulator:
    accumulator.total_queries += 1
    num_documents = len(element.prediction)
    accumulator.total_documents += num_documents
    accumulator.min_documents = min(accumulator.min_documents, num_documents)
    accumulator.max_documents = max(accumulator.max_documents, num_documents)
    return accumulator

  def merge_accumulators(
      self, accumulators: List[_QueryStatisticsAccumulator]
  ) -> _QueryStatisticsAccumulator:
    result = self.create_accumulator()
    for accumulator in accumulators:
      result.total_queries += accumulator.total_queries
      result.total_documents += accumulator.total_documents
      result.min_documents = min(result.min_documents,
                                 accumulator.min_documents)
      result.max_documents = max(result.max_documents,
                                 accumulator.max_documents)
    return result

  def extract_output(
      self, accumulator: _QueryStatisticsAccumulator
  ) -> Dict[metric_types.MetricKey, int]:
    return {
        self._total_queries_key: accumulator.total_queries,
        self._total_documents_key: accumulator.total_documents,
        self._min_documents_key: accumulator.min_documents,
        self._max_documents_key: accumulator.max_documents
    }
