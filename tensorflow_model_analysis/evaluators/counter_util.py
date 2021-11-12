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
"""Utility for evaluator to add / update beam counters."""

from typing import List, Set

import apache_beam as beam

from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.proto import config_pb2


def _IncrementMetricsCounters(metric_name: str, version: str, model_type: str):
  # LINT.IfChange
  metric_name = 'metric_computed_%s_%s_%s' % (metric_name, version, model_type)
  # LINT.ThenChange(../../../../learning/fairness/infra/plx/scripts/tfma_metrics_computed_tracker_macros.sql)
  metrics_counter = beam.metrics.Metrics.counter(constants.METRICS_NAMESPACE,
                                                 metric_name)
  metrics_counter.inc(1)


@beam.ptransform_fn
# TODO(b/148788775): These type hints fail Beam type checking.
# @beam.typehints.with_input_types(beam.Pipeline)
# @beam.typehints.with_output_types(beam.pvalue.PDone)
def IncrementMetricsCallbacksCounters(
    pipeline: beam.Pipeline,
    metrics_callbacks: List[types.AddMetricsCallbackType], model_type: str):
  """To track count of all the metrics being computed using TFMA."""

  def _MakeAndIncrementCounters(_):
    for callback in metrics_callbacks:
      if hasattr(callback, 'name'):
        _IncrementMetricsCounters(callback.name, 'v1', model_type)

  return (pipeline
          | 'CreateSole' >> beam.Create([None])
          | 'Count' >> beam.Map(_MakeAndIncrementCounters))


@beam.ptransform_fn
# TODO(b/148788775): These type hints fail Beam type checking.
# @beam.typehints.with_input_types(beam.Pipeline)
# @beam.typehints.with_output_types(beam.pvalue.PDone)
def IncrementSliceSpecCounters(pipeline: beam.Pipeline):
  """To track count of all slicing spec computed using TFMA."""

  def _MakeAndIncrementCounters(slice_list):
    for slice_key, slice_value in slice_list:
      # LINT.IfChange
      slice_name = 'slice_computed_%s_%s' % (slice_key, slice_value)
      # LINT.ThenChange(../../../../learning/fairness/infra/plx/scripts/tfma_metrics_computed_tracker_macros.sql)
      slice_counter = beam.metrics.Metrics.counter(constants.METRICS_NAMESPACE,
                                                   slice_name)
      slice_counter.inc(1)

  return (pipeline
          | 'GetSliceCountKeys' >> beam.Keys()
          | 'Count' >> beam.Map(_MakeAndIncrementCounters))


@beam.ptransform_fn
# TODO(b/148788775): These type hints fail Beam type checking.
# @beam.typehints.with_input_types(beam.Pipeline)
# @beam.typehints.with_output_types(beam.pvalue.PDone)
def IncrementMetricsSpecsCounters(pipeline: beam.Pipeline,
                                  metrics_specs: List[config_pb2.MetricsSpec],
                                  model_types: Set[str]):
  """To track count of all metrics specs in TFMA."""

  def _MakeAndIncrementCounters(_):
    for model_type in model_types:
      for metrics_spec in metrics_specs:
        for metric in metrics_spec.metrics:
          _IncrementMetricsCounters(metric.class_name, 'v2', model_type)

  return (pipeline
          | 'CreateSole' >> beam.Create([None])
          | 'Count' >> beam.Map(_MakeAndIncrementCounters))
