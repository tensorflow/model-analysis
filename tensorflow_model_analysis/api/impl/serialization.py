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
"""Serialization library. For internal use only."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import os
import pickle
import apache_beam as beam
import numpy as np
import tensorflow as tf

from tensorflow_model_analysis.api.impl import api_types
from tensorflow_model_analysis.slicer import slicer
from tensorflow_model_analysis.types_compat import Any, Callable, Dict, List, Tuple, TypeVar

# File names for files written out to the result directory.
_METRICS_OUTPUT_FILE = 'metrics'
_PLOTS_OUTPUT_FILE = 'plots'
_EVAL_CONFIG_FILE = 'eval_config'

# Keys for the serialized final dictionary.
_SLICE_METRICS_LIST_KEY = 'slice_metrics_list'
_VERSION_KEY = 'version'
_METRICS_TYPE_KEY = 'metrics_type'
_EVAL_CONFIG_KEY = 'eval_config'

# Metric types for the _METRICS_TYPE_KEY field.
_PLOTS_METRICS_TYPE = 'plots'
_METRICS_METRICS_TYPE = 'metrics'


def _serialize_eval_config(eval_config):
  final_dict = {}
  final_dict[_VERSION_KEY] = '0.0.1'
  final_dict[_EVAL_CONFIG_KEY] = eval_config
  return pickle.dumps(final_dict)


def _deserialize_eval_config_raw(serialized):
  return pickle.loads(serialized)


def load_eval_config(output_path):
  serialized_record = tf.python_io.tf_record_iterator(
      os.path.join(output_path, _EVAL_CONFIG_FILE)).next()
  final_dict = _deserialize_eval_config_raw(serialized_record)
  return final_dict[_EVAL_CONFIG_KEY]


def _deserialize_metrics_raw(serialized):
  """Deserializes metrics serialized with _serialize_metrics.

  Returns the "raw" serialized final dictionary, including the version and
  other metadata. Metrics can be extracted from the _SLICE_METRICS_LIST_KEY key.

  Args:
    serialized: Metrics serialized with _serialize_metrics

  Returns:
    "Raw" serialized final dictionary.
  """
  return pickle.loads(serialized)


def _load_and_deserialize_metrics(
    path,
    metrics_type):
  serialized_record = tf.python_io.tf_record_iterator(path).next()
  final_dict = _deserialize_metrics_raw(serialized_record)
  if final_dict[_METRICS_TYPE_KEY] != metrics_type:
    raise ValueError('when deserializing file %s, metrics type mismatch: '
                     'expecting %s, got %s' % (path, metrics_type,
                                               final_dict[_METRICS_TYPE_KEY]))
  return final_dict[_SLICE_METRICS_LIST_KEY]


def load_plots_and_metrics(
    output_path):
  slicing_metrics = _load_and_deserialize_metrics(
      path=os.path.join(output_path, _METRICS_OUTPUT_FILE),
      metrics_type=_METRICS_METRICS_TYPE)
  plots = _load_and_deserialize_metrics(
      path=os.path.join(output_path, _PLOTS_OUTPUT_FILE),
      metrics_type=_PLOTS_METRICS_TYPE)
  return slicing_metrics, plots




class _AccumulateCombineFn(beam.CombineFn):
  """CombineFn that accumulates all elements, applies a function, outputs."""

  def __init__(self, transform_fn):
    self._transform_fn = transform_fn

  def create_accumulator(self):  # pytype: disable=invalid-annotation
    return []

  def add_input(self, accumulator, elem):
    accumulator.append(elem)
    return accumulator

  def merge_accumulators(self, accumulators):
    result = []
    for accumulator in accumulators:
      result.extend(accumulator)
    return result

  def extract_output(
      self, accumulator):  # pytype: disable=invalid-annotation
    return self._transform_fn(accumulator)


def _serialize_metrics(
    slice_metrics_list,
    metrics_type,
):
  """Serialize the given slice metrics list.

  Implementation details: we create a dictionary containing a verison key,
  and a slice_metrics_list key. The slice_metrics_list key contains the
  given slice_metrics_list with the appropriate conversions.

  Args:
    slice_metrics_list: List of slice metrics.
    metrics_type: The metrics type to store in the metadata.

  Returns:
    Serialized version of the slice metrics list.
  """
  formatted_slice_metrics_list = []
  for slice_metric in slice_metrics_list:
    slice_key, slice_metrics = slice_metric
    formatted_slice_metrics = {}
    for k, v in slice_metrics.items():
      if isinstance(v, np.ndarray):
        formatted_slice_metrics[k] = v.tolist()
      else:
        # v is a numpy scalar value, e.g. np.float32, np.int64, etc
        formatted_slice_metrics[k] = v.item()
    formatted_slice_metrics_list.append((slice_key, formatted_slice_metrics))

  final_dict = {
      _VERSION_KEY: '0.0.1',
      _SLICE_METRICS_LIST_KEY: formatted_slice_metrics_list,
      _METRICS_TYPE_KEY: metrics_type,
  }
  return pickle.dumps(final_dict)


def _make_serialize_metrics_fn(
    metrics_type
):
  return lambda s: _serialize_metrics(s, metrics_type)


@beam.typehints.with_output_types(beam.pvalue.PDone)
class WriteMetricsPlotsAndConfig(beam.PTransform):
  """Writes metrics, plots and config to the given path.

  This is the internal implementation. Users should call
  model_eval_lib.WriteMetricsAndPlots instead of this.
  """

  def __init__(self, output_path,
               eval_config):
    self._output_path = output_path
    self._eval_config = eval_config

  def expand(self, metrics_and_plots
            ):
    metrics_output_file = os.path.join(self._output_path, _METRICS_OUTPUT_FILE)
    plots_output_file = os.path.join(self._output_path, _PLOTS_OUTPUT_FILE)
    eval_config_file = os.path.join(self._output_path, _EVAL_CONFIG_FILE)

    metrics, plots = metrics_and_plots
    if metrics.pipeline != plots.pipeline:
      raise ValueError('metrics and plots should come from the same pipeline '
                       'but pipelines were metrics: %s and plots: %s' %
                       (metrics.pipeline, plots.pipeline))
    _ = (
        metrics
        | 'CombineMetricsForWriting' >> beam.CombineGlobally(
            _AccumulateCombineFn(
                _make_serialize_metrics_fn(_METRICS_METRICS_TYPE)))
        | 'WriteMetricsToFile' >> beam.io.WriteToTFRecord(
            metrics_output_file, shard_name_template=''))
    _ = (
        plots
        | 'CombinePlotsForWriting' >> beam.CombineGlobally(
            _AccumulateCombineFn(
                _make_serialize_metrics_fn(_PLOTS_METRICS_TYPE)))
        | 'WritePlotsToFile' >> beam.io.WriteToTFRecord(
            plots_output_file, shard_name_template=''))

    _ = (
        metrics.pipeline
        | 'CreateEvalConfig' >> beam.Create(
            [_serialize_eval_config(self._eval_config)])
        | 'WriteEvalConfig' >> beam.io.WriteToTFRecord(
            eval_config_file, shard_name_template=''))

    return beam.pvalue.PDone(metrics.pipeline)
