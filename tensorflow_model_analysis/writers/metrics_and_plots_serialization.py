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
"""Serialization functions for metrics and plots."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

# Standard Imports

import apache_beam as beam

import numpy as np
import six
import tensorflow as tf

from tensorflow_model_analysis import math_util
from tensorflow_model_analysis import types
from tensorflow_model_analysis.post_export_metrics import metric_keys
from tensorflow_model_analysis.proto import metrics_for_slice_pb2
from tensorflow_model_analysis.slicer import slicer

from typing import Any, Dict, List, Optional, Text, Tuple

from google.protobuf import json_format


# The input proto_map is a google.protobuf.internal.containers.MessageMap where
# the keys are strings and the values are some protocol buffer field. Note that
# MessageMap is not a protobuf message, none of the exising utility methods work
# on it. We must iterate over its values and call the utility function
# individually.
def _convert_proto_map_to_dict(proto_map: Any) -> Dict[Text, Dict[Text, Any]]:
  """Converts a metric map (metrics in MetricsForSlice protobuf) into a dict.

  Args:
    proto_map: A protocol buffer MessageMap that has behaviors like dict. The
      keys are strings while the values are protocol buffers. However, it is not
      a protobuf message and cannot be passed into json_format.MessageToDict
      directly. Instead, we must iterate over its values.

  Returns:
    A dict representing the proto_map. For example:
    Assume myProto contains
    {
      metrics: {
        key: 'double'
        value: {
          double_value: {
            value: 1.0
          }
        }
      }
      metrics: {
        key: 'bounded'
        value: {
          bounded_value: {
            lower_bound: {
              double_value: {
                value: 0.8
              }
            }
            upper_bound: {
              double_value: {
                value: 0.9
              }
            }
            value: {
              double_value: {
                value: 0.86
              }
            }
          }
        }
      }
    }

    The output of _convert_proto_map_to_dict(myProto.metrics) would be

    {
      'double': {
        'doubleValue': 1.0,
      },
      'bounded': {
        'boundedValue': {
          'lowerBound': 0.8,
          'upperBound': 0.9,
          'value': 0.86,
        },
      },
    }

    Note that field names are converted to lowerCamelCase and the field value in
    google.protobuf.DoubleValue is collapsed automatically.
  """
  return {k: json_format.MessageToDict(proto_map[k]) for k in proto_map}


def _get_multi_class_key_id(multi_class_key):
  if multi_class_key.HasField('class_id'):
    return 'classId:' + str(multi_class_key.class_id)
  elif multi_class_key.HasField('top_k'):
    return 'topK:' + str(multi_class_key.top_k)
  elif multi_class_key.HasField('k'):
    return 'k:' + str(multi_class_key.k)


def load_and_deserialize_metrics(
    path: Text,
    model_name: Optional[Text] = None) -> List[Tuple[slicer.SliceKeyType, Any]]:
  """Loads metrics from the given location and builds a metric map for it."""
  result = []
  for record in tf.compat.v1.python_io.tf_record_iterator(path):
    metrics_for_slice = metrics_for_slice_pb2.MetricsForSlice.FromString(record)

    model_metrics_map = {}
    if metrics_for_slice.metrics:
      model_metrics_map[''] = {
          '': {
              '': _convert_proto_map_to_dict(metrics_for_slice.metrics)
          }
      }

    if metrics_for_slice.metric_keys:
      for key, value in zip(metrics_for_slice.metric_keys,
                            metrics_for_slice.metric_values):
        current_model_name = key.model_name

        if current_model_name not in model_metrics_map:
          model_metrics_map[current_model_name] = {}
        output_name = key.output_name
        if output_name not in model_metrics_map[current_model_name]:
          model_metrics_map[current_model_name][output_name] = {}

        multi_class_metrics_map = model_metrics_map[current_model_name][
            output_name]
        multi_class_key_id = _get_multi_class_key_id(
            key.multi_class_key) if key.HasField('multi_class_key') else ''
        if multi_class_key_id not in multi_class_metrics_map:
          multi_class_metrics_map[multi_class_key_id] = {}
        metric_name = key.name
        multi_class_metrics_map[multi_class_key_id][
            metric_name] = json_format.MessageToDict(value)

    metrics_map = None
    keys = list(model_metrics_map.keys())
    if model_name in model_metrics_map:
      # Use the provided model name if there is a match.
      metrics_map = model_metrics_map[model_name]
    elif model_name is None and len(keys) == 1:
      # Show result of the only model if no model name is specified.
      metrics_map = model_metrics_map[keys[0]]
    else:
      # No match found.
      raise ValueError('Fail to find metrics for model name: %s . '
                       'Available model names are [%s]' %
                       (model_name, ', '.join(keys)))

    result.append((
        slicer.deserialize_slice_key(metrics_for_slice.slice_key),  # pytype: disable=wrong-arg-types
        metrics_map))
  return result


def load_and_deserialize_plots(
    path: Text) -> List[Tuple[slicer.SliceKeyType, Any]]:
  """Returns deserialized plots loaded from given path."""
  result = []
  for record in tf.compat.v1.python_io.tf_record_iterator(path):
    plots_for_slice = metrics_for_slice_pb2.PlotsForSlice.FromString(record)
    plots_map = {}
    if plots_for_slice.plots:
      plot_dict = _convert_proto_map_to_dict(plots_for_slice.plots)
      keys = list(plot_dict.keys())
      # If there is only one label, choose it automatically.
      plot_data = plot_dict[keys[0]] if len(keys) == 1 else plot_dict
      plots_map[''] = {'': plot_data}
    elif plots_for_slice.HasField('plot_data'):
      plots_map[''] = {'': json_format.MessageToDict(plots_for_slice.plot_data)}

    if plots_for_slice.plot_keys:
      for key, value in zip(plots_for_slice.plot_keys,
                            plots_for_slice.plot_values):
        output_name = key.output_name
        if output_name not in plots_map:
          plots_map[output_name] = {}
        multi_class_key_id = _get_multi_class_key_id(
            key.multi_class_key) if key.HasField('multi_class_key') else ''
        plots_map[output_name][multi_class_key_id] = json_format.MessageToDict(
            value)

    result.append((
        slicer.deserialize_slice_key(plots_for_slice.slice_key),  # pytype: disable=wrong-arg-types
        plots_map))

  return result


def _convert_to_array_value(
    array: np.ndarray) -> metrics_for_slice_pb2.ArrayValue:
  """Converts NumPy array to ArrayValue."""
  result = metrics_for_slice_pb2.ArrayValue()
  result.shape[:] = array.shape
  if array.dtype == 'int32':
    result.data_type = metrics_for_slice_pb2.ArrayValue.INT32
    result.int32_values[:] = array.flatten()
  elif array.dtype == 'int64':
    result.data_type = metrics_for_slice_pb2.ArrayValue.INT64
    result.int64_values[:] = array.flatten()
  elif array.dtype == 'float32':
    result.data_type = metrics_for_slice_pb2.ArrayValue.FLOAT32
    result.float32_values[:] = array.flatten()
  elif array.dtype == 'float64':
    result.data_type = metrics_for_slice_pb2.ArrayValue.FLOAT64
    result.float64_values[:] = array.flatten()
  else:
    # For all other types, cast to string and convert to bytes.
    result.data_type = metrics_for_slice_pb2.ArrayValue.BYTES
    result.bytes_values[:] = [
        tf.compat.as_bytes(x) for x in array.astype(six.text_type).flatten()
    ]
  return result


def convert_slice_metrics(
    slice_key: slicer.SliceKeyType, slice_metrics: Dict[Text, Any],
    post_export_metrics: List[types.AddMetricsCallbackType],
    metrics_for_slice: metrics_for_slice_pb2.MetricsForSlice) -> None:
  """Converts slice_metrics into the given metrics_for_slice proto."""

  slice_metrics_copy = slice_metrics.copy()
  # Prevent further references to this, so we don't accidentally mutate it.
  del slice_metrics

  # Convert the metrics from post_export_metrics to the structured output if
  # defined.
  for post_export_metric in post_export_metrics:
    if hasattr(post_export_metric, 'populate_stats_and_pop'):
      post_export_metric.populate_stats_and_pop(slice_key, slice_metrics_copy,
                                                metrics_for_slice.metrics)
  for name, value in slice_metrics_copy.items():
    if isinstance(value, types.ValueWithTDistribution):
      # Convert to a bounded value. 95% confidence level is computed here.
      # Will populate t distribution value instead after migration.
      sample_mean, lower_bound, upper_bound = math_util.calculate_confidence_interval(
          value)
      metrics_for_slice.metrics[name].bounded_value.value.value = sample_mean
      metrics_for_slice.metrics[
          name].bounded_value.lower_bound.value = lower_bound
      metrics_for_slice.metrics[
          name].bounded_value.upper_bound.value = upper_bound
      metrics_for_slice.metrics[name].bounded_value.methodology = (
          metrics_for_slice_pb2.BoundedValue.POISSON_BOOTSTRAP)
    elif isinstance(value, (six.binary_type, six.text_type)):
      # Convert textual types to string metrics.
      metrics_for_slice.metrics[name].bytes_value = value
    elif isinstance(value, np.ndarray):
      # Convert NumPy arrays to ArrayValue.
      metrics_for_slice.metrics[name].array_value.CopyFrom(
          _convert_to_array_value(value))
    else:
      # We try to convert to float values.
      try:
        metrics_for_slice.metrics[name].double_value.value = float(value)
      except (TypeError, ValueError) as e:
        metrics_for_slice.metrics[name].unknown_type.value = str(value)
        metrics_for_slice.metrics[name].unknown_type.error = e.message  # pytype: disable=attribute-error


def _serialize_metrics(
    metrics: Tuple[slicer.SliceKeyType, Dict[Text, Any]],
    post_export_metrics: List[types.AddMetricsCallbackType]) -> bytes:
  """Converts the given slice metrics into serialized proto MetricsForSlice.

  Args:
    metrics: The slice metrics.
    post_export_metrics: A list of metric callbacks. This should be the same
      list as the one passed to tfma.Evaluate().

  Returns:
    The serialized proto MetricsForSlice.

  Raises:
    TypeError: If the type of the feature value in slice key cannot be
      recognized.
  """
  result = metrics_for_slice_pb2.MetricsForSlice()
  slice_key, slice_metrics = metrics

  if metric_keys.ERROR_METRIC in slice_metrics:
    tf.compat.v1.logging.warning('Error for slice: %s with error message: %s ',
                                 slice_key,
                                 slice_metrics[metric_keys.ERROR_METRIC])
    metrics = metrics_for_slice_pb2.MetricsForSlice()
    metrics.slice_key.CopyFrom(slicer.serialize_slice_key(slice_key))
    metrics.metrics[metric_keys.ERROR_METRIC].debug_message = slice_metrics[
        metric_keys.ERROR_METRIC]
    return metrics.SerializeToString()

  # Convert the slice key.
  result.slice_key.CopyFrom(slicer.serialize_slice_key(slice_key))

  # Convert the slice metrics.
  convert_slice_metrics(slice_key, slice_metrics, post_export_metrics, result)
  return result.SerializeToString()


def _convert_slice_plots(
    slice_plots: Dict[Text, Any],
    post_export_metrics: List[types.AddMetricsCallbackType],
    plot_data: Dict[Text, metrics_for_slice_pb2.PlotData]):
  """Converts slice_plots into the given plot_data proto."""
  slice_plots_copy = slice_plots.copy()
  # Prevent further references to this, so we don't accidentally mutate it.
  del slice_plots

  for post_export_metric in post_export_metrics:
    if hasattr(post_export_metric, 'populate_plots_and_pop'):
      post_export_metric.populate_plots_and_pop(slice_plots_copy, plot_data)
  if slice_plots_copy:
    raise NotImplementedError(
        'some plots were not converted or popped. keys: %s. post_export_metrics'
        'were: %s' % (
            slice_plots_copy.keys(),
            [
                x.name for x in post_export_metrics  # pytype: disable=attribute-error
            ]))


def _serialize_plots(
    plots: Tuple[slicer.SliceKeyType, Dict[Text, Any]],
    post_export_metrics: List[types.AddMetricsCallbackType]) -> bytes:
  """Converts the given slice plots into serialized proto PlotsForSlice..

  Args:
    plots: The slice plots.
    post_export_metrics: A list of metric callbacks. This should be the same
      list as the one passed to tfma.Evaluate().

  Returns:
    The serialized proto PlotsForSlice.
  """
  result = metrics_for_slice_pb2.PlotsForSlice()
  slice_key, slice_plots = plots

  if metric_keys.ERROR_METRIC in slice_plots:
    tf.compat.v1.logging.warning('Error for slice: %s with error message: %s ',
                                 slice_key,
                                 slice_plots[metric_keys.ERROR_METRIC])
    metrics = metrics_for_slice_pb2.PlotsForSlice()
    metrics.slice_key.CopyFrom(slicer.serialize_slice_key(slice_key))
    metrics.plots[metric_keys.ERROR_METRIC].debug_message = slice_plots[
        metric_keys.ERROR_METRIC]
    return metrics.SerializeToString()

  # Convert the slice key.
  result.slice_key.CopyFrom(slicer.serialize_slice_key(slice_key))

  # Convert the slice plots.
  _convert_slice_plots(slice_plots, post_export_metrics, result.plots)  # pytype: disable=wrong-arg-types

  return result.SerializeToString()


class SerializeMetrics(beam.PTransform):  # pylint: disable=invalid-name
  """Converts metrics to serialized protos."""

  def __init__(self, post_export_metrics: List[types.AddMetricsCallbackType]):
    self._post_export_metrics = post_export_metrics

  def expand(self, metrics: beam.pvalue.PCollection):
    """Converts the given metrics into serialized proto.

    Args:
      metrics: PCollection of (slice key, slice metrics).

    Returns:
      PCollection of serialized proto MetricsForSlice.
    """
    metrics = metrics | 'SerializeMetrics' >> beam.Map(
        _serialize_metrics, post_export_metrics=self._post_export_metrics)
    return metrics


class SerializePlots(beam.PTransform):  # pylint: disable=invalid-name
  """Converts plots serialized protos."""

  def __init__(self, post_export_metrics: List[types.AddMetricsCallbackType]):
    self._post_export_metrics = post_export_metrics

  def expand(self, plots: beam.pvalue.PCollection):
    """Converts the given plots into serialized proto.

    Args:
      plots: PCollection of (slice key, slice plots).

    Returns:
      PCollection of serialized proto MetricsForSlice.
    """
    plots = plots | 'SerializePlots' >> beam.Map(
        _serialize_plots, post_export_metrics=self._post_export_metrics)
    return plots


# No typehint for input or output, since it's a multi-output DoFn result that
# Beam doesn't support typehints for yet (BEAM-3280).
class SerializeMetricsAndPlots(beam.PTransform):  # pylint: disable=invalid-name
  """Converts metrics and plots into serialized protos."""

  def __init__(self, post_export_metrics: List[types.AddMetricsCallbackType]):
    self._post_export_metrics = post_export_metrics

  def expand(self, metrics_and_plots: Tuple[beam.pvalue.PCollection,
                                            beam.pvalue.PCollection]):
    """Converts the given metrics_and_plots into serialized proto.

    Args:
      metrics_and_plots: A Tuple of (slice metrics, slice plots).

    Returns:
      A Tuple of PCollection of Serialized proto MetricsForSlice.
    """
    metrics, plots = metrics_and_plots
    metrics = metrics | 'SerializeMetrics' >> beam.Map(
        _serialize_metrics, post_export_metrics=self._post_export_metrics)
    plots = plots | 'SerializePlots' >> beam.Map(
        _serialize_plots, post_export_metrics=self._post_export_metrics)
    return (metrics, plots)
