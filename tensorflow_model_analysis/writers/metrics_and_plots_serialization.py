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

from typing import Any, Dict, List, Text, Tuple


# The desired output type is
# List[Tuple[slicer.SliceKeyType,
# protobuf.python.google.internal.containers.MessageMap[Union[str, unicode],
# metrics_for_slice_pb2.MetricValue]], while the metrics type isn't visible to
# this module.
def load_and_deserialize_metrics(path: Text
                                ) -> List[Tuple[slicer.SliceKeyType, Any]]:
  result = []
  for record in tf.compat.v1.python_io.tf_record_iterator(path):
    metrics_for_slice = metrics_for_slice_pb2.MetricsForSlice.FromString(record)
    result.append((
        slicer.deserialize_slice_key(metrics_for_slice.slice_key),  # pytype: disable=wrong-arg-types
        metrics_for_slice.metrics))
  return result


def load_and_deserialize_plots(path: Text
                              ) -> List[Tuple[slicer.SliceKeyType, Any]]:
  """Returns deserialized plots loaded from given path."""
  result = []
  for record in tf.compat.v1.python_io.tf_record_iterator(path):
    plots_for_slice = metrics_for_slice_pb2.PlotsForSlice.FromString(record)
    if plots_for_slice.HasField('plot_data'):
      if plots_for_slice.plots:
        raise RuntimeError('Both plots and plot_data are set.')

      # For backward compatibility, plots data geneated with old code are added
      # to the plots map with default key empty string.
      plots_for_slice.plots[''].CopyFrom(plots_for_slice.plot_data)

    result.append((
        slicer.deserialize_slice_key(plots_for_slice.slice_key),  # pytype: disable=wrong-arg-types
        plots_for_slice.plots))
  return result


def _convert_to_array_value(array: np.ndarray
                           ) -> metrics_for_slice_pb2.ArrayValue:
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
    slice_metrics: Dict[Text, Any],
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
      post_export_metric.populate_stats_and_pop(slice_metrics_copy,
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


def _serialize_metrics(metrics: Tuple[slicer.SliceKeyType, Dict[Text, Any]],
                       post_export_metrics: List[types.AddMetricsCallbackType]
                      ) -> bytes:
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
  convert_slice_metrics(slice_metrics, post_export_metrics, result)
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


def _serialize_plots(plots: Tuple[slicer.SliceKeyType, Dict[Text, Any]],
                     post_export_metrics: List[types.AddMetricsCallbackType]
                    ) -> bytes:
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

  def expand(
      self,
      metrics_and_plots: Tuple[beam.pvalue.PCollection, beam.pvalue.PCollection]
  ):
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
