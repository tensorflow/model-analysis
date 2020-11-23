# Lint as: python3
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
"""Metrics, plots, and validations writer."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import itertools
import os

from typing import Any, Dict, Iterable, Iterator, List, Optional, Text, Tuple, Union

from absl import logging
import apache_beam as beam
import numpy as np
import pyarrow as pa
import six
import tensorflow as tf
from tensorflow_model_analysis import config
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import math_util
from tensorflow_model_analysis import types
from tensorflow_model_analysis.evaluators import evaluator
from tensorflow_model_analysis.evaluators import metrics_validator
from tensorflow_model_analysis.metrics import metric_specs
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.post_export_metrics import metric_keys
from tensorflow_model_analysis.proto import metrics_for_slice_pb2
from tensorflow_model_analysis.proto import validation_result_pb2
from tensorflow_model_analysis.slicer import slicer_lib as slicer
from tensorflow_model_analysis.writers import writer


_PARQUET_FORMAT = 'parquet'
_TFRECORD_FORMAT = 'tfrecord'
_SUPPORTED_FORMATS = (_PARQUET_FORMAT, _TFRECORD_FORMAT)
_SLICE_KEY_PARQUET_COLUMN_NAME = 'slice_key'
_SERIALIZED_VALUE_PARQUET_COLUMN_NAME = 'serialized_value'
_SINGLE_SLICE_KEYS_PARQUET_FIELD_NAME = 'single_slice_specs'
_SLICE_KEY_ARROW_TYPE = pa.struct([(pa.field(
    _SINGLE_SLICE_KEYS_PARQUET_FIELD_NAME,
    pa.list_(
        pa.struct([
            pa.field('column', pa.string()),
            pa.field('bytes_value', pa.binary()),
            pa.field('float_value', pa.float32()),
            pa.field('int64_value', pa.int64())
        ]))))])
_SLICED_PARQUET_SCHEMA = pa.schema([
    pa.field(_SLICE_KEY_PARQUET_COLUMN_NAME, _SLICE_KEY_ARROW_TYPE),
    pa.field(_SERIALIZED_VALUE_PARQUET_COLUMN_NAME, pa.binary())
])
_UNSLICED_PARQUET_SCHEMA = pa.schema(
    [pa.field(_SERIALIZED_VALUE_PARQUET_COLUMN_NAME, pa.binary())])

_SliceKeyDictPythonType = Dict[Text, List[Dict[Text, Union[bytes, float, int]]]]


def _match_all_files(file_path: Text) -> Text:
  """Return expression to match all files at given path."""
  return file_path + '*'


def _parquet_column_iterator(paths: Iterable[str],
                             column_name: str) -> Iterator[pa.Buffer]:
  """Yields values from a bytes column in a set of parquet file partitions."""
  dataset = pa.parquet.ParquetDataset(paths)
  table = dataset.read(columns=[column_name])
  for record_batch in table.to_batches():
    # always read index 0 because we filter to one column
    value_array = record_batch.column(0)
    for value in value_array:
      yield value.as_buffer()


def _raw_value_iterator(
    paths: Iterable[Text],
    output_file_format: Text) -> Iterator[Union[pa.Buffer, bytes]]:
  """Returns an iterator of raw per-record values from supported file formats.

  When reading parquet format files, values from the column with name
  _SERIALIZED_VALUE_PARQUET_COLUMN_NAME will be read.

  Args:
    paths: The paths from which to read records
    output_file_format: The format of the files from which to read records.

  Returns:
    An iterator which yields serialized values.

  Raises:
    ValueError when the output_file_format is unknown.
  """
  if output_file_format == _PARQUET_FORMAT:
    return _parquet_column_iterator(paths,
                                    _SERIALIZED_VALUE_PARQUET_COLUMN_NAME)
  elif not output_file_format or output_file_format == _TFRECORD_FORMAT:
    return itertools.chain(*(tf.compat.v1.python_io.tf_record_iterator(path)
                             for path in paths))
  raise ValueError('Formats "{}" are currently supported but got '
                   'output_file_format={}'.format(_SUPPORTED_FORMATS,
                                                  output_file_format))


def load_and_deserialize_metrics(
    output_path: Text,
    output_file_format: Text = '',
    slice_specs: Optional[Iterable[slicer.SingleSliceSpec]] = None
) -> Iterator[metrics_for_slice_pb2.MetricsForSlice]:
  """Read and deserialize the MetricsForSlice records.

  Args:
    output_path: Path or pattern to search for metrics files under. If a
      directory is passed, files matching 'metrics*' will be searched for.
    output_file_format: Optional file extension to filter files by.
    slice_specs: A set of SingleSliceSpecs to use for filtering returned
      metrics. The metrics for a given slice key will be returned if that slice
      key matches any of the slice_specs.

  Yields:
    MetricsForSlice protos found in matching files.
  """
  if tf.io.gfile.isdir(output_path):
    output_path = os.path.join(output_path, constants.METRICS_KEY)
  pattern = _match_all_files(output_path)
  if output_file_format:
    pattern = pattern + '.' + output_file_format
  paths = tf.io.gfile.glob(pattern)
  for value in _raw_value_iterator(paths, output_file_format):
    metrics = metrics_for_slice_pb2.MetricsForSlice.FromString(value)
    if slice_specs and not slicer.slice_key_matches_slice_specs(
        slicer.deserialize_slice_key(metrics.slice_key), slice_specs):
      continue
    yield metrics


def load_and_deserialize_plots(
    output_path: Text,
    output_file_format: Text = '',
    slice_specs: Optional[Iterable[slicer.SingleSliceSpec]] = None
) -> Iterator[metrics_for_slice_pb2.PlotsForSlice]:
  """Read and deserialize the PlotsForSlice records.

  Args:
    output_path: Path or pattern to search for plots files under. If a directory
      is passed, files matching 'plots*' will be searched for.
    output_file_format: Optional file extension to filter files by.
    slice_specs: A set of SingleSliceSpecs to use for filtering returned plots.
      The plots for a given slice key will be returned if that slice key matches
      any of the slice_specs.

  Yields:
    PlotsForSlice protos found in matching files.
  """
  if tf.io.gfile.isdir(output_path):
    output_path = os.path.join(output_path, constants.PLOTS_KEY)
  pattern = _match_all_files(output_path)
  if output_file_format:
    pattern = pattern + '.' + output_file_format
  paths = tf.io.gfile.glob(pattern)
  for value in _raw_value_iterator(paths, output_file_format):
    plots = metrics_for_slice_pb2.PlotsForSlice.FromString(value)
    if slice_specs and not slicer.slice_key_matches_slice_specs(
        slicer.deserialize_slice_key(plots.slice_key), slice_specs):
      continue
    yield plots


def load_and_deserialize_attributions(
    output_path: Text,
    output_file_format: Text = '',
    slice_specs: Optional[Iterable[slicer.SingleSliceSpec]] = None
) -> Iterator[metrics_for_slice_pb2.AttributionsForSlice]:
  """Read and deserialize the AttributionsForSlice records.

  Args:
    output_path: Path or pattern to search for attribution files under. If a
      directory is passed, files matching 'attributions*' will be searched for.
    output_file_format: Optional file extension to filter files by.
    slice_specs: A set of SingleSliceSpecs to use for filtering returned
      attributions. The attributions for a given slice key will be returned if
      that slice key matches any of the slice_specs.

  Yields:
    AttributionsForSlice protos found in matching files.
  """
  if tf.io.gfile.isdir(output_path):
    output_path = os.path.join(output_path, constants.ATTRIBUTIONS_KEY)
  pattern = _match_all_files(output_path)
  if output_file_format:
    pattern = pattern + '.' + output_file_format
  paths = tf.io.gfile.glob(pattern)
  for value in _raw_value_iterator(paths, output_file_format):
    attributions = metrics_for_slice_pb2.AttributionsForSlice.FromString(value)
    if slice_specs and not slicer.slice_key_matches_slice_specs(
        slicer.deserialize_slice_key(attributions.slice_key), slice_specs):
      continue
    yield attributions


def load_and_deserialize_validation_result(
    output_path: Text,
    output_file_format: Text = '') -> validation_result_pb2.ValidationResult:
  """Read and deserialize the ValidationResult record.

  Args:
    output_path: Path or pattern to search for validation file under. If a
      directory is passed, a file matching 'validations*' will be searched for.
    output_file_format: Optional file extension to filter file by.

  Returns:
    ValidationResult proto.
  """
  if tf.io.gfile.isdir(output_path):
    output_path = os.path.join(output_path, constants.VALIDATIONS_KEY)
  pattern = _match_all_files(output_path)
  if output_file_format:
    pattern = pattern + '.' + output_file_format
  validation_records = []
  paths = tf.io.gfile.glob(pattern)
  for value in _raw_value_iterator(paths, output_file_format):
    validation_records.append(
        validation_result_pb2.ValidationResult.FromString(value))
  assert len(validation_records) == 1
  return validation_records[0]


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


def convert_slice_metrics_to_proto(
    metrics: Tuple[slicer.SliceKeyOrCrossSliceKeyType, Dict[Any, Any]],
    add_metrics_callbacks: List[types.AddMetricsCallbackType]
) -> metrics_for_slice_pb2.MetricsForSlice:
  """Converts the given slice metrics into serialized proto MetricsForSlice.

  Args:
    metrics: The slice metrics.
    add_metrics_callbacks: A list of metric callbacks. This should be the same
      list as the one passed to tfma.Evaluate().

  Returns:
    The MetricsForSlice proto.

  Raises:
    TypeError: If the type of the feature value in slice key cannot be
      recognized.
  """
  result = metrics_for_slice_pb2.MetricsForSlice()
  slice_key, slice_metrics = metrics

  if slicer.is_cross_slice_key(slice_key):
    result.cross_slice_key.CopyFrom(slicer.serialize_cross_slice_key(slice_key))
  else:
    result.slice_key.CopyFrom(slicer.serialize_slice_key(slice_key))

  slice_metrics = slice_metrics.copy()

  if metric_keys.ERROR_METRIC in slice_metrics:
    logging.warning('Error for slice: %s with error message: %s ', slice_key,
                    slice_metrics[metric_keys.ERROR_METRIC])
    result.metrics[metric_keys.ERROR_METRIC].debug_message = slice_metrics[
        metric_keys.ERROR_METRIC]
    return result

  # Convert the metrics from add_metrics_callbacks to the structured output if
  # defined.
  if add_metrics_callbacks and (not any(
      isinstance(k, metric_types.MetricKey) for k in slice_metrics.keys())):
    for add_metrics_callback in add_metrics_callbacks:
      if hasattr(add_metrics_callback, 'populate_stats_and_pop'):
        add_metrics_callback.populate_stats_and_pop(slice_key, slice_metrics,
                                                    result.metrics)
  for key in sorted(slice_metrics.keys()):
    value = slice_metrics[key]
    metric_value = metrics_for_slice_pb2.MetricValue()
    if isinstance(value, metrics_for_slice_pb2.ConfusionMatrixAtThresholds):
      metric_value.confusion_matrix_at_thresholds.CopyFrom(value)
    elif isinstance(
        value, metrics_for_slice_pb2.MultiClassConfusionMatrixAtThresholds):
      metric_value.multi_class_confusion_matrix_at_thresholds.CopyFrom(value)
    elif isinstance(value, types.ValueWithTDistribution):
      # Currently we populate both bounded_value and confidence_interval.
      # Avoid populating bounded_value once the UI handles confidence_interval.
      # Convert to a bounded value. 95% confidence level is computed here.
      _, lower_bound, upper_bound = (
          math_util.calculate_confidence_interval(value))
      metric_value.bounded_value.value.value = value.unsampled_value
      metric_value.bounded_value.lower_bound.value = lower_bound
      metric_value.bounded_value.upper_bound.value = upper_bound
      metric_value.bounded_value.methodology = (
          metrics_for_slice_pb2.BoundedValue.POISSON_BOOTSTRAP)
      # Populate confidence_interval
      metric_value.confidence_interval.lower_bound.value = lower_bound
      metric_value.confidence_interval.upper_bound.value = upper_bound
      t_dist_value = metrics_for_slice_pb2.TDistributionValue()
      t_dist_value.sample_mean.value = value.sample_mean
      t_dist_value.sample_standard_deviation.value = (
          value.sample_standard_deviation)
      t_dist_value.sample_degrees_of_freedom.value = (
          value.sample_degrees_of_freedom)
      # Once the UI handles confidence interval, we will avoid setting this and
      # instead use the double_value.
      t_dist_value.unsampled_value.value = value.unsampled_value
      metric_value.confidence_interval.t_distribution_value.CopyFrom(
          t_dist_value)
    elif isinstance(value, six.binary_type):
      # Convert textual types to string metrics.
      metric_value.bytes_value = value
    elif isinstance(value, six.text_type):
      # Convert textual types to string metrics.
      metric_value.bytes_value = value.encode('utf8')
    elif isinstance(value, np.ndarray):
      # Convert NumPy arrays to ArrayValue.
      metric_value.array_value.CopyFrom(_convert_to_array_value(value))
    else:
      # We try to convert to float values.
      try:
        metric_value.double_value.value = float(value)
      except (TypeError, ValueError) as e:
        metric_value.unknown_type.value = str(value)
        metric_value.unknown_type.error = e.message  # pytype: disable=attribute-error

    if isinstance(key, metric_types.MetricKey):
      key_and_value = result.metric_keys_and_values.add()
      key_and_value.key.CopyFrom(key.to_proto())
      key_and_value.value.CopyFrom(metric_value)
    else:
      result.metrics[key].CopyFrom(metric_value)

  return result


def convert_slice_plots_to_proto(
    plots: Tuple[slicer.SliceKeyOrCrossSliceKeyType, Dict[Any, Any]],
    add_metrics_callbacks: List[types.AddMetricsCallbackType]
) -> metrics_for_slice_pb2.PlotsForSlice:
  """Converts the given slice plots into PlotsForSlice proto.

  Args:
    plots: The slice plots.
    add_metrics_callbacks: A list of metric callbacks. This should be the same
      list as the one passed to tfma.Evaluate().

  Returns:
    The PlotsForSlice proto.
  """
  result = metrics_for_slice_pb2.PlotsForSlice()
  slice_key, slice_plots = plots

  if slicer.is_cross_slice_key(slice_key):
    result.cross_slice_key.CopyFrom(slicer.serialize_cross_slice_key(slice_key))
  else:
    result.slice_key.CopyFrom(slicer.serialize_slice_key(slice_key))

  slice_plots = slice_plots.copy()

  if metric_keys.ERROR_METRIC in slice_plots:
    logging.warning('Error for slice: %s with error message: %s ', slice_key,
                    slice_plots[metric_keys.ERROR_METRIC])
    error_metric = slice_plots.pop(metric_keys.ERROR_METRIC)
    result.plots[metric_keys.ERROR_METRIC].debug_message = error_metric
    return result

  if add_metrics_callbacks and (not any(
      isinstance(k, metric_types.MetricKey) for k in slice_plots.keys())):
    for add_metrics_callback in add_metrics_callbacks:
      if hasattr(add_metrics_callback, 'populate_plots_and_pop'):
        add_metrics_callback.populate_plots_and_pop(slice_plots, result.plots)
  plots_by_key = {}
  for key in sorted(slice_plots.keys()):
    value = slice_plots[key]
    # Remove plot name from key (multiple plots are combined into a single
    # proto).
    if isinstance(key, metric_types.MetricKey):
      parent_key = key._replace(name=None)
    else:
      continue
    if parent_key not in plots_by_key:
      key_and_value = result.plot_keys_and_values.add()
      key_and_value.key.CopyFrom(parent_key.to_proto())
      plots_by_key[parent_key] = key_and_value.value

    if isinstance(value, metrics_for_slice_pb2.CalibrationHistogramBuckets):
      plots_by_key[parent_key].calibration_histogram_buckets.CopyFrom(value)
      slice_plots.pop(key)
    elif isinstance(value, metrics_for_slice_pb2.ConfusionMatrixAtThresholds):
      plots_by_key[parent_key].confusion_matrix_at_thresholds.CopyFrom(value)
      slice_plots.pop(key)
    elif isinstance(
        value, metrics_for_slice_pb2.MultiClassConfusionMatrixAtThresholds):
      plots_by_key[
          parent_key].multi_class_confusion_matrix_at_thresholds.CopyFrom(value)
      slice_plots.pop(key)
    elif isinstance(
        value, metrics_for_slice_pb2.MultiLabelConfusionMatrixAtThresholds):
      plots_by_key[
          parent_key].multi_label_confusion_matrix_at_thresholds.CopyFrom(value)
      slice_plots.pop(key)

  if slice_plots:
    if add_metrics_callbacks is None:
      add_metrics_callbacks = []
    raise NotImplementedError(
        'some plots were not converted or popped. keys: %s. '
        'add_metrics_callbacks were: %s' % (
            slice_plots.keys(),
            [
                x.name for x in add_metrics_callbacks  # pytype: disable=attribute-error
            ]))

  return result


def convert_slice_attributions_to_proto(
    attributions: Tuple[slicer.SliceKeyType, Dict[Any, Dict[Text, Any]]]
) -> metrics_for_slice_pb2.AttributionsForSlice:
  """Converts the given slice attributions into serialized AtributionsForSlice.

  Args:
    attributions: The slice attributions.

  Returns:
    The AttributionsForSlice proto.

  Raises:
    TypeError: If the type of the feature value in slice key cannot be
      recognized.
  """
  result = metrics_for_slice_pb2.AttributionsForSlice()
  slice_key, slice_attributions = attributions

  result.slice_key.CopyFrom(slicer.serialize_slice_key(slice_key))

  slice_attributions = slice_attributions.copy()
  for key in sorted(slice_attributions.keys()):
    key_and_value = result.attributions_keys_and_values.add()
    key_and_value.key.CopyFrom(key.to_proto())
    for feature, value in slice_attributions[key].items():
      attribution_value = metrics_for_slice_pb2.MetricValue()
      if isinstance(value, six.binary_type):
        # Convert textual types to string metrics.
        attribution_value.bytes_value = value
      elif isinstance(value, six.text_type):
        # Convert textual types to string metrics.
        attribution_value.bytes_value = value.encode('utf8')
      elif isinstance(value, np.ndarray) and value.size != 1:
        # Convert NumPy arrays to ArrayValue.
        attribution_value.array_value.CopyFrom(_convert_to_array_value(value))
      else:
        # We try to convert to float values.
        try:
          attribution_value.double_value.value = float(value)
        except (TypeError, ValueError) as e:
          attribution_value.unknown_type.value = str(value)
          attribution_value.unknown_type.error = e.message  # pytype: disable=attribute-error
      key_and_value.values[feature].CopyFrom(attribution_value)

  return result


def MetricsPlotsAndValidationsWriter(  # pylint: disable=invalid-name
    output_paths: Dict[Text, Text],
    eval_config: config.EvalConfig,
    add_metrics_callbacks: Optional[List[types.AddMetricsCallbackType]] = None,
    metrics_key: Text = constants.METRICS_KEY,
    plots_key: Text = constants.PLOTS_KEY,
    attributions_key: Text = constants.ATTRIBUTIONS_KEY,
    validations_key: Text = constants.VALIDATIONS_KEY,
    output_file_format: Text = '',
    rubber_stamp: Optional[bool] = False) -> writer.Writer:
  """Returns metrics and plots writer.

  Note, sharding will be enabled by default if a output_file_format is provided.
  The files will be named <output_path>-SSSSS-of-NNNNN.<output_file_format>
  where SSSSS is the shard number and NNNNN is the number of shards.

  Args:
    output_paths: Output paths keyed by output key (e.g. 'metrics', 'plots',
      'validation').
    eval_config: Eval config.
    add_metrics_callbacks: Optional list of metric callbacks (if used).
    metrics_key: Name to use for metrics key in Evaluation output.
    plots_key: Name to use for plots key in Evaluation output.
    attributions_key: Name to use for attributions key in Evaluation output.
    validations_key: Name to use for validations key in Evaluation output.
    output_file_format: File format to use when saving files. Currently
      'tfrecord' and 'parquet' are supported. If using parquet, the output
      metrics and plots files will contain two columns: 'slice_key' and
      'serialized_value'. The 'slice_key' column will be a structured column
      matching the metrics_for_slice_pb2.SliceKey proto. the 'serialized_value'
      column will contain a serialized MetricsForSlice or PlotsForSlice
      proto. The validation result file will contain a single column
      'serialized_value' which will contain a single serialized ValidationResult
      proto.
    rubber_stamp: True if this model is being rubber stamped. When a model is
      rubber stamped diff thresholds will be ignored if an associated baseline
      model is not passed.
  """
  return writer.Writer(
      stage_name='WriteMetricsAndPlots',
      ptransform=_WriteMetricsPlotsAndValidations(  # pylint: disable=no-value-for-parameter
          output_paths=output_paths,
          eval_config=eval_config,
          add_metrics_callbacks=add_metrics_callbacks or [],
          metrics_key=metrics_key,
          plots_key=plots_key,
          attributions_key=attributions_key,
          validations_key=validations_key,
          output_file_format=output_file_format,
          rubber_stamp=rubber_stamp))


@beam.typehints.with_input_types(validation_result_pb2.ValidationResult)
@beam.typehints.with_output_types(validation_result_pb2.ValidationResult)
class CombineValidations(beam.CombineFn):
  """Combines the ValidationResults protos.

  Combines PCollection of ValidationResults for different metrics and slices.
  """

  def __init__(self,
               eval_config: config.EvalConfig,
               rubber_stamp: bool = False):
    self._eval_config = eval_config
    self._rubber_stamp = rubber_stamp

  def create_accumulator(self) -> None:
    return

  def add_input(
      self, result: 'Optional[validation_result_pb2.ValidationResult]',
      new_input: 'Optional[validation_result_pb2.ValidationResult]'
  ) -> 'Optional[validation_result_pb2.ValidationResult]':
    if new_input is None:
      return None
    if result is None:
      result = validation_result_pb2.ValidationResult(validation_ok=True)
    result.validation_ok &= new_input.validation_ok
    result.metric_validations_per_slice.extend(
        new_input.metric_validations_per_slice)
    metrics_validator.merge_details(result, new_input)
    return result

  def merge_accumulators(
      self,
      accumulators: 'List[Optional[validation_result_pb2.ValidationResult]]'
  ) -> 'Optional[validation_result_pb2.ValidationResult]':
    accumulators = [accumulator for accumulator in accumulators if accumulator]
    if not accumulators:
      return None
    result = validation_result_pb2.ValidationResult(validation_ok=True)
    for new_input in accumulators:
      result.metric_validations_per_slice.extend(
          new_input.metric_validations_per_slice)
      metrics_validator.merge_details(result, new_input)
      result.validation_ok &= new_input.validation_ok
    return result

  def extract_output(
      self, accumulator: 'Optional[validation_result_pb2.ValidationResult]'
  ) -> 'Optional[validation_result_pb2.ValidationResult]':
    # Verification fails if there is empty input.
    if not accumulator:
      accumulator = validation_result_pb2.ValidationResult(validation_ok=False)
    thresholds = metric_specs.metric_thresholds_from_metrics_specs(
        self._eval_config.metrics_specs)
    if not thresholds:
      # Default is to validation NOT ok when not rubber stamping.
      accumulator.validation_ok = self._rubber_stamp
      # Default is to missing thresholds when not rubber stamping.
      accumulator.missing_thresholds = not self._rubber_stamp
    missing = metrics_validator.get_missing_slices(
        accumulator.validation_details.slicing_details, self._eval_config)
    if missing:
      missing_slices = []
      missing_cross_slices = []
      for m in missing:
        if isinstance(m, config.SlicingSpec):
          missing_slices.append(m)
        elif isinstance(m, config.CrossSlicingSpec):
          missing_cross_slices.append(m)
      accumulator.validation_ok = False
      if missing_slices:
        accumulator.missing_slices.extend(missing_slices)
      if missing_cross_slices:
        accumulator.missing_cross_slices.extend(missing_cross_slices)
    return accumulator


@beam.ptransform_fn
# TODO(b/157600974): Add typehint.
@beam.typehints.with_output_types(beam.pvalue.PDone)
def _WriteMetricsPlotsAndValidations(  # pylint: disable=invalid-name
    evaluation: evaluator.Evaluation,
    output_paths: Dict[Text, Text],
    eval_config: config.EvalConfig,
    add_metrics_callbacks: List[types.AddMetricsCallbackType],
    metrics_key: Text,
    plots_key: Text,
    attributions_key: Text,
    validations_key: Text,
    output_file_format: Text,
    rubber_stamp: bool = False) -> beam.pvalue.PDone:
  """PTransform to write metrics and plots."""

  if output_file_format and output_file_format not in _SUPPORTED_FORMATS:
    raise ValueError('only "{}" formats are currently supported but got '
                     'output_file_format={}'.format(_SUPPORTED_FORMATS,
                                                    output_file_format))

  def convert_slice_key_to_parquet_dict(
      slice_key: metrics_for_slice_pb2.SliceKey) -> _SliceKeyDictPythonType:
    single_slice_key_dicts = []
    for single_slice_key in slice_key.single_slice_keys:
      kind = single_slice_key.WhichOneof('kind')
      if not kind:
        continue
      single_slice_key_dicts.append({kind: getattr(single_slice_key, kind)})
    return {_SINGLE_SLICE_KEYS_PARQUET_FIELD_NAME: single_slice_key_dicts}

  def convert_to_parquet_columns(
      value: Union[metrics_for_slice_pb2.MetricsForSlice,
                   metrics_for_slice_pb2.PlotsForSlice,
                   metrics_for_slice_pb2.AttributionsForSlice]
  ) -> Dict[Text, Union[_SliceKeyDictPythonType, bytes]]:
    return {
        _SLICE_KEY_PARQUET_COLUMN_NAME:
            convert_slice_key_to_parquet_dict(value.slice_key),
        _SERIALIZED_VALUE_PARQUET_COLUMN_NAME:
            value.SerializeToString()
    }

  if metrics_key in evaluation and constants.METRICS_KEY in output_paths:
    metrics = (
        evaluation[metrics_key] | 'ConvertSliceMetricsToProto' >> beam.Map(
            convert_slice_metrics_to_proto,
            add_metrics_callbacks=add_metrics_callbacks))

    file_path_prefix = output_paths[constants.METRICS_KEY]
    if output_file_format == _PARQUET_FORMAT:
      _ = (
          metrics
          | 'ConvertToParquetColumns' >> beam.Map(convert_to_parquet_columns)
          | 'WriteMetricsToParquet' >> beam.io.WriteToParquet(
              file_path_prefix=file_path_prefix,
              schema=_SLICED_PARQUET_SCHEMA,
              file_name_suffix='.' + output_file_format))
    elif not output_file_format or output_file_format == _TFRECORD_FORMAT:
      _ = metrics | 'WriteMetrics' >> beam.io.WriteToTFRecord(
          file_path_prefix=file_path_prefix,
          shard_name_template=None if output_file_format else '',
          file_name_suffix=('.' +
                            output_file_format if output_file_format else ''),
          coder=beam.coders.ProtoCoder(metrics_for_slice_pb2.MetricsForSlice))

  if plots_key in evaluation and constants.PLOTS_KEY in output_paths:
    plots = (
        evaluation[plots_key] | 'ConvertSlicePlotsToProto' >> beam.Map(
            convert_slice_plots_to_proto,
            add_metrics_callbacks=add_metrics_callbacks))

    file_path_prefix = output_paths[constants.PLOTS_KEY]
    if output_file_format == _PARQUET_FORMAT:
      _ = (
          plots
          |
          'ConvertPlotsToParquetColumns' >> beam.Map(convert_to_parquet_columns)
          | 'WritePlotsToParquet' >> beam.io.WriteToParquet(
              file_path_prefix=file_path_prefix,
              schema=_SLICED_PARQUET_SCHEMA,
              file_name_suffix='.' + output_file_format))
    elif not output_file_format or output_file_format == _TFRECORD_FORMAT:
      _ = plots | 'WritePlotsToTFRecord' >> beam.io.WriteToTFRecord(
          file_path_prefix=file_path_prefix,
          shard_name_template=None if output_file_format else '',
          file_name_suffix=('.' +
                            output_file_format if output_file_format else ''),
          coder=beam.coders.ProtoCoder(metrics_for_slice_pb2.PlotsForSlice))

  if (attributions_key in evaluation and
      constants.ATTRIBUTIONS_KEY in output_paths):
    attributions = (
        evaluation[attributions_key] | 'ConvertSliceAttributionsToProto' >>
        beam.Map(convert_slice_attributions_to_proto))

    file_path_prefix = output_paths[constants.ATTRIBUTIONS_KEY]
    if output_file_format == _PARQUET_FORMAT:
      _ = (
          attributions
          | 'ConvertAttributionsToParquetColumns' >>
          beam.Map(convert_to_parquet_columns)
          | 'WriteAttributionsToParquet' >> beam.io.WriteToParquet(
              file_path_prefix=file_path_prefix,
              schema=_SLICED_PARQUET_SCHEMA,
              file_name_suffix='.' + output_file_format))
    elif not output_file_format or output_file_format == _TFRECORD_FORMAT:
      _ = attributions | 'WriteAttributionsToTFRecord' >> beam.io.WriteToTFRecord(
          file_path_prefix=file_path_prefix,
          shard_name_template=None if output_file_format else '',
          file_name_suffix=('.' +
                            output_file_format if output_file_format else ''),
          coder=beam.coders.ProtoCoder(
              metrics_for_slice_pb2.AttributionsForSlice))

  if (validations_key in evaluation and
      constants.VALIDATIONS_KEY in output_paths):
    validations = (
        evaluation[validations_key]
        | 'MergeValidationResults' >> beam.CombineGlobally(
            CombineValidations(eval_config, rubber_stamp=rubber_stamp)))

    file_path_prefix = output_paths[constants.VALIDATIONS_KEY]
    # We only use a single shard here because validations are usually single
    # values. Setting the shard_name_template to the empty string forces this.
    shard_name_template = ''
    if output_file_format == _PARQUET_FORMAT:
      _ = (
          validations
          | 'ConvertValidationsToParquetColumns' >> beam.Map(
              lambda v:  # pylint: disable=g-long-lambda
              {_SERIALIZED_VALUE_PARQUET_COLUMN_NAME: v.SerializeToString()})
          | 'WriteValidationsToParquet' >> beam.io.WriteToParquet(
              file_path_prefix=file_path_prefix,
              shard_name_template=shard_name_template,
              schema=_UNSLICED_PARQUET_SCHEMA,
              file_name_suffix='.' + output_file_format))
    elif not output_file_format or output_file_format == _TFRECORD_FORMAT:
      _ = (
          validations
          | 'WriteValidationsToTFRecord' >> beam.io.WriteToTFRecord(
              file_path_prefix=file_path_prefix,
              shard_name_template=shard_name_template,
              file_name_suffix=('.' + output_file_format
                                if output_file_format else ''),
              coder=beam.coders.ProtoCoder(
                  validation_result_pb2.ValidationResult)))

  return beam.pvalue.PDone(list(evaluation.values())[0].pipeline)
