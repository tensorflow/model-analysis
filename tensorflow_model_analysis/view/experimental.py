# Copyright 2022 Google LLC
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
"""Experimental output format and util for Tensorflow Model Analysis."""

import collections
from typing import Any, Iterable, Iterator, Optional, Tuple, Union

import pandas as pd
from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.proto import metrics_for_slice_pb2

# Dataframe output columns.
_DISPLAY_VALUE = 'display_value'
_METRIC_VALUE = 'metric_value'
_CONFIDENCE_INTERVAL = 'confidence_interval'
_SLICE = 'slice'


def _flatten_namedtuple(
    key: Union[metric_types.MetricKey, metric_types.SubKey,
               metric_types.AggregationType],
    include_empty_columns: bool) -> Iterator[Tuple[str, Any]]:
  """Flatten the named tuple recursively until non-namedtuple element."""
  for k, v in key._asdict().items():
    # Check the named tuple.
    if isinstance(v, tuple) and hasattr(v, '_fields'):
      yield from _flatten_namedtuple(v, include_empty_columns)
    elif not include_empty_columns:
      if v is not None:
        yield (k, v)
    else:
      yield (k, v)


def _get_slice_value(
    slice_key: metrics_for_slice_pb2.SingleSliceKey
) -> Optional[Union[float, bytes, int]]:
  value_type = slice_key.WhichOneof('kind')
  if value_type == 'float_value':
    return slice_key.float_value
  elif value_type == 'bytes_value':
    return slice_key.bytes_value
  elif value_type == 'int64_value':
    return slice_key.int64_value


def _slice_key_str(slice_key: metrics_for_slice_pb2.SliceKey) -> str:
  return '; '.join(
      f'{single_slice_key.column} = {_get_slice_value(single_slice_key)}'
      for single_slice_key in slice_key.single_slice_keys)


def metrics_as_dataframe(
    metrics_for_slices: Iterable[metrics_for_slice_pb2.MetricsForSlice],
    include_empty_columns: bool = False,
) -> pd.DataFrame:
  """Convert the deserialized MetricForSlice protos to Pandas dataframe.

  Args:
    metrics_for_slices: an interable of MetricForSlice proto.
    include_empty_columns: include a column if its value is not empty (None) in
      corresponding field in the MetricKey.

  Returns:
    A dataframe with the following columns if the value is not None:
    * slice: the string representation of the slice, e.g., "age=10;sex=male".
    * recursively flattened items in metric_key: name, model_name, output_name,
      is_diff, example_weighted, flattened sub_key (class_id, k, top_k),
      flattened aggregation_type (micro_average, macro_average,
      weighted_macro_average).
    * metric_value: a tfma.metrics.MetricValue proto.
    * confidence_interval: a tfma.metrics.ConfidenceInterval proto.
    * display_value: a best effort string reprensentation of the underlying
      metric value and confidence interval if turned on.
  """

  def _get_display_value(val: metrics_for_slice_pb2.MetricValue) -> Any:
    # Best effort to convert the metric value to a concise representation.
    value_type = val.WhichOneof('type')
    if value_type == 'double_value':
      return str(val.double_value.value)
    # TODO(b/230781704): support other types of metric values.
    else:
      return str(val)

  # Create a pd series per column for the dataframe.
  columns = collections.defaultdict(list)
  index = 0
  for metrics_for_slice in metrics_for_slices:
    slice_str = _slice_key_str(metrics_for_slice.slice_key)
    for key_and_value in metrics_for_slice.metric_keys_and_values:
      columns[_SLICE].append((slice_str, index))
      for k, v in _flatten_namedtuple(
          metric_types.MetricKey.from_proto(key_and_value.key),
          include_empty_columns):
        columns[k].append((v, index))
      columns[_DISPLAY_VALUE].append(
          (_get_display_value(key_and_value.value), index))
      columns[_METRIC_VALUE].append((key_and_value.value, index))
      if include_empty_columns:
        if key_and_value.HasField(_CONFIDENCE_INTERVAL):
          columns[_CONFIDENCE_INTERVAL].append(
              (key_and_value.confidence_interval, index))
        else:
          columns[_CONFIDENCE_INTERVAL].append((None, index))
      index += 1

  return pd.DataFrame({
      column_name: pd.Series(*zip(*values))
      for column_name, values in columns.items()
  })


def load_metrics_as_dataframe(
    output_path: str,
    output_file_format: str = 'tfrecord',
    include_empty_columns: bool = False,
) -> pd.DataFrame:
  """Read and deserialize the MetricForSlice records as Pandas dataframe.

  One typical use of this dataframe table is to re-organize it in the form of
  slices vs. metrics table.
  E.g., for single model single output:
    result.pivot(index='slice', columns='name', values='display_value').
  This only works when there is one unique value as the pivot values.
  Otherewise, a user needs to specify more columns or indices to make sure that
  the metric value is unique per column and per index.
  E.g., for single model and multiple outputs:
    result.pivot(index='slice', columns=['output_name', 'name'],
                 values='display_value').
  Args:
    output_path: the directory path of the metrics file.
    output_file_format: the file format of the metrics file, such as tfrecord.
    include_empty_columns: include a column if its value is not empty (None) in
      corresponding field in the MetricKey.

  Returns:
    A dataframe with the following columns if the value is not None:
    * slice: the string representation of the slice, e.g., "age=10;sex=male".
    * recursively flattened items in metric_key: name, model_name, output_name,
      is_diff, example_weighted, flattened sub_key (class_id, k, top_k),
      flattened aggregation_type (micro_average, macro_average,
      weighted_macro_average).
    * metric_value: a tfma.metrics.MetricValue proto.
    * confidence_interval: a tfma.metrics.ConfidenceInterval proto.
    * display_value: a best effort string reprensentation of the underlying
      metric value and confidence interval if turned on.
  """

  return metrics_as_dataframe(
      metrics_for_slices=model_eval_lib.load_metrics(output_path,
                                                     output_file_format),
      include_empty_columns=include_empty_columns)
