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
"""Pandas DataFrame utils for Tensorflow Model Analysis."""

import collections
import itertools
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import attr
import numpy as np
import pandas as pd
from tensorflow_model_analysis.proto import metrics_for_slice_pb2

from google.protobuf import wrappers_pb2
from google.protobuf import descriptor
from google.protobuf import message

MetricsForSlice = metrics_for_slice_pb2.MetricsForSlice
PlotsForSlice = metrics_for_slice_pb2.PlotsForSlice

_OVERALL = 'Overall'
# DataFrame output columns.
_METRIC_VALUES = 'metric_values'
_PLOT_DATA = 'plot_data'
_SLICE_STR = 'stringified_slices'
_SLICES = 'slices'
_METRIC_KEYS = 'metric_keys'
_PLOT_KEYS = 'plot_keys'


@attr.define
class _ColumnData:
  metric_keys: Dict[str, List[Tuple[Any, int]]] = attr.Factory(
      lambda: collections.defaultdict(list))
  values: Dict[str, List[Tuple[Any, int]]] = attr.Factory(
      lambda: collections.defaultdict(list))
  slices: Dict[str, List[Tuple[Any, int]]] = attr.Factory(
      lambda: collections.defaultdict(list))


@attr.frozen
class MetricsDataFrames:
  double_value: Optional[pd.DataFrame] = None
  confusion_matrix_at_thresholds: Optional[pd.DataFrame] = None
  multi_class_confusion_matrix_at_thresholds: Optional[pd.DataFrame] = None
  bytes_value: Optional[pd.DataFrame] = None
  array_value: Optional[pd.DataFrame] = None


@attr.frozen
class PlotsDataFrames:
  calibration_histogram_buckets: pd.DataFrame = None
  confusion_matrix_at_thresholds: pd.DataFrame = None
  multi_class_confusion_matrix_at_thresholds: pd.DataFrame = None
  multi_label_confusion_matrix_at_thresholds: pd.DataFrame = None
  debug_message: pd.DataFrame = None


@attr.frozen
class _ColumnPrefixes:
  slices: str
  metric_keys: str
  metric_values: str


_metric_columns = _ColumnPrefixes(_SLICES, _METRIC_KEYS, _METRIC_VALUES)
_plot_columns = _ColumnPrefixes(_SLICES, _PLOT_KEYS, _PLOT_DATA)

_WRAPPED_PRIMITIVES = (wrappers_pb2.DoubleValue, wrappers_pb2.FloatValue,
                       wrappers_pb2.BoolValue, wrappers_pb2.BytesValue,
                       wrappers_pb2.StringValue, wrappers_pb2.Int64Value,
                       wrappers_pb2.Int32Value)


def _flatten_proto(
    root_field: message.Message,
    field_name: str,
    index: int,
    include_empty_columns: bool = False) -> Iterator[Tuple[str, Any, int]]:
  """Generates the leaf primitive fields by traversing the proto recursively.

  Traverses a proto and emits a tuple of the name, value, index at which the
  value should be inserted. If include_empty_columns is True, unset fields are
  also emitted with value of None. The index is the order of which this
  primitive should be inserted to the DataFrame.
  Note: that nested or misaligned repeated fields are not supported and will
  lead to undefined behavior.

  Args:
    root_field: The root message field.
    field_name: The child field name under the root field where the traversal
      begins.
    index: The starting row index where the DataFrame was at.
    include_empty_columns: If True, the unset fields are also emitted.

  Returns:
    An iterator of the field name, field value, and index of the underlying
    proto primitives.
  """

  def _is_repeated_field(parent, field_name):
    return (parent.DESCRIPTOR.fields_by_name[field_name].label ==
            descriptor.FieldDescriptor.LABEL_REPEATED)

  def _flatten_proto_in(
      parent: message.Message,
      field_name: str,
      field_value: Any,
      index: int,
      is_repeated_field: bool = False) -> Iterator[Tuple[str, Any, int]]:
    if isinstance(field_value, message.Message):
      # Test the message field is unset.
      if not is_repeated_field and not parent.HasField(field_name):
        if include_empty_columns:
          yield (field_name, None, index)
        return
      elif isinstance(field_value, _WRAPPED_PRIMITIVES):
        # Preserve the field_name of the wrapped primitives.
        yield (field_name, field_value.value, index)
      else:
        for field in field_value.DESCRIPTOR.fields:
          yield from _flatten_proto_in(field_value, field.name,
                                       getattr(field_value, field.name), index)
    # Handling repeated field.
    elif _is_repeated_field(parent, field_name):
      for i, single_field_value in enumerate(field_value):
        yield from _flatten_proto_in(
            parent,
            field_name,
            single_field_value,
            index + i,
            is_repeated_field=True)
    # Python primitives.
    else:
      yield (field_name, field_value, index)

  field_value = getattr(root_field, field_name)
  return _flatten_proto_in(root_field, field_name, field_value, index)


def _get_slice_value(
    slice_key: metrics_for_slice_pb2.SingleSliceKey
) -> Union[float, bytes, int]:
  """Determines the primitive value stored by the slice."""
  value_type = slice_key.WhichOneof('kind')
  if value_type == 'float_value':
    return slice_key.float_value
  elif value_type == 'bytes_value':
    return slice_key.bytes_value
  elif value_type == 'int64_value':
    return slice_key.int64_value
  else:
    raise NotImplementedError(f'{value_type} in {slice_key} is not supported.')


def _to_dataframes(
    metrics_or_plots: Iterable[Union[MetricsForSlice, PlotsForSlice]],
    column_prefixes: _ColumnPrefixes,
    include_empty_columns: bool = False,
) -> Dict[str, pd.DataFrame]:
  """The implementation of loading TFMA metrics or plots as DataFrames.

  Args:
    metrics_or_plots: an iterable of MetricsForSlice or PlotsForSlice.
    column_prefixes: the column names of the first layer columns of the
      multi-index columns.
    include_empty_columns: if True, keeps all the unset fields with value set to
      None.

  Returns:
    A map of the DataFrame type to the corresponding DataFrame.
  """
  slice_key_name = column_prefixes.slices
  metric_key_name = column_prefixes.metric_keys
  metric_value_name = column_prefixes.metric_values
  column_data = collections.defaultdict(_ColumnData)
  index = 0
  # For each slice.
  for metrics_or_plots_for_slice in metrics_or_plots:
    slices = [(single_slice_key.column, _get_slice_value(single_slice_key))
              for single_slice_key in
              metrics_or_plots_for_slice.slice_key.single_slice_keys]
    # For each metric inside the slice.
    if isinstance(metrics_or_plots_for_slice, MetricsForSlice):
      key_and_values = metrics_or_plots_for_slice.metric_keys_and_values
    elif isinstance(metrics_or_plots_for_slice, PlotsForSlice):
      key_and_values = metrics_or_plots_for_slice.plot_keys_and_values
    else:
      raise NotImplementedError(
          f'{type(metrics_or_plots_for_slice)} is not supported.')
    for key_and_value in key_and_values:
      metric_value = key_and_value.value
      for field in metric_value.DESCRIPTOR.fields:
        metric_type = field.name
        metric = getattr(metric_value, metric_type)
        if (isinstance(metric, message.Message) and
            not metric_value.HasField(metric_type) or not metric):
          continue
        # Flattens the metric_values.
        # Initializes index_end to 'index - 1' to indicate that there is no item
        # added yet. It is set to 'index' once the loop starts, indicating that
        # at least one item is found.
        index_end = index - 1
        for k, v, index_end in _flatten_proto(metric_value, metric_type, index):
          column_data[metric_type].values[k].append((v, index_end))
        # index_end is later used as the exclusive range end, thus, the +1.
        index_end += 1
        # Insert a column per leaf field in MetricKey.
        for k, v, _ in _flatten_proto(
            key_and_value,
            'key',
            index,
            include_empty_columns=include_empty_columns):
          column_data[metric_type].metric_keys[k].extend([
              (v, i) for i in range(index, index_end)
          ])
        # Insert each slice
        if slices:
          for slice_name, slice_value in slices:
            column_data[metric_type].slices[slice_name].extend(
                [slice_value, i] for i in range(index, index_end))
        else:
          column_data[metric_type].slices[_OVERALL].extend(
              ['', i] for i in range(index, index_end))
        index = index_end
  dfs = {}
  for metric_type, data in column_data.items():
    columns = pd.MultiIndex.from_tuples(
        [(slice_key_name, key) for key in data.slices]
        + [(metric_key_name, key) for key in data.metric_keys]
        + [(metric_value_name, key) for key in data.values]
    )
    all_data = itertools.chain(data.slices.values(), data.metric_keys.values(),
                               data.values.values())
    df = pd.DataFrame({
        column: pd.Series(*zip(*values))
        for column, values in zip(columns, all_data)
    })
    dfs[metric_type] = df
  return dfs


def metrics_as_dataframes(
    metrics: Iterable[MetricsForSlice],
    *,  # Keyword only args below.
    include_empty_columns: bool = False,
) -> MetricsDataFrames:
  """Convert the deserialized MetricsForSlice protos to Pandas DataFrame.

  To load all metrics:
    dfs = metrics_as_dataframe(tfma.load_metrics(output_path))
  then users can load "double_value" DataFrame as
    dfs.double_value
  and confusion_metrics_at_thresholds DataFrame as:
    dfs.confusion_metrics_at_thresholds:

  For example, if the input metrics proto looks like this:
        slice_key {
           single_slice_keys {
             column: "age"
             float_value: 38.0
           }
         }
         metric_keys_and_values {
           key {
             name: "mean_absolute_error"
           }
           value {
             double_value {
               value: 0.1
             }
           }
         }
         metric_keys_and_values {
           key {
             name: "mean_squared_logarithmic_error"
           }
           value {
             double_value {
               value: 0.02
             }
           }
         }

  The corresponding output table will look like this:

  |    | metric_values| metric_keys                    |               slices |
  |    | double_value |           name                 |       age |      sex |
  |---:|-------------:|:-------------------------------|----------:|:---------|
  |  0 |         0.1  | mean_absolute_error            |        38 | male     |
  |  1 |         0.02 | mean_squared_logarithmic_error |        38 | female   |

  One typical use of this DataFrame table is to re-organize it in the form of
  slices vs. metrics table. For single model single output:
    `auto_pivot(dfs.double_value)`
  This will pivot on the non-unique "metric_keys.*" columns.

  Args:
    metrics: The directory path of the metrics file or an iterable of
      MetricsForSlice proto.
    include_empty_columns: Include a column if its value is not empty (None) in
      corresponding field in the MetricKey.

  Returns:
    A DataFrame with the following columns if the value is not None:
    * slices.<slice_feature_column>: feature columns derived from
      tfma.proto.metrics_for_slice_pb2.SliceKey. There are multiple feature
      columns if there is any feature-cross, e.g., slice.featureA and
      slice.featureB if there is FeatureA and FeatureB cross.
    * metric_keys.<metric_key_column>: recursively flattened items in
      metric_key: name, model_name, output_name, is_diff, example_weighted,
      flattened sub_key (class_id, k, top_k), flattened aggregation_type
      (micro_average, macro_average, weighted_macro_average).
    * metric_values.<metric_value_columns>: column(s) of metric_value(s);
      metric_value specified by metric_type and flattened to its leaf
      primitives. E.g.,
        'double_value' if metric_type is 'double_value';
        'true_positives', 'false_negatives', etc. if metric_type is
          'confusion_matrix_at_thresholds'.
  """
  dfs = _to_dataframes(
      metrics,
      column_prefixes=_metric_columns,
      include_empty_columns=include_empty_columns)
  return MetricsDataFrames(**dfs)


def plots_as_dataframes(
    plots: Iterable[PlotsForSlice],
    *,  # Keyword only args below.
    include_empty_columns: bool = False,
) -> PlotsDataFrames:
  """Read and deserialize the PlotsForSlice records as Pandas DataFrame.

  To load confusion_matrix_at_thresholds:
    df = plots_as_dataframes(tfma.load_plots(eval_path))

  For example, if the input plots_for_slice proto looks like this:
        slice_key {
           single_slice_keys {
             column: "age"
             float_value: 38.0
           }
         }
         plot_keys_and_values {
           key {}
           value {
             confusion_matrix_at_thresholds {
               matrices {
                 threshold: 0.5
                 false_negatives: 10
                 true_negatives: 10
                 false_positives: 10
                 true_positives: 10
                 precision: 0.9
                 recall: 0.8
               }
               matrices {
                 threshold: 0.5
                 false_negatives: 10
                 true_negatives: 10
                 false_positives: 10
                 true_positives: 10
                 precision: 0.9
                 recall: 0.8
               }
             }
           }
         }

  The corresponding output table will look like this:

  |                         plot_data |            plot_keys |          slices |
  | threshold | false_negatives | ... |              is_diff | ... |       age |
  |----------:|----------------:|----:|:---------------------| ... |:----------|
  |       0.5 |              10 | ... |                False | ... |        38 |
  |       0.5 |              10 | ... |                False | ... |        38 |
  # pylint: enable=line-too-long

  One typical use of this DataFrame table is to re-organize it in the form of
  slices vs. plots table.
  E.g., for single model single output:
    result.pivot(index='slice', columns='name', values='plot_data').
  This only works when there is one unique value as the pivot values.
  Otherewise, a user needs to specify more columns or indices to make sure that
  the metric value is unique per column and per index.
  E.g., for single model and multiple outputs:
    result.pivot(index='slice', columns=['output_name', 'name'],
                 values='plot_data').

  Args:
    plots: a path to the evaluation directory or an iterable of PlotsForSlice
      proto.
    include_empty_columns: include a column if its value is not empty (None) in
      corresponding field in the MetricKey.

  Returns:
    A DataFrame with the following columns if the value is not None:
    * slice_str: the string representation of the slice.
      E.g., "age: 10; sex: male".
    * <slice_feature_column>: feature columns derived from
      tfma.proto.metrics_for_slice_pb2.SliceKey. There are multiple feature
      columns if there is any feature-cross, e.g., slice.featureA and
      slice.featureB if there is FeatureA and FeatureB cross.
    * recursively flattened items in metric_key: name, model_name, output_name,
      is_diff, example_weighted, flattened sub_key (class_id, k, top_k),
      flattened aggregation_type (micro_average, macro_average,
      weighted_macro_average).
    * multiple columns of plot_values: plot_value specified by plot_type
      and flattened to its leaf primitives. E.g., there will be columns
      false_negatives, false_positives, true_negatives, true_positives,
      precision, recall when plot_type is `confusion_matrix_at_thresholds`.
  """
  dfs = _to_dataframes(
      plots,
      column_prefixes=_plot_columns,
      include_empty_columns=include_empty_columns)
  return PlotsDataFrames(**dfs)


def _collapse_column_names(columns: pd.MultiIndex) -> pd.Index:
  """Reduce multi-index column names by removing layers with the same value."""
  dropables = [i for i, x in enumerate(zip(*columns)) if len(set(x)) == 1]
  return columns.droplevel(dropables)


def _stringify_slices(df: pd.DataFrame,
                      slice_key_name: str = _SLICES,
                      drop_slices: bool = True) -> pd.DataFrame:
  """Stringify all the slice columns into one column.

  For example, if there are two slice column of 'sex' and 'age', the function
  takes all the slice columns, convert them to string in the format of
  <slice_name>:<slive_value> and concatenate all slice columns using '; '. The
  final result looks like:
    '<slice1_name>:<slice1_value>; <slice2_name>:<slice2_value>.'

  Args:
    df: a metrics DataFrame or plots DataFrame.
    slice_key_name: the first level column name that groups all slice columns.
    drop_slices: if True, drops the original slice columns.

  Returns:
    A DataFrame with all slice columns stringified and concatenated.
  """

  def _concatenate(x):
    return '; '.join(
        f'{col}:{val}' for col, val in zip(df[slice_key_name].columns, x)
        if pd.notnull(val))

  t = df.slices.agg(_concatenate, axis=1)
  df = df.drop(slice_key_name, axis=1, level=0) if drop_slices else df.copy()
  df.insert(0, (slice_key_name, _SLICE_STR), t)
  return df.sort_values((slice_key_name, _SLICE_STR))


def _auto_pivot(df: pd.DataFrame, column_prefixes: _ColumnPrefixes,
                stringify_slices: bool,
                collapse_column_names: bool) -> pd.DataFrame:
  """Implements auto_pivot."""
  df = _stringify_slices(df, column_prefixes.slices) if stringify_slices else df
  # TODO(b/277280388): Use Series.sort_values after upgrading to pandas > 1.2.
  # See https://github.com/pandas-dev/pandas/issues/35922
  # Replace the df_unique logic with the following block.
  # df_unique = (
  #     df[column_prefixes.metric_keys]
  #     .nunique(dropna=False)
  #     .sort_index()
  #     .sort_values(ascending=False, kind='stable')
  # )
  df_unique = df[column_prefixes.metric_keys].nunique(dropna=False).sort_index()
  _, tags = np.unique(df_unique.values, return_inverse=True)
  df_unique = df_unique.iloc[np.argsort(-tags, kind='stable')]

  pivot_columns = [(column_prefixes.metric_keys, column)
                   for column, nunique in df_unique.items()
                   if nunique > 1]
  metric_value_columns = df[column_prefixes.metric_values].columns
  slice_columns = df[column_prefixes.slices].columns
  value_columns = [
      (column_prefixes.metric_values, c) for c in metric_value_columns
  ]
  index_columns = [(column_prefixes.slices, c) for c in slice_columns]
  result = df.pivot(
      index=index_columns, columns=pivot_columns, values=value_columns)
  if stringify_slices:
    result.index.name = column_prefixes.slices
  if collapse_column_names and isinstance(result.columns, pd.MultiIndex):
    result.columns = _collapse_column_names(result.columns)
  return result


def auto_pivot(df: pd.DataFrame,
               stringify_slices: bool = True,
               collapse_column_names: bool = True) -> pd.DataFrame:
  """Automatically pivots a metric or plots DataFrame.

  Given a DataFrame provided by metrics/plots_as_dataframes, one can
  automatically pivot the table on all non-unique metric_key columns, with the
  slices as the index columns, metric_values as the values columns.
  E.g., given this raw DataFrame:

  |    | metric_values| metric_keys                    |               slices |
  |    | double_value |           name                 |       age |      sex |
  |---:|-------------:|:-------------------------------|----------:|:---------|
  |  0 |         0.1  | mean_absolute_error            |       nan | nan      |
  |  1 |         0.02 | mean_squared_logarithmic_error |        38 | male     |

  Since the only non-unique metric_key column is metric_keys.name, auto_pivot
  with stringify_slices set to True will generates the following DataFrame:

  |    slices      | 'mean_absolute_error' | 'mean_squared_logarithmic_error' |
  |:---------------|---------------------------------------------------------:|
  |Overall         |                   0.1 |                              nan |
  |sex:male; age:38|                   nan |                             0.02 |

  Args:
    df: a DataFrame from one of the MetricsDataFrames or PlotsDataFrames.
    stringify_slices: stringify all the slice columns and collapse them into one
      column by concatenating the corresponding strings. This is turned on by
      default.
    collapse_column_names: collapsing the multi-index column names by removing
      layer(s) with only the same string. This is turned on by default.
  Returns:
    A DataFrame that pivoted from the metrics DataFrame or plots DataFrame.
  """
  if _SLICES in df:
    if _PLOT_KEYS in df and _PLOT_DATA in df:
      return _auto_pivot(
          df,
          _plot_columns,
          stringify_slices=stringify_slices,
          collapse_column_names=collapse_column_names)
    elif _METRIC_KEYS in df and _METRIC_VALUES in df:
      return _auto_pivot(
          df,
          _metric_columns,
          stringify_slices=stringify_slices,
          collapse_column_names=collapse_column_names)

  raise NotImplementedError(
      'Only a metrics or a plots DataFrame is supported. This DataFrame has'
      f'the following columns: {df.columns}')
