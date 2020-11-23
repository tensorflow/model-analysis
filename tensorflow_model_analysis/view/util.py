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
"""View API for Tensorflow Model Analysis."""
from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import json
import os

from typing import Any, Dict, List, Optional, Text, Tuple, Union
from absl import logging

from tensorflow_model_analysis import config
from tensorflow_model_analysis.metrics import example_count
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import weighted_example_count
from tensorflow_model_analysis.post_export_metrics import metric_keys
from tensorflow_model_analysis.proto import metrics_for_slice_pb2
from tensorflow_model_analysis.slicer import slicer_lib as slicer
from tensorflow_model_analysis.view import view_types

from google.protobuf import json_format


def get_slicing_metrics(
    results: List[Tuple[slicer.SliceKeyType, Dict[Text, Any]]],
    slicing_column: Optional[Text] = None,
    slicing_spec: Optional[slicer.SingleSliceSpec] = None,
) -> List[Dict[Text, Union[Dict[Text, Any], Text]]]:
  """Util function that extracts slicing metrics from the results.

  If neither slicing_column nor slicing_spec is provided, get Overall. If
  slicing_column is set, use it to filter metrics from results. Otherwise, use
  slicing_spec for filtering.

  Args:
    results: A list of records. Each record is a tuple of (slice_name,
      {output_name: {sub_key: {metric_name, metric_value}}}).
    slicing_column: The column to filter the resuslts with.
    slicing_spec: The slicer.SingleSliceSpec to filter the resutls with.

  Returns:
    A list of {slice, metrics}

  Raises:
    ValueError: The provided slicing_column does not exist in results or more
    than one set of overall result is found.
  """

  if slicing_column:
    data = find_all_slices(results,
                           slicer.SingleSliceSpec(columns=[slicing_column]))
  elif not slicing_spec:
    data = find_all_slices(results, slicer.SingleSliceSpec())
  else:
    data = find_all_slices(results, slicing_spec)

  slice_count = len(data)
  if not slice_count:
    if not slicing_spec:
      if not slicing_column:
        slicing_column = slicer.OVERALL_SLICE_NAME
      raise ValueError('No slices found for %s' % slicing_column)
    else:
      raise ValueError('No slices found for %s' % slicing_spec)
  elif not slicing_column and not slicing_spec and slice_count > 1:
    raise ValueError('More than one slice found for %s' %
                     slicer.OVERALL_SLICE_NAME)
  else:
    return data


def find_all_slices(
    results: List[Tuple[slicer.SliceKeyType,
                        Dict[Text, Any]]], slicing_spec: slicer.SingleSliceSpec
) -> List[Dict[Text, Union[Dict[Text, Any], Text]]]:
  """Util function that extracts slicing metrics for the named column.

  Args:
    results: A list of records. Each record is a tuple of (slice_name,
      {metric_name, metric_value}).
    slicing_spec: The spec to slice on.

  Returns:
    A list of {slice, metrics}
  """
  data = []
  for (slice_key, metric_value) in results:
    if slicing_spec.is_slice_applicable(slice_key):
      data.append({
          'slice': slicer.stringify_slice_key(slice_key),
          'metrics': metric_value
      })

  return data  # pytype: disable=bad-return-type


def get_time_series(
    results: view_types.EvalResults, slicing_spec: slicer.SingleSliceSpec,
    display_full_path: bool
) -> List[Dict[Text, Union[Dict[Union[float, Text], Any], Text]]]:
  """Util function that extracts time series data for the specified slice.

  Args:
    results: A collection of EvalResult whose metrics should be visualized in a
      time series.
    slicing_spec: The spec specifying the slice to show in the time series.
    display_full_path: Whether to display the full path or just the file name.

  Returns:
    A list of dictionaries, where each dictionary contains the config and the
    metrics for the specified slice for a single eval run.

  Raises:
    ValueError: if the given slice spec matches more than one slice for any eval
    run in results or if the slicing spec matches nothing in all eval runs.
  """
  data = []
  for result in results.get_results():
    matching_slices = find_all_slices(result.slicing_metrics, slicing_spec)
    slice_count = len(matching_slices)
    if slice_count == 1:
      data.append({
          'metrics': matching_slices[0]['metrics'],
          'config': {
              'modelIdentifier':
                  _get_identifier(result.model_location, display_full_path),
              'dataIdentifier':
                  _get_identifier(result.data_location, display_full_path),
          }
      })
    elif slice_count > 1:
      raise ValueError('Given slice spec matches more than one slice.')

  run_count = len(data)
  if not run_count:
    raise ValueError('Given slice spec has no matches in any eval run.')

  return data  # pytype: disable=bad-return-type


def _get_identifier(path: Text, use_full_path: bool) -> Text:
  """"Returns the desired identifier based on the path to the file.

  Args:
    path: The full path to the file.
    use_full_path: Whether to use the full path or just the file name as the
      identifier.

  Returns:
    A string containing the identifier
  """
  return path if use_full_path else os.path.basename(path)


# Passing the keys from python means that it is possible to reuse the plot UI
# with other data by overwriting the config on python side.
_SUPPORTED_PLOT_KEYS = {
    'calibrationPlot': {
        'metricName': 'calibrationHistogramBuckets',
        'dataSeries': 'buckets',
    },
    'confusionMatrixPlot': {
        'metricName': 'confusionMatrixAtThresholds',
        'dataSeries': 'matrices',
    },
    'multiClassConfusionMatrixPlot': {
        'metricName': 'multiClassConfusionMatrixAtThresholds',
        'dataSeries': 'matrices',
    },
    'multiLabelConfusionMatrixPlot': {
        'metricName': 'multiLabelConfusionMatrixAtThresholds',
        'dataSeries': 'matrices',
    }
}


def _replace_nan_with_none(
    plot_data: Union[Dict[Text, Any], Text],
    plot_keys: Dict[Text, Dict[Text, Text]]) -> Union[Dict[Text, Any], Text]:
  """Replaces all instances of nan with None in plot data.

  This is necessary for Colab integration where we serializes the data into json
  string as NaN is not supported by json standard. Turning nan into None will
  make the value null once parsed. The visualization already handles falsy
  values by setting them to zero.

  Args:
    plot_data: The original plot data
    plot_keys: A dictionary containing field names of plot data.

  Returns:
    Transformed plot data where all nan has been replaced with None.
  """
  output_metrics = {}
  for plot_type in plot_keys:
    metric_name = plot_keys[plot_type]['metricName']
    if metric_name in plot_data:
      data_series_name = plot_keys[plot_type]['dataSeries']
      if data_series_name in plot_data[metric_name]:
        data_series = plot_data[metric_name][data_series_name]
        outputs = []
        for entry in data_series:
          output = {}
          for key in entry:
            value = entry[key]
            # When converting protocol buffer into dict, float value nan is
            # automatically converted into the string 'NaN'.
            output[key] = None if value == 'NaN' else value
          outputs.append(output)
        output_metrics[metric_name] = {data_series_name: outputs}

  return output_metrics


def get_plot_data_and_config(
    results: List[Tuple[slicer.SliceKeyType, Dict[Text, Any]]],
    slicing_spec: slicer.SingleSliceSpec,
    output_name: Optional[Text] = None,
    class_id: Optional[int] = None,
    top_k: Optional[int] = None,
    k: Optional[int] = None,
    label: Optional[Text] = None,
) -> Tuple[Union[Dict[Text, Any], Text], Dict[Text, Union[Dict[Text, Dict[
    Text, Text]], Text]]]:
  """Util function that extracts plot for a particular slice from the results.

  Args:
    results: A list of records. Each record is a tuple of (slice_name,
      {metric_name, metric_value}).
    slicing_spec: The slicer.SingleSliceSpec to identify the slice to fetch plot
      for.
    output_name: The name of the output.
    class_id: An int representing the class id if model is multi-class.
    top_k: The k used to compute prediction in the top k position.
    k: The k used to compute prediciton at the kth position.
    label: A partial label used to match a set of plots in the results. This is
      kept for backward compatibility.

  Returns:
    (plot_data, plot_config) for the specified slice.

    Note that plot_data should be of type Dict[Text, Any]. However, PyType
    can't figure it out. As a result, the annotation has to be
    Union[Dict[Text, Any], Text].

  Raises:
    ValueError: The provided slicing_column does not exist in results or more
    than one result is found; or there is not plot data available; or there are
    multiple sets of plot while label is not provided; or label matches to more
    than one set of plots; or label does not match any set of plots.
  """
  if label is not None and (output_name is not None or class_id is not None or
                            top_k is not None or k is not None):
    # Plot key (specified by output_name and class_id / top k / k) and label (
    # for backward compatibiility only) should not be provided together.
    raise ValueError('Do not specify both label and output_name / class_id')

  sub_key_oneof_check = 0
  sub_key_id = None

  if class_id is not None:
    sub_key_oneof_check = sub_key_oneof_check + 1
    sub_key_id = 'classId:' + str(class_id)
  if top_k is not None:
    sub_key_oneof_check = sub_key_oneof_check + 1
    sub_key_id = 'topK:' + str(top_k)
  if k is not None:
    sub_key_oneof_check = sub_key_oneof_check + 1
    sub_key_id = 'k:' + str(k)
  if sub_key_oneof_check > 1:
    raise ValueError('Up to one of class_id, top_k and k can be provided.')

  output_name = '' if output_name is None else output_name

  matching_slices = find_all_slices(results, slicing_spec)
  count = len(matching_slices)

  if count == 0:
    raise ValueError('No slice matching slicing spec is found.')
  elif count > 1:
    raise ValueError('More than one slice matching slicing spec is found.')

  target_slice = matching_slices[0]

  plot_config = {
      'sliceName': target_slice['slice'],
      'metricKeys': _SUPPORTED_PLOT_KEYS,
  }

  if output_name not in target_slice['metrics']:
    if output_name:
      raise ValueError('No plot data found for output name %s.' % output_name)
    else:
      raise ValueError('No plot data found without output name.')

  output = target_slice['metrics'][output_name]
  class_id_to_use = (sub_key_id if sub_key_id is not None else '')

  if class_id_to_use not in output:
    if class_id is None:
      raise ValueError(
          'No plot data found for output name %s with no class id.' %
          output_name)
    else:
      raise ValueError(
          'No plot data found for output name %s with class id %d.' %
          (output_name, class_id))

  plot_data = target_slice['metrics'][output_name][class_id_to_use]

  # Backward compatibility support for plot data stored in a map before the plot
  # key was introduced.
  if label is not None:
    plot_set = None
    for key in plot_data:
      if label in key:
        if not plot_set:
          plot_set = plot_data[key]
        else:
          raise ValueError(
              'Label %s matches more than one key. Keys are %s. Please make it more specific.'
              % (label, ', '.join([key for key, _ in plot_data])))

    if not plot_set:
      raise ValueError(
          'Label %s does not match any key. Keys are %s. Please provide a new one.'
          % (label, ', '.join([key for key, _ in plot_data])))
    plot_data = plot_set
  else:
    # Make sure that we are not looking at actual plot data instead of a map of
    # plot.
    contains_supported_plot_data = False
    for key in plot_data:
      for _, plot in _SUPPORTED_PLOT_KEYS.items():
        contains_supported_plot_data = contains_supported_plot_data or plot[
            'metricName'] == key

    if not contains_supported_plot_data:
      raise ValueError('No plot data found. Maybe provide a label? %s' %
                       json.dumps(plot_data))

  plot_data = _replace_nan_with_none(plot_data, _SUPPORTED_PLOT_KEYS)

  return plot_data, plot_config  # pytype: disable=bad-return-type


# TODO(paulyang): Add support for multi-model / multi-output selection.
def get_slicing_config(
    eval_config: config.EvalConfig,
    weighted_example_column_to_use: Text = None) -> Dict[Text, Text]:
  """Util function that generates config for visualization.

  Args:
    eval_config: EvalConfig that contains example_weight_metric_key.
    weighted_example_column_to_use: Override for weighted example column in the
      slicing metrics browser.

  Returns:
    A dictionary containing configuration of the slicing metrics evalaution.
  """
  if eval_config.metrics_specs:
    default_column = example_count.EXAMPLE_COUNT_NAME
  else:
    # Legacy post_export_metric
    default_column = metric_keys.EXAMPLE_COUNT

  # For now we only support one candidate model, so pick first.
  model_spec = None
  if eval_config.model_specs:
    for spec in eval_config.model_specs:
      if not spec.is_baseline:
        model_spec = spec
        break

  if model_spec:
    if model_spec.example_weight_key:
      if eval_config.metrics_specs:
        # Legacy post_export_metric
        default_column = weighted_example_count.WEIGHTED_EXAMPLE_COUNT_NAME
      else:
        default_column = metric_keys.EXAMPLE_WEIGHT

  return {
      'weightedExamplesColumn':
          weighted_example_column_to_use
          if weighted_example_column_to_use else default_column
  }


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


def convert_metrics_proto_to_dict(
    metrics_for_slice: metrics_for_slice_pb2.MetricsForSlice,
    model_name: Optional[Text] = None
) -> Optional[Tuple[slicer.SliceKeyType,
                    Optional[view_types.MetricsByOutputName]]]:
  """Converts metrics proto to dict."""
  model_metrics_map = {}
  if metrics_for_slice.metrics:
    model_metrics_map[''] = {
        '': {
            '': _convert_proto_map_to_dict(metrics_for_slice.metrics)
        }
    }

  default_model_name = None
  if metrics_for_slice.metric_keys_and_values:
    for kv in metrics_for_slice.metric_keys_and_values:
      current_model_name = kv.key.model_name
      if current_model_name not in model_metrics_map:
        model_metrics_map[current_model_name] = {}
      output_name = kv.key.output_name
      if output_name not in model_metrics_map[current_model_name]:
        model_metrics_map[current_model_name][output_name] = {}

      sub_key_metrics_map = model_metrics_map[current_model_name][output_name]
      sub_key_id = str(metric_types.SubKey.from_proto(
          kv.key.sub_key)) if kv.key.HasField('sub_key') else ''
      if sub_key_id not in sub_key_metrics_map:
        sub_key_metrics_map[sub_key_id] = {}
      if kv.key.is_diff:
        if default_model_name is None:
          default_model_name = current_model_name
        elif default_model_name != current_model_name:
          # Setting '' to trigger no match found ValueError below.
          default_model_name = ''
        metric_name = '{}_diff'.format(kv.key.name)
      elif kv.key.HasField('aggregation_type'):
        metric_name = '{}_{}'.format(kv.key.aggregation_type, kv.key.name)
      else:
        metric_name = kv.key.name
      sub_key_metrics_map[sub_key_id][metric_name] = json_format.MessageToDict(
          kv.value)

  metrics_map = None
  keys = list(model_metrics_map.keys())
  tmp_model_name = model_name or default_model_name
  if tmp_model_name in model_metrics_map:
    # Use the provided model name if there is a match.
    metrics_map = model_metrics_map[tmp_model_name]
    # Add model-independent (e.g. example_count) metrics to all models.
    if tmp_model_name and '' in model_metrics_map:
      for output_name, output_dict in model_metrics_map[''].items():
        for sub_key_id, sub_key_dict in output_dict.items():
          for name, value in sub_key_dict.items():
            metrics_map.setdefault(output_name, {}).setdefault(sub_key_id,
                                                               {})[name] = value
  elif not tmp_model_name and len(keys) == 1:
    # Show result of the only model if no model name is specified.
    metrics_map = model_metrics_map[keys[0]]
  elif keys:
    # No match found.
    logging.warning(
        'Fail to find metrics for model name: %s . '
        'Available model names are [%s]', model_name, ', '.join(keys))
    return None

  return (slicer.deserialize_slice_key(metrics_for_slice.slice_key),
          metrics_map)


def convert_plots_proto_to_dict(
    plots_for_slice: metrics_for_slice_pb2.PlotsForSlice,
    model_name: Optional[Text] = None
) -> Optional[Tuple[slicer.SliceKeyType,
                    Optional[view_types.PlotsByOutputName]]]:
  """Converts plots proto to dict."""
  model_plots_map = {}
  if plots_for_slice.plots:
    plot_dict = _convert_proto_map_to_dict(plots_for_slice.plots)
    keys = list(plot_dict.keys())
    # If there is only one label, choose it automatically.
    plot_data = plot_dict[keys[0]] if len(keys) == 1 else plot_dict
    model_plots_map[''] = {'': {'': plot_data}}
  elif plots_for_slice.HasField('plot_data'):
    model_plots_map[''] = {
        '': {
            '': {json_format.MessageToDict(plots_for_slice.plot_data)}
        }
    }

  if plots_for_slice.plot_keys_and_values:
    for kv in plots_for_slice.plot_keys_and_values:
      current_model_name = kv.key.model_name
      if current_model_name not in model_plots_map:
        model_plots_map[current_model_name] = {}
      output_name = kv.key.output_name
      if output_name not in model_plots_map[current_model_name]:
        model_plots_map[current_model_name][output_name] = {}

      sub_key_plots_map = model_plots_map[current_model_name][output_name]
      sub_key_id = str(metric_types.SubKey.from_proto(
          kv.key.sub_key)) if kv.key.HasField('sub_key') else ''
      sub_key_plots_map[sub_key_id] = json_format.MessageToDict(kv.value)

  plots_map = None
  keys = list(model_plots_map.keys())
  if model_name in model_plots_map:
    # Use the provided model name if there is a match.
    plots_map = model_plots_map[model_name]
  elif not model_name and len(keys) == 1:
    # Show result of the only model if no model name is specified.
    plots_map = model_plots_map[keys[0]]
  elif keys:
    # No match found.
    logging.warning(
        'Fail to find plots for model name: %s . '
        'Available model names are [%s]', model_name, ', '.join(keys))
    return None

  return (slicer.deserialize_slice_key(plots_for_slice.slice_key), plots_map)


def convert_attributions_proto_to_dict(
    attributions_for_slice: metrics_for_slice_pb2.AttributionsForSlice,
    model_name: Optional[Text] = None
) -> Tuple[slicer.SliceKeyType, Optional[view_types.AttributionsByOutputName]]:
  """Converts attributions proto to dict."""
  model_metrics_map = {}
  default_model_name = None
  if attributions_for_slice.attributions_keys_and_values:
    for kv in attributions_for_slice.attributions_keys_and_values:
      current_model_name = kv.key.model_name
      if current_model_name not in model_metrics_map:
        model_metrics_map[current_model_name] = {}
      output_name = kv.key.output_name
      if output_name not in model_metrics_map[current_model_name]:
        model_metrics_map[current_model_name][output_name] = {}
      sub_key_metrics_map = model_metrics_map[current_model_name][output_name]
      if kv.key.HasField('sub_key'):
        sub_key_id = str(metric_types.SubKey.from_proto(kv.key.sub_key))
      else:
        sub_key_id = ''
      if sub_key_id not in sub_key_metrics_map:
        sub_key_metrics_map[sub_key_id] = {}
      if kv.key.is_diff:
        if default_model_name is None:
          default_model_name = current_model_name
        elif default_model_name != current_model_name:
          # Setting '' to possibly trigger no match found ValueError below.
          default_model_name = ''
        metric_name = '{}_diff'.format(kv.key.name)
      else:
        metric_name = kv.key.name
      attributions = {}
      for k in kv.values:
        attributions[k] = json_format.MessageToDict(kv.values[k])
      sub_key_metrics_map[sub_key_id][metric_name] = attributions

  metrics_map = None
  keys = list(model_metrics_map.keys())
  tmp_model_name = model_name or default_model_name
  if tmp_model_name in model_metrics_map:
    # Use the provided model name if there is a match.
    metrics_map = model_metrics_map[tmp_model_name]
  elif (not tmp_model_name) and len(keys) == 1:
    # Show result of the only model if no model name is specified.
    metrics_map = model_metrics_map[keys[0]]
  elif keys:
    # No match found.
    raise ValueError('Fail to find attribution metrics for model name: %s . '
                     'Available model names are [%s]' %
                     (model_name, ', '.join(keys)))

  return (slicer.deserialize_slice_key(attributions_for_slice.slice_key),
          metrics_map)
