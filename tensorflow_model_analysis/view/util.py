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
import os

from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.slicer import slicer
from typing import Any, Dict, List, Optional, Text, Tuple, Union


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
      {metric_name, metric_value}).
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
    if slicing_spec is None:
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


def find_all_slices(results: List[Tuple[slicer.SliceKeyType, Dict[Text, Any]]],
                    slicing_spec: slicer.SingleSliceSpec
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
    results: model_eval_lib.EvalResults, slicing_spec: slicer.SingleSliceSpec,
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
                  _get_identifier(result.config.model_location,
                                  display_full_path),
              'dataIdentifier':
                  _get_identifier(result.config.data_location,
                                  display_full_path),
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
    'aucPlot': {
        'metricName': 'confusionMatrixAtThresholds',
        'dataSeries': 'matrices',
    }
}


def _replace_nan_with_none(plot_data: Union[Dict[Text, Any], Text],
                           plot_keys: Dict[Text, Dict[Text, Text]]
                          ) -> Union[Dict[Text, Any], Text]:
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
    label: Optional[Text] = None,
) -> Tuple[Union[Dict[Text, Any], Text],
           Dict[Text, Union[Dict[Text, Dict[Text, Text]], Text]]]:
  """Util function that extracts plot for a particular slice from the results.

  Args:
    results: A list of records. Each record is a tuple of (slice_name,
      {metric_name, metric_value}).
    slicing_spec: The slicer.SingleSliceSpec to identify the slice to fetch plot
      for.
    label: A partial label used to match a set of plots in the results.

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

  plot_sets = list(target_slice['metrics'].items())  # pytype: disable=attribute-error

  plot_sets_count = len(plot_sets)

  if plot_sets_count == 0:
    raise ValueError('No plot data found for the slice.')

  plot_set = None

  if not label:
    if plot_sets_count == 1:
      # If there is only one set of plot, label is not necessary and we simply
      # return the only set of plots available.
      plot_set = plot_sets[0][1]
    else:
      # If more than one set of plots are available, ask the user to provide a
      # label to help determine which set to return.
      raise ValueError(
          'More than one set of plots available. Please provide a partial '
          'label for matching.')
  else:
    for (key, plots) in plot_sets:
      if label in key:
        if not plot_set:
          plot_set = plots
        else:
          raise ValueError(
              'Label %s matches more than one key. Keys are %s. Please make it more specific.'
              % (label, ', '.join([key for key, _ in plot_sets])))
    if not plot_set:
      raise ValueError(
          'Label %s does not match any key. Keys are %s. Please provide a new one.'
          % (label, ', '.join([key for key, _ in plot_sets])))

  plot_data = _replace_nan_with_none(plot_set, _SUPPORTED_PLOT_KEYS)

  return plot_data, plot_config  # pytype: disable=bad-return-type


def get_slicing_config(config: model_eval_lib.EvalConfig,
                       weighted_example_column_to_use: Text = None
                      ) -> Dict[Text, Text]:
  """Util function that generates config for visualization.

  Args:
    config: EvalConfig that contains example_weight_metric_key.
    weighted_example_column_to_use: Override for weighted example column in the
      slicing metrics browser.

  Returns:
    A dictionary containing configuration of the slicing metrics evalaution.
  """
  default_column = config.example_weight_metric_key
  if not default_column or isinstance(default_column, dict):
    default_column = config.example_count_metric_key
  return {
      'weightedExamplesColumn':
          weighted_example_column_to_use
          if weighted_example_column_to_use else default_column
  }
