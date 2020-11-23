# Lint as: python3
# Copyright 2020 Google LLC
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
"""View types for Tensorflow Model Analysis."""

# Standard __future__ imports

import copy

from typing import Any, Dict, List, Sequence, Text, NamedTuple, Optional, Union
from tensorflow_model_analysis import config
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.proto import metrics_for_slice_pb2
from tensorflow_model_analysis.slicer import slicer_lib as slicer

Plots = Any
PlotsBySubKey = Dict[Text, Plots]
PlotsByOutputName = Dict[Text, PlotsBySubKey]


class SlicedPlots(
    NamedTuple('SlicedPlots', [('slice', slicer.SliceKeyType),
                               ('plot', PlotsByOutputName)])):
  """A tuple containing the plots belonging to a slice.

  Attributes:
    slice: A 2-element tuple representing a slice. The first element is the key
      of a feature (ex: 'color'), and the second element is the value (ex:
        'green'). An empty tuple represents an 'overall' slice (i.e. one that
        encompasses the entire dataset.
    plot: A dict mapping `output_name` and `sub_key_id` to plot data. The data
      contains histograms and confusion matrices, which can be rendered with the
      `tfma.view.render_plot` function.
  """


MetricsByTextKey = Dict[Text, metrics_for_slice_pb2.MetricValue]
MetricsBySubKey = Dict[Text, MetricsByTextKey]
MetricsByOutputName = Dict[Text, MetricsBySubKey]


class SlicedMetrics(
    NamedTuple('SlicedMetrics', [('slice', slicer.SliceKeyType),
                                 ('metrics', MetricsByOutputName)])):
  """A tuple containing the metrics belonging to a slice.

  The metrics are stored in a nested dictionary with the following levels:

   1. output_name: Optional output name associated with metric (for multi-output
   models). '' by default.
   2. sub_key: Optional sub key associated with metric (for multi-class models).
   '' by default. See `tfma.metrics.SubKey` for more info.
   3. metric_name: Name of the metric (`auc`, `accuracy`, etc).
   4. metric_value: A dictionary containing the metric's value. See
   [`tfma.proto.metrics_for_slice_pb2.MetricValue`](https://github.com/tensorflow/model-analysis/blob/cdb6790dcd7a37c82afb493859b3ef4898963fee/tensorflow_model_analysis/proto/metrics_for_slice.proto#L194)
   for more info.

  Below is a sample SlicedMetrics:

  ```python
  (
    (('color', 'green')),
    {
      '': {  # default for single-output models
        '': {  # default sub_key for non-multiclass-classification models
          'auc': {
            'doubleValue': 0.7243943810462952
          },
          'accuracy': {
            'doubleValue': 0.6488351225852966
          }
        }
      }
    }
  )
  ```

  Attributes:
    slice: A 2-element tuple representing a slice. The first element is the key
      of a feature (ex: 'color'), and the second element is the value (ex:
        'green'). An empty tuple represents an 'overall' slice (i.e. one that
        encompasses the entire dataset.
    metrics: A nested dictionary containing metric names and values.
  """


AttributionsByFeatureKey = Dict[Text, metrics_for_slice_pb2.MetricValue]
AttributionsByMetricName = Dict[Text, AttributionsByFeatureKey]
AttributionsBySubKey = Dict[Text, AttributionsByMetricName]
AttributionsByOutputName = Dict[Text, AttributionsBySubKey]


class SlicedAttributions(
    NamedTuple('SlicedAttributions',
               [('slice', slicer.SliceKeyType),
                ('attributions', AttributionsByOutputName)])):
  """A tuple containing the attributions belonging to a slice.

  The attributions are stored in a nested dictionary with the following levels:

   1. output_name: Optional output name associated with metric (for multi-output
   models). '' by default.
   2. sub_key: Optional sub key associated with metric (for multi-class models).
   '' by default. See `tfma.metrics.SubKey` for more info.
   3. metric_name: Name of the metric (`auc`, `accuracy`, etc).
   4. feature_key: Key of feature ('age', etc).
   5. metric_value: A dictionary containing the metric's value. See
   [tfma.proto.metrics_for_slice_pb2.MetricValue](https://github.com/tensorflow/model-analysis/blob/cdb6790dcd7a37c82afb493859b3ef4898963fee/tensorflow_model_analysis/proto/metrics_for_slice.proto#L194)
   for more info.

  Below is a sample SlicedAttributions:

  ```python
  (
    (('color', 'green')),
    {
      '': {  # default for single-output models
        '': {  # default sub_key for non-multiclass-classification models
          'total_attributions': {
            'feature1': {
              'doubleValue': 100.32
            },
            'feature2': {
              'doubleValue': 54.2
            }
          }
        }
      }
    }
  )
  ```

  Attributes:
    slice: A 2-element tuple representing a slice. The first element is the key
      of a feature (ex: 'color'), and the second element is the value (ex:
        'green'). An empty tuple represents an 'overall' slice (i.e. one that
        encompasses the entire dataset.
    attributions: A nested dictionary containing attribution names and values.
  """


class EvalResult(
    NamedTuple('EvalResult', [('slicing_metrics', List[SlicedMetrics]),
                              ('plots', List[SlicedPlots]),
                              ('attributions', List[SlicedAttributions]),
                              ('config', config.EvalConfig),
                              ('data_location', Text), ('file_format', Text),
                              ('model_location', Text)])):
  """The result of a single model analysis run.

  Attributes:
    slicing_metrics: a list of `tfma.SlicedMetrics`, containing metric values
      for each slice.
    plots: List of slice-plot pairs.
    attributions: List of SlicedAttributions containing attribution values for
      each slice.
    config: The config containing slicing and metrics specification.
    data_location: Optional location for data used with config.
    file_format: Optional format for data used with config.
    model_location: Optional location(s) for model(s) used with config.
  """

  def get_metrics_for_slice(
      self,
      slice_name: slicer.SliceKeyType = (),
      output_name: Text = '',
      class_id: Optional[int] = None,
      k: Optional[int] = None,
      top_k: Optional[int] = None) -> Union[MetricsByTextKey, None]:
    """Get metric names and values for a slice.

    Args:
      slice_name: A tuple of the form (column, value), indicating which slice to
        get metrics from. Optional; if excluded, return overall metrics.
      output_name: The name of the output. Optional, only used for multi-output
        models.
      class_id: Used with multi-class metrics to identify a specific class ID.
      k: Used with multi-class metrics to identify the kth predicted value.
      top_k: Used with multi-class and ranking metrics to identify top-k
        predicted values.

    Returns:
      Dictionary containing metric names and values for the specified slice.
    """

    if class_id or k or top_k:
      sub_key = str(metric_types.SubKey(class_id, k, top_k))
    else:
      sub_key = ''

    def equals_slice_name(slice_key):
      if not slice_key:
        return not slice_name
      else:
        return slice_key == slice_name

    for slicing_metric in self.slicing_metrics:
      slice_key = slicing_metric[0]
      slice_val = slicing_metric[1]
      if equals_slice_name(slice_key):
        return slice_val[output_name][sub_key]

    # if slice could not be found, return None
    return None

  def get_metrics_for_all_slices(
      self,
      output_name: Text = '',
      class_id: Optional[int] = None,
      k: Optional[int] = None,
      top_k: Optional[int] = None) -> Dict[Text, MetricsByTextKey]:
    """Get metric names and values for every slice.

    Args:
      output_name: The name of the output (optional, only used for multi-output
        models).
      class_id: Used with multi-class metrics to identify a specific class ID.
      k: Used with multi-class metrics to identify the kth predicted value.
      top_k: Used with multi-class and ranking metrics to identify top-k
        predicted values.

    Returns:
      Dictionary mapping slices to metric names and values.
    """

    if class_id or k or top_k:
      sub_key = str(metric_types.SubKey(class_id, k, top_k))
    else:
      sub_key = ''

    sliced_metrics = {}
    for slicing_metric in self.slicing_metrics:
      slice_name = slicing_metric[0]
      metrics = slicing_metric[1][output_name][sub_key]
      sliced_metrics[slice_name] = {
          metric_name: metric_value
          for metric_name, metric_value in metrics.items()
      }
    return sliced_metrics  # pytype: disable=bad-return-type

  def get_metric_names(self) -> Sequence[Text]:
    """Get names of metrics.

    Returns:
      List of metric names.
    """

    metric_names = set()
    for slicing_metric in self.slicing_metrics:
      for output_name in slicing_metric[1]:
        for metrics in slicing_metric[1][output_name].values():
          metric_names.update(metrics)
    return list(metric_names)

  def get_attributions_for_slice(
      self,
      slice_name: slicer.SliceKeyType = (),
      metric_name: Text = '',
      output_name: Text = '',
      class_id: Optional[int] = None,
      k: Optional[int] = None,
      top_k: Optional[int] = None) -> Union[AttributionsByFeatureKey, None]:
    """Get attribution features names and values for a slice.

    Args:
      slice_name: A tuple of the form (column, value), indicating which slice to
        get attributions from. Optional; if excluded, use overall slice.
      metric_name: Name of metric to get attributions for. Optional if only one
        metric used.
      output_name: The name of the output. Optional, only used for multi-output
        models.
      class_id: Used with multi-class models to identify a specific class ID.
      k: Used with multi-class models to identify the kth predicted value.
      top_k: Used with multi-class models to identify top-k attribution values.

    Returns:
      Dictionary containing feature keys and values for the specified slice.

    Raises:
      ValueError: If metric_name is required.
    """

    if class_id or k or top_k:
      sub_key = str(metric_types.SubKey(class_id, k, top_k))
    else:
      sub_key = ''

    def equals_slice_name(slice_key):
      if not slice_key:
        return not slice_name
      else:
        return slice_key == slice_name

    for sliced_attributions in self.attributions:
      slice_key = sliced_attributions[0]
      slice_val = sliced_attributions[1]
      if equals_slice_name(slice_key):
        if metric_name:
          return slice_val[output_name][sub_key][metric_name]
        elif len(slice_val[output_name][sub_key]) == 1:
          return list(slice_val[output_name][sub_key].values())[0]
        else:
          raise ValueError(
              'metric_name must be one of the following: {}'.format(
                  slice_val[output_name][sub_key].keys()))

    # if slice could not be found, return None
    return None

  def get_attributions_for_all_slices(
      self,
      metric_name: Text = '',
      output_name: Text = '',
      class_id: Optional[int] = None,
      k: Optional[int] = None,
      top_k: Optional[int] = None) -> Dict[Text, AttributionsByFeatureKey]:
    """Get attribution feature keys and values for every slice.

    Args:
      metric_name: Name of metric to get attributions for. Optional if only one
        metric used.
      output_name: The name of the output (optional, only used for multi-output
        models).
      class_id: Used with multi-class metrics to identify a specific class ID.
      k: Used with multi-class metrics to identify the kth predicted value.
      top_k: Used with multi-class and ranking metrics to identify top-k
        predicted values.

    Returns:
      Dictionary mapping slices to attribution feature keys and values.
    """

    if class_id or k or top_k:
      sub_key = str(metric_types.SubKey(class_id, k, top_k))
    else:
      sub_key = ''

    all_sliced_attributions = {}
    for sliced_attributions in self.attributions:
      slice_name = sliced_attributions[0]
      attributions = sliced_attributions[1][output_name][sub_key]
      if metric_name:
        attributions = attributions[metric_name]
      elif len(attributions) == 1:
        attributions = list(attributions.values())[0]
      else:
        raise ValueError('metric_name must be one of the following: {}'.format(
            attributions.keys()))
      all_sliced_attributions[slice_name] = copy.copy(attributions)
    return all_sliced_attributions  # pytype: disable=bad-return-type

  def get_slice_names(self) -> Sequence[Text]:
    """Get names of slices.

    Returns:
      List of slice names.
    """

    return [slicing_metric[0] for slicing_metric in self.slicing_metrics]  # pytype: disable=bad-return-type


class EvalResults(object):
  """The results from multiple TFMA runs, or a TFMA run on multiple models."""

  def __init__(self,
               results: List[EvalResult],
               mode: Text = constants.UNKNOWN_EVAL_MODE):
    supported_modes = [
        constants.DATA_CENTRIC_MODE,
        constants.MODEL_CENTRIC_MODE,
    ]
    if mode not in supported_modes:
      raise ValueError('Mode ' + mode + ' must be one of ' +
                       Text(supported_modes))

    self._results = results
    self._mode = mode

  def get_results(self) -> List[EvalResult]:
    return self._results

  def get_mode(self) -> Text:
    return self._mode
