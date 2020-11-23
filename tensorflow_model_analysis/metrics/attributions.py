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
"""Attribution related metrics."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

from typing import Any, Dict, Iterable, List, Optional, Text, Union

import apache_beam as beam
import numpy as np
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis import util
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.metrics import weighted_example_count

TOTAL_ATTRIBUTIONS_NAME = 'total_attributions'
TOTAL_ABSOLUTE_ATTRIBUTIONS_NAME = 'total_absolute_attributions'
MEAN_ATTRIBUTIONS_NAME = 'mean_attributions'
MEAN_ABSOLUTE_ATTRIBUTIONS_NAME = 'mean_absolute_attributions'


class AttributionsMetric(metric_types.Metric):
  """Base type for attribution metrics."""


class MeanAttributions(AttributionsMetric):
  """Mean attributions metric."""

  def __init__(self, name: Text = MEAN_ATTRIBUTIONS_NAME):
    """Initializes mean attributions metric.

    Args:
      name: Attribution metric name.
    """
    super(MeanAttributions, self).__init__(
        metric_util.merge_per_key_computations(_mean_attributions),
        absolute=False,
        name=name)


metric_types.register_metric(MeanAttributions)


class MeanAbsoluteAttributions(AttributionsMetric):
  """Mean aboslute attributions metric."""

  def __init__(self, name: Text = MEAN_ABSOLUTE_ATTRIBUTIONS_NAME):
    """Initializes mean absolute attributions metric.

    Args:
      name: Attribution metric name.
    """
    super(MeanAbsoluteAttributions, self).__init__(
        metric_util.merge_per_key_computations(_mean_attributions),
        absolute=True,
        name=name)


metric_types.register_metric(MeanAbsoluteAttributions)


def _mean_attributions(
    absolute: bool = True,
    name: Text = MEAN_ATTRIBUTIONS_NAME,
    model_name: Text = '',
    output_name: Text = '',
    sub_key: Optional[metric_types.SubKey] = None,
) -> metric_types.MetricComputations:
  """Returns metric computations for mean attributions."""

  key = metric_types.AttributionsKey(
      name=name,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key)

  # Make sure total_attributions is calculated.
  computations = _total_attributions_computations(
      absolute=absolute,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key,
  )
  total_attributions_key = computations[-1].keys[-1]
  # Make sure weighted_example_count is calculated
  computations.extend(
      weighted_example_count.weighted_example_count(
          model_names=[model_name],
          output_names=[output_name],
          sub_keys=[sub_key]))
  weighted_example_count_key = computations[-1].keys[-1]

  def result(
      metrics: Dict[metric_types.MetricKey, Any]
  ) -> Dict[metric_types.AttributionsKey, Dict[Text, Union[float, np.ndarray]]]:
    """Returns mean attributions."""
    total_attributions = metrics[total_attributions_key]
    weighted_count = metrics[weighted_example_count_key]
    attributions = {}
    for k, v in total_attributions.items():
      if np.isclose(weighted_count, 0.0):
        attributions[k] = float('nan')
      else:
        attributions[k] = v / weighted_count
    return {key: attributions}

  derived_computation = metric_types.DerivedMetricComputation(
      keys=[key], result=result)
  computations.append(derived_computation)
  return computations


class TotalAttributions(AttributionsMetric):
  """Total attributions metric."""

  def __init__(self, name: Text = TOTAL_ATTRIBUTIONS_NAME):
    """Initializes total attributions metric.

    Args:
      name: Attribution metric name.
    """
    super(TotalAttributions, self).__init__(
        metric_util.merge_per_key_computations(_total_attributions),
        absolute=False,
        name=name)


metric_types.register_metric(TotalAttributions)


class TotalAbsoluteAttributions(AttributionsMetric):
  """Total absolute attributions metric."""

  def __init__(self, name: Text = TOTAL_ABSOLUTE_ATTRIBUTIONS_NAME):
    """Initializes total absolute attributions metric.

    Args:
      name: Attribution metric name.
    """
    super(TotalAbsoluteAttributions, self).__init__(
        metric_util.merge_per_key_computations(_total_attributions),
        absolute=True,
        name=name)


metric_types.register_metric(TotalAbsoluteAttributions)


def _total_attributions(
    absolute: bool = True,
    name: Text = '',
    model_name: Text = '',
    output_name: Text = '',
    sub_key: Optional[metric_types.SubKey] = None,
) -> metric_types.MetricComputations:
  """Returns metric computations for total attributions."""

  key = metric_types.AttributionsKey(
      name=name,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key)

  # Make sure total_attributions is calculated.
  computations = _total_attributions_computations(
      absolute=absolute,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key,
  )
  private_key = computations[-1].keys[-1]

  def result(
      metrics: Dict[metric_types.MetricKey, Any]
  ) -> Dict[metric_types.AttributionsKey, Dict[Text, Union[float, np.ndarray]]]:
    """Returns total attributions."""
    return {key: metrics[private_key]}

  derived_computation = metric_types.DerivedMetricComputation(
      keys=[key], result=result)
  computations.append(derived_computation)
  return computations


def _total_attributions_computations(
    absolute: bool = True,
    name: Text = '',
    model_name: Text = '',
    output_name: Text = '',
    sub_key: Optional[metric_types.SubKey] = None
) -> metric_types.MetricComputations:
  """Returns metric computations for total attributions.

  Args:
    absolute: True to use absolute value when summing.
    name: Metric name.
    model_name: Optional model name (if multi-model evaluation).
    output_name: Optional output name (if multi-output model type).
    sub_key: Optional sub key.
  """
  if not name:
    if absolute:
      name = '_' + TOTAL_ABSOLUTE_ATTRIBUTIONS_NAME
    else:
      name = '_' + TOTAL_ATTRIBUTIONS_NAME
  key = metric_types.AttributionsKey(
      name=name,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key)
  return [
      metric_types.MetricComputation(
          keys=[key],
          preprocessor=_AttributionsPreprocessor(),
          combiner=_TotalAttributionsCombiner(key, absolute))
  ]


@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(Dict[Text, Any])
class _AttributionsPreprocessor(beam.DoFn):
  """Attributions preprocessor."""

  def process(self, extracts: types.Extracts) -> Iterable[Dict[Text, Any]]:
    if constants.ATTRIBUTIONS_KEY not in extracts:
      raise ValueError(
          '{} missing from extracts {}\n\n. An attribution extractor is '
          'required to use attribution metrics'.format(
              constants.ATTRIBUTIONS_KEY, extracts))
    yield extracts[constants.ATTRIBUTIONS_KEY]


@beam.typehints.with_input_types(Dict[Text, Any])
@beam.typehints.with_output_types(Dict[metric_types.AttributionsKey,
                                       Dict[Text, Union[float, np.ndarray]]])
class _TotalAttributionsCombiner(beam.CombineFn):
  """Computes total attributions."""

  def __init__(self, key: metric_types.AttributionsKey, absolute: bool):
    self._key = key
    self._absolute = absolute

  def _sum(self, a: List[float], b: Union[np.ndarray, List[float]]):
    """Adds values in b to a at matching offsets."""
    if isinstance(b, float) or (isinstance(b, np.ndarray) and b.size == 1):
      if len(a) != 1:
        raise ValueError(
            'Attributions have different array sizes {} != {}'.format(a, b))
      a[0] += abs(float(b)) if self._absolute else float(b)
    else:
      if len(a) != len(b):
        raise ValueError(
            'Attributions have different array sizes {} != {}'.format(a, b))
      for i, v in enumerate(b):
        a[i] += abs(v) if self._absolute else v

  def create_accumulator(self) -> Dict[Text, List[float]]:
    return {}

  def add_input(self, accumulator: Dict[Text, List[float]],
                attributions: Dict[Text, Any]) -> Dict[Text, List[float]]:
    if self._key.model_name:
      attributions = util.get_by_keys(attributions, [self._key.model_name])
    if self._key.output_name:
      attributions = util.get_by_keys(attributions, [self._key.output_name])
    for k, v in attributions.items():
      v = metric_util.to_numpy(v)
      if self._key.sub_key is not None:
        if self._key.sub_key.class_id is not None:
          v = _scores_by_class_id(self._key.sub_key.class_id, v)
        elif self._key.sub_key.k is not None:
          v = _scores_by_top_k(self._key.sub_key.k, v)
          v = np.array(v[self._key.sub_key.k - 1])
        elif self._key.sub_key.top_k is not None:
          v = _scores_by_top_k(self._key.sub_key.top_k, v)
      if k not in accumulator:
        accumulator[k] = [0.0] * v.size
      self._sum(accumulator[k], v)
    return accumulator

  def merge_accumulators(
      self, accumulators: List[Dict[Text,
                                    List[float]]]) -> Dict[Text, List[float]]:
    result = accumulators[0]
    for accumulator in accumulators[1:]:
      for k, v in accumulator.items():
        if k in result:
          self._sum(result[k], v)
        else:
          result[k] = v
    return result

  def extract_output(
      self, accumulator: Dict[Text, List[float]]
  ) -> Dict[metric_types.AttributionsKey, Dict[Text, Union[float, np.ndarray]]]:
    result = {}
    for k, v in accumulator.items():
      result[k] = v[0] if len(v) == 1 else np.array(v)
    return {self._key: result}


def _scores_by_class_id(class_id: int, scores: np.ndarray) -> np.ndarray:
  """Returns selected class ID or raises ValueError."""
  if class_id < 0 or class_id >= len(scores):
    raise ValueError('class_id "{}" out of range for attribution {}'.format(
        class_id, scores))
  return scores[class_id]


def _scores_by_top_k(top_k: int, scores: np.ndarray) -> np.ndarray:
  """Returns top_k scores or raises ValueError if invalid value for top_k."""
  if scores.shape[-1] < top_k:
    raise ValueError(
        'not enough attributions were provided to perform the requested '
        'calcuations for top k. The requested value for k is {}, but the '
        'values are {}\n\nThis may be caused by a metric configuration error '
        'or an error in the pipeline.'.format(top_k, scores))

  indices = np.argpartition(scores, -top_k)[-top_k:]
  indices = indices[np.argsort(-scores[indices])]
  return scores[indices]
