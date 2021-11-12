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

import functools

from typing import Any, Dict, Iterable, List, Optional, Union

import apache_beam as beam
import numpy as np
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.metrics import example_count
from tensorflow_model_analysis.metrics import metric_specs
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.utils import util

TOTAL_ATTRIBUTIONS_NAME = 'total_attributions'
TOTAL_ABSOLUTE_ATTRIBUTIONS_NAME = 'total_absolute_attributions'
MEAN_ATTRIBUTIONS_NAME = 'mean_attributions'
MEAN_ABSOLUTE_ATTRIBUTIONS_NAME = 'mean_absolute_attributions'


class AttributionsMetric(metric_types.Metric):
  """Base type for attribution metrics."""


def has_attributions_metrics(
    metrics_specs: Iterable[config_pb2.MetricsSpec]) -> bool:
  """Returns true if any of the metrics_specs have attributions metrics."""
  tfma_metric_classes = metric_types.registered_metrics()
  for metrics_spec in metrics_specs:
    for metric_config in metrics_spec.metrics:
      instance = metric_specs.metric_instance(metric_config,
                                              tfma_metric_classes)
      if isinstance(instance, AttributionsMetric):
        return True
  return False


class MeanAttributions(AttributionsMetric):
  """Mean attributions metric."""

  def __init__(self, name: str = MEAN_ATTRIBUTIONS_NAME):
    """Initializes mean attributions metric.

    Args:
      name: Attribution metric name.
    """
    super().__init__(
        metric_util.merge_per_key_computations(
            functools.partial(_mean_attributions, False)),
        name=name)


metric_types.register_metric(MeanAttributions)


class MeanAbsoluteAttributions(AttributionsMetric):
  """Mean aboslute attributions metric."""

  def __init__(self, name: str = MEAN_ABSOLUTE_ATTRIBUTIONS_NAME):
    """Initializes mean absolute attributions metric.

    Args:
      name: Attribution metric name.
    """
    super().__init__(
        metric_util.merge_per_key_computations(
            functools.partial(_mean_attributions, True)),
        name=name)


metric_types.register_metric(MeanAbsoluteAttributions)


def _mean_attributions(
    absolute: bool = True,
    name: str = MEAN_ATTRIBUTIONS_NAME,
    eval_config: Optional[config_pb2.EvalConfig] = None,
    model_name: str = '',
    output_name: str = '',
    sub_key: Optional[metric_types.SubKey] = None,
    example_weighted: bool = False,
) -> metric_types.MetricComputations:
  """Returns metric computations for mean attributions."""
  key = metric_types.AttributionsKey(
      name=name,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key,
      example_weighted=example_weighted)

  # Make sure total_attributions is calculated.
  computations = _total_attributions_computations(
      absolute=absolute,
      eval_config=eval_config,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key,
      example_weighted=example_weighted)
  total_attributions_key = computations[-1].keys[-1]
  # Make sure example_count is calculated
  computations.extend(
      example_count.example_count(
          model_names=[model_name],
          output_names=[output_name],
          sub_keys=[sub_key],
          example_weighted=example_weighted))
  example_count_key = computations[-1].keys[-1]

  def result(
      metrics: Dict[metric_types.MetricKey, Any]
  ) -> Dict[metric_types.AttributionsKey, Dict[str, Union[float, np.ndarray]]]:
    """Returns mean attributions."""
    total_attributions = metrics[total_attributions_key]
    count = metrics[example_count_key]
    attributions = {}
    for k, v in total_attributions.items():
      if np.isclose(count, 0.0):
        attributions[k] = float('nan')
      else:
        attributions[k] = v / count
    return {key: attributions}

  derived_computation = metric_types.DerivedMetricComputation(
      keys=[key], result=result)
  computations.append(derived_computation)
  return computations


class TotalAttributions(AttributionsMetric):
  """Total attributions metric."""

  def __init__(self, name: str = TOTAL_ATTRIBUTIONS_NAME):
    """Initializes total attributions metric.

    Args:
      name: Attribution metric name.
    """
    super().__init__(
        metric_util.merge_per_key_computations(
            functools.partial(_total_attributions, False)),
        name=name)


metric_types.register_metric(TotalAttributions)


class TotalAbsoluteAttributions(AttributionsMetric):
  """Total absolute attributions metric."""

  def __init__(self, name: str = TOTAL_ABSOLUTE_ATTRIBUTIONS_NAME):
    """Initializes total absolute attributions metric.

    Args:
      name: Attribution metric name.
    """
    super().__init__(
        metric_util.merge_per_key_computations(
            functools.partial(_total_attributions, True)),
        name=name)


metric_types.register_metric(TotalAbsoluteAttributions)


def _total_attributions(
    absolute: bool = True,
    name: str = '',
    eval_config: Optional[config_pb2.EvalConfig] = None,
    model_name: str = '',
    output_name: str = '',
    sub_key: Optional[metric_types.SubKey] = None,
    example_weighted: bool = False) -> metric_types.MetricComputations:
  """Returns metric computations for total attributions."""
  key = metric_types.AttributionsKey(
      name=name,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key,
      example_weighted=example_weighted)

  # Make sure total_attributions is calculated.
  computations = _total_attributions_computations(
      absolute=absolute,
      eval_config=eval_config,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key,
      example_weighted=example_weighted)
  private_key = computations[-1].keys[-1]

  def result(
      metrics: Dict[metric_types.MetricKey, Any]
  ) -> Dict[metric_types.AttributionsKey, Dict[str, Union[float, np.ndarray]]]:
    """Returns total attributions."""
    return {key: metrics[private_key]}

  derived_computation = metric_types.DerivedMetricComputation(
      keys=[key], result=result)
  computations.append(derived_computation)
  return computations


def _total_attributions_computations(
    absolute: bool = True,
    name: str = '',
    eval_config: Optional[config_pb2.EvalConfig] = None,
    model_name: str = '',
    output_name: str = '',
    sub_key: Optional[metric_types.SubKey] = None,
    example_weighted: bool = False) -> metric_types.MetricComputations:
  """Returns metric computations for total attributions.

  Args:
    absolute: True to use absolute value when summing.
    name: Metric name.
    eval_config: Eval config.
    model_name: Optional model name (if multi-model evaluation).
    output_name: Optional output name (if multi-output model type).
    sub_key: Optional sub key.
    example_weighted: True if example weights should be applied.
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
      sub_key=sub_key,
      example_weighted=example_weighted)
  return [
      metric_types.MetricComputation(
          keys=[key],
          preprocessor=metric_types.AttributionPreprocessor(feature_keys={}),
          combiner=_TotalAttributionsCombiner(key, eval_config, absolute))
  ]


@beam.typehints.with_input_types(metric_types.StandardMetricInputs)
@beam.typehints.with_output_types(Dict[metric_types.AttributionsKey,
                                       Dict[str, Union[float, np.ndarray]]])
class _TotalAttributionsCombiner(beam.CombineFn):
  """Computes total attributions."""

  def __init__(self, key: metric_types.AttributionsKey,
               eval_config: Optional[config_pb2.EvalConfig], absolute: bool):
    self._key = key
    self._eval_config = eval_config
    self._absolute = absolute

  def _sum(self, a: List[float], b: Union[np.ndarray, List[float]]):
    """Adds values in b to a at matching offsets."""
    if (isinstance(b, (float, np.floating)) or
        (isinstance(b, np.ndarray) and b.size == 1)):
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

  def create_accumulator(self) -> Dict[str, List[float]]:
    return {}

  def add_input(
      self, accumulator: Dict[str, List[float]],
      extracts: metric_types.StandardMetricInputs) -> Dict[str, List[float]]:
    if constants.ATTRIBUTIONS_KEY not in extracts:
      raise ValueError(
          '{} missing from extracts {}\n\n. An attribution extractor is '
          'required to use attribution metrics'.format(
              constants.ATTRIBUTIONS_KEY, extracts))
    attributions = extracts[constants.ATTRIBUTIONS_KEY]
    if self._key.model_name:
      attributions = util.get_by_keys(attributions, [self._key.model_name])
    if self._key.output_name:
      attributions = util.get_by_keys(attributions, [self._key.output_name])
    _, _, example_weight = next(
        metric_util.to_label_prediction_example_weight(
            extracts,
            eval_config=self._eval_config,
            model_name=self._key.model_name,
            output_name=self._key.output_name,
            sub_key=self._key.sub_key,
            example_weighted=self._key.example_weighted,
            allow_none=True,
            flatten=False))
    example_weight = float(example_weight)
    for k, v in attributions.items():
      v = util.to_numpy(v)
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
      self._sum(accumulator[k], v * example_weight)
    return accumulator

  def merge_accumulators(
      self,
      accumulators: Iterable[Dict[str, List[float]]]) -> Dict[str, List[float]]:
    accumulators = iter(accumulators)
    result = next(accumulators)
    for accumulator in accumulators:
      for k, v in accumulator.items():
        if k in result:
          self._sum(result[k], v)
        else:
          result[k] = v
    return result

  def extract_output(
      self, accumulator: Dict[str, List[float]]
  ) -> Dict[metric_types.AttributionsKey, Dict[str, Union[float, np.ndarray]]]:
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
