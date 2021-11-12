# Copyright 2019 Google LLC
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
"""Aggregation metrics."""

from typing import Any, Dict, Iterable, List, Optional

import apache_beam as beam
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.proto import config_pb2

_CLASS_WEIGHTS_FROM_LABELS_NAME = '_class_weights_from_labels'


def output_average(
    metric_name: str,
    output_weights: Dict[str, float],
    eval_config: Optional[config_pb2.EvalConfig] = None,
    model_name: str = '',
    sub_key: Optional[metric_types.SubKey] = None,
    example_weighted: bool = False) -> metric_types.MetricComputations:
  """Returns metric computations for computing output average of given metric.

  Args:
    metric_name: Name of underlying metric average is being computed for.
    output_weights: Output weights to use to compute metric.
    eval_config: Eval config.
    model_name: Optional model name.
    sub_key: Optional sub key associated with metric (e.g. top_k).
    example_weighted: True if example weights should be applied.

  Returns:
    Computation for performing the output average.
  """
  del eval_config

  key = metric_types.MetricKey(
      name=metric_name,
      model_name=model_name,
      sub_key=sub_key,
      example_weighted=example_weighted)

  def result(
      metrics: Dict[metric_types.MetricKey, float]
  ) -> Dict[metric_types.MetricKey, float]:
    """Returns output average."""
    total_value = 0.0
    total_weight = 0.0
    for output_name, output_weight in output_weights.items():
      child_key = metric_types.MetricKey(
          name=metric_name,
          model_name=model_name,
          output_name=output_name,
          sub_key=sub_key,
          example_weighted=example_weighted)
      total_value += _to_float(metrics[child_key]) * output_weight
      total_weight += output_weight
    average = total_value / total_weight if total_weight else float('nan')
    return {key: average}

  return [metric_types.DerivedMetricComputation(keys=[key], result=result)]


def macro_average(
    metric_name: str,
    sub_keys: Iterable[metric_types.SubKey],
    eval_config: Optional[config_pb2.EvalConfig] = None,
    model_name: str = '',
    output_name: str = '',
    sub_key: Optional[metric_types.SubKey] = None,
    class_weights: Optional[Dict[int, float]] = None,
    example_weighted: bool = False) -> metric_types.MetricComputations:
  """Returns metric computations for computing macro average of given metric.

  Args:
    metric_name: Name of underlying metric average is being computed for.
    sub_keys: Sub keys used to compute the metric (e.g. class_ids, etc).
    eval_config: Eval config.
    model_name: Optional model name.
    output_name: Optional output name.
    sub_key: Optional sub key associated with aggregation metric (e.g. top_k).
    class_weights: Optional class weights to apply. Required if sub_key is not
      provided. If class_weights are provided, but a sub_key.class_id (if
      sub_key is None) or sub_key.k (if sub_key is top_k) is not set or not
      found in the dictionary then 0.0 is assumed.
    example_weighted: True if example weights should be applied.

  Returns:
    Computation for performing the macro average.
  """
  del eval_config

  key = metric_types.MetricKey(
      name=metric_name,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key,
      aggregation_type=metric_types.AggregationType(macro_average=True),
      example_weighted=example_weighted)

  def result(
      metrics: Dict[metric_types.MetricKey, float]
  ) -> Dict[metric_types.MetricKey, float]:
    """Returns macro average."""
    total_value = 0.0
    total_weight = 0.0
    for sub_key in sub_keys:
      child_key = metric_types.MetricKey(
          name=metric_name,
          model_name=model_name,
          output_name=output_name,
          sub_key=sub_key,
          example_weighted=example_weighted)
      if child_key not in metrics:
        # Use private name if not found under metric name
        child_key = metric_types.MetricKey(
            name='_' + metric_name,
            model_name=model_name,
            output_name=output_name,
            sub_key=sub_key,
            example_weighted=example_weighted)
      weight = 1.0 if not class_weights else 0.0
      offset = None
      if (child_key.sub_key is not None and
          child_key.sub_key.class_id is not None):
        offset = child_key.sub_key.class_id
      elif child_key.sub_key is not None and child_key.sub_key.k is not None:
        offset = child_key.sub_key.k
      if offset is not None and offset in class_weights:
        weight = class_weights[offset]
      total_value += _to_float(metrics[child_key]) * weight
      total_weight += weight
    average = total_value / total_weight if total_weight else float('nan')
    return {key: average}

  return [metric_types.DerivedMetricComputation(keys=[key], result=result)]


def weighted_macro_average(
    metric_name: str,
    sub_keys: Iterable[metric_types.SubKey],
    eval_config: Optional[config_pb2.EvalConfig] = None,
    model_name: str = '',
    output_name: str = '',
    sub_key: Optional[metric_types.SubKey] = None,
    class_weights: Optional[Dict[int, float]] = None,
    example_weighted: bool = False) -> metric_types.MetricComputations:
  """Returns metric computations for computing weighted macro average of metric.

  The weights per class are based on the percentage of positive labels for each
  class.

  Args:
    metric_name: Name of metric weighted average is being computed for.
    sub_keys: Sub keys used to compute the metric (e.g. class_ids, etc).
    eval_config: Eval config.
    model_name: Optional model name.
    output_name: Optional output name.
    sub_key: Optional sub key associated with aggregation metric (e.g. top_k).
    class_weights: Optional class weights to apply. Required if sub_key is not
      provided. If class_weights are provided, but a sub_key.class_id (if
      sub_key is None) or sub_key.k (if sub_key is top_k) is not set or not
      found in the dictionary then 0.0 is assumed. Note that these weights are
      applied in addition to the weights based on the positive labels for each
      class.
    example_weighted: True if example weights should be applied.

  Returns:
    Computation for performing the weighted macro average.
  """
  key = metric_types.MetricKey(
      name=metric_name,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key,
      aggregation_type=metric_types.AggregationType(macro_average=True),
      example_weighted=example_weighted)

  class_ids = [k.class_id for k in sub_keys if k.class_id is not None]

  # Compute the weights for labels.
  computations = _class_weights_from_labels(
      class_ids=class_ids,
      eval_config=eval_config,
      model_name=model_name,
      output_name=output_name,
      example_weighted=example_weighted)
  # Class weights metrics are based on a single computation and key.
  class_weights_from_labels_key = computations[0].keys[0]

  def result(
      metrics: Dict[metric_types.MetricKey, Any]
  ) -> Dict[metric_types.MetricKey, float]:
    """Returns weighted macro average."""
    class_weights_from_labels = metrics[class_weights_from_labels_key]
    total_value = 0.0
    total_weight = 0.0
    for sub_key in sub_keys:
      child_key = metric_types.MetricKey(
          name=metric_name,
          model_name=model_name,
          output_name=output_name,
          sub_key=sub_key,
          example_weighted=example_weighted)
      if child_key not in metrics:
        # Use private name if not found under metric name
        child_key = metric_types.MetricKey(
            name='_' + metric_name,
            model_name=model_name,
            output_name=output_name,
            sub_key=sub_key,
            example_weighted=example_weighted)
      weight = 1.0 if not class_weights else 0.0
      offset = None
      if (child_key.sub_key is not None and
          child_key.sub_key.class_id is not None):
        offset = child_key.sub_key.class_id
      elif child_key.sub_key is not None and child_key.sub_key.k is not None:
        offset = child_key.sub_key.k
      if offset is not None:
        if (class_weights_from_labels and
            child_key.sub_key.class_id in class_weights_from_labels):
          weight = class_weights_from_labels[offset]
        if class_weights and child_key.sub_key.class_id in class_weights:
          weight *= class_weights[offset]
      total_value += _to_float(metrics[child_key]) * weight
      total_weight += weight
    average = total_value / total_weight if total_weight else float('nan')
    return {key: average}

  derived_computation = metric_types.DerivedMetricComputation(
      keys=[key], result=result)
  computations.append(derived_computation)
  return computations


def _to_float(value: Any) -> float:
  try:
    return float(value)
  except (ValueError, TypeError):
    raise ValueError(
        '{} is not aggregatable: value={}\n\nThis is most likely caused by a '
        'configuration error in which the aggregate option was applied '
        'incorrectly.'.format(value.__class__.__name__, value))


def _class_weights_from_labels(
    class_ids: List[int],
    name: str = _CLASS_WEIGHTS_FROM_LABELS_NAME,
    eval_config: Optional[config_pb2.EvalConfig] = None,
    model_name: str = '',
    output_name: str = '',
    example_weighted: bool = False) -> metric_types.MetricComputations:
  """Returns metric computations for class weights based on labels.

  Args:
    class_ids: List of class Ids to compute weighted labels from.
    name: Metric name.
    eval_config: Eval config.
    model_name: Optional model name (if multi-model evaluation).
    output_name: Optional output name (if multi-output model type).
    example_weighted: True if example weights should be applied.
  """
  key = metric_types.MetricKey(
      name=name,
      model_name=model_name,
      output_name=output_name,
      example_weighted=example_weighted)
  return [
      metric_types.MetricComputation(
          keys=[key],
          preprocessor=None,  # Use default
          combiner=_ClassWeightsFromLabelsCombiner(
              key,
              eval_config=eval_config,
              example_weighted=example_weighted,
              class_ids=class_ids))
  ]


class _ClassWeightsFromLabelsCombiner(beam.CombineFn):
  """Computes class weights from labels."""

  def __init__(self, key: metric_types.MetricKey,
               eval_config: Optional[config_pb2.EvalConfig],
               class_ids: List[int], example_weighted: bool):
    self._key = key
    self._eval_config = eval_config
    self._class_ids = class_ids
    self._example_weighted = example_weighted

  def create_accumulator(self) -> Dict[int, float]:
    return {i: 0.0 for i in self._class_ids}

  def add_input(self, accumulator: Dict[int, float],
                element: metric_types.StandardMetricInputs) -> Dict[int, float]:
    for label, _, example_weight in (
        metric_util.to_label_prediction_example_weight(
            element,
            eval_config=self._eval_config,
            model_name=self._key.model_name,
            output_name=self._key.output_name,
            example_weighted=self._example_weighted,
            flatten=False,
            allow_none=True,
            require_single_example_weight=True)):
      example_weight = float(example_weight)
      if label is not None:
        for class_id in self._class_ids:
          if label.size == 1:
            label_value = float(label.item() == class_id)
          else:
            if class_id >= len(label):
              raise ValueError(
                  'class_id {} used with weighted_macro_average is outside the '
                  'range of the label provided: label={}, '
                  'StandardMetricInput={}'.format(class_id, label, element))
            label_value = float(label[class_id])
          accumulator[class_id] += label_value * example_weight
    return accumulator

  def merge_accumulators(
      self, accumulators: Iterable[Dict[int, float]]) -> Dict[int, float]:
    accumulators = iter(accumulators)
    result = next(accumulators)
    for accumulator in accumulators:
      for k, v in accumulator.items():
        result[k] += v
    return result

  def extract_output(
      self, accumulator: Dict[int, float]
  ) -> Dict[metric_types.MetricKey, Dict[int, float]]:
    total = sum(v for v in accumulator.values())
    class_weights = {
        k: (v / total) if total else 0.0 for k, v in accumulator.items()
    }
    return {self._key: class_weights}
