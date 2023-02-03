# Copyright 2023 Google LLC
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
"""Cross entropy metrics."""

import abc
import dataclasses
from typing import Iterable, Optional, Dict

import apache_beam as beam
import numpy as np
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.proto import config_pb2


BINARY_CROSSENTROPY_NAME = 'binary_crossentropy'
CATEGORICAL_CROSSENTROPY_NAME = 'categorical_crossentropy'


class BinaryCrossEntropy(metric_types.Metric):
  """Calculates the binary cross entropy.

  The metric computes the cross entropy when there are only two label classes
  (0 and 1). See definition at: https://en.wikipedia.org/wiki/Cross_entropy
  """

  def __init__(
      self,
      name: str = BINARY_CROSSENTROPY_NAME,
      from_logits: bool = False,
      label_smoothing: float = 0.0,
  ):
    """Initializes binary cross entropy metric.

    Args:
      name: The name of the metric.
      from_logits: (Optional) Whether output is expected to be a logits tensor.
        By default, we consider that output encodes a probability distribution.
      label_smoothing: Float in [0, 1]. If > `0` then smooth the labels by
        squeezing them towards 0.5 That is, using `1. - 0.5 * label_smoothing`
        for the target class and `0.5 * label_smoothing` for the non-target
        class.
    """
    super().__init__(
        metric_util.merge_per_key_computations(
            _binary_cross_entropy_computations
        ),
        name=name,
        from_logits=from_logits,
        label_smoothing=label_smoothing,
    )


def _binary_cross_entropy_computations(
    name: Optional[str] = None,
    from_logits: bool = False,
    label_smoothing: float = 0.0,
    eval_config: Optional[config_pb2.EvalConfig] = None,
    model_name: Optional[str] = None,
    output_name: Optional[str] = None,
    example_weighted: bool = False,
    sub_key: Optional[metric_types.SubKey] = None,
    aggregation_type: Optional[metric_types.AggregationType] = None,
    class_weights: Optional[Dict[int, float]] = None,
) -> metric_types.MetricComputations:
  """Returns metric computations for binary cross entropy.

  Args:
    name: The name of the metric.
    from_logits: (Optional) Whether output is expected to be a logits tensor. By
      default, we consider that output encodes a probability distribution.
    label_smoothing: Float in [0, 1]. If > `0` then smooth the labels by
      squeezing them towards 0.5 That is, using `1. - 0.5 * label_smoothing` for
      the target class and `0.5 * label_smoothing` for the non-target class.
    eval_config: The configurations for TFMA pipeline.
    model_name: The name of the model to get predictions from.
    output_name: The name of the output under the model to get predictions from.
    example_weighted: Whether the examples have specified weights.
    sub_key: The key includes class, top-k, k information. It should only be in
      classfication problems.
    aggregation_type: The method to aggregate over classes. It should only be in
      classfication problems.
    class_weights: The weight of classes. It should only be in classfication
      problems.
  """
  key = metric_types.MetricKey(
      name=name,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key,
      example_weighted=example_weighted,
  )
  return [
      metric_types.MetricComputation(
          keys=[key],
          preprocessors=None,
          combiner=_BinaryCrossEntropyCombiner(
              from_logits=from_logits,
              label_smoothing=label_smoothing,
              eval_config=eval_config,
              model_name=model_name,
              output_name=output_name,
              metric_key=key,
              example_weighted=example_weighted,
              aggregation_type=aggregation_type,
              class_weights=class_weights,
          ),
      )
  ]


metric_types.register_metric(BinaryCrossEntropy)


class CategoricalCrossEntropy(metric_types.Metric):
  """Calculates the categorical cross entropy.

  The metric computes the cross entropy when there are multiple classes.
  It outputs a numpy array.
  """

  def __init__(
      self,
      name: str = CATEGORICAL_CROSSENTROPY_NAME,
      from_logits: bool = False,
      label_smoothing: float = 0.0,
  ):
    """Initializes categorical cross entropy metric.

    Args:
      name: The name of the metric.
      from_logits: (Optional) Whether output is expected to be a logits tensor.
        By default, we consider that output encodes a probability distribution.
      label_smoothing: Float in [0, 1]. If > `0` then smooth the labels. For
        example, if `0.1`, use `0.1 / num_classes` for non-target labels and
        `0.9 + 0.1 / num_classes` for target labels.
    """
    super().__init__(
        metric_util.merge_per_key_computations(
            _categorical_cross_entropy_computations
        ),
        name=name,
        from_logits=from_logits,
        label_smoothing=label_smoothing,
    )


def _categorical_cross_entropy_computations(
    name: Optional[str] = None,
    from_logits: bool = False,
    label_smoothing: float = 0.0,
    eval_config: Optional[config_pb2.EvalConfig] = None,
    model_name: Optional[str] = None,
    output_name: Optional[str] = None,
    example_weighted: bool = False,
    sub_key: Optional[metric_types.SubKey] = None,
    aggregation_type: Optional[metric_types.AggregationType] = None,
    class_weights: Optional[Dict[int, float]] = None,
) -> metric_types.MetricComputations:
  """Returns metric computations for categorical cross entropy.

  Args:
    name: The name of the metric.
    from_logits: (Optional) Whether output is expected to be a logits tensor. By
      default, we consider that output encodes a probability distribution.
    label_smoothing: Float in [0, 1]. If > `0` then smooth the labels. For
      example, if `0.1`, use `0.1 / num_classes` for non-target labels and `0.9
      + 0.1 / num_classes` for target labels.
    eval_config: The configurations for TFMA pipeline.
    model_name: The name of the model to get predictions from.
    output_name: The name of the output under the model to get predictions from.
    example_weighted: Whether the examples have specified weights.
    sub_key: The key includes class, top-k, k information. It should only be in
      classfication problems.
    aggregation_type: The method to aggregate over classes. It should only be in
      classfication problems.
    class_weights: The weight of classes. It should only be in classfication
      problems.
  """
  key = metric_types.MetricKey(
      name=name,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key,
      example_weighted=example_weighted,
  )
  return [
      metric_types.MetricComputation(
          keys=[key],
          preprocessors=None,
          combiner=_CategoricalCrossEntropyCombiner(
              from_logits=from_logits,
              label_smoothing=label_smoothing,
              eval_config=eval_config,
              model_name=model_name,
              output_name=output_name,
              metric_key=key,
              example_weighted=example_weighted,
              aggregation_type=aggregation_type,
              class_weights=class_weights,
          ),
      )
  ]


metric_types.register_metric(CategoricalCrossEntropy)


@dataclasses.dataclass
class _CrossEntropyAccumulator:
  """Accumulator for computing cross entropy metrics."""

  total_cross_entropy: float = 0.0
  total_example_weights: float = 0.0

  def merge(self, other: '_CrossEntropyAccumulator'):
    self.total_cross_entropy += other.total_cross_entropy
    self.total_example_weights += other.total_example_weights


class _CrossEntropyCombiner(beam.CombineFn, metaclass=abc.ABCMeta):
  """A combiner which computes cross entropy metrics.

  Two importnat parameters for cross entropy calcualtion.
    from_logits: (Optional) Whether output is expected to be a logits tensor.
      By default, we consider that output encodes a probability distribution.
    label_smoothing: Float in [0, 1]. If > `0` then smooth the labels. For
      example, if `0.1`, use `0.1 / num_classes` for non-target labels and
      `0.9 + 0.1 / num_classes` for target labels.
  """

  def __init__(
      self,
      eval_config: config_pb2.EvalConfig,
      model_name: str,
      output_name: str,
      metric_key: metric_types.MetricKey,
      aggregation_type: Optional[metric_types.AggregationType],
      class_weights: Optional[Dict[int, float]],
      example_weighted: bool,
      from_logits: bool = False,
      label_smoothing: float = 0.0,
  ):
    self._eval_config = eval_config
    self._model_name = model_name
    self._output_name = output_name
    self._metric_key = metric_key
    self._example_weighted = example_weighted
    self._aggregation_type = aggregation_type
    self._class_weights = class_weights
    self._from_logits = from_logits
    self._label_smoothing = label_smoothing

  @abc.abstractmethod
  def _cross_entropy(self, label, prediction) -> float:
    """Returns the cross entropy between the label and prediction.

    Subclasses must override this method. Preditctions should encode the
    probability distribution. The output of cross entropy is a numpy array.

    Args:
      label: The numpy array of floats. It should be the class probabilities
      prediction: The numpy array of floats. It should be probabilities or
        logits.
    """
    raise NotImplementedError('Must be implemented in subclasses.')

  def create_accumulator(self) -> _CrossEntropyAccumulator:
    return _CrossEntropyAccumulator()

  def add_input(
      self,
      accumulator: _CrossEntropyAccumulator,
      element: metric_types.StandardMetricInputs,
  ) -> _CrossEntropyAccumulator:
    lpe_iterator = metric_util.to_label_prediction_example_weight(
        element,
        eval_config=self._eval_config,
        model_name=self._metric_key.model_name,
        output_name=self._metric_key.output_name,
        aggregation_type=self._aggregation_type,
        class_weights=self._class_weights,
        example_weighted=self._example_weighted,
        sub_key=self._metric_key.sub_key,
        flatten=False,
    )
    for label, prediction, example_weight in lpe_iterator:
      # The np.item method makes sure the result is a one element numpy array
      # and returns the single element as a float.
      accumulator.total_cross_entropy += (
          self._cross_entropy(label, prediction) * example_weight.item()
      )
      accumulator.total_example_weights += example_weight.item()

    return accumulator

  def merge_accumulators(
      self, accumulators: Iterable[_CrossEntropyAccumulator]
  ) -> _CrossEntropyAccumulator:
    result = next(iter(accumulators))
    for accumulator in accumulators:
      result.merge(accumulator)
    return result

  def extract_output(
      self, accumulator: _CrossEntropyAccumulator
  ) -> metric_types.MetricsDict:
    result = np.divide(
        accumulator.total_cross_entropy, accumulator.total_example_weights
    )
    return {self._metric_key: result}


class _BinaryCrossEntropyCombiner(_CrossEntropyCombiner):
  """A combiner which computes binary cross entropy."""

  def _cross_entropy(
      self,
      label: np.ndarray,
      prediction: np.ndarray,
  ) -> float:
    # smooth labels
    label = label * (1.0 - self._label_smoothing) + 0.5 * self._label_smoothing

    # If predictions are logits rather than probability, then the probability
    # should be sigmoid(logits). In this case, starting from logits, we can
    # derive the formula for cross entropy which is expressed in logits.
    # Let y = label, x = prediction logits
    # If x > 0,
    #  Cross entropy loss = y * - log(sigmoid(x)) + (1-y) * -log(1-sigmoid(x))
    #                    = x - x * y + log(1 + exp(-x))
    # If x < 0,
    #  Cross entropy loss = -x * y + log(1 + exp(x))
    # In summary, to merge the x > 0 and x < 0 cases, we obtain,
    # Cross entryopy loss = max(x, 0) - x * y + log(1 + exp(-abs(x)))
    if self._from_logits:
      elementwise_binary_cross_entropy = (
          np.maximum(prediction, 0)
          - np.multiply(prediction, label)
          + np.log(1 + np.exp(-np.abs(prediction)))
      )
    else:
      elementwise_binary_cross_entropy = -np.multiply(
          label, np.log(prediction)
      ) - np.multiply((1 - label), np.log(1 - prediction))
    binary_cross_entropy = np.mean(elementwise_binary_cross_entropy)
    # The np.item method makes sure the result is a one element numpy array and
    # returns the single element as a float.
    return binary_cross_entropy.item()


class _CategoricalCrossEntropyCombiner(_CrossEntropyCombiner):
  """A combiner which computes categorical cross entropy."""

  def _cross_entropy(
      self,
      label: np.ndarray,
      prediction: np.ndarray,
  ) -> float:
    # smooth labels
    num_classes = prediction.shape[0]
    label = (
        label * (1.0 - self._label_smoothing)
        + self._label_smoothing / num_classes
    )

    if self._from_logits:
      # Let z_i be the logits of probability p_i
      # z_i = log( p_i / sum_(j!=i) p_j )
      # p_i = exp(z_i) / sum(exp(z))
      prediction = np.exp(prediction)
    # Normalize prediction probability to 1
    prediction /= np.sum(prediction)

    # It assumes each row is a single prediction and each column is a class.
    # The reduction on axis -1 is a classwise reduction.
    categorical_cross_entropy = -np.sum(
        np.multiply(label, np.ma.log(prediction))
    )

    # The np.item method makes sure the result is a one element numpy array and
    # returns the single element as a float.
    return categorical_cross_entropy.item()
