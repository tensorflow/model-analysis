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
"""Mean absolute error and mean absolute percentage error."""

import abc
import dataclasses
from typing import Iterable, Optional, Dict

import apache_beam as beam
import numpy as np
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.proto import config_pb2


MEAN_ABSOLUTE_ERROR_NAME = 'mean_absolute_error'
MEAN_SQUARED_ERROR_NAME = 'mean_squared_error'
MEAN_ABSOLUTE_PERCENTAGE_ERROR_NAME = 'mean_absolute_percentage_error'
MEAN_SQUARED_LOGARITHMIC_ERROR_NAME = 'mean_squared_logarithmic_error'


class MeanAbsoluteError(metric_types.Metric):
  """Calculates the mean of absolute error between labels and predictions.

  Formula: error = abs(label - prediction)

  The metric computes the mean of absolute error between labels and
  predictions. The labels and predictions should be floats.
  """

  def __init__(self, name: str = MEAN_ABSOLUTE_ERROR_NAME):
    """Initializes mean regression error metric.

    Args:
      name: The name of the metric.
    """
    super().__init__(
        metric_util.merge_per_key_computations(
            _mean_absolute_error_computations),
        name=name)


def _mean_absolute_error_computations(
    name: Optional[str] = None,
    eval_config: Optional[config_pb2.EvalConfig] = None,
    model_name: Optional[str] = None,
    output_name: Optional[str] = None,
    example_weighted: bool = False,
    sub_key: Optional[metric_types.SubKey] = None,
    aggregation_type: Optional[metric_types.AggregationType] = None,
    class_weights: Optional[Dict[int, float]] = None
) -> metric_types.MetricComputations:
  """Returns metric computations for mean absolute error computations.

  Args:
    name: The name of the metric.
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
      example_weighted=example_weighted)
  return [
      metric_types.MetricComputation(
          keys=[key],
          preprocessors=None,
          combiner=_MeanAbsoluteErrorCombiner(
              eval_config=eval_config,
              model_name=model_name,
              output_name=output_name,
              metric_key=key,
              class_weights=class_weights,
              aggregation_type=aggregation_type,
              example_weighted=example_weighted,
          ))
  ]


metric_types.register_metric(MeanAbsoluteError)


class MeanAbsolutePercentageError(metric_types.Metric):
  """Calculates the mean of absolute percentage error.

  Formula: error = 100 * abs( (label - prediction) / label )

  The metric computes the mean of absolute percentage error between labels and
  predictions. The labels and predictions should be floats.
  """

  def __init__(self, name: str = MEAN_ABSOLUTE_PERCENTAGE_ERROR_NAME):
    """Initializes mean regression error metric.

    Args:
      name: The name of the metric.
    """
    super().__init__(
        metric_util.merge_per_key_computations(
            _mean_absolute_percentage_error_computations),
        name=name)


def _mean_absolute_percentage_error_computations(
    name: Optional[str] = None,
    eval_config: Optional[config_pb2.EvalConfig] = None,
    model_name: Optional[str] = None,
    output_name: Optional[str] = None,
    example_weighted: bool = False,
    sub_key: Optional[metric_types.SubKey] = None,
    aggregation_type: Optional[metric_types.AggregationType] = None,
    class_weights: Optional[Dict[int, float]] = None
) -> metric_types.MetricComputations:
  """Returns metric computations for mean absolute percentage error.

  Args:
    name: The name of the metric.
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
      example_weighted=example_weighted)
  return [
      metric_types.MetricComputation(
          keys=[key],
          preprocessors=None,
          combiner=_MeanAbsolutePercentageErrorCombiner(
              eval_config=eval_config,
              model_name=model_name,
              output_name=output_name,
              metric_key=key,
              class_weights=class_weights,
              aggregation_type=aggregation_type,
              example_weighted=example_weighted,
          ))
  ]


metric_types.register_metric(MeanAbsolutePercentageError)


class MeanSquaredError(metric_types.Metric):
  """Calculates the mean of squared error between labels and predictions.

  Formula: error = L2_norm(label - prediction)**2

  The metric computes the mean of squared error (square of L2 norm) between
  labels and predictions. The labels and predictions could be arrays of
  arbitrary dimensions. Their dimension should match.
  """

  def __init__(self, name: str = MEAN_SQUARED_ERROR_NAME):
    """Initializes mean regression error metric.

    Args:
      name: The name of the metric.
    """
    super().__init__(
        metric_util.merge_per_key_computations(
            _mean_squared_error_computations),
        name=name)


def _mean_squared_error_computations(
    name: Optional[str] = None,
    eval_config: Optional[config_pb2.EvalConfig] = None,
    model_name: Optional[str] = None,
    output_name: Optional[str] = None,
    example_weighted: bool = False,
    sub_key: Optional[metric_types.SubKey] = None,
    aggregation_type: Optional[metric_types.AggregationType] = None,
    class_weights: Optional[Dict[int, float]] = None
) -> metric_types.MetricComputations:
  """Returns metric computations for mean squared error computations.

  Args:
    name: The name of the metric.
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
      example_weighted=example_weighted)
  return [
      metric_types.MetricComputation(
          keys=[key],
          preprocessors=None,
          combiner=_MeanSquaredErrorCombiner(
              eval_config=eval_config,
              model_name=model_name,
              output_name=output_name,
              metric_key=key,
              example_weighted=example_weighted,
              aggregation_type=aggregation_type,
              class_weights=class_weights,
          ))
  ]


metric_types.register_metric(MeanSquaredError)


class MeanSquaredLogarithmicError(metric_types.Metric):
  """Calculates the mean of squared logarithmic error.

  Formula: error = L2_norm(log(label + 1) - log(prediction + 1))**2
  Note: log of an array will be elementwise,
    i.e. log([x1, x2]) = [log(x1), log(x2)]

  The metric computes the mean of squared logarithmic error (square of L2 norm)
  between labels and predictions. The labels and predictions could be arrays of
  arbitrary dimensions. Their dimension should match.
  """

  def __init__(self, name: str = MEAN_SQUARED_LOGARITHMIC_ERROR_NAME):
    """Initializes mean regression error metric.

    Args:
      name: The name of the metric.
    """
    super().__init__(
        metric_util.merge_per_key_computations(
            _mean_squared_logarithmic_error_computations),
        name=name)


def _mean_squared_logarithmic_error_computations(
    name: Optional[str] = None,
    eval_config: Optional[config_pb2.EvalConfig] = None,
    model_name: Optional[str] = None,
    output_name: Optional[str] = None,
    example_weighted: bool = False,
    sub_key: Optional[metric_types.SubKey] = None,
    aggregation_type: Optional[metric_types.AggregationType] = None,
    class_weights: Optional[Dict[int, float]] = None
) -> metric_types.MetricComputations:
  """Returns metric computations for mean squared logarithmic error.

  Args:
    name: The name of the metric.
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
      example_weighted=example_weighted)
  return [
      metric_types.MetricComputation(
          keys=[key],
          preprocessors=None,
          combiner=_MeanSquaredLogarithmicErrorCombiner(
              eval_config=eval_config,
              model_name=model_name,
              output_name=output_name,
              metric_key=key,
              example_weighted=example_weighted,
              aggregation_type=aggregation_type,
              class_weights=class_weights,
          ))
  ]


metric_types.register_metric(MeanSquaredLogarithmicError)


@dataclasses.dataclass
class _MeanRegressionErrorAccumulator:
  """Accumulator for computing MeanRegressionError."""
  total_example_weights: float = 0.0
  total_regression_error: float = 0.0

  def merge(self, other: '_MeanRegressionErrorAccumulator'):
    self.total_example_weights += other.total_example_weights
    self.total_regression_error += other.total_regression_error


class _MeanRegressionErrorCombiner(beam.CombineFn, metaclass=abc.ABCMeta):
  """A combiner which computes metrics averaging regression errors."""

  def __init__(self, eval_config: config_pb2.EvalConfig, model_name: str,
               output_name: str, metric_key: metric_types.MetricKey,
               aggregation_type: Optional[metric_types.AggregationType],
               class_weights: Optional[Dict[int,
                                            float]], example_weighted: bool):
    self._eval_config = eval_config
    self._model_name = model_name
    self._output_name = output_name
    self._metric_key = metric_key
    self._example_weighted = example_weighted
    self._aggregation_type = aggregation_type
    self._class_weights = class_weights

  @abc.abstractmethod
  def _regression_error(self, label, prediction) -> float:
    """Returns the regression error between the label and prediction.

    Subclasses must override this method. Labels and preditctions could be
    an array, a float, a string and etc. But the output of regression error must
    be a float.

    Args:
      label: label.
      prediction: prediction from the model.
    """
    raise NotImplementedError('Must be implemented in subclasses.')

  def create_accumulator(self) -> _MeanRegressionErrorAccumulator:
    return _MeanRegressionErrorAccumulator()

  def add_input(
      self, accumulator: _MeanRegressionErrorAccumulator,
      element: metric_types.StandardMetricInputs
  ) -> _MeanRegressionErrorAccumulator:

    lpe_iterator = metric_util.to_label_prediction_example_weight(
        element,
        eval_config=self._eval_config,
        model_name=self._metric_key.model_name,
        output_name=self._metric_key.output_name,
        aggregation_type=self._aggregation_type,
        class_weights=self._class_weights,
        example_weighted=self._example_weighted,
        sub_key=self._metric_key.sub_key,
    )
    for label, prediction, example_weight in lpe_iterator:
      # The np.item method makes sure the result is a one element numpy array
      # and returns the single element as a float.
      error = self._regression_error(label, prediction)
      if not np.isnan(error):
        accumulator.total_regression_error += error * example_weight.item()
        accumulator.total_example_weights += example_weight.item()

    return accumulator

  def merge_accumulators(
      self, accumulators: Iterable[_MeanRegressionErrorAccumulator]
  ) -> _MeanRegressionErrorAccumulator:
    result = next(iter(accumulators))
    for accumulator in accumulators:
      result.merge(accumulator)
    return result

  def extract_output(
      self,
      accumulator: _MeanRegressionErrorAccumulator) -> metric_types.MetricsDict:
    if accumulator.total_example_weights != 0.0:
      result = accumulator.total_regression_error / accumulator.total_example_weights
    else:
      result = float('nan')
    return {self._metric_key: result}


class _MeanAbsoluteErrorCombiner(_MeanRegressionErrorCombiner):
  """A combiner which computes metrics averaging absolute errors."""

  def _regression_error(self, label: np.ndarray,
                        prediction: np.ndarray) -> float:
    # The np.item method makes sure the result is a one element numpy array and
    # returns the single element as a float.
    return np.absolute(label - prediction).item()


class _MeanSquaredErrorCombiner(_MeanRegressionErrorCombiner):
  """A combiner which computes metrics averaging squared errors."""

  def _regression_error(self, label: np.ndarray,
                        prediction: np.ndarray) -> float:
    # The np.item method makes sure the result is a one element numpy array and
    # returns the single element as a float.
    return np.linalg.norm(label - prediction).item()**2


class _MeanAbsolutePercentageErrorCombiner(_MeanRegressionErrorCombiner):
  """A combiner which computes metrics averaging absolute percentage errors."""

  def _regression_error(self, label: np.ndarray,
                        prediction: np.ndarray) -> float:
    # The np.item method makes sure the result is a one element numpy array and
    # returns the single element as a float.
    # The error also requires the label to be a one element numpy array.
    if label.item() == 0:
      return float('nan')
    return 100 * np.absolute((label - prediction) / label).item()


class _MeanSquaredLogarithmicErrorCombiner(_MeanRegressionErrorCombiner):
  """A combiner which computes metrics averaging squared logarithmic errors."""

  def _regression_error(self, label: np.ndarray,
                        prediction: np.ndarray) -> float:
    # The np.item method makes sure the result is a one element numpy array and
    # returns the single element as a float.
    return np.linalg.norm(np.log(label + 1) - np.log(prediction + 1)).item()**2
