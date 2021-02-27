# Lint as: python3
# Copyright 2021 Google LLC
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
"""TF metric accumulators."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

from typing import Any, Callable, List, Optional, Text, Tuple, Union

import numpy as np
from tensorflow_model_analysis import config
from tensorflow_model_analysis import size_estimator
from tensorflow_model_analysis.metrics import metric_util


class TFMetricsAccumulator(object):
  """Accumulator for TF metrics.

  Attributes:
    inputs: Accumulated batch of inputs. The inputs are stored in a
      multi-dimensional list. The first dimension is used to index the
      associated output (for single-output models this will only have one item).
      The second dimension is used to store the args used by the combiner. For
      example the args might be a tf.Example if feeding a model or they might be
      (y_true, y_pred, example_weight) for calling update_state directly.
      Batching is done on the last dimension.
    weights: Accumulated weights. The weights are stored in a multi-dimensional
      list where the first dimension is used to index the associated output (for
      single-output models this will only have one item). The second dimension
      is used to store the accumulated weights for each metric associated with
      the output dimension.
    size_estimator: Batch size estimator.
    desired_batch_size: Desired batch size.
  """

  # We really want the batch size to be adaptive like it is in
  # beam.BatchElements(), but there isn't an easy way to make it so. For now
  # we will limit stored inputs to a max overall byte size.
  # TODO(b/73789023): Figure out how to make this batch size dynamic.
  _TOTAL_INPUT_BYTE_SIZE_THRESHOLD = 16 << 20  # 16MiB
  _DEFAULT_DESIRED_BATCH_SIZE = 1000

  __slots__ = ['_inputs', '_weights', '_size_estimator', '_desired_batch_size']

  def __init__(self,
               input_counts: List[int],
               metric_counts: List[int],
               size_estimator_fn: Callable[[Any], int],
               desired_batch_size: Optional[int] = None):
    """Initializes accumulator using a list of metric counts per output.

    Args:
      input_counts: Number of inputs associated with each output index.
      metric_counts: Number of metrics associated with each output index.
      size_estimator_fn: Function to use for estimating the size of the inputs.
      desired_batch_size: FOR TESTING ONLY.
    """
    # Inputs have shape (num_outputs, num_metrics, num_accumulated_inputs)
    self._inputs = []
    # Weights have shape (num_outputs, num_metrics)
    self._weights = []  # type: List[List[Optional[np.ndarray]]]
    for input_count in input_counts:
      self._inputs.append(tuple([] for _ in range(input_count)))
    for output_metric_count in metric_counts:
      self._weights.append([None] * output_metric_count)
    self._size_estimator = size_estimator.SizeEstimator(
        size_threshold=self._TOTAL_INPUT_BYTE_SIZE_THRESHOLD,
        size_fn=size_estimator_fn)
    if desired_batch_size and desired_batch_size > 0:
      self._desired_batch_size = desired_batch_size
    else:
      self._desired_batch_size = self._DEFAULT_DESIRED_BATCH_SIZE

  def len_inputs(self) -> int:
    """Returns length of inputs."""
    return len(self._inputs[0][0])

  def add_input(self, output_index: int, *args):
    """Adds new inputs to the lists of input args stored at output_index."""
    for i, v in enumerate(args):
      self._inputs[output_index][i].append(v)
      if v is not None:
        self._size_estimator.update(v)

  def get_inputs(self, output_index: int) -> Any:
    """Returns input args for output at given offset."""
    return self._inputs[output_index]

  def clear_inputs(self):
    """Clears currently stored inputs."""
    for output_index in range(len(self._inputs)):
      for i in range(len(self._inputs[output_index])):
        del self._inputs[output_index][i][:]
    self._size_estimator.clear()

  def add_weights(self, output_index: int, metric_index: int,
                  weights: np.ndarray):
    """Adds weights for metric at given metric_index and output_index."""
    cur_weights = self._weights[output_index][metric_index]
    if cur_weights is None:
      self._weights[output_index][metric_index] = weights
    else:
      self._weights[output_index][metric_index] = np.add(cur_weights, weights)

  def get_weights(self, output_index: int,
                  metric_index: int) -> Optional[np.ndarray]:
    """Gets currently stored weights for given metric_index and output_index."""
    return self._weights[output_index][metric_index]

  def should_flush(self) -> bool:
    """Returns true if size estimator indicates flush is needed."""
    return (self.len_inputs() >= self._desired_batch_size or
            self._size_estimator.should_flush())

  def get_size_estimate(self) -> int:
    """Returns size estimator associated with accumulator."""
    return self._size_estimator.get_estimate()


def _numpy_array_size_fn(array: np.ndarray) -> int:
  """Size estimator for numpy arrays."""
  return array.nbytes


class TFCompilableMetricsAccumulator(TFMetricsAccumulator):
  """Accumulator for compilable TF metrics.

  Attributes:
    inputs: Accumulated batch of inputs. The inputs are stored in a
      multi-dimensional list. The first dimension is used to index the
      associated output (for single-output models this will only have one item).
      The second dimension is used to store the args passed to update_state
      (i.e. (y_true, y_pred, example_weight)). Batching is done on the last
      dimension.calling update_state directly. Batching is done on the last
      dimension.
    weights: Accumulated weights. The weights are stored in a multi-dimensional
      list where the first dimension is used to index the associated output (for
      single-output models this will only have one item). The second dimension
      is used to store the accumulated weights for each metric associated with
      the output dimension.
    pad: True if padding needed.
    last_dim: Max size of the last dimension of labels or predictions (used with
      padding).
    size_estimator: Batch size estimator.
    desired_batch_size: Desired batch size.
  """

  __slots__ = [
      '_inputs', '_weights', '_pad', '_pad_to_dim', '_label_padding',
      '_prediction_padding', '_size_estimator', '_desired_batch_size'
  ]

  def __init__(self,
               padding_options: Optional[config.PaddingOptions],
               metric_counts: List[int],
               desired_batch_size: Optional[int] = None):
    """Initializes accumulator using a list of metric counts per output."""
    super(TFCompilableMetricsAccumulator, self).__init__(
        # Input args of labels, predictions, example_weights for each output.
        input_counts=[3] * len(metric_counts),
        metric_counts=metric_counts,
        size_estimator_fn=_numpy_array_size_fn,
        desired_batch_size=desired_batch_size)

    self._pad = False
    if padding_options is not None:

      def get_padding_value(oneof_name):
        oneof = padding_options.WhichOneof(oneof_name)
        return None if oneof is None else getattr(padding_options, oneof)

      self._pad = True
      self._label_padding = get_padding_value('label_padding')
      self._prediction_padding = get_padding_value('prediction_padding')
      self._pad_to_dim = 0

  def add_input(self, output_index: int, label: np.ndarray,
                prediction: np.ndarray, example_weight: np.ndarray):
    """Adds label, prediction, and example weight to output_index."""
    super(TFCompilableMetricsAccumulator,
          self).add_input(output_index, label, prediction, example_weight)
    if self._pad:
      self._pad_to_dim = max(self._pad_to_dim, label.shape[-1],
                             prediction.shape[-1])

  def get_inputs(
      self, output_index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns labels, predictions, and weights for output at given offset."""
    labels, preds, example_weights = super(TFCompilableMetricsAccumulator,
                                           self).get_inputs(output_index)
    if self._pad:

      def pad_value(
          name: Text, a: np.ndarray,
          configured_value: Optional[Union[float, int]]) -> Union[int, float]:
        if configured_value is None:
          return 0 if a.dtype.kind == 'i' else .0
        if isinstance(configured_value, int) and a.dtype.kind == 'i':
          return configured_value
        if isinstance(configured_value, float) and a.dtype.kind == 'f':
          return configured_value
        raise ValueError('%s padding is configured to be %s but data is %s' %
                         (name, type(configured_value), a.dtype))

      labels = [
          metric_util.pad(l, self._pad_to_dim,
                          pad_value('label', l, self._label_padding))
          for l in labels
      ]
      preds = [
          metric_util.pad(p, self._pad_to_dim,
                          pad_value('prediction', p, self._prediction_padding))
          for p in preds
      ]
    return (np.array(labels), np.array(preds), np.array(example_weights))

  def clear_inputs(self):
    """Clears currently stored inputs."""
    super(TFCompilableMetricsAccumulator, self).clear_inputs()
    self._pad_to_dim = 0
