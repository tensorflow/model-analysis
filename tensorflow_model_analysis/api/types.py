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
"""Types."""

import abc
import datetime
import operator
from typing import Any, Callable, Dict, Iterable, List, MutableMapping, NamedTuple, Optional, Tuple, TypeVar, Union

from apache_beam.utils import shared
import numpy as np
import six
import tensorflow as tf
from tensorflow_model_analysis.proto import metrics_for_slice_pb2


class RaggedTensorValue(
    NamedTuple(
        'RaggedTensorValue',
        [('values', np.ndarray), ('nested_row_splits', List[np.ndarray])],
    )
):
  """RaggedTensorValue encapsulates a batch of ragged tensor values.

  Attributes:
    values: A np.ndarray of values.
    nested_row_splits: A list of np.ndarray values representing the row splits
      (one per dimension including the batch dimension).
  """


class SparseTensorValue(
    NamedTuple(
        'SparseTensorValue',
        [
            ('values', np.ndarray),
            ('indices', np.ndarray),
            ('dense_shape', np.ndarray),
        ],
    )
):
  """SparseTensorValue encapsulates a batch of sparse tensor values.

  Attributes:
    values: A np.ndarray of values.
    indices: A np.ndarray of indices.
    dense_shape: A np.ndarray representing the dense shape.
  """


class VarLenTensorValue(
    NamedTuple(
        'VarLenTensorValue',
        [
            ('values', np.ndarray),
            ('indices', np.ndarray),
            ('dense_shape', np.ndarray),
        ],
    )
):
  """VarLenTensorValue encapsulates a batch of varlen dense tensor values.

  Attributes:
    values: A np.ndarray of values.
    indices: A np.ndarray of indices.
    dense_shape: A np.ndarray representing the dense shape of the entire tensor.
      Note that each row (i.e. set of values sharing the same value for the
      first / batch dimension) is considered to have its own shape based on the
      presence of values.
  """

  def __new__(
      cls, values: np.ndarray, indices: np.ndarray, dense_shape: np.ndarray
  ):
    # we keep the sparse representation despite not needing it so that we can
    # convert back to TF sparse tensors for free.
    if len(dense_shape) != 2:
      raise ValueError(
          'A VarLenTensorValue can only be used to represent a '
          '2D tensor in which the size of the second dimension '
          'varies over rows. However, the provided dense_shape '
          f'({dense_shape}) implies a {dense_shape.size}D tensor'
      )
    row_index_diffs = np.diff(indices[:, 0])
    column_index_diffs = np.diff(indices[:, 1])
    # Enforce row-major ordering of indices by checking that row indices are
    # always increasing, and column indices within the same row are also always
    # increasing.
    bad_index_mask = (row_index_diffs < 0) | (
        (row_index_diffs == 0) & (column_index_diffs < 0)
    )
    if np.any(bad_index_mask):
      raise ValueError(
          'The values and indices arrays must be provided in a '
          'row major order, and represent a set of variable '
          'length dense lists. However, indices['
          f'{np.nonzero(bad_index_mask)[0] + 1}, :] did not '
          'follow this pattern. The full indices array was: '
          f'{indices}.'
      )
    return super().__new__(
        cls, values=values, indices=indices, dense_shape=dense_shape
    )

  class DenseRowIterator:
    """An Iterator over rows of a VarLenTensorValue as dense np.arrays.

    Because the VarLenTensorValue was created from a set of variable length
    (dense) arrays, we can invert this process to turn a VarLenTensorValue back
    into the original dense arrays.
    """

    def __init__(self, tensor):
      self._tensor = tensor
      self._offset = 0

    def __iter__(self):
      return self

    def __next__(self):
      if (
          not self._tensor.indices.size
          or self._offset >= self._tensor.dense_shape[0]
      ):
        raise StopIteration
      row_mask = self._tensor.indices[:, 0] == self._offset
      self._offset += 1
      if not row_mask.any():
        # handle empty rows
        return np.array([])
      # we rely on slice indexing (a[start:end] rather than fancy indexing
      # (a[mask]) to avoid making a copy of each row. For details, see:
      # https://scipy-cookbook.readthedocs.io/items/ViewsVsCopies.html
      row_mask_indices = np.nonzero(row_mask)[0]
      row_start_index = row_mask_indices[0]
      row_end = row_mask_indices[-1] + 1
      assert (row_end - row_start_index) == len(row_mask_indices), (
          'The values for each row in the represented tensor must be '
          'contiguous in the values and indices arrays but found '
          f'row_start_index: {row_start_index}, row_end: {row_end}'
          f'len(row_mask_indices): {len(row_mask_indices)}'
      )
      return self._tensor.values[row_start_index:row_end]

  def dense_rows(self):
    return self.DenseRowIterator(self)

  @classmethod
  def from_dense_rows(
      cls, dense_rows: Iterable[np.ndarray]
  ) -> 'VarLenTensorValue':
    """Converts a collection of variable length dense arrays into a tensor.

    Args:
      dense_rows: A sequence of possibly variable length 1D arrays.

    Returns:
      A new VarLenTensorValue containing the sparse representation of the
      vertically stacked dense rows. The dense_shape attribute on the result
      will be (num_rows, max_row_len).
    """
    rows = []
    index_arrays = []
    max_row_len = 0
    num_rows = 0
    for i, row in enumerate(dense_rows):
      num_rows += 1
      if row.size:
        if row.ndim <= 1:
          # Add a dimension for unsized numpy array. This will solve the problem
          # where scalar numpy arrays like np.array(None), np.array(0) can not
          # be merged with other numpy arrays.
          row = row.reshape(-1)
          rows.append(row)
        else:
          raise ValueError(
              'Each non-empty dense row should be 1D or scalar but'
              f' found row with shape {row.shape}.'
          )
        index_arrays.append(np.array([[i, j] for j in range(len(row))]))
      max_row_len = max(max_row_len, row.size)
    if index_arrays:
      values = np.concatenate(rows, axis=0)
      indices = np.concatenate(index_arrays, axis=0)
    else:
      # empty case
      values = np.array([])
      indices = np.empty((0, 2))
    dense_shape = np.array([num_rows, max_row_len])
    return cls.__new__(
        cls, values=values, indices=indices, dense_shape=dense_shape
    )


# pylint: disable=invalid-name

TensorType = Union[tf.Tensor, tf.SparseTensor, tf.RaggedTensor]
TensorOrOperationType = Union[TensorType, tf.Operation]
DictOfTensorType = Dict[str, TensorType]
TensorTypeMaybeDict = Union[TensorType, DictOfTensorType]
DictOfTensorTypeMaybeDict = Dict[str, TensorTypeMaybeDict]
TensorTypeMaybeMultiLevelDict = Union[
    TensorTypeMaybeDict, DictOfTensorTypeMaybeDict
]

DictOfTypeSpec = Dict[str, tf.TypeSpec]
TypeSpecMaybeDict = Union[tf.TypeSpec, DictOfTypeSpec]
DictOfTypeSpecMaybeDict = Dict[str, TypeSpecMaybeDict]
TypeSpecMaybeMultiLevelDict = Union[TypeSpecMaybeDict, DictOfTypeSpecMaybeDict]

# TODO(b/171992041): Remove tf.compat.v1.SparseTensorValue.
TensorValue = Union[
    np.ndarray,
    SparseTensorValue,
    RaggedTensorValue,
    tf.compat.v1.SparseTensorValue,
]
DictOfTensorValue = Dict[str, TensorValue]
TensorValueMaybeDict = Union[TensorValue, DictOfTensorValue]
DictOfTensorValueMaybeDict = Dict[str, TensorValueMaybeDict]
TensorValueMaybeMultiLevelDict = Union[
    TensorValueMaybeDict, DictOfTensorValueMaybeDict
]

MetricVariablesType = List[Any]

PrimitiveMetricValueType = Union[float, int, np.number]

ConcreteStructuredMetricValue = TypeVar(
    'ConcreteStructuredMetricValue', bound='StructuredMetricValue'
)


class StructuredMetricValue(abc.ABC):
  """The base class for all structured metrics used within TFMA.

  This class allows custom metrics to control how proto serialization happens,
  and how to handle basic algebraic operations used in computing confidence
  intervals and model diffs. By implementing the _apply_binary_op methods,
  subclasses can then be treated like primitive numeric types.
  """

  @abc.abstractmethod
  def to_proto(self) -> metrics_for_slice_pb2.MetricValue:
    ...

  @abc.abstractmethod
  def _apply_binary_op_elementwise(
      self: ConcreteStructuredMetricValue,
      other: ConcreteStructuredMetricValue,
      op: Callable[[float, float], float],
  ) -> ConcreteStructuredMetricValue:
    """Applies the binary operator elementwise on self and `other`.

    Given two structures of the same type, this function's job is to find
    corresponding pairs of elements within both structures, invoke `op` on each
    pair, and store the result in a corresponding location within a new
    structure. For example, to implement for a list, this function could be
    implemented as:

        return [op(elem, other_elem) for elem, other_elem in zip(self, other)]

    Args:
      other: A structure containing elements which should be the second operand
        when applying `op`. `Other` must be a structured metric of the same type
        as self.
      op: A binary operator which should be applied elementwise to corresponding
        primitive values in self and `other`.

    Returns:
      A new structured metric that is the result of elementwise applying `op`
      on corresponding elements within self and `other`.
    """
    ...

  @abc.abstractmethod
  def _apply_binary_op_broadcast(
      self: ConcreteStructuredMetricValue,
      other: float,
      op: Callable[[float, float], float],
  ) -> ConcreteStructuredMetricValue:
    """Applies the binary operator on each element in self and a single float.

    This function supports broadcasting operations on the structured metric by
    applying `op` on each element in self, paired with the primitive value
    `other`. This makes it possible do things like add a fixed quantity to every
    element in a structure. For example, to implement for a list, this function
    could be implemented as:

        return [op(elem, other) for elem in self]

    Args:
      other: The value to be used as the second operand when applying `op`.
      op: A binary operator which should be applied elementwise to each element
        in self and `other`.

    Returns:
      A new structured metric that is the result of applying `op` on each
      element within self and a single value, `other`.
    """
    ...

  def _apply_binary_op(
      self: ConcreteStructuredMetricValue,
      other: Union[PrimitiveMetricValueType, ConcreteStructuredMetricValue],
      op: Callable[[float, float], float],
  ) -> ConcreteStructuredMetricValue:
    if type(other) is type(self):  # pylint: disable=unidiomatic-typecheck
      return self._apply_binary_op_elementwise(other, op)
    elif isinstance(other, (float, int, np.number)):
      return self._apply_binary_op_broadcast(float(other), op)
    else:
      raise ValueError(
          'Binary ops can only be applied elementwise on two instances of the '
          'same StructuredMetricValue subclass or using broadcasting with one '
          'StructuredMetricValue and a primitive numeric type (int, float, '
          'np.number). Cannot apply binary op on objects of type '
          '{} and {}'.format(type(self), type(other))
      )

  def __add__(
      self: ConcreteStructuredMetricValue,
      other: Union[ConcreteStructuredMetricValue, float],
  ):
    return self._apply_binary_op(other, operator.add)

  def __sub__(
      self: ConcreteStructuredMetricValue,
      other: Union[ConcreteStructuredMetricValue, float],
  ):
    return self._apply_binary_op(other, operator.sub)

  def __mul__(
      self: ConcreteStructuredMetricValue,
      other: Union[ConcreteStructuredMetricValue, float],
  ):
    return self._apply_binary_op(other, operator.mul)

  def __truediv__(
      self: ConcreteStructuredMetricValue,
      other: Union[ConcreteStructuredMetricValue, float],
  ):
    return self._apply_binary_op(other, operator.truediv)

  def __pow__(
      self: ConcreteStructuredMetricValue,
      other: Union[ConcreteStructuredMetricValue, float],
  ):
    return self._apply_binary_op(other, operator.pow)


MetricValueType = Union[
    PrimitiveMetricValueType, np.ndarray, StructuredMetricValue
]


class ValueWithTDistribution(
    NamedTuple(
        'ValueWithTDistribution',
        [
            ('sample_mean', MetricValueType),
            ('sample_standard_deviation', MetricValueType),
            ('sample_degrees_of_freedom', int),
            ('unsampled_value', MetricValueType),
        ],
    )
):
  r"""Represents the t-distribution value.

  It includes sample_mean, sample_standard_deviation,
  sample_degrees_of_freedom. And also unsampled_value is also stored here to
  record the value calculated without bootstrapping.
  The sample_standard_deviation is calculated as:
  \sqrt{ \frac{1}{N-1} \sum_{i=1}^{N}{(x_i - \bar{x})^2} }
  """

  def __new__(
      cls,
      sample_mean: float,
      sample_standard_deviation: Optional[float] = None,
      sample_degrees_of_freedom: Optional[int] = None,
      unsampled_value: Optional[float] = None,
  ):
    return super(ValueWithTDistribution, cls).__new__(
        cls,
        sample_mean,
        sample_standard_deviation,
        sample_degrees_of_freedom,
        unsampled_value,
    )

  def __float__(self):
    # unsampled_value can be numpy.float which is a subclass of float, but here
    # need to return a strict float.
    return float(self.unsampled_value)


# AddMetricsCallback should have the following prototype:
#   def add_metrics_callback(features_dict, predictions_dict, labels_dict):
#
# It should create and return a metric_ops dictionary, such that
# metric_ops['metric_name'] = (value_op, update_op), just as in the Trainer.
#
# Note that features_dict, predictions_dict and labels_dict are not
# necessarily dictionaries - they might also be Tensors, depending on what the
# model's eval_input_receiver_fn returns.
# pyformat: disable
AddMetricsCallbackType = Callable[[
    TensorTypeMaybeDict, TensorTypeMaybeDict, TensorTypeMaybeDict
], Dict[str, Tuple[TensorType, TensorType]]]
# pyformat: enable

# Type of keys we support for prediction, label and features dictionaries.
FPLKeyType = Union[str, Tuple[str, ...]]

# Dictionary of Tensor values fetched. The dictionary maps original dictionary
# keys => ('node' => value). This type exists for backward compatibility with
# FeaturesPredictionsLabels, new code should use DictOfTensorValue instead.
DictOfFetchedTensorValues = Dict[FPLKeyType, Dict[str, TensorValue]]

FeaturesPredictionsLabels = NamedTuple(
    'FeaturesPredictionsLabels',
    [
        ('input_ref', int),
        ('features', DictOfFetchedTensorValues),
        ('predictions', DictOfFetchedTensorValues),
        ('labels', DictOfFetchedTensorValues),
    ],
)

# Used in building the model diagnostics table, a MaterializedColumn is a value
# inside of Extracts that will be emitted to file. Note that for strings, the
# values are raw byte strings rather than unicode strings. This is by design, as
# features can have arbitrary bytes values.
MaterializedColumn = NamedTuple(
    'MaterializedColumn',
    [
        ('name', str),
        (
            'value',
            Union[List[bytes], List[int], List[float], bytes, int, float],
        ),
    ],
)

# Extracts represent data extracted during pipeline processing. In order to
# provide a flexible API, these types are just dicts where the keys are defined
# (reserved for use) by different extractor implementations. For example, the
# FeaturesExtractor stores the data for the features under the key "features",
# LabelsExtractor stores the data for the labels under the key "labels", etc.
Extracts = MutableMapping[str, Any]

# pylint: enable=invalid-name


class ModelLoader:
  """Model loader is responsible for loading shared model types.

  Attributes:
    construct_fn: A callable which creates the model instance. The callable
      should take no args as input (typically a closure is used to capture
      necessary parameters).
    tags: Optional model tags (e.g. 'serve' for serving or 'eval' for
      EvalSavedModel).
  """

  __slots__ = ['construct_fn', 'tags', '_shared_handle']

  def __init__(
      self, construct_fn: Callable[[], Any], tags: Optional[List[str]] = None
  ):
    self.construct_fn = construct_fn
    self.tags = tags
    self._shared_handle = shared.Shared()

  def load(
      self, model_load_time_callback: Optional[Callable[[int], None]] = None
  ) -> Any:
    """Returns loaded model.

    Args:
      model_load_time_callback: Optional callback to track load time.
    """
    if model_load_time_callback:
      construct_fn = self._construct_fn_with_load_time(model_load_time_callback)
    else:
      construct_fn = self.construct_fn
    return self._shared_handle.acquire(construct_fn)

  def _construct_fn_with_load_time(
      self, model_load_time_callback: Callable[[int], None]
  ) -> Callable[[], Any]:
    """Wraps actual construct fn to allow for load time metrics."""

    def with_load_times():
      start_time = datetime.datetime.now()
      model = self.construct_fn()
      end_time = datetime.datetime.now()
      model_load_time_callback(int((end_time - start_time).total_seconds()))
      return model

    return with_load_times


class EvalSharedModel(
    NamedTuple(
        'EvalSharedModel',
        [
            ('model_path', str),
            (
                'add_metrics_callbacks',
                List[Callable],
            ),  # List[AnyMetricsCallbackType]
            ('include_default_metrics', bool),
            ('example_weight_key', Union[str, Dict[str, str]]),
            ('additional_fetches', List[str]),
            ('model_loader', ModelLoader),
            ('model_name', str),
            ('model_type', str),
            ('rubber_stamp', bool),
            ('is_baseline', bool),
            ('resource_hints', Optional[Dict[str, Any]]),
            ('backend_config', Optional[Any]),
        ],
    )
):
  # pyformat: disable
  """Shared model used during extraction and evaluation.

  Attributes:
    model_path: Path to EvalSavedModel (containing the saved_model.pb file).
    add_metrics_callbacks: Optional list of callbacks for adding additional
      metrics to the graph. The names of the metrics added by the callbacks
      should not conflict with existing metrics. See below for more details
      about what each callback should do. The callbacks are only used during
      evaluation.
    include_default_metrics: True to include the default metrics that are part
      of the saved model graph during evaluation.
    example_weight_key: Example weight key (single-output model) or dict of
      example weight keys (multi-output model) keyed by output_name.
    additional_fetches: Prefixes of additional tensors stored in
      signature_def.inputs that should be fetched at prediction time. The
      "features" and "labels" tensors are handled automatically and should not
      be included in this list.
    model_loader: Model loader.
    model_name: Model name (should align with ModelSpecs.name).
    model_type: Model type (tfma.TF_KERAS, tfma.TF_LITE, tfma.TF_ESTIMATOR, ..).
    rubber_stamp: True if this model is being rubber stamped. When a
      model is rubber stamped diff thresholds will be ignored if an associated
      baseline model is not passed.
    is_baseline: The model is the baseline for comparison or not.
    resource_hints: The beam resource hints to apply to the PTransform which
      runs inference for this model.
    backend_config: The backend config for running model inference.


  More details on add_metrics_callbacks:

    Each add_metrics_callback should have the following prototype:
      def add_metrics_callback(features_dict, predictions_dict, labels_dict):

    Note that features_dict, predictions_dict and labels_dict are not
    necessarily dictionaries - they might also be Tensors, depending on what the
    model's eval_input_receiver_fn returns.

    It should create and return a metric_ops dictionary, such that
    metric_ops['metric_name'] = (value_op, update_op), just as in the Trainer.

    Short example:

    def add_metrics_callback(features_dict, predictions_dict, labels):
      metrics_ops = {}
      metric_ops['mean_label'] = tf.metrics.mean(labels)
      metric_ops['mean_probability'] = tf.metrics.mean(tf.slice(
        predictions_dict['probabilities'], [0, 1], [2, 1]))
      return metric_ops
  """
  # pyformat: enable

  def __new__(
      cls,
      model_path: Optional[str] = None,
      add_metrics_callbacks: Optional[List[AddMetricsCallbackType]] = None,
      include_default_metrics: Optional[bool] = True,
      example_weight_key: Optional[Union[str, Dict[str, str]]] = None,
      additional_fetches: Optional[List[str]] = None,
      model_loader: Optional[ModelLoader] = None,
      model_name: str = '',
      model_type: str = '',
      rubber_stamp: bool = False,
      is_baseline: bool = False,
      resource_hints: Optional[Dict[str, Any]] = None,
      backend_config: Optional[Any] = None,
      construct_fn: Optional[Callable[[], Any]] = None,
  ):
    if not add_metrics_callbacks:
      add_metrics_callbacks = []
    if model_loader and construct_fn:
      raise ValueError(
          'only one of model_loader or construct_fn should be used'
      )
    if construct_fn:
      model_loader = ModelLoader(tags=None, construct_fn=construct_fn)
    if model_path is not None:
      model_path = six.ensure_str(model_path)
    if is_baseline and rubber_stamp:
      raise ValueError('Baseline model cannot be rubber stamped.')
    return super(EvalSharedModel, cls).__new__(
        cls,
        model_path,
        add_metrics_callbacks,
        include_default_metrics,
        example_weight_key,
        additional_fetches,
        model_loader,
        model_name,
        model_type,
        rubber_stamp,
        is_baseline,
        resource_hints,
        backend_config,
    )


# MaybeMultipleEvalSharedModels represents a parameter that can take on a single
# model or a list of models.
#
# TODO(b/150416505): Deprecate support for dict.
MaybeMultipleEvalSharedModels = Union[
    EvalSharedModel, List[EvalSharedModel], Dict[str, EvalSharedModel]
]

__all__ = [
  'AddMetricsCallbackType',
  'ConcreteStructuredMetricValue',
  'DictOfFetchedTensorValues',
  'DictOfTensorType',
  'DictOfTensorTypeMaybeDict',
  'DictOfTensorValue',
  'DictOfTensorValueMaybeDict',
  'DictOfTypeSpec',
  'DictOfTypeSpecMaybeDict',
  'EvalSharedModel',
  'Extracts',
  'FeaturesPredictionsLabels',
  'FPLKeyType',
  'MaterializedColumn',
  'MaybeMultipleEvalSharedModels',
  'MetricValueType',
  'MetricVariablesType',
  'ModelLoader',
  'PrimitiveMetricValueType',
  'RaggedTensorValue',
  'SparseTensorValue',
  'StructuredMetricValue',
  'TensorOrOperationType',
  'TensorType',
  'TensorTypeMaybeDict',
  'TensorTypeMaybeMultiLevelDict',
  'TensorValue',
  'TensorValueMaybeDict',
  'TensorValueMaybeMultiLevelDict',
  'TypeSpecMaybeDict',
  'TypeSpecMaybeMultiLevelDict',
  'ValueWithTDistribution',
  'VarLenTensorValue'
]
