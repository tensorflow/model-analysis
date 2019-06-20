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

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

# Standard Imports
import numpy as np
import tensorflow as tf
from tensorflow_transform.beam import shared

from typing import Any, Callable, Dict, List, Optional, Text, Tuple, Union, NamedTuple

# pylint: disable=invalid-name

TensorType = Union[tf.Tensor, tf.SparseTensor]
TensorOrOperationType = Union[TensorType, tf.Operation]
DictOfTensorType = Dict[Text, TensorType]
TensorTypeMaybeDict = Union[TensorType, DictOfTensorType]

# Value of a Tensor fetched using session.run.
TensorValue = Union[tf.compat.v1.SparseTensorValue, np.ndarray]
DictOfTensorValue = Dict[Text, TensorValue]
TensorValueMaybeDict = Union[TensorValue, DictOfTensorValue]

MetricVariablesType = List[Any]


class ValueWithTDistribution(
    NamedTuple('ValueWithTDistribution', [
        ('sample_mean', float),
        ('sample_standard_deviation', float),
        ('sample_degrees_of_freedom', int),
        ('unsampled_value', float),
    ])):
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
    return super(ValueWithTDistribution,
                 cls).__new__(cls, sample_mean, sample_standard_deviation,
                              sample_degrees_of_freedom, unsampled_value)

# AddMetricsCallback should have the following prototype:
#   def add_metrics_callback(features_dict, predictions_dict, labels_dict):
#
# It should create and return a metric_ops dictionary, such that
# metric_ops['metric_name'] = (value_op, update_op), just as in the Trainer.
#
# Note that features_dict, predictions_dict and labels_dict are not
# necessarily dictionaries - they might also be Tensors, depending on what the
# model's eval_input_receiver_fn returns.
AddMetricsCallbackType = Any

# Type of keys we support for prediction, label and features dictionaries.
FPLKeyType = Union[Text, Tuple[Text, ...]]

# Dictionary of Tensor values fetched. The dictionary maps original dictionary
# keys => ('node' => value). This type exists for backward compatibility with
# FeaturesPredictionsLabels, new code should use DictOfTensorValue instead.
DictOfFetchedTensorValues = Dict[FPLKeyType, Dict[Text, TensorValue]]

FeaturesPredictionsLabels = NamedTuple(
    'FeaturesPredictionsLabels', [('input_ref', int),
                                  ('features', DictOfFetchedTensorValues),
                                  ('predictions', DictOfFetchedTensorValues),
                                  ('labels', DictOfFetchedTensorValues)])

# Used in building the model diagnostics table, a MaterializedColumn is a value
# inside of Extracts that will be emitted to file. Note that for strings, the
# values are raw byte strings rather than unicode strings. This is by design, as
# features can have arbitrary bytes values.
MaterializedColumn = NamedTuple(
    'MaterializedColumn',
    [('name', Text),
     ('value', Union[List[bytes], List[int], List[float], bytes, int, float])])

# Extracts represent data extracted during pipeline processing. In order to
# provide a flexible API, these types are just dicts where the keys are defined
# (reserved for use) by different extractor implementations. For example, the
# PredictExtractor stores the data for the features, labels, and predictions
# under the keys "features", "labels", and "predictions".
Extracts = Dict[Text, Any]

# pylint: enable=invalid-name


def is_tensor(obj):
  return isinstance(obj, tf.Tensor) or isinstance(obj, tf.SparseTensor)


class EvalSharedModel(
    NamedTuple(
        'EvalSharedModel',
        [
            ('model_path', Text),
            ('add_metrics_callbacks',
             List[Callable]),  # List[AnyMetricsCallbackType]
            ('include_default_metrics', bool),
            ('example_weight_key', Union[Text, Dict[Text, Text]]),
            ('additional_fetches', List[Text]),
            ('shared_handle', shared.Shared),
            ('construct_fn', Callable)
        ])):
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
    shared_handle: Optional handle to a shared.Shared object for sharing the
      in-memory model within / between stages.
    construct_fn: A callable which creates a construct function
      to set up the tensorflow graph. Callable takes a beam.metrics distribution
      to track graph construction time.

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
      model_path: Optional[Text] = None,
      add_metrics_callbacks: Optional[List[AddMetricsCallbackType]] = None,
      include_default_metrics: Optional[bool] = True,
      example_weight_key: Optional[Union[Text, Dict[Text, Text]]] = None,
      additional_fetches: Optional[List[Text]] = None,
      shared_handle: Optional[shared.Shared] = None,
      construct_fn: Optional[Callable[..., Any]] = None):
    if not add_metrics_callbacks:
      add_metrics_callbacks = []
    if not shared_handle:
      shared_handle = shared.Shared()
    return super(EvalSharedModel,
                 cls).__new__(cls, model_path, add_metrics_callbacks,
                              include_default_metrics, example_weight_key,
                              additional_fetches, shared_handle, construct_fn)
