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

from __future__ import print_function

import copy

import numpy as np
import tensorflow as tf
from tensorflow_transform.beam import shared

from tensorflow_model_analysis.types_compat import Any, Callable, Dict, List, Optional, Text, Tuple, Union, NamedTuple

# pylint: disable=invalid-name

TensorType = Union[tf.Tensor, tf.SparseTensor]
TensorOrOperationType = Union[TensorType, tf.Operation]
DictOfTensorType = Dict[Text, TensorType]
TensorTypeMaybeDict = Union[TensorType, DictOfTensorType]

# Type of keys we support for prediction, label and features dictionaries.
KeyType = Union[Text, Tuple[Text, Ellipsis]]

# Value of a Tensor fetched using session.run.
FetchedTensorValue = Union[tf.SparseTensorValue, np.ndarray]

# Dictionary of Tensor values fetched.
# The dictionary maps original dictionary keys => ('node' => value).
DictOfFetchedTensorValues = Dict[KeyType, Dict[Text, FetchedTensorValue]]

MetricVariablesType = List[Any]


# Used in building the model diagnostics table, a MatrializedColumn is a value
# inside of ExampleAndExtract that will be emitted to file. Note that for
# strings, the values are raw byte strings rather than unicode strings. This is
# by design, as features can have arbitrary bytes values.
MaterializedColumn = NamedTuple(
    'MaterializedColumn',
    [('name', Text),
     ('value', Union[List[bytes], List[int], List[float], bytes, int, float])])

# Used in building model diagnostics table, the ExampleAndExtracts holds an
# example and all its "extractions." Extractions that should be emitted to file.
# Each Extract has a name, stored as the key of the DictOfExtractedValues.
DictOfExtractedValues = Dict[Text, Any]

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
            ('example_weight_key', Text),
            ('shared_handle', shared.Shared)
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
    example_weight_key: The key of the example weight column. If None, weight
      will be 1 for each example.
    shared_handle: Optional handle to a shared.Shared object for sharing the
      in-memory model within / between stages.

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
      model_path,
      add_metrics_callbacks = None,
      example_weight_key = None,
      shared_handle = None):
    if not add_metrics_callbacks:
      add_metrics_callbacks = []
    if not shared_handle:
      shared_handle = shared.Shared()
    return super(EvalSharedModel,
                 cls).__new__(cls, model_path, add_metrics_callbacks,
                              example_weight_key, shared_handle)


class ExampleAndExtracts(
    NamedTuple('ExampleAndExtracts', [('example', bytes),
                                      ('extracts', DictOfExtractedValues)])):
  """Example and extracts."""

  def create_copy_with_shallow_copy_of_extracts(self):
    """Returns a new copy of this with a shallow copy of extracts.

    This is NOT equivalent to making a shallow copy with copy.copy(this).
    That does NOT make a shallow copy of the dictionary. An illustration of
    the differences:
      a = ExampleAndExtracts(example='content', extracts=dict(apple=[1, 2]))

      # The dictionary is shared (and hence the elements are also shared)
      b = copy.copy(a)
      b.extracts['banana'] = 10
      assert a.extracts['banana'] == 10

      # The dictionary is not shared (but the elements are)
      c = a.create_copy_with_shallow_copy_of_extracts()
      c.extracts['cherry'] = 10
      assert 'cherry' not in a.extracts  # The dictionary is not shared
      c.extracts['apple'][0] = 100
      assert a.extracts['apple'][0] == 100  # But the elements are

    Returns:
      A shallow copy of this object.
    """
    return ExampleAndExtracts(
        example=self.example, extracts=copy.copy(self.extracts))
