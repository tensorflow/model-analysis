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
"""Abstract class representing the TFMA metrics graph API.

This abstract class is intended to be used as a base class for other graph-based
methods of computing metrics, such as EvalSavedModel (where the metrics graph
is part of the exported model and is loaded from disk), or ModeAgnosticEvalGraph
(where the metrics graph is constructed in-memory based on user-configuration).
It provides structures and expectations with regards to annotations to the graph
and a common set of metric APIs on which to operate on the graph. This way,
different graph implementations can reuse most other TFMA code infrastructure.

Note that this class captures only the metrics computation and aggregation part
of TFMA. The prediction part does not have a separate base class, and can be
found in EvalSavedModel.
"""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import abc
import itertools
import threading
# Standard Imports
import tensorflow as tf

from tensorflow_model_analysis import types
from tensorflow_model_analysis import util as general_util
from tensorflow_model_analysis.eval_saved_model import constants
from tensorflow_model_analysis.eval_saved_model import util
from typing import Any, Dict, List, NamedTuple, Text, Tuple

# Config for defining the input tensor feed into the EvalMetricsGraph. This
# is needed for model agnostic use cases where a graph must be constructed.
FPLFeedConfig = NamedTuple(  # pylint: disable=invalid-name
    'FPLFeedConfig', [('features', Dict[Text, Any]),
                      ('predictions', Dict[Text, Any]),
                      ('labels', Dict[Text, Any])])


class EvalMetricsGraph(object):  # pytype: disable=ignored-metaclass
  """Abstraction for a graph that is used for computing and aggregating metrics.

  This abstract class contains methods and lays out the API to handle metrics
  computation and aggregation as part of the TFMA flow. Inheritors of this class
  are responsible for setting up the metrics graph and setting the class
  variables which are required to do metric calculations.
  """

  __metaclass__ = abc.ABCMeta

  def __init__(self):
    """Initializes this class and attempts to create the graph.

    This method attempts to create the graph through _construct_graph and
    also creates all class variables that need to be populated by the override
    function _construct_graph.
    """
    self._graph = tf.Graph()
    self._session = tf.compat.v1.Session(graph=self._graph)

    # This lock is for  multi-threaded contexts where multiple threads
    # share the same EvalSavedModel.
    #
    # Locking is required in the case where there are multiple threads using
    # the same EvalMetricsGraph. Because the metrics variables are part of the
    # session, and all threads share the same session, without a lock, the
    # "reset-update-get" steps may not be atomic and there can be races.
    #
    # Having each thread have its own session would also work, but would
    # require a bigger refactor.
    # TODO(b/131727905): Investigate whether it's possible / better to have
    # each thread have its own session.
    self._lock = threading.Lock()

    # Variables that need to be populated.

    # The names of the metric.
    self._metric_names = []

    # Ops associated with reading and writing the metric variables.
    self._metric_value_ops = []
    self._metric_update_ops = []
    self._metric_variable_assign_ops = []

    # Nodes associated with the metric variables.
    self._metric_variable_nodes = []

    # Placeholders and feed input for the metric variables.
    self._metric_variable_placeholders = []
    self._perform_metrics_update_fn_feed_list = []
    self._perform_metrics_update_fn_feed_list_keys = []

    # OrderedDicts that map features, predictions, and labels keys to their
    # tensors.
    self._features_map = {}
    self._predictions_map = {}
    self._labels_map = {}

    # Ops to set/update/reset all metric variables.
    self._all_metric_variable_assign_ops = None
    self._all_metric_update_ops = None
    self._reset_variables_op = None

    # Callable to perform metric update.
    self._perform_metrics_update_fn = None

    # OrderedDict produced by graph_ref's load_(legacy_)inputs, mapping input
    # key to tensor value.
    self._input_map = None

    try:
      self._construct_graph()
    except (RuntimeError, TypeError, ValueError,
            tf.errors.OpError) as exception:
      general_util.reraise_augmented(exception, 'Failed to create graph.')

  @abc.abstractmethod
  def _construct_graph(self):
    """Abstract function that is responsible for graph construction.

    This method is called as part of init. Subclasses are also responsible for
    populating the variables in the __init__ method as part of graph
    construction.
    """
    raise NotImplementedError

  def register_add_metric_callbacks(
      self, add_metrics_callbacks: List[types.AddMetricsCallbackType]) -> None:
    """Register additional metric callbacks.

    Runs the given list of callbacks for adding additional metrics to the graph.

    For more details about add_metrics_callbacks, see the docstring for
    EvalSharedModel.add_metrics_callbacks in types.py.

    Args:
      add_metrics_callbacks: A list of metric callbacks to add to the metrics
        graph.

    Raises:
      ValueError: There was a metric name conflict: a callback tried to add a
        metric whose name conflicted with a metric that was added by an earlier
        callback.

    """
    with self._graph.as_default():
      features_dict, predictions_dict, labels_dict = (
          self.get_features_predictions_labels_dicts())
      features_dict = util.wrap_tensor_or_dict_of_tensors_in_identity(
          features_dict)
      predictions_dict = util.wrap_tensor_or_dict_of_tensors_in_identity(
          predictions_dict)
      labels_dict = util.wrap_tensor_or_dict_of_tensors_in_identity(labels_dict)

      metric_ops = {}
      for add_metrics_callback in add_metrics_callbacks:
        new_metric_ops = add_metrics_callback(features_dict, predictions_dict,
                                              labels_dict)
        overlap = set(new_metric_ops.keys()) & set(metric_ops.keys())
        if overlap:
          raise ValueError('metric keys should not conflict, but an '
                           'earlier callback already added the metrics '
                           'named %s' % overlap)
        metric_ops.update(new_metric_ops)
      self.register_additional_metric_ops(metric_ops)

  def graph_as_default(self):
    return self._graph.as_default()

  def graph_finalize(self):
    self._graph.finalize()

  def register_additional_metric_ops(
      self, metric_ops: Dict[Text, Tuple[tf.Tensor, tf.Tensor]]) -> None:
    """Register additional metric ops that were added.

    Args:
      metric_ops: Dictionary of metric ops, just like in the Trainer.

    Raises:
      ValueError: One or more of the metric ops already exist in the graph.
    """
    for metric_name, (value_op, update_op) in metric_ops.items():
      if metric_name in self._metric_names:
        raise ValueError('tried to register new metric with name %s, but a '
                         'metric with that name already exists.' % metric_name)
      self._metric_names.append(metric_name)
      self._metric_value_ops.append(value_op)
      self._metric_update_ops.append(update_op)

    # Update metric variables incrementally with only the new elements in the
    # metric_variables collection.
    collection = self._graph.get_collection(
        tf.compat.v1.GraphKeys.METRIC_VARIABLES)
    collection = collection[len(self._metric_variable_nodes):]

    # Note that this is a node_list - it's not something that TFMA
    # configures, but something that TF.Learn configures.
    #
    # As such, we also use graph.get_tensor_by_name directly, instead of
    # TFMA's version which expects names encoded by TFMA.
    for node in collection:
      self._metric_variable_nodes.append(node)
      with self._graph.as_default():
        placeholder = tf.compat.v1.placeholder(
            dtype=node.dtype, shape=node.get_shape())
        self._metric_variable_placeholders.append(placeholder)
        self._metric_variable_assign_ops.append(
            tf.compat.v1.assign(node, placeholder))

    with self._graph.as_default():
      self._all_metric_variable_assign_ops = tf.group(
          *self._metric_variable_assign_ops)
      self._all_metric_update_ops = tf.group(*self._metric_update_ops)
      self._reset_variables_op = tf.compat.v1.local_variables_initializer()
      self._session.run(self._reset_variables_op)

    self._perform_metrics_update_fn = self._session.make_callable(
        fetches=self._all_metric_update_ops,
        feed_list=self._perform_metrics_update_fn_feed_list)

  def _log_debug_message_for_tracing_feed_errors(
      self, fetches: List[types.TensorOrOperationType],
      feed_list: List[types.TensorOrOperationType]) -> None:
    """Logs debug message for tracing feed errors."""

    def create_tuple_list(tensor: types.TensorOrOperationType):
      """Create a list of tuples describing a Tensor."""
      result = None
      if isinstance(tensor, tf.Operation):
        result = [('Op', tensor.name)]
      elif isinstance(tensor, tf.SparseTensor):
        result = [
            ('SparseTensor.indices', tensor.indices.name),
            ('SparseTensor.values', tensor.values.name),
            ('SparseTensor.dense_shape', tensor.dense_shape.name),
        ]
      elif isinstance(tensor, tf.Tensor):
        result = [('Tensor', tensor.name)]
      else:
        result = [('Unknown', str(tensor))]
      return result

    def flatten(target: List[List[Any]]) -> List[Any]:
      return list(itertools.chain.from_iterable(target))

    def log_list(name: Text, target: List[Any]) -> None:
      tf.compat.v1.logging.info('%s = [', name)
      for elem_type, elem_name in flatten(
          [create_tuple_list(x) for x in target]):
        tf.compat.v1.logging.info('(\'%s\', \'%s\'),', elem_type, elem_name)
      tf.compat.v1.logging.info(']')

    tf.compat.v1.logging.info(
        '-------------------- fetches and feeds information')
    log_list('fetches', fetches)
    tf.compat.v1.logging.info('')
    log_list('feed_list', feed_list)
    tf.compat.v1.logging.info(
        '-------------------- end fetches and feeds information')

  def get_features_predictions_labels_dicts(
      self) -> Tuple[types.TensorTypeMaybeDict, types.TensorTypeMaybeDict, types
                     .TensorTypeMaybeDict]:
    """Returns features, predictions, labels dictionaries (or values).

    The dictionaries contain references to the nodes, so they can be used
    to construct new metrics similarly to how metrics can be constructed in
    the Trainer.

    Returns:
      Tuple of features, predictions, labels dictionaries (or values).
    """
    features = {}
    for key, value in self._features_map.items():
      features[key] = value

    predictions = {}
    for key, value in self._predictions_map.items():
      predictions[key] = value
    # Unnest if it wasn't a dictionary to begin with.
    default_predictions_key = util.default_dict_key(constants.PREDICTIONS_NAME)
    if list(predictions.keys()) == [default_predictions_key]:
      predictions = predictions[default_predictions_key]

    labels = {}
    for key, value in self._labels_map.items():
      labels[key] = value
    # Unnest if it wasn't a dictionary to begin with.
    default_labels_key = util.default_dict_key(constants.LABELS_NAME)
    if list(labels.keys()) == [default_labels_key]:
      labels = labels[default_labels_key]

    return (features, predictions, labels)

  def _perform_metrics_update_list(self, examples_list: List[Any]) -> None:
    """Run a metrics update on a list of examples."""
    try:
      self._perform_metrics_update_fn(*[examples_list])

    except (RuntimeError, TypeError, ValueError,
            tf.errors.OpError) as exception:
      general_util.reraise_augmented(exception,
                                     'raw_input = %s' % (examples_list))

  def metrics_reset_update_get(
      self, features_predictions_labels: types.FeaturesPredictionsLabels
  ) -> List[Any]:
    """Run the metrics reset, update, get operations on a single FPL."""
    return self.metrics_reset_update_get_list([features_predictions_labels])

  def metrics_reset_update_get_list(self,
                                    examples_list: List[bytes]) -> List[Any]:
    """Run the metrics reset, update, get operations on a list of FPLs."""
    with self._lock:
      # Note that due to tf op reordering issues on some hardware, DO NOT merge
      # these operations into a single atomic reset_update_get operation.
      self._reset_metric_variables()
      self._perform_metrics_update_list(examples_list)
      return self._get_metric_variables()

  def _get_metric_variables(self) -> List[Any]:
    # Lock should be acquired before calling this function.
    return self._session.run(fetches=self._metric_variable_nodes)

  def get_metric_variables(self) -> List[Any]:
    """Returns a list containing the metric variable values."""
    with self._lock:
      return self._get_metric_variables()

  def _create_feed_for_metric_variables(self, metric_variable_values: List[Any]
                                       ) -> Dict[types.TensorType, Any]:
    """Returns a feed dict for feeding metric variables values to set them.

    Args:
      metric_variable_values: Metric variable values retrieved using
        get_metric_variables, for instance.

    Returns:
      A feed dict for feeding metric variables values to the placeholders
      constructed for setting the metric variable values to the fed values.
    """
    result = {}
    for node, value in zip(self._metric_variable_placeholders,
                           metric_variable_values):
      result[node] = value
    return result

  def _set_metric_variables(self, metric_variable_values: List[Any]) -> None:
    # Lock should be acquired before calling this function.
    return self._session.run(
        fetches=self._all_metric_variable_assign_ops,
        feed_dict=self._create_feed_for_metric_variables(
            metric_variable_values))

  def set_metric_variables(self, metric_variable_values: List[Any]) -> None:
    """Set metric variable values to the given values."""
    with self._lock:
      self._set_metric_variables(metric_variable_values)

  def _reset_metric_variables(self) -> None:
    # Lock should be acquired before calling this function.
    self._session.run(self._reset_variables_op)

  def reset_metric_variables(self) -> None:
    """Reset metric variable values to their initial values."""
    with self._lock:
      self._reset_metric_variables()

  def _get_metric_values(self) -> Dict[Text, Any]:
    # Lock should be acquired before calling this function.
    metric_values = self._session.run(fetches=self._metric_value_ops)
    return dict(zip(self._metric_names, metric_values))

  def get_metric_values(self) -> Dict[Text, Any]:
    """Retrieve metric values."""
    with self._lock:
      return self._get_metric_values()

  def metrics_set_variables_and_get_values(self,
                                           metric_variable_values: List[Any]
                                          ) -> Dict[Text, Any]:
    with self._lock:
      self._set_metric_variables(metric_variable_values)
      return self._get_metric_values()
