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
"""Library for loading and applying the EvalSavedModel."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function


import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import types
from tensorflow_model_analysis.eval_saved_model import encoding
from tensorflow_model_analysis.eval_saved_model import graph_ref
from tensorflow_model_analysis.types_compat import Any, Dict, List, NamedTuple, Optional, Tuple  # pytype: disable=not-supported-yet

FeaturesPredictionsLabels = NamedTuple(  # pylint: disable=invalid-name
    'FeaturesPredictionsLabels',
    [('features', types.DictOfFetchedTensorValues),
     ('predictions', types.DictOfFetchedTensorValues),
     ('labels', types.DictOfFetchedTensorValues)])


class EvalSavedModel(object):
  """Abstraction for using a EvalSavedModel."""

  def __init__(self, path):
    self._path = path
    self._graph = tf.Graph()
    self._session = tf.Session(graph=self._graph)
    self._load_and_parse_graph()

  def _load_and_parse_graph(self):
    """Actually load and parse the graph.

    This is factored out from __init__ in case we want to support delayed-loads
    in the future.
    """
    meta_graph_def = tf.saved_model.loader.load(
        self._session, [tf.saved_model.tag_constants.SERVING], self._path)

    with self._graph.as_default():
      # Get references to "named" nodes.
      self._input_example_node = graph_ref.get_node_in_graph(
          meta_graph_def, encoding.INPUT_EXAMPLE_COLLECTION, self._graph)
      self._labels_map = graph_ref.get_node_map_in_graph(
          meta_graph_def, encoding.LABELS_COLLECTION, [encoding.NODE_SUFFIX],
          self._graph)
      self._predictions_map = graph_ref.get_node_map_in_graph(
          meta_graph_def, encoding.PREDICTIONS_COLLECTION,
          [encoding.NODE_SUFFIX], self._graph)
      self._features_map = graph_ref.get_node_map_in_graph(
          meta_graph_def, encoding.FEATURES_COLLECTION, [encoding.NODE_SUFFIX],
          self._graph)

      metrics = graph_ref.get_node_map_in_graph(
          meta_graph_def, encoding.METRICS_COLLECTION,
          [encoding.VALUE_OP_SUFFIX, encoding.UPDATE_OP_SUFFIX], self._graph)
      metric_ops = {}
      for metric_name, ops in metrics.items():
        metric_ops[metric_name] = (ops[encoding.VALUE_OP_SUFFIX],
                                   ops[encoding.UPDATE_OP_SUFFIX])

      self._metric_names = []
      self._metric_value_ops = []
      self._metric_update_ops = []
      self._metric_variable_nodes = []
      self._metric_variable_placeholders = []
      self._metric_variable_assign_ops = []
      self.register_additional_metric_ops(metric_ops)

  def graph_as_default(self):
    return self._graph.as_default()

  def register_additional_metric_ops(
      self, metric_ops):
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
    collection = self._graph.get_collection(tf.GraphKeys.METRIC_VARIABLES)
    collection = collection[len(self._metric_variable_nodes):]

    # Note that this is a node_list - it's not something that TFMA
    # configures, but something that TF.Learn configures.
    #
    # As such, we also use graph.get_tensor_by_name directly, instead of
    # TFMA's version which expects names encoded by TFMA.
    for node in collection:
      self._metric_variable_nodes.append(node)
      with self._graph.as_default():
        placeholder = tf.placeholder(dtype=node.dtype, shape=node.get_shape())
        self._metric_variable_placeholders.append(placeholder)
        self._metric_variable_assign_ops.append(tf.assign(node, placeholder))

    with self._graph.as_default():
      self._all_metric_variable_assign_ops = tf.group(
          *self._metric_variable_assign_ops)
      self._all_metric_update_ops = tf.group(*self._metric_update_ops)
      self._reset_variables_op = tf.local_variables_initializer()
      self._session.run(self._reset_variables_op)

  def get_features_predictions_labels_dicts(
      self):
    """Returns features, predictions, labels dictionaries (or values).

    The dictionaries contain references to the nodes, so they can be used
    to construct new metrics similarly to how metrics can be constructed in
    the Trainer.

    Returns:
      Tuple of features, predictions, labels dictionaries (or values).
    """
    features = {}
    for key, value in self._features_map.items():
      features[key] = value[encoding.NODE_SUFFIX]

    predictions = {}
    for key, value in self._predictions_map.items():
      predictions[key] = value[encoding.NODE_SUFFIX]
    # Unnest if it wasn't a dictionary to begin with.
    if predictions.keys() == [encoding.DEFAULT_PREDICTIONS_DICT_KEY]:
      predictions = predictions[encoding.DEFAULT_PREDICTIONS_DICT_KEY]

    labels = {}
    for key, value in self._labels_map.items():
      labels[key] = value[encoding.NODE_SUFFIX]
    # Unnest if it wasn't a dictionary to begin with.
    if labels.keys() == [encoding.DEFAULT_LABELS_DICT_KEY]:
      labels = labels[encoding.DEFAULT_LABELS_DICT_KEY]

    return (features, predictions, labels)

  def predict(self, input_example_bytes):
    """Feed an example, get features, predictions, labels.

    Args:
      input_example_bytes: Bytes to feed the input example with. Could be a
        serialised tf.Example, a CSV row, JSON data, or something else depending
        on what the model's input_fn was configured to ingest.

    Returns:
      FeaturesPredictionsLabels.
    """
    (features, predictions, labels) = self._session.run(
        fetches=(self._features_map, self._predictions_map, self._labels_map),
        feed_dict={
            self._input_example_node: [input_example_bytes]
        })
    return FeaturesPredictionsLabels(
        features=features, predictions=predictions, labels=labels)

  def predict_list(self, input_example_bytes_list
                  ):
    """Like predict, but takes a list of examples."""
    (features, predictions, labels) = self._session.run(
        fetches=(self._features_map, self._predictions_map, self._labels_map),
        feed_dict={
            self._input_example_node: input_example_bytes_list,
        })

    split_labels = {}
    for label_key in self._labels_map.keys():
      split_labels[label_key] = self._split_tensor_value(
          labels[label_key][encoding.NODE_SUFFIX])
    split_features = {}
    for feature_key in self._features_map.keys():
      split_features[feature_key] = self._split_tensor_value(
          features[feature_key][encoding.NODE_SUFFIX])
    split_predictions = {}
    for prediction_key in self._predictions_map.keys():
      split_predictions[prediction_key] = self._split_tensor_value(
          predictions[prediction_key][encoding.NODE_SUFFIX])

    result = []
    for i in range(len(input_example_bytes_list)):
      labels = {}
      for label_key in self._labels_map.keys():
        labels[label_key] = {encoding.NODE_SUFFIX: split_labels[label_key][i]}
      features = {}
      for feature_key in self._features_map.keys():
        features[feature_key] = {
            encoding.NODE_SUFFIX: split_features[feature_key][i]
        }
      predictions = {}
      for prediction_key in self._predictions_map.keys():
        predictions[prediction_key] = {
            encoding.NODE_SUFFIX: split_predictions[prediction_key][i]
        }
      result.append(
          FeaturesPredictionsLabels(
              features=features, predictions=predictions, labels=labels))

    return result

  def _create_feed_for_features_predictions_labels(
      self, features_predictions_labels
  ):
    """Create feed dict for feeding the given features, predictions, labels."""
    feed_dict = {}
    for label_key, label_dict in self._labels_map.items():
      feed_dict[label_dict[encoding.NODE_SUFFIX]] = (
          features_predictions_labels.labels[label_key][encoding.NODE_SUFFIX])
    for feature_key, feature_dict in self._features_map.items():
      feed_dict[feature_dict[encoding.NODE_SUFFIX]] = (
          features_predictions_labels.features[feature_key][
              encoding.NODE_SUFFIX])
    for prediction_key, prediction_dict in self._predictions_map.items():
      feed_dict[prediction_dict[encoding.NODE_SUFFIX]] = (
          features_predictions_labels.predictions[prediction_key][
              encoding.NODE_SUFFIX])
    return feed_dict

  def _split_tensor_value(
      self, tensor_value):
    """Split a single batch of Tensor values into a list of Tensor values.

    Args:
      tensor_value: A single Tensor value that represents a batch of Tensor
        values. The zeroth dimension should be batch size.

    Returns:
      A list of Tensor values, one per element of the zeroth dimension.

    Raises:
      TypeError: tensor_value had unknown type.
    """
    if isinstance(tensor_value, tf.SparseTensorValue):
      result = []
      offset = 0
      for batch_num in range(tensor_value.dense_shape[0]):
        indices = []
        values = []
        while (offset < len(tensor_value.indices) and
               tensor_value.indices[offset][0] == batch_num):
          indices.append([0, tensor_value.indices[offset][1]])
          values.append(tensor_value.values[offset])
          offset += 1
        if not indices:
          # Empty indices array still needs the correct type and shape.
          indices = np.array([], dtype=np.int64)
          indices.shape = (0, 2)
        result.append(
            tf.SparseTensorValue(
                indices=indices, values=values, dense_shape=[1, len(values)]))
      return result
    elif isinstance(tensor_value, np.ndarray):
      return np.split(
          tensor_value, indices_or_sections=tensor_value.shape[0], axis=0)
    else:
      raise TypeError('tensor_value had unknown type: %s, value was: %s' %
                      (type(tensor_value), tensor_value))

  def _merge_tensor_values(self, tensor_values
                          ):
    """Merge a list of Tensor values into a single batch of Tensor values.

    Args:
      tensor_values: A list of Tensor values, all fetched from the same node
        in the same graph. Each Tensor value should be for a single example.

    Returns:
      A single Tensor value that represents a batch of all the Tensor values
      in the given list.

    Raises:
      ValueError: Got a SparseTensor with more than 1 row (i.e. that is likely
        to be for more than one example).
      TypeError: tensor_value had unknown type.
    """
    if not tensor_values:
      return tensor_values

    if isinstance(tensor_values[0], tf.SparseTensorValue):
      indices = []
      values = []
      dense_shape = []
      for i, tensor_value in enumerate(tensor_values):
        if tensor_value.dense_shape[1] == 0:
          continue
        index = np.array(tensor_value.indices)
        index[:, 0] += i
        indices.append(index)
        values.extend(tensor_value.values)
        if tensor_value.dense_shape[0] != 1:
          raise ValueError(
              'expecting SparseTensor to be for only 1 example. '
              'but got dense_shape %s instead' % tensor_value.dense_shape)
      dense_shape = tensor_values[0].dense_shape[:]
      dense_shape[0] = len(tensor_values)

      concatenated_indices = []
      if not indices:
        # indices can be [] if all the SparseTensors are "empty".
        # Empty indices array still needs the correct type and shape.
        concatenated_indices = np.array([], dtype=np.int64)
        concatenated_indices.shape = (0, 2)
      else:
        concatenated_indices = np.concatenate(indices, axis=0)

      return tf.SparseTensorValue(
          indices=concatenated_indices, values=values, dense_shape=dense_shape)
    elif isinstance(tensor_values[0], np.ndarray):
      return np.concatenate(tensor_values, axis=0)
    else:
      raise TypeError('tensor_values[0] had unknown type: %s, value was: %s' %
                      (type(tensor_values[0]), tensor_values[0]))

  def _create_feed_for_features_predictions_labels_list(
      self, features_predictions_labels_list
  ):
    """Create feed dict for feeding a list of features, predictions, labels."""
    result = {}
    for label_key, label_dict in self._labels_map.items():
      result[label_dict[encoding.NODE_SUFFIX]] = self._merge_tensor_values([
          fpl.labels[label_key][encoding.NODE_SUFFIX]
          for fpl in features_predictions_labels_list
      ])
    for feature_key, feature_dict in self._features_map.items():
      result[feature_dict[encoding.NODE_SUFFIX]] = self._merge_tensor_values([
          fpl.features[feature_key][encoding.NODE_SUFFIX]
          for fpl in features_predictions_labels_list
      ])
    for prediction_key, prediction_dict in self._predictions_map.items():
      result[prediction_dict[encoding.NODE_SUFFIX]] = (
          self._merge_tensor_values([
              fpl.predictions[prediction_key][encoding.NODE_SUFFIX]
              for fpl in features_predictions_labels_list
          ]))
    return result

  def perform_metrics_update(
      self, features_predictions_labels):
    """Run a single metrics update step on a single FPL."""
    self._session.run(
        fetches=self._all_metric_update_ops,
        feed_dict=self._create_feed_for_features_predictions_labels(
            features_predictions_labels))

  def metrics_reset_update_get(
      self,
      features_predictions_labels):
    """Run the metrics reset, update, get operations on a single FPL."""
    self.reset_metric_variables()
    [_, result] = self._session.run(
        fetches=[self._all_metric_update_ops, self._metric_variable_nodes],
        feed_dict=self._create_feed_for_features_predictions_labels(
            features_predictions_labels))
    return result

  def metrics_reset_update_get_list(
      self, features_predictions_labels_list
  ):
    """Run the metrics reset, update, get operations on a list of FPLs."""
    self.reset_metric_variables()
    [_, result] = self._session.run(
        fetches=[self._all_metric_update_ops, self._metric_variable_nodes],
        feed_dict=self._create_feed_for_features_predictions_labels_list(
            features_predictions_labels_list))
    return result

  def get_metric_variables(self):
    """Returns a list containing the metric variable values."""
    result = self._session.run(fetches=self._metric_variable_nodes)
    return result

  def _create_feed_for_metric_variables(
      self, metric_variable_values):
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

  def set_metric_variables(self, metric_variable_values):
    """Set metric variable values to the given values."""
    self._session.run(
        fetches=self._all_metric_variable_assign_ops,
        feed_dict=self._create_feed_for_metric_variables(
            metric_variable_values))

  def reset_metric_variables(self):
    """Reset metric variable values to their initial values."""
    self._session.run(self._reset_variables_op)

  def get_metric_values(self):
    """Retrieve metric values."""
    metric_values = self._session.run(fetches=self._metric_value_ops)
    return dict(zip(self._metric_names, metric_values))

  def check_metric_compatibility(self, input_example_bytes
                                ):
    """Checks for metrics that cannot be evaluated using EvalSavedModel.

    These may be metrics that violate the "separation property" - we can
    only evaluate metrics that have the following properties, among others:
      (a) update_op only needs predictions, labels, features
      (b) value_op needs NO fetches

    Note that this check only checks that the above properties are probably
    satisfied. Some other properties that metrics should have but are not
    checked here include:
      - Metrics should add all their state variables to the METRIC_VARIABLES
        collection.
      - The state variables should be combined using addition, i.e. calling
        update_op twice on examples X and Y should be equivalent to calling
        update_op on X, saving the metric variables, independently calling
        update_on on Y, saving the metric variables, and adding the two sets
        of metric variables together.

    Args:
      input_example_bytes: Bytes to feed the input example with. Could be a
        serialised tf.Example, a CSV row, JSON data, or something else depending
        on what the model's input_fn was configured to ingest.

    Returns:
      Dictionary mapping metric names to Tuple(compatible, errors if any).
    """
    result = {}
    for metric_name in self._metric_names:
      result[metric_name] = (True, None)

    features_predictions_labels = self.predict(input_example_bytes)
    feed_dict = self._create_feed_for_features_predictions_labels(
        features_predictions_labels)

    for metric_name, update_op in zip(self._metric_names,
                                      self._metric_update_ops):
      try:
        self._session.run(fetches=update_op, feed_dict=feed_dict)
      except tf.errors.InvalidArgumentError as e:
        result[metric_name] = (False, 'update_op failed: %s' % e)

    for metric_name, value_op in zip(self._metric_names,
                                     self._metric_value_ops):
      try:
        self._session.run(fetches=value_op)
      except tf.errors.InvalidArgumentError:
        result[metric_name] = (False, 'value_op failed: %s' % e)

    return result
