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

import collections

import tensorflow as tf
from tensorflow_model_analysis import types
from tensorflow_model_analysis import util as general_util
from tensorflow_model_analysis.eval_saved_model import constants
from tensorflow_model_analysis.eval_saved_model import encoding
from tensorflow_model_analysis.eval_saved_model import graph_ref
from tensorflow_model_analysis.eval_saved_model import util
from tensorflow_model_analysis.types_compat import Any, Dict, List, NamedTuple, Optional, Tuple  # pytype: disable=not-supported-yet

from tensorflow.core.protobuf import meta_graph_pb2

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
    try:
      self._load_and_parse_graph()
    except (RuntimeError, ValueError) as exception:
      general_util.reraise_augmented(exception,
                                     'for saved_model at path %s' % self._path)

  def _check_version(self, meta_graph_def):
    version = meta_graph_def.collection_def.get(
        encoding.TFMA_VERSION_COLLECTION)
    if version is None:
      raise ValueError(
          'could not find TFMA version in graph (at path %s)' % self._path)
    # We don't actually do any checking for now, since we don't have any
    # compatibility issues.

  def _load_and_parse_graph(self):
    """Actually load and parse the graph.

    This is factored out from __init__ in case we want to support delayed-loads
    in the future.

    Raises:
      ValueError: Could not find signature keyed with EVAL_TAG; or
        signature_def did not have exactly one input; or there was a signature
        output with the metric prefix but an unrecognised suffix.
    """
    meta_graph_def = tf.saved_model.loader.load(
        self._session, [constants.EVAL_TAG], self._path)

    self._check_version(meta_graph_def)
    with self._graph.as_default():
      signature_def = meta_graph_def.signature_def.get(constants.EVAL_TAG)
      if signature_def is None:
        raise ValueError('could not find signature with name %s. signature_def '
                         'was %s' % (constants.EVAL_TAG, signature_def))

      # Note that there are two different encoding schemes in use here:
      #
      # 1. The scheme used by TFMA for the TFMA-specific extra collections
      #    for the features and labels.
      # 2. The scheme used by TensorFlow Estimators in the SignatureDefs for the
      #    input example node, predictions, metrics and so on.

      # Features and labels are in TFMA-specific extra collections.
      self._features_map = graph_ref.get_node_map_in_graph(
          meta_graph_def, encoding.FEATURES_COLLECTION, [encoding.NODE_SUFFIX],
          self._graph)
      self._labels_map = graph_ref.get_node_map_in_graph(
          meta_graph_def, encoding.LABELS_COLLECTION, [encoding.NODE_SUFFIX],
          self._graph)

      if len(signature_def.inputs) != 1:
        raise ValueError('there should be exactly one input. signature_def '
                         'was: %s' % signature_def)

      # The input node, predictions and metrics are in the signature.
      input_node = list(signature_def.inputs.values())[0]
      self._input_example_node = (
          tf.saved_model.utils.get_tensor_from_tensor_info(
              input_node, self._graph))

      predictions = graph_ref.extract_signature_outputs_with_prefix(
          constants.PREDICTIONS_NAME, signature_def.outputs)
      predictions_map = {}
      for k, v in predictions.items():
        # Extract to dictionary with a single key for consistency with
        # how features and labels are extracted.
        predictions_map[k] = {
            encoding.NODE_SUFFIX:
                tf.saved_model.utils.get_tensor_from_tensor_info(
                    v, self._graph)
        }
      self._predictions_map = predictions_map

      metrics = graph_ref.extract_signature_outputs_with_prefix(
          constants.METRICS_NAME, signature_def.outputs)
      metrics_map = collections.defaultdict(dict)
      for k, v in metrics.items():
        node = tf.saved_model.utils.get_tensor_from_tensor_info(v, self._graph)

        if k.endswith('/' + constants.METRIC_VALUE_SUFFIX):
          key = k[:-len(constants.METRIC_VALUE_SUFFIX) - 1]
          metrics_map[key][encoding.VALUE_OP_SUFFIX] = node
        elif k.endswith('/' + constants.METRIC_UPDATE_SUFFIX):
          key = k[:-len(constants.METRIC_UPDATE_SUFFIX) - 1]
          metrics_map[key][encoding.UPDATE_OP_SUFFIX] = node
        else:
          raise ValueError('unrecognised suffix for metric. key was: %s' % k)

      metric_ops = {}
      for metric_name, ops in metrics_map.items():
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
    for label_key in self._labels_map:
      split_labels[label_key] = util.split_tensor_value(
          labels[label_key][encoding.NODE_SUFFIX])
    split_features = {}
    for feature_key in self._features_map:
      split_features[feature_key] = util.split_tensor_value(
          features[feature_key][encoding.NODE_SUFFIX])
    split_predictions = {}
    for prediction_key in self._predictions_map:
      split_predictions[prediction_key] = util.split_tensor_value(
          predictions[prediction_key][encoding.NODE_SUFFIX])

    result = []
    for i in range(len(input_example_bytes_list)):
      labels = {}
      for label_key in self._labels_map:
        labels[label_key] = {encoding.NODE_SUFFIX: split_labels[label_key][i]}
      features = {}
      for feature_key in self._features_map:
        features[feature_key] = {
            encoding.NODE_SUFFIX: split_features[feature_key][i]
        }
      predictions = {}
      for prediction_key in self._predictions_map:
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

  def _create_feed_for_features_predictions_labels_list(
      self, features_predictions_labels_list
  ):
    """Create feed dict for feeding a list of features, predictions, labels."""
    result = {}
    for label_key, label_dict in self._labels_map.items():
      result[label_dict[encoding.NODE_SUFFIX]] = util.merge_tensor_values([
          fpl.labels[label_key][encoding.NODE_SUFFIX]
          for fpl in features_predictions_labels_list
      ])
    for feature_key, feature_dict in self._features_map.items():
      result[feature_dict[encoding.NODE_SUFFIX]] = util.merge_tensor_values([
          fpl.features[feature_key][encoding.NODE_SUFFIX]
          for fpl in features_predictions_labels_list
      ])
    for prediction_key, prediction_dict in self._predictions_map.items():
      result[prediction_dict[encoding.NODE_SUFFIX]] = (
          util.merge_tensor_values([
              fpl.predictions[prediction_key][encoding.NODE_SUFFIX]
              for fpl in features_predictions_labels_list
          ]))
    return result

  def perform_metrics_update(
      self, features_predictions_labels):
    """Run a single metrics update step on a single FPL."""
    feed_dict = self._create_feed_for_features_predictions_labels(
        features_predictions_labels)
    try:
      self._session.run(
          fetches=self._all_metric_update_ops, feed_dict=feed_dict)
    except (RuntimeError, TypeError, ValueError) as exception:
      general_util.reraise_augmented(
          exception, 'features_predictions_labels = %s, feed_dict = %s' %
          (features_predictions_labels, feed_dict))

  def metrics_reset_update_get(
      self,
      features_predictions_labels):
    """Run the metrics reset, update, get operations on a single FPL."""
    self.reset_metric_variables()
    feed_dict = self._create_feed_for_features_predictions_labels(
        features_predictions_labels)
    try:
      [_, result] = self._session.run(
          fetches=[self._all_metric_update_ops, self._metric_variable_nodes],
          feed_dict=feed_dict)
    except (RuntimeError, TypeError, ValueError) as exception:
      general_util.reraise_augmented(
          exception, 'features_predictions_labels = %s, feed_dict = %s' %
          (features_predictions_labels, feed_dict))
    return result

  def metrics_reset_update_get_list(
      self, features_predictions_labels_list
  ):
    """Run the metrics reset, update, get operations on a list of FPLs."""
    self.reset_metric_variables()

    feed_dict = self._create_feed_for_features_predictions_labels_list(
        features_predictions_labels_list)
    try:
      [_, result] = self._session.run(
          fetches=[self._all_metric_update_ops, self._metric_variable_nodes],
          feed_dict=feed_dict)
    except (RuntimeError, TypeError, ValueError) as exception:
      general_util.reraise_augmented(
          exception, 'features_predictions_labels_list = %s, feed_dict = %s' %
          (features_predictions_labels_list, feed_dict))

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
