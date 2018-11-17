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
import itertools

import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import types
from tensorflow_model_analysis import util as general_util
from tensorflow_model_analysis.api.impl import api_types
from tensorflow_model_analysis.eval_saved_model import constants
from tensorflow_model_analysis.eval_saved_model import encoding
from tensorflow_model_analysis.eval_saved_model import graph_ref
from tensorflow_model_analysis.eval_saved_model import util
from tensorflow_model_analysis.types_compat import Any, Dict, Generator, List, Text, Tuple

from tensorflow.core.protobuf import meta_graph_pb2


class EvalSavedModel(object):
  """Abstraction for using a EvalSavedModel."""

  def __init__(self, path):
    self._path = path
    self._graph = tf.Graph()
    self._session = tf.Session(graph=self._graph)
    try:
      self._load_and_parse_graph()
    except (RuntimeError, TypeError, ValueError,
            tf.errors.OpError) as exception:
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

  def _iterate_fpl_maps_in_canonical_order(
      self
  ):
    """Iterate through features, predictions, labels maps in canonical order.

    We need to fix a canonical order because to use `make_callable`, we must use
    a list for feeding Tensor values. This means we need to generate the
    values to feed in the same order that we generated the list of parameters
    for `make_callable`.

    Each of features_map, predictions_map, and labels_map are OrderedDicts,
    so the iteration order is stable.

    The map names returned are chosen to correspond to the fields of
    FeaturesPredictionsLabels, so callers can do getattr(fpl, map_name) to
    get the corresponding field.

    Yields:
      Tuples of (map name, map key, map value)
    """
    for feature_key, feature_value in self._features_map.items():
      yield 'features', feature_key, feature_value  # pytype: disable=bad-return-type
    for prediction_key, prediction_value in self._predictions_map.items():
      yield 'predictions', prediction_key, prediction_value  # pytype: disable=bad-return-type
    for label_key, label_value in self._labels_map.items():
      yield 'labels', label_key, label_value  # pytype: disable=bad-return-type

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
      #
      # We use OrderedDict because the ordering of the keys matters:
      # we need to fix a canonical ordering for passing feed_list arguments
      # into make_callable.
      self._features_map = collections.OrderedDict(
          graph_ref.get_node_map_in_graph(meta_graph_def,
                                          encoding.FEATURES_COLLECTION,
                                          [encoding.NODE_SUFFIX], self._graph))
      self._labels_map = collections.OrderedDict(
          graph_ref.get_node_map_in_graph(meta_graph_def,
                                          encoding.LABELS_COLLECTION,
                                          [encoding.NODE_SUFFIX], self._graph))

      if len(signature_def.inputs) != 1:
        raise ValueError('there should be exactly one input. signature_def '
                         'was: %s' % signature_def)

      # The input node, predictions and metrics are in the signature.
      input_node = list(signature_def.inputs.values())[0]
      self._input_example_node = (
          tf.saved_model.utils.get_tensor_from_tensor_info(
              input_node, self._graph))

      # The example reference node. If not defined in the graph, use the
      # input examples as example references.
      try:
        self._example_ref_tensor = graph_ref.get_node_in_graph(
            meta_graph_def, encoding.EXAMPLE_REF_COLLECTION, self._graph)
      except KeyError:
        # If we can't find the ExampleRef collection, then this is probably a
        # model created before we introduced the ExampleRef parameter to
        # EvalInputReceiver. In that case, we default to a tensor of range(0,
        # len(input_example)).
        self._example_ref_tensor = tf.range(tf.size(self._input_example_node))

      # We use OrderedDict because the ordering of the keys matters:
      # we need to fix a canonical ordering for passing feed_dict arguments
      # into make_callable.
      #
      # The canonical ordering we use here is simply the ordering we get
      # from the predictions collection.
      predictions = graph_ref.extract_signature_outputs_with_prefix(
          constants.PREDICTIONS_NAME, signature_def.outputs)
      predictions_map = collections.OrderedDict()
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

      # Create feed_list for metrics_reset_update_get_fn
      #
      # We need to save this because we need to update the
      # metrics_reset_update_get_fn when additional metric ops are registered
      # (the feed_list will stay the same though).
      feed_list = []
      feed_list_keys = []
      for which_map, key, map_dict in (
          self._iterate_fpl_maps_in_canonical_order()):
        feed_list.append(map_dict[encoding.NODE_SUFFIX])
        feed_list_keys.append((which_map, key))
      self._metrics_reset_update_get_fn_feed_list = feed_list
      # We also keep the associated keys for better error messages.
      self._metrics_reset_update_get_fn_feed_list_keys = feed_list_keys

      self._metric_names = []
      self._metric_value_ops = []
      self._metric_update_ops = []
      self._metric_variable_nodes = []
      self._metric_variable_placeholders = []
      self._metric_variable_assign_ops = []
      self.register_additional_metric_ops(metric_ops)

      # Make callable for predict_list. The callable for
      # metrics_reset_update_get is updated in register_additional_metric_ops.
      # Repeated calls to a callable made using make_callable are faster than
      # doing repeated calls to session.run.
      self._predict_list_fn = self._session.make_callable(
          fetches=(self._features_map, self._predictions_map, self._labels_map,
                   self._example_ref_tensor),
          feed_list=[self._input_example_node])

  def graph_as_default(self):
    return self._graph.as_default()

  def graph_finalize(self):
    self._graph.finalize()

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

    self._metrics_reset_update_get_fn = self._session.make_callable(
        fetches=[self._all_metric_update_ops, self._metric_variable_nodes],
        feed_list=self._metrics_reset_update_get_fn_feed_list)
    self._perform_metrics_update_fn = self._session.make_callable(
        fetches=self._all_metric_update_ops,
        feed_list=self._metrics_reset_update_get_fn_feed_list)

  def _log_debug_message_for_tracing_feed_errors(
      self, fetches,
      feed_list):
    """Logs debug message for tracing feed errors."""

    def create_tuple_list(tensor):
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

    def flatten(target):
      return list(itertools.chain.from_iterable(target))

    def log_list(name, target):
      tf.logging.info('%s = [', name)
      for elem_type, elem_name in flatten(
          [create_tuple_list(x) for x in target]):
        tf.logging.info('(\'%s\', \'%s\'),', elem_type, elem_name)
      tf.logging.info(']')

    tf.logging.info('-------------------- fetches and feeds information')
    log_list('fetches', fetches)
    tf.logging.info('')
    log_list('feed_list', feed_list)
    tf.logging.info('-------------------- end fetches and feeds information')

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
    if list(predictions.keys()) == [encoding.DEFAULT_PREDICTIONS_DICT_KEY]:
      predictions = predictions[encoding.DEFAULT_PREDICTIONS_DICT_KEY]

    labels = {}
    for key, value in self._labels_map.items():
      labels[key] = value[encoding.NODE_SUFFIX]
    # Unnest if it wasn't a dictionary to begin with.
    if list(labels.keys()) == [encoding.DEFAULT_LABELS_DICT_KEY]:
      labels = labels[encoding.DEFAULT_LABELS_DICT_KEY]

    return (features, predictions, labels)

  def predict(self, input_example_bytes
             ):
    """Feed an input_example_bytes, get features, predictions, labels.

    Args:
      input_example_bytes: Bytes to feed the input example with. Could be a
        serialised tf.Example, a CSV row, JSON data, or something else depending
        on what the model's input_fn was configured to ingest.

    Returns:
      A list of FeaturesPredictionsLabels (while in most
      cases one input_example_bytes will result in one FPL thus the list
      contains only one element, in some cases, e.g. where examples are
      dynamically decoded and generated within the graph, one
      input_example_bytes might result in zero to many examples).
    """
    return self.predict_list([input_example_bytes])

  def predict_list(self, input_example_bytes_list
                  ):
    """Like predict, but takes a list of examples.

    Args:
      input_example_bytes_list: a list of input example bytes.

    Returns:
       A list of FeaturesPredictionsLabels (while in most cases one
       input_example_bytes will result in one FPL, in some cases, e.g.
       where examples are dynamically decoded and generated within the graph,
       one input_example_bytes might result in multiple examples).

    Raises:
      ValueError: if the example_ref is not a 1-D tensor integer tensor or
        it is not batch aligned with features, predictions and labels or
        it is out of range (< 0 or >= len(input_example_bytes_list)).
    """
    (features, predictions, labels,
     example_refs) = self._predict_list_fn(input_example_bytes_list)

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

    if (not isinstance(example_refs, np.ndarray) or example_refs.ndim != 1 or
        not np.issubdtype(example_refs.dtype, np.integer)):
      raise ValueError(
          'example_ref should be an 1-D array of integers. example_ref was {}.'
          .format(example_refs))

    for result_key, split_values in itertools.chain(split_labels.items(),
                                                    split_features.items(),
                                                    split_predictions.items()):
      if len(split_values) != example_refs.shape[0]:
        raise ValueError(
            'example_ref should be batch-aligned with features, predictions'
            ' and labels; key {} had {} slices but ExampleRef had batch size'
            ' of {}'.format(result_key, len(split_values),
                            example_refs.shape[0]))

    for i, example_ref in enumerate(example_refs):
      if example_ref < 0 or example_ref >= len(input_example_bytes_list):
        raise ValueError('An index in example_ref is out of range: {} vs {}; '
                         'input_example_bytes: {}'.format(
                             example_ref, len(input_example_bytes_list),
                             input_example_bytes_list))
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
          api_types.FeaturesPredictionsLabels(
              example_ref=example_ref,
              features=features,
              predictions=predictions,
              labels=labels))

    return result

  def _create_feed_for_features_predictions_labels(
      self, features_predictions_labels
  ):
    """Create feed list for feeding the given features, predictions, labels."""
    result = []
    for which_map, key, _ in self._iterate_fpl_maps_in_canonical_order():
      result.append(
          getattr(features_predictions_labels,
                  which_map)[key][encoding.NODE_SUFFIX])
    return result

  def _create_feed_for_features_predictions_labels_list(
      self, features_predictions_labels_list
  ):
    """Create feed list for feeding a list of features, predictions, labels."""
    result = []
    for which_map, key, _ in self._iterate_fpl_maps_in_canonical_order():
      result.append(
          util.merge_tensor_values([
              getattr(fpl, which_map)[key][encoding.NODE_SUFFIX]
              for fpl in features_predictions_labels_list
          ]))
    return result

  def perform_metrics_update(
      self,
      features_predictions_labels):
    """Run a single metrics update step on a single FPL."""
    feed_list = self._create_feed_for_features_predictions_labels(
        features_predictions_labels)
    try:
      self._perform_metrics_update_fn(*feed_list)
    except (RuntimeError, TypeError, ValueError,
            tf.errors.OpError) as exception:
      feed_dict = dict(
          zip(self._metrics_reset_update_get_fn_feed_list_keys, feed_list))
      self._log_debug_message_for_tracing_feed_errors(
          fetches=[self._all_metric_update_ops] + self._metric_variable_nodes,
          feed_list=self._metrics_reset_update_get_fn_feed_list)
      general_util.reraise_augmented(
          exception, 'features_predictions_labels = %s, feed_dict = %s' %
          (features_predictions_labels, feed_dict))

  def metrics_reset_update_get(
      self, features_predictions_labels
  ):
    """Run the metrics reset, update, get operations on a single FPL."""
    self.reset_metric_variables()
    feed_list = self._create_feed_for_features_predictions_labels(
        features_predictions_labels)
    try:
      [_, result] = self._metrics_reset_update_get_fn(*feed_list)
    except (RuntimeError, TypeError, ValueError,
            tf.errors.OpError) as exception:
      feed_dict = dict(
          zip(self._metrics_reset_update_get_fn_feed_list_keys, feed_list))
      self._log_debug_message_for_tracing_feed_errors(
          fetches=[self._all_metric_update_ops],
          feed_list=self._metrics_reset_update_get_fn_feed_list)
      general_util.reraise_augmented(
          exception, 'features_predictions_labels = %s, feed_dict = %s' %
          (features_predictions_labels, feed_dict))
    return result

  def metrics_reset_update_get_list(
      self, features_predictions_labels_list):
    """Run the metrics reset, update, get operations on a list of FPLs."""
    self.reset_metric_variables()

    feed_list = self._create_feed_for_features_predictions_labels_list(
        features_predictions_labels_list)
    try:
      [_, result] = self._metrics_reset_update_get_fn(*feed_list)
    except (RuntimeError, TypeError, ValueError,
            tf.errors.OpError) as exception:
      feed_dict = dict(
          zip(self._metrics_reset_update_get_fn_feed_list_keys, feed_list))
      self._log_debug_message_for_tracing_feed_errors(
          fetches=[self._all_metric_update_ops],
          feed_list=self._metrics_reset_update_get_fn_feed_list)
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
