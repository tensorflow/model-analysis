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

import itertools

import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import types
from tensorflow_model_analysis.eval_metrics_graph import eval_metrics_graph
from tensorflow_model_analysis.eval_saved_model import constants
from tensorflow_model_analysis.eval_saved_model import encoding
from tensorflow_model_analysis.eval_saved_model import graph_ref
from tensorflow_model_analysis.eval_saved_model import util
from tensorflow_model_analysis.types_compat import Any, Dict, Generator, List, Optional, Text, Tuple, Union

from tensorflow.core.protobuf import meta_graph_pb2

# Type used to feed a single input to the model. This will be converted to a
# MultipleInputFeedType using a batch of size one.
SingleInputFeedType = Union[bytes, Dict[bytes, Any]]  # pylint: disable=invalid-name
# Type used to feed a batch of inputs to the model. This should match the values
# expected by the receiver_tensor placeholders used with the EvalInputReceiver.
# Typically this will be a batch of serialized `tf.train.Example` protos.
MultipleInputFeedType = Union[List[bytes], Dict[bytes, List[Any]]]  # pylint: disable=invalid-name


class EvalSavedModel(eval_metrics_graph.EvalMetricsGraph):
  """Abstraction for using a EvalSavedModel.

  Note that this class overrides eval_metrics_graph.EvalMetricsGraph
  which contains most of the functionality for handling the metric ops.
  In the eval saved model path, a common graph is shared between generation
  FeaturesPredictionLabels through Predict and doing the metric ops.
  The specific methods of this class constructs the graph and handles
  the prediction ops.
  """

  def __init__(self, path,
               include_default_metrics = True):
    self._path = path
    self._include_default_metrics = include_default_metrics
    super(EvalSavedModel, self).__init__()

  def _check_version(self, version_node):
    version = self._session.run(version_node)
    if not version:
      raise ValueError(
          'invalid TFMA version in graph (at path %s)' % self._path)
    # We don't actually do any checking for now, since we don't have any
    # compatibility issues.

  def _legacy_check_version(self, meta_graph_def):
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

  def _construct_graph(self):
    """Actually load and parse the graph.

    This is factored out from __init__ in case we want to support delayed-loads
    in the future.

    Raises:
      ValueError: Could not find signature keyed with
        DEFAULT_EVAL_SIGNATURE_DEF_KEY; or signature_def did not have exactly
        one input; or there was a signature output with the metric prefix but an
        unrecognised suffix.
    """
    meta_graph_def = tf.saved_model.loader.load(
        self._session, [constants.EVAL_TAG], self._path)

    with self._graph.as_default():
      signature_def = meta_graph_def.signature_def.get(
          constants.DEFAULT_EVAL_SIGNATURE_DEF_KEY)
      if signature_def is None:
        raise ValueError('could not find signature with name %s. signature_def '
                         'was %s' % (constants.EVAL_TAG, signature_def))

      # If features and labels are not stored in the signature_def.inputs then
      # only a single input will be present. We will use this as our flag to
      # indicate whether the features and labels should be read using the legacy
      # collections or using new signature_def.inputs.
      if len(signature_def.inputs) == 1:
        self._legacy_check_version(meta_graph_def)
        self._input_map, self._input_refs_node = graph_ref.load_legacy_inputs(
            meta_graph_def, signature_def, self._graph)
        self._features_map, self._labels_map = (
            graph_ref.load_legacy_features_and_labels(meta_graph_def,
                                                      self._graph))
      else:
        self._check_version(
            graph_ref.load_tfma_version(signature_def, self._graph))
        self._input_map, self._input_refs_node = graph_ref.load_inputs(
            signature_def, self._graph)
        self._features_map, self._labels_map = (
            graph_ref.load_features_and_labels(signature_def, self._graph))

      self._predictions_map = graph_ref.load_predictions(
          signature_def, self._graph)

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

      if self._include_default_metrics:
        metrics_map = graph_ref.load_metrics(signature_def, self._graph)
        metric_ops = {}
        for metric_name, ops in metrics_map.items():
          metric_ops[metric_name] = (ops[encoding.VALUE_OP_SUFFIX],
                                     ops[encoding.UPDATE_OP_SUFFIX])
        self.register_additional_metric_ops(metric_ops)

      # Make callable for predict_list. The callable for
      # metrics_reset_update_get is updated in register_additional_metric_ops.
      # Repeated calls to a callable made using make_callable are faster than
      # doing repeated calls to session.run.
      self._predict_list_fn = self._session.make_callable(
          fetches=(self._features_map, self._predictions_map, self._labels_map,
                   self._input_refs_node),
          feed_list=list(self._input_map.values()))

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
    # Unnest if it wasn't a dictionary to begin with.
    if list(features.keys()) == [constants.DEFAULT_FEATURES_DICT_KEY]:
      features = features[constants.DEFAULT_FEATURES_DICT_KEY]

    predictions = {}
    for key, value in self._predictions_map.items():
      predictions[key] = value[encoding.NODE_SUFFIX]
    # Unnest if it wasn't a dictionary to begin with.
    if list(predictions.keys()) == [constants.DEFAULT_PREDICTIONS_DICT_KEY]:
      predictions = predictions[constants.DEFAULT_PREDICTIONS_DICT_KEY]

    labels = {}
    for key, value in self._labels_map.items():
      labels[key] = value[encoding.NODE_SUFFIX]
    # Unnest if it wasn't a dictionary to begin with.
    if list(labels.keys()) == [constants.DEFAULT_LABELS_DICT_KEY]:
      labels = labels[constants.DEFAULT_LABELS_DICT_KEY]

    return (features, predictions, labels)

  def predict(self, single_input
             ):
    """Returns features, predictions, and labels for a single_input.

    Args:
      single_input: Data to use to feed the input tensors. This must align with
        the receiver_tensors passed to EvalInputReceiver. For example, if
        receiver_tensors was a placeholder for parsing `tf.train.Example`
        protos, then this will be a serialised `tf.train.Example`.

    Returns:
      A list of FeaturesPredictionsLabels (one per example used by model). In
      most cases a single input will result in one example and the returned list
      will contain only one element, but in some cases (e.g. where examples are
      dynamically decoded and generated within the graph), the single input
      might result in zero to many examples).
    """
    return self.predict_list([single_input])

  def predict_list(self, inputs
                  ):
    """Like predict, but takes a list of inputs.

    Args:
      inputs: A list of input data (or a dict of keys to lists of input data).
        See predict for more details.

    Returns:
       A list of FeaturesPredictionsLabels. See predict for more details.

    Raises:
      ValueError: If the original input_refs tensor passed to the
        EvalInputReceiver does not align with the features, predictions and
        labels returned after feeding the inputs.
    """
    if isinstance(inputs, dict):
      input_args = []
      # Only add values for keys that are in the input map (in order).
      for key in self._input_map:
        if key in inputs:
          input_args.append(inputs[key])
    else:
      input_args = [inputs]

    (features, predictions, labels,
     input_refs) = self._predict_list_fn(*input_args)

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

    if (not isinstance(input_refs, np.ndarray) or input_refs.ndim != 1 or
        not np.issubdtype(input_refs.dtype, np.integer)):
      raise ValueError(
          'input_refs should be an 1-D array of integers. input_refs was {}.'
          .format(input_refs))

    for result_key, split_values in itertools.chain(split_labels.items(),
                                                    split_features.items(),
                                                    split_predictions.items()):
      if len(split_values) != input_refs.shape[0]:
        raise ValueError(
            'input_refs should be batch-aligned with features, predictions'
            ' and labels; key {} had {} slices but input_refs had batch size'
            ' of {}'.format(result_key, len(split_values), input_refs.shape[0]))

    for i, input_ref in enumerate(input_refs):
      if input_ref < 0 or input_ref >= len(inputs):
        raise ValueError('An index in input_refs is out of range: {} vs {}; '
                         'inputs: {}'.format(input_ref, len(inputs), inputs))
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
          types.FeaturesPredictionsLabels(
              input_ref=input_ref,
              features=features,
              predictions=predictions,
              labels=labels))

    return result

  def _create_feed_for_features_predictions_labels_list(
      self,
      features_predictions_labels_list
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
