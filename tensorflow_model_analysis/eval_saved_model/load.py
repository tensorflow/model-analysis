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
# Standard __future__ imports
from __future__ import print_function

# Standard Imports
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import types
from tensorflow_model_analysis.eval_metrics_graph import eval_metrics_graph
from tensorflow_model_analysis.eval_saved_model import constants
from tensorflow_model_analysis.eval_saved_model import encoding
from tensorflow_model_analysis.eval_saved_model import graph_ref
from tensorflow_model_analysis.eval_saved_model import util
from typing import Any, Dict, Generator, List, NamedTuple, Optional, Text, Tuple, Union

from tensorflow.core.protobuf import meta_graph_pb2

# pylint: disable=invalid-name
# Type used to feed a single input to the model. This will be converted to a
# MultipleInputFeedType using a batch of size one.
SingleInputFeedType = Union[bytes, Dict[bytes, Any]]
# Type used to feed a batch of inputs to the model. This should match the values
# expected by the receiver_tensor placeholders used with the EvalInputReceiver.
# Typically this will be a batch of serialized `tf.train.Example` protos.
MultipleInputFeedType = Union[List[bytes], Dict[bytes, List[Any]]]
# Type used to return the tensor values fetched from the model.
FetchedTensorValues = NamedTuple(
    'FetchedTensorValues',
    [
        ('input_ref', int),
        # Dict is keyed by group ('features', 'predictions', 'labels', etc).
        ('values', Dict[Text, types.TensorValueMaybeDict])
    ])

# pylint: enable=invalid-name


class EvalSavedModel(eval_metrics_graph.EvalMetricsGraph):
  """Abstraction for using a EvalSavedModel.

  Note that this class overrides eval_metrics_graph.EvalMetricsGraph
  which contains most of the functionality for handling the metric ops.
  In the eval saved model path, a common graph is shared between generation
  FeaturesPredictionLabels through Predict and doing the metric ops.
  The specific methods of this class constructs the graph and handles
  the prediction ops.
  """

  def __init__(self,
               path: Text,
               include_default_metrics: Optional[bool] = True,
               additional_fetches: Optional[List[Text]] = None,
               blacklist_feature_fetches: Optional[List[Text]] = None,
               tags: Optional[List[Text]] = None):
    """Initializes EvalSavedModel.

    Args:
      path: Path to model.
      include_default_metrics: True to include the in-graph metrics by default.
      additional_fetches: Prefixes of additional tensors stored in
        signature_def.inputs that should be fetched at prediction time. The
        "features" and "labels" tensors are handled automatically and should not
        be included in this list.
      blacklist_feature_fetches: List of tensor names in the features dictionary
        which should be excluded from the fetches request. This is useful in
        scenarios where features are large (e.g. images) and can lead to
        excessive memory use if stored.
      tags: Tags to use when loading the saved model.

    Raises:
      ValueError: If "features" or "labels" included in additional_fetches.
    """
    self._path = path
    self._include_default_metrics = include_default_metrics
    if additional_fetches:
      if constants.FEATURES_NAME in additional_fetches:
        raise ValueError('additional_fetches should not contain "features"')
      if constants.LABELS_NAME in additional_fetches:
        raise ValueError('additional_fetches should not contain "labels"')
    self._additional_fetches = additional_fetches
    self._blacklist_feature_fetches = blacklist_feature_fetches
    if tags:
      self._tags = tags
    else:
      self._tags = [constants.EVAL_TAG]
    super(EvalSavedModel, self).__init__()

  def _check_version(self, version_node: types.TensorType):
    version = self._session.run(version_node)
    if not version:
      raise ValueError('invalid TFMA version in graph (at path %s)' %
                       self._path)
    # We don't actually do any checking for now, since we don't have any
    # compatibility issues.

  # TODO(b/119308261): Remove once all exported EvalSavedModels are updated.
  def _legacy_check_version(self, meta_graph_def: meta_graph_pb2.MetaGraphDef):
    version = meta_graph_def.collection_def.get(
        encoding.TFMA_VERSION_COLLECTION)
    if version is None:
      raise ValueError('could not find TFMA version in graph (at path %s)' %
                       self._path)
    # We don't actually do any checking for now, since we don't have any
    # compatibility issues.

  def _get_op_from_tensor(self, op_name_tensor: types.TensorType):
    """Returns the operation based on name stored in a tensor."""
    if op_name_tensor is None:
      return None
    op_name = self._session.run(op_name_tensor).decode('utf-8')
    return self._graph.get_operation_by_name(op_name)

  def _iterate_fpl_maps_in_canonical_order(
      self
  ) -> Generator[Tuple[Text, types.FPLKeyType, types.TensorType], None, None]:
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
    meta_graph_def = tf.compat.v1.saved_model.loader.load(
        self._session, self._tags, self._path)

    with self._graph.as_default():
      signature_def = meta_graph_def.signature_def.get(
          constants.DEFAULT_EVAL_SIGNATURE_DEF_KEY)
      if signature_def is None:
        raise ValueError('could not find signature with name %s. signature_def '
                         'was %s' % (constants.EVAL_TAG, signature_def))

      self._additional_fetches_map = {}
      iterator_initializer = None

      # If features and labels are not stored in the signature_def.inputs then
      # only a single input will be present. We will use this as our flag to
      # indicate whether the features and labels should be read using the legacy
      # collections or using new signature_def.inputs.
      # TODO(b/119308261): Remove once all exported EvalSavedModels are updated.
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
        self._features_map = graph_ref.load_additional_inputs(
            constants.FEATURES_NAME, signature_def, self._graph)
        if self._blacklist_feature_fetches:
          for feature_name in self._blacklist_feature_fetches:
            self._features_map.pop(feature_name, None)
        self._labels_map = graph_ref.load_additional_inputs(
            constants.LABELS_NAME, signature_def, self._graph)
        if self._additional_fetches:
          for prefix in self._additional_fetches:
            self._additional_fetches_map[prefix] = (
                graph_ref.load_additional_inputs(prefix, signature_def,
                                                 self._graph))
        iterator_initializer = self._get_op_from_tensor(
            graph_ref.load_iterator_initializer_name(signature_def,
                                                     self._graph))

      self._predictions_map = graph_ref.load_predictions(
          signature_def, self._graph)

      # Create feed_list for metrics_reset_update_get_fn
      #
      # We need to save this because we need to update the
      # metrics_reset_update_get_fn when additional metric ops are registered
      # (the feed_list will stay the same though).
      self._perform_metrics_update_fn_feed_list = list(self._input_map.values())

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
      if iterator_initializer:
        # When iterator is used, the initializer is used to feed the inputs. The
        # values are then fetched by repeated calls to the predict_list_fn until
        # OutOfRange is thrown.
        self._iterator_initializer_fn = self._session.make_callable(
            fetches=(iterator_initializer),
            feed_list=list(self._input_map.values()))
        self._predict_list_fn = self._session.make_callable(
            fetches=(self._features_map, self._predictions_map,
                     self._labels_map, self._input_refs_node,
                     self._additional_fetches_map))
      else:
        self._iterator_initializer_fn = None
        self._predict_list_fn = self._session.make_callable(
            fetches=(self._features_map, self._predictions_map,
                     self._labels_map, self._input_refs_node,
                     self._additional_fetches_map),
            feed_list=list(self._input_map.values()))

  def get_inputs_dict(self) -> types.TensorValueMaybeDict:
    """Returns tensors used for model inputs."""
    return self._input_map

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
    # Unnest if it wasn't a dictionary to begin with.
    features = util.extract_tensor_maybe_dict(constants.FEATURES_NAME, features)

    predictions = {}
    for key, value in self._predictions_map.items():
      predictions[key] = value
    # Unnest if it wasn't a dictionary to begin with.
    predictions = util.extract_tensor_maybe_dict(constants.PREDICTIONS_NAME,
                                                 predictions)

    labels = {}
    for key, value in self._labels_map.items():
      labels[key] = value
    # Unnest if it wasn't a dictionary to begin with.
    labels = util.extract_tensor_maybe_dict(constants.LABELS_NAME, labels)

    return (features, predictions, labels)

  def predict(self,
              single_input: SingleInputFeedType) -> List[FetchedTensorValues]:
    """Returns fetches (features, predictions, labels, etc) for single_input.

    Args:
      single_input: Data to use to feed the input tensors. This must align with
        the receiver_tensors passed to EvalInputReceiver. For example, if
        receiver_tensors was a placeholder for parsing `tf.train.Example`
        protos, then this will be a serialised `tf.train.Example`.

    Returns:
      A list of FetchedTensorValues (one per example used by model). In most
      cases a single input will result in one example and the returned list will
      contain only one element, but in some cases (e.g. where examples are
      dynamically decoded and generated within the graph), the single input
      might result in zero to many examples).
    """
    return self.predict_list([single_input])

  def predict_list(self,
                   inputs: MultipleInputFeedType) -> List[FetchedTensorValues]:
    """Like predict, but takes a list of inputs.

    Args:
      inputs: A list of input data (or a dict of keys to lists of input data).
        See predict for more details.

    Returns:
       A list of FetchedTensorValues. See predict for more details.

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

    if self._iterator_initializer_fn:
      self._iterator_initializer_fn(*input_args)
      input_args = []

    result = []

    while True:
      try:
        (features, predictions, labels, input_refs,
         additional_fetches) = self._predict_list_fn(*input_args)

        all_fetches = additional_fetches
        all_fetches[constants.FEATURES_NAME] = features
        all_fetches[constants.LABELS_NAME] = labels
        all_fetches[constants.PREDICTIONS_NAME] = predictions

        # TODO(cyfoo): Optimise this.
        split_fetches = {}
        for group, tensors in all_fetches.items():
          split_tensors = {}
          for key in tensors:
            if not np.isscalar(tensors[key]):
              split_tensors[key] = util.split_tensor_value(tensors[key])
          split_fetches[group] = split_tensors

        if (not isinstance(input_refs, np.ndarray) or input_refs.ndim != 1 or
            not np.issubdtype(input_refs.dtype, np.integer)):
          raise ValueError('input_refs should be an 1-D array of integers. '
                           'input_refs was {}.'.format(input_refs))

        for group, tensors in split_fetches.items():
          for result_key, split_values in tensors.items():
            if len(split_values) != input_refs.shape[0]:
              raise ValueError(
                  'input_refs should be batch-aligned with fetched values; '
                  '{} key {} had {} slices but input_refs had batch size of '
                  '{}'.format(group, result_key, len(split_values),
                              input_refs.shape[0]))

        for i, input_ref in enumerate(input_refs):
          if input_ref < 0 or input_ref >= len(inputs):
            raise ValueError(
                'An index in input_refs is out of range: {} vs {}; '
                'inputs: {}'.format(input_ref, len(inputs), inputs))
          values = {}
          for group, split_tensors in split_fetches.items():
            tensor_values = {}
            for key, split_value in split_tensors.items():
              tensor_values[key] = split_value[i]
            values[group] = util.extract_tensor_maybe_dict(group, tensor_values)

          result.append(FetchedTensorValues(input_ref=input_ref, values=values))

        if self._iterator_initializer_fn is None:
          break
      except tf.errors.OutOfRangeError:
        break

    return result

  def as_features_predictions_labels(self,
                                     fetched_values: List[FetchedTensorValues]
                                    ) -> List[types.FeaturesPredictionsLabels]:
    """Gets features, predictions, labels as FeaturesPredictionsLabelsType."""

    def fpl_dict(fetched: FetchedTensorValues,
                 group: Text) -> types.DictOfFetchedTensorValues:
      native = fetched.values[group]
      wrapped = {}
      if not isinstance(native, dict):
        native = {util.default_dict_key(group): native}
      for key in native:
        wrapped[key] = {encoding.NODE_SUFFIX: native[key]}
      return wrapped

    fpls = []
    for fetched in fetched_values:
      fpls.append(
          types.FeaturesPredictionsLabels(
              input_ref=fetched.input_ref,
              features=fpl_dict(fetched, constants.FEATURES_NAME),
              predictions=fpl_dict(fetched, constants.PREDICTIONS_NAME),
              labels=fpl_dict(fetched, constants.LABELS_NAME)))
    return fpls
