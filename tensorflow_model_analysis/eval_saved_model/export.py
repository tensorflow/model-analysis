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
"""Library for exporting the EvalSavedModel."""

# TODO(b/72233799): Have TF.Learn review this.

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import types as python_types

# Standard Imports
import six
import tensorflow as tf

from tensorflow_model_analysis import types
from tensorflow_model_analysis import util as tfma_util
from tensorflow_model_analysis import version
from tensorflow_model_analysis.eval_saved_model import constants
from tensorflow_model_analysis.eval_saved_model import encoding
from tensorflow_model_analysis.eval_saved_model import util
from typing import Callable, Dict, Optional, Text, Union

from tensorflow.python.estimator.export import export as export_lib

# TODO(b/110472071): Temporary for tf.contrib.learn Estimator support only.
IS_TF_1 = (
    hasattr(tf, 'version') and hasattr(tf.version, 'VERSION') and
    tf.version.VERSION[0] == '1')

# Return type of EvalInputReceiver function.
EvalInputReceiverType = Union[  # pylint: disable=invalid-name
    export_lib.SupervisedInputReceiver, export_lib.UnsupervisedInputReceiver]


@tfma_util.kwargs_only
def EvalInputReceiver(  # pylint: disable=invalid-name
    features: types.TensorTypeMaybeDict,
    labels: Optional[types.TensorTypeMaybeDict],
    receiver_tensors: types.TensorTypeMaybeDict,
    input_refs: Optional[types.TensorType] = None,
    iterator_initializer: Optional[Text] = None) -> EvalInputReceiverType:
  """Returns an appropriate receiver for eval_input_receiver_fn.

  This is a wrapper around TensorFlow's InputReceiver that adds additional
  entries and prefixes to the input tensors so that features and labels can be
  discovered at evaluation time. It also wraps the features and labels tensors
  in identity to workaround TensorFlow issue #17568.

  The resulting signature_def.inputs will have the following form:
    inputs/<input>     - placeholders that are used for input processing (i.e
                         receiver_tensors). If receiver_tensors is a tensor and
                         not a dict, then this will just be named 'inputs'.
    input_refs         - reference to input_refs tensor (see below).
    features/<feature> - references to tensors passed in features. If features
                         is a tensor and not a dict, then this will just be
                         named 'features'.
    labels/<label>     - references to tensors passed in labels. If labels is
                         a tensor and not a dict, then this will just be named
                         'labels'.

  Args:
    features: A `Tensor`, `SparseTensor`, or dict of string to `Tensor` or
      `SparseTensor`, specifying the features to be passed to the model.
    labels: A `Tensor`, `SparseTensor`, or dict of string to `Tensor` or
      `SparseTensor`, specifying the labels to be passed to the model. If your
      model is an unsupervised model whose `model_fn` does not accept a `labels`
      argument, you may pass None instead.
    receiver_tensors: A dict of string to `Tensor` containing exactly key named
      'examples', which maps to the single input node that the receiver expects
      to be fed by default. Typically this is a placeholder expecting serialized
      `tf.Example` protos.
    input_refs: Optional (unless iterator_initializer used). A 1-D integer
      `Tensor` that is batch-aligned with `features` and `labels` which is an
      index into receiver_tensors['examples'] indicating where this slice of
      features / labels came from. If not provided, defaults to range(0,
      len(receiver_tensors['examples'])).
    iterator_initializer: Optional name of tf.compat.v1.data.Iterator
      initializer used when the inputs are fed using an iterator. This is
      intended to be used by models that cannot handle a single large input due
      to memory resource constraints. For example, a model that takes a
      tf.train.SequenceExample record as input but only processes smaller
      batches of examples within the overall sequence at a time. The caller is
      responsible for setting the input_refs appropriately (i.e. all examples
      belonging to the same tf.train.Sequence should have the same input_ref).

  Raises:
    ValueError: receiver_tensors did not contain exactly one key named
      "examples" or iterator_initializer used without input_refs.
  """
  if list(receiver_tensors.keys()) != ['examples']:
    raise ValueError('receiver_tensors must contain exactly one key named '
                     'examples.')

  if input_refs is None:
    if iterator_initializer is not None:
      raise ValueError('input_refs is required if iterator_initializer is used')
    input_refs = tf.range(tf.size(input=list(receiver_tensors.values())[0]))

  updated_receiver_tensors = {}

  def add_tensors(prefix, tensor_or_dict):
    if isinstance(tensor_or_dict, dict):
      for key in tensor_or_dict:
        updated_receiver_tensors[prefix + '/' + key] = tensor_or_dict[key]
    else:
      updated_receiver_tensors[prefix] = tensor_or_dict

  add_tensors(constants.SIGNATURE_DEF_INPUTS_PREFIX, receiver_tensors)
  add_tensors(constants.FEATURES_NAME, features)
  if labels is not None:
    add_tensors(constants.LABELS_NAME, labels)
  updated_receiver_tensors[constants.SIGNATURE_DEF_INPUT_REFS_KEY] = (
      input_refs)
  if iterator_initializer:
    updated_receiver_tensors[
        constants.SIGNATURE_DEF_ITERATOR_INITIALIZER_KEY] = (
            tf.constant(iterator_initializer))
  updated_receiver_tensors[constants.SIGNATURE_DEF_TFMA_VERSION_KEY] = (
      tf.constant(version.VERSION_STRING))

  # TODO(b/119308261): Remove once all evaluator binaries have been updated.
  _add_tfma_collections(features, labels, input_refs)
  util.add_build_data_collection()

  # Workaround for TensorFlow issue #17568. Note that we pass the
  # identity-wrapped features and labels to model_fn, but we have to feed
  # the non-identity wrapped Tensors during evaluation.
  #
  # Also note that we can't wrap predictions, so metrics that have control
  # dependencies on predictions will cause the predictions to be recomputed
  # during their evaluation.
  wrapped_features = util.wrap_tensor_or_dict_of_tensors_in_identity(features)
  if labels is not None:
    wrapped_labels = util.wrap_tensor_or_dict_of_tensors_in_identity(labels)
    return export_lib.SupervisedInputReceiver(
        features=wrapped_features,
        labels=wrapped_labels,
        receiver_tensors=updated_receiver_tensors)
  else:
    return export_lib.UnsupervisedInputReceiver(
        features=wrapped_features, receiver_tensors=updated_receiver_tensors)
  # pylint: enable=unreachable


# TODO(b/119308261): Remove once all exported EvalSavedModels are updated.
@tfma_util.kwargs_only
def _LegacyEvalInputReceiver(  # pylint: disable=invalid-name
    features: types.TensorTypeMaybeDict,
    labels: Optional[types.TensorTypeMaybeDict],
    receiver_tensors: Dict[Text, types.TensorType],
    input_refs: Optional[types.TensorType] = None) -> EvalInputReceiverType:
  """Returns a legacy eval_input_receiver_fn.

  This is for testing purposes only.

  Args:
    features: A `Tensor`, `SparseTensor`, or dict of string to `Tensor` or
      `SparseTensor`, specifying the features to be passed to the model.
    labels: A `Tensor`, `SparseTensor`, or dict of string to `Tensor` or
      `SparseTensor`, specifying the labels to be passed to the model. If your
      model is an unsupervised model whose `model_fn` does not accept a `labels`
      argument, you may pass None instead.
    receiver_tensors: A dict of string to `Tensor` containing exactly key named
      'examples', which maps to the single input node that the receiver expects
      to be fed by default. Typically this is a placeholder expecting serialized
      `tf.Example` protos.
    input_refs: Optional. A 1-D integer `Tensor` that is batch-aligned with
      `features` and `labels` which is an index into
      receiver_tensors['examples'] indicating where this slice of features /
      labels came from. If not provided, defaults to range(0,
      len(receiver_tensors['examples'])).

  Raises:
    ValueError: receiver_tensors did not contain exactly one key named
      "examples".
  """
  # Force list representation for Python 3 compatibility.
  if list(receiver_tensors.keys()) != ['examples']:
    raise ValueError('receiver_tensors must contain exactly one key named '
                     'examples.')

  # Workaround for TensorFlow issue #17568. Note that we pass the
  # identity-wrapped features and labels to model_fn, but we have to feed
  # the non-identity wrapped Tensors during evaluation.
  #
  # Also note that we can't wrap predictions, so metrics that have control
  # dependencies on predictions will cause the predictions to be recomputed
  # during their evaluation.
  wrapped_features = util.wrap_tensor_or_dict_of_tensors_in_identity(features)
  if labels is not None:
    wrapped_labels = util.wrap_tensor_or_dict_of_tensors_in_identity(labels)
    receiver = export_lib.SupervisedInputReceiver(
        features=wrapped_features,
        labels=wrapped_labels,
        receiver_tensors=receiver_tensors)
  else:
    receiver = export_lib.UnsupervisedInputReceiver(
        features=wrapped_features, receiver_tensors=receiver_tensors)

  if input_refs is None:
    input_refs = tf.range(tf.size(input=receiver_tensors['examples']))
  # Note that in the collection we store the unwrapped versions, because
  # we want to feed the unwrapped versions.
  _add_tfma_collections(features, labels, input_refs)
  util.add_build_data_collection()
  return receiver


def _add_tfma_collections(features: types.TensorTypeMaybeDict,
                          labels: Optional[types.TensorTypeMaybeDict],
                          input_refs: types.TensorType):
  """Add extra collections for features, labels, input_refs, version.

  This should be called within the Graph that will be saved. Typical usage
  would be when features and labels have been parsed, i.e. in the
  input_receiver_fn.

  Args:
    features: dict of strings to tensors representing features
    labels: dict of strings to tensors or a single tensor
    input_refs: See EvalInputReceiver().
  """
  # Clear existing collections first, in case the EvalInputReceiver was called
  # multiple times.
  del tf.compat.v1.get_collection_ref(
      encoding.with_suffix(encoding.FEATURES_COLLECTION,
                           encoding.KEY_SUFFIX))[:]
  del tf.compat.v1.get_collection_ref(
      encoding.with_suffix(encoding.FEATURES_COLLECTION,
                           encoding.NODE_SUFFIX))[:]
  del tf.compat.v1.get_collection_ref(
      encoding.with_suffix(encoding.LABELS_COLLECTION, encoding.KEY_SUFFIX))[:]
  del tf.compat.v1.get_collection_ref(
      encoding.with_suffix(encoding.LABELS_COLLECTION, encoding.NODE_SUFFIX))[:]
  del tf.compat.v1.get_collection_ref(encoding.EXAMPLE_REF_COLLECTION)[:]
  del tf.compat.v1.get_collection_ref(encoding.TFMA_VERSION_COLLECTION)[:]

  for feature_name, feature_node in features.items():
    _encode_and_add_to_node_collection(encoding.FEATURES_COLLECTION,
                                       feature_name, feature_node)

  if labels is not None:
    # Labels can either be a Tensor, or a dict of Tensors.
    if not isinstance(labels, dict):
      labels = {util.default_dict_key(constants.LABELS_NAME): labels}

    for label_key, label_node in labels.items():
      _encode_and_add_to_node_collection(encoding.LABELS_COLLECTION, label_key,
                                         label_node)
  # Previously input_refs was called example_ref. This code is being deprecated
  # so it was not renamed.
  example_ref_collection = tf.compat.v1.get_collection_ref(
      encoding.EXAMPLE_REF_COLLECTION)
  example_ref_collection.append(encoding.encode_tensor_node(input_refs))

  tf.compat.v1.add_to_collection(encoding.TFMA_VERSION_COLLECTION,
                                 version.VERSION_STRING)


def _encode_and_add_to_node_collection(collection_prefix: Text,
                                       key: types.FPLKeyType,
                                       node: types.TensorType) -> None:
  tf.compat.v1.add_to_collection(
      encoding.with_suffix(collection_prefix, encoding.KEY_SUFFIX),
      encoding.encode_key(key))
  tf.compat.v1.add_to_collection(
      encoding.with_suffix(collection_prefix, encoding.NODE_SUFFIX),
      encoding.encode_tensor_node(node))


def build_parsing_eval_input_receiver_fn(
    feature_spec,
    label_key: Optional[Text]) -> Callable[[], EvalInputReceiverType]:
  """Build a eval_input_receiver_fn expecting fed tf.Examples.

  Creates a eval_input_receiver_fn that expects a serialized tf.Example fed
  into a string placeholder.  The function parses the tf.Example according to
  the provided feature_spec, and returns all parsed Tensors as features.

  Args:
    feature_spec: A dict of string to `VarLenFeature`/`FixedLenFeature`.
    label_key: The key for the label column in the feature_spec. Note that the
      label must be part of the feature_spec. If None, does not pass a label to
      the EvalInputReceiver (note that label_key must be None and not simply the
      empty string for this case).

  Returns:
    A eval_input_receiver_fn suitable for use with TensorFlow model analysis.
  """

  def eval_input_receiver_fn():
    """An input_fn that expects a serialized tf.Example."""
    # Note it's *required* that the batch size should be variable for TFMA.
    serialized_tf_example = tf.compat.v1.placeholder(
        dtype=tf.string, shape=[None], name='input_example_tensor')
    features = tf.io.parse_example(
        serialized=serialized_tf_example, features=feature_spec)
    labels = None if label_key is None else features[label_key]
    return EvalInputReceiver(
        features=features,
        labels=labels,
        receiver_tensors={'examples': serialized_tf_example})

  return eval_input_receiver_fn


@tfma_util.kwargs_only
def export_eval_savedmodel(
    estimator,
    export_dir_base: Text,
    eval_input_receiver_fn: Callable[[], EvalInputReceiverType],
    serving_input_receiver_fn: Optional[
        Callable[[], tf.estimator.export.ServingInputReceiver]] = None,
    assets_extra: Optional[Dict[Text, Text]] = None,
    checkpoint_path: Optional[Text] = None) -> bytes:
  """Export a EvalSavedModel for the given estimator.

  Args:
    estimator: Estimator to export the graph for.
    export_dir_base: Base path for export. Graph will be exported into a
      subdirectory of this base path.
    eval_input_receiver_fn: Eval input receiver function.
    serving_input_receiver_fn: (Optional) Serving input receiver function. We
      recommend that you provide this as well, so that the exported SavedModel
      also contains the serving graph. If not privded, the serving graph will
      not be included in the exported SavedModel.
    assets_extra: An optional dict specifying how to populate the assets.extra
      directory within the exported SavedModel.  Each key should give the
      destination path (including the filename) relative to the assets.extra
      directory.  The corresponding value gives the full path of the source file
      to be copied.  For example, the simple case of copying a single file
      without renaming it is specified as
      `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.
    checkpoint_path: Path to a specific checkpoint to export. If set to None,
      exports the latest checkpoint.

  Returns:
    Path to the directory where the EvalSavedModel was exported.

  Raises:
    ValueError: Could not find a checkpoint to export.
  """
  # TODO(b/110472071): Temporary for tf.contrib.learn Estimator support only.
  if IS_TF_1 and isinstance(estimator, tf.contrib.learn.Estimator):
    return _export_eval_savedmodel_contrib_estimator(
        estimator=estimator,
        export_dir_base=export_dir_base,
        eval_input_receiver_fn=eval_input_receiver_fn,
        serving_input_receiver_fn=serving_input_receiver_fn,
        checkpoint_path=checkpoint_path)

  return estimator.experimental_export_all_saved_models(
      export_dir_base=export_dir_base,
      input_receiver_fn_map={
          tf.estimator.ModeKeys.EVAL: eval_input_receiver_fn,
          tf.estimator.ModeKeys.PREDICT: serving_input_receiver_fn,
      },
      assets_extra=assets_extra,
      checkpoint_path=checkpoint_path)


# TODO(b/110472071): Temporary for tf.contrib.learn Estimator support only.
if IS_TF_1:

  @tfma_util.kwargs_only
  def _export_eval_savedmodel_contrib_estimator(
      estimator,
      export_dir_base: Text,
      eval_input_receiver_fn: Callable[[], EvalInputReceiverType],
      serving_input_receiver_fn: Optional[
          Callable[[], tf.estimator.export.ServingInputReceiver]] = None,
      checkpoint_path: Optional[Text] = None) -> bytes:
    """Export a EvalSavedModel for the given tf.contrib.learn Estimator.

    This is a compatibility shim for exporting tf.contrib.learn Estiamtors.
    The way we do this is we copy over the methods we need from
    tf.estimator.Estimator, add a shim for converting ModelFnOps to
    EstimatorSpec
    so we can use those methods, and then call the copied methods.

    Args:
      estimator: Estimator to export the graph for.
      export_dir_base: Base path for export. Graph will be exported into a
        subdirectory of this base path.
      eval_input_receiver_fn: Eval input receiver function.
      serving_input_receiver_fn: (Optional) Serving input receiver function. If
        not privded, the serving graph will not be included in the exported
        SavedModel.
      checkpoint_path: Path to a specific checkpoint to export. If set to None,
        exports the latest checkpoint.

    Returns:
      Path to the directory where the eval graph was exported.

    Raises:
      ValueError: Could not find a checkpoint to export; Or the Estimator had
        an _export_all_saved_models method, which suggests that it wasn't
        actually a contrib Estimator.
    """

    if hasattr(estimator, 'experimental_export_all_saved_models'):
      raise ValueError(
          'not expecting contrib estimator to have '
          'experimental_export_all_saved_models method, but the estimator had '
          'the method')

    # Copy the TF Core Estimator methods we need for exporting the model
    # into the tf.contrib.learn Estimator we got.
    methods = [
        'experimental_export_all_saved_models',
        '_add_meta_graph_for_mode',
        '_export_all_saved_models',
        '_create_and_assert_global_step',
        '_create_global_step',
    ]
    for method in methods:
      if six.PY2:
        estimator_function = getattr(tf.estimator.Estimator, method).__func__
      else:
        estimator_function = getattr(tf.estimator.Estimator, method)
      fn = python_types.MethodType(estimator_function, estimator)
      setattr(estimator, method, fn)

    # We need our own version of this because the TF version calls into its
    # parent class, which isn't what it expects for contrib Estimators.
    def call_add_meta_graph_and_variables(
        self,  # pylint: disable=unused-argument
        save_variables,
        builder,
        session,
        kwargs):
      if save_variables:
        builder.add_meta_graph_and_variables(session, **kwargs)
      else:
        builder.add_meta_graph(**kwargs)

    setattr(
        estimator, '_call_add_meta_graph_and_variables',
        python_types.MethodType(call_add_meta_graph_and_variables, estimator))
    setattr(estimator, '_strip_default_attrs', False)

    def call_model_fn_override(self, *args, **kwargs):
      """Shim that converts ModelFnOps to EstimatorSpec."""
      model_fn_ops = self._old_call_model_fn(*args, **kwargs)  # pylint: disable=protected-access
      estimator_spec = tf.estimator.EstimatorSpec(
          loss=tf.constant(0.0),
          mode=tf.estimator.ModeKeys.EVAL,
          predictions=model_fn_ops.predictions,
          eval_metric_ops=model_fn_ops.eval_metric_ops,
          scaffold=model_fn_ops.scaffold)
      return estimator_spec

    estimator._old_call_model_fn = estimator._call_model_fn  # pylint: disable=protected-access
    estimator._call_model_fn = python_types.MethodType(  # pylint: disable=protected-access
        call_model_fn_override, estimator)

    if serving_input_receiver_fn:

      def wrapped_serving_input_receiver_fn():
        """Shim to maybe upconvert legacy InputFnOps to ServingInputReceiver."""
        maybe_input_fn_ops = serving_input_receiver_fn()
        if isinstance(maybe_input_fn_ops, tf.contrib.learn.InputFnOps):
          return tf.estimator.export.ServingInputReceiver(
              features=maybe_input_fn_ops.features,
              receiver_tensors=maybe_input_fn_ops.default_inputs)
        return maybe_input_fn_ops
    else:
      wrapped_serving_input_receiver_fn = None

    result = tf.contrib.estimator.export_all_saved_models(
        estimator,
        export_dir_base=export_dir_base,
        input_receiver_fn_map={
            tf.estimator.ModeKeys.EVAL: eval_input_receiver_fn,
            tf.estimator.ModeKeys.PREDICT: wrapped_serving_input_receiver_fn,
        },
        checkpoint_path=checkpoint_path)

    # Undo all the changes we made.
    estimator._call_model_fn = estimator._old_call_model_fn  # pylint: disable=protected-access
    del estimator._old_call_model_fn
    for method in methods:
      delattr(estimator, method)
    delattr(estimator, '_call_add_meta_graph_and_variables')
    delattr(estimator, '_strip_default_attrs')

    return result

  @tfma_util.kwargs_only
  def make_export_strategy(
      eval_input_receiver_fn: Callable[[], EvalInputReceiverType],
      serving_input_receiver_fn: Optional[
          Callable[[], tf.estimator.export.ServingInputReceiver]] = None,
      exports_to_keep: Optional[int] = 5,
  ) -> tf.contrib.learn.ExportStrategy:
    """Create an ExportStrategy for EvalSavedModel.

    Note: The strip_default_attrs is not used for EvalSavedModel export. And
    writing the EvalSavedModel proto in text format is not supported for now.

    Args:
      eval_input_receiver_fn: Eval input receiver function.
      serving_input_receiver_fn: (Optional) Serving input receiver function. We
        recommend that you provide this as well, so that the exported SavedModel
        also contains the serving graph. If not privded, the serving graph will
        not be included in the exported SavedModel.
      exports_to_keep: Number of exports to keep.  Older exports will be
        garbage-collected.  Defaults to 5.  Set to None to disable garbage
        collection.

    Returns:
      An ExportStrategy for EvalSavedModel that can be passed to the
      tf.contrib.learn.Experiment constructor.
    """

    def export_fn(estimator,
                  export_dir_base,
                  checkpoint_path=None,
                  strip_default_attrs=False):
      """Export function."""
      del strip_default_attrs
      export_dir = export_eval_savedmodel(
          estimator=estimator,
          export_dir_base=export_dir_base,
          eval_input_receiver_fn=eval_input_receiver_fn,
          serving_input_receiver_fn=serving_input_receiver_fn,
          checkpoint_path=checkpoint_path)
      tf.contrib.learn.utils.saved_model_export_utils.garbage_collect_exports(
          export_dir_base, exports_to_keep)
      return export_dir

    return tf.contrib.learn.ExportStrategy(
        constants.EVAL_SAVED_MODEL_EXPORT_NAME, export_fn)
