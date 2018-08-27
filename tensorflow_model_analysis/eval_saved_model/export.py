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


from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import types as python_types


import tensorflow as tf

from tensorflow_model_analysis import types
from tensorflow_model_analysis import version
from tensorflow_model_analysis.eval_saved_model import constants
from tensorflow_model_analysis.eval_saved_model import encoding
from tensorflow_model_analysis.eval_saved_model import util
from tensorflow_model_analysis.types_compat import Callable, Optional

from tensorflow.python.estimator.export import export as export_lib





class EvalInputReceiver(export_lib.SupervisedInputReceiver):
  """A return type for eval_input_receiver_fn.

  The expected return values are:
    features: A `Tensor`, `SparseTensor`, or dict of string to `Tensor` or
      `SparseTensor`, specifying the features to be passed to the model.
    labels: A `Tensor`, `SparseTensor`, or dict of string to `Tensor` or
      `SparseTensor`, specifying the labels to be passed to the model.
    receiver_tensors: A `Tensor`, or dict of string to `Tensor`, specifying
      input nodes where this receiver expects to be fed by default. Typically
      this is a single placeholder expecting serialized `tf.Example` protos.

  This is a wrapper around TensorFlow InputReceiver that explicitly adds
  collections needed by TFMA, and also wraps the features and labels Tensors
  in identity to workaround TensorFlow issue #17568.
  """

  def __new__(cls, features, labels, receiver_tensors):
    # Workaround for TensorFlow issue #17568. Note that we pass the
    # identity-wrapped features and labels to model_fn, but we have to feed
    # the non-identity wrapped Tensors during evaluation.
    #
    # Also note that we can't wrap predictions, so metrics that have control
    # dependencies on predictions will cause the predictions to be recomputed
    # during their evaluation.
    wrapped_features = util.wrap_tensor_or_dict_of_tensors_in_identity(features)
    wrapped_labels = util.wrap_tensor_or_dict_of_tensors_in_identity(labels)
    receiver = super(EvalInputReceiver, cls).__new__(
        cls,
        features=wrapped_features,
        labels=wrapped_labels,
        receiver_tensors=receiver_tensors)
    # Note that in the collection we store the unwrapped versions, because
    # we want to feed the unwrapped versions.
    _add_features_labels_version_collections(features, labels)
    return receiver


def _add_features_labels_version_collections(features, labels):
  """Add extra collections for features, labels, version.

  This should be called within the Graph that will be saved. Typical usage
  would be when features and labels have been parsed, i.e. in the
  input_receiver_fn.

  Args:
    features: dict of strings to tensors representing features
    labels: dict of strings to tensors or a single tensor
  """
  # Labels can either be a Tensor, or a dict of Tensors.
  if not isinstance(labels, dict):
    labels = {encoding.DEFAULT_LABELS_DICT_KEY: labels}

  for label_key, label_node in labels.items():
    _encode_and_add_to_node_collection(encoding.LABELS_COLLECTION, label_key,
                                       label_node)

  # Save features.
  for feature_name, feature_node in features.items():
    _encode_and_add_to_node_collection(encoding.FEATURES_COLLECTION,
                                       feature_name, feature_node)

  tf.add_to_collection(encoding.TFMA_VERSION_COLLECTION, version.VERSION_STRING)


def _encode_and_add_to_node_collection(collection_prefix,
                                       key,
                                       node):
  tf.add_to_collection('%s/%s' % (collection_prefix, encoding.KEY_SUFFIX),
                       encoding.encode_key(key))
  tf.add_to_collection('%s/%s' % (collection_prefix, encoding.NODE_SUFFIX),
                       encoding.encode_tensor_node(node))


def build_parsing_eval_input_receiver_fn(
    feature_spec, label_key):
  """Build a eval_input_receiver_fn expecting fed tf.Examples.

  Creates a eval_input_receiver_fn that expects a serialized tf.Example fed
  into a string placeholder.  The function parses the tf.Example according to
  the provided feature_spec, and returns all parsed Tensors as features.

  Args:
    feature_spec: A dict of string to `VarLenFeature`/`FixedLenFeature`.
    label_key: The key for the label column in the feature_spec. Note that
      the label must be part of the feature_spec.

  Returns:
    A eval_input_receiver_fn suitable for use with TensorFlow model analysis.
  """

  def eval_input_receiver_fn():
    """An input_fn that expects a serialized tf.Example."""
    # Note it's *required* that the batch size should be variable for TFMA.
    serialized_tf_example = tf.placeholder(
        dtype=tf.string, shape=[None], name='input_example_tensor')
    features = tf.parse_example(serialized_tf_example, feature_spec)
    return EvalInputReceiver(
        features=features,
        receiver_tensors={'examples': serialized_tf_example},
        labels=features[label_key])

  return eval_input_receiver_fn




def export_eval_savedmodel(
    estimator,
    export_dir_base,
    eval_input_receiver_fn,
    checkpoint_path = None):
  """Export a EvalSavedModel for the given estimator.

  Args:
    estimator: Estimator to export the graph for.
    export_dir_base: Base path for export. Graph will be exported into a
      subdirectory of this base path.
    eval_input_receiver_fn: Eval input receiver function.
    checkpoint_path: Path to a specific checkpoint to export. If set to None,
      exports the latest checkpoint.

  Returns:
    Path to the directory where the eval graph was exported.

  Raises:
    ValueError: Could not find a checkpoint to export.
  """

  return tf.contrib.estimator.export_all_saved_models(
      estimator,
      export_dir_base=export_dir_base,
      input_receiver_fn_map={
          tf.estimator.ModeKeys.EVAL: eval_input_receiver_fn
      },
      checkpoint_path=checkpoint_path)


def make_export_strategy(
    eval_input_receiver_fn,
    exports_to_keep = 5):
  """Create an ExportStrategy for EvalSavedModel.

  Note: The strip_default_attrs is not used for EvalSavedModel export. And
  writing the EvalSavedModel proto in text format is not supported for now.

  Args:
    eval_input_receiver_fn: Eval input receiver function.
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
    del strip_default_attrs
    export_dir = export_eval_savedmodel(
        estimator=estimator,
        export_dir_base=export_dir_base,
        eval_input_receiver_fn=eval_input_receiver_fn,
        checkpoint_path=checkpoint_path)
    tf.contrib.learn.utils.saved_model_export_utils.garbage_collect_exports(
        export_dir_base, exports_to_keep)
    return export_dir

  return tf.contrib.learn.ExportStrategy(constants.EVAL_SAVED_MODEL_EXPORT_NAME,
                                         export_fn)
