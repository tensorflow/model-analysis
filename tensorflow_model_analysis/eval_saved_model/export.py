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

import os
import time


import tensorflow as tf
from tensorflow_model_analysis import types
from tensorflow_model_analysis.eval_saved_model import encoding
from tensorflow_model_analysis.eval_saved_model import util
from tensorflow_model_analysis.types_compat import Callable, Optional, NamedTuple  # pytype: disable=not-supported-yet

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat


class EvalInputReceiver(
    NamedTuple('EvalInputReceiver',
               [('features', types.TensorTypeMaybeDict),
                ('receiver_tensors', types.TensorTypeMaybeDict),
                ('labels', types.TensorTypeMaybeDict)])):
  """A return type for eval_input_receiver_fn.

  The expected return values are:
    features: A `Tensor`, `SparseTensor`, or dict of string to `Tensor` or
      `SparseTensor`, specifying the features to be passed to the model.
    receiver_tensors: A `Tensor`, or dict of string to `Tensor`, specifying
      input nodes where this receiver expects to be fed by default. Typically
      this is a single placeholder expecting serialized `tf.Example` protos.
    labels: A `Tensor`, `SparseTensor`, or dict of string to `Tensor` or
      `SparseTensor`, specifying the labels to be passed to the model.
  """

  # When we create a timestamped directory, there is a small chance that the


# directory already exists because another worker is also writing exports.
# In this case we just wait one second to get a new timestamp and try again.
# If this fails several times in a row, then something is seriously wrong.
MAX_DIRECTORY_CREATION_ATTEMPTS = 10


def _get_timestamped_export_dir(export_dir_base):
  """Builds a path to a new subdirectory within the base directory.

  Each export is written into a new subdirectory named using the
  current time.  This guarantees monotonically increasing version
  numbers even across multiple runs of the pipeline.
  The timestamp used is the number of seconds since epoch UTC.

  Args:
    export_dir_base: A string containing a directory to write the exported
        graph and checkpoints.
  Returns:
    The full path of the new subdirectory (which is not actually created yet).

  Raises:
    RuntimeError: if repeated attempts fail to obtain a unique timestamped
      directory name.
  """
  attempts = 0
  while attempts < MAX_DIRECTORY_CREATION_ATTEMPTS:
    export_timestamp = int(time.time())

    export_dir = os.path.join(
        compat.as_bytes(export_dir_base), compat.as_bytes(
            str(export_timestamp)))
    if not gfile.Exists(export_dir):
      # Collisions are still possible (though extremely unlikely): this
      # directory is not actually created yet, but it will be almost
      # instantly on return from this function.
      return export_dir
    time.sleep(1)
    attempts += 1
    tf.logging.warn(
        'Export directory {} already exists; retrying (attempt {}/{})'.format(
            export_dir, attempts, MAX_DIRECTORY_CREATION_ATTEMPTS))
  raise RuntimeError('Failed to obtain a unique export directory name after '
                     '{} attempts.'.format(MAX_DIRECTORY_CREATION_ATTEMPTS))


def _get_temp_export_dir(timestamped_export_dir):
  """Builds a directory name based on the argument but starting with 'temp-'.

  This relies on the fact that TensorFlow Serving ignores subdirectories of
  the base directory that can't be parsed as integers.

  Args:
    timestamped_export_dir: the name of the eventual export directory, e.g.
      /foo/bar/<timestamp>

  Returns:
    A sister directory prefixed with 'temp-', e.g. /foo/bar/temp-<timestamp>.
  """
  (dirname, basename) = os.path.split(timestamped_export_dir)
  temp_export_dir = os.path.join(
      compat.as_bytes(dirname), compat.as_bytes('temp-{}'.format(basename)))
  return temp_export_dir


def _encode_and_add_to_node_collection(collection_prefix,
                                       key,
                                       node):
  tf.add_to_collection('%s/%s' % (collection_prefix, encoding.KEY_SUFFIX),
                       encoding.encode_key(key))
  tf.add_to_collection('%s/%s' % (collection_prefix, encoding.NODE_SUFFIX),
                       encoding.encode_tensor_node(node))


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
  with tf.Graph().as_default() as g:
    eval_input_receiver = eval_input_receiver_fn()
    tf.train.create_global_step(g)
    tf.set_random_seed(estimator.config.tf_random_seed)

    # Workaround for TensorFlow issue #17568. Note that we pass the
    # identity-wrapped features and labels to model_fn, but we have to feed
    # the non-identity wrapped Tensors during evaluation.
    #
    # Also note that we can't wrap predictions, so metrics that have control
    # dependencies on predictions will cause the predictions to be recomputed
    # during their evaluation.
    wrapped_features = util.wrap_tensor_or_dict_of_tensors_in_identity(
        eval_input_receiver.features)
    wrapped_labels = util.wrap_tensor_or_dict_of_tensors_in_identity(
        eval_input_receiver.labels)

    if isinstance(estimator, tf.estimator.Estimator):
      # This is a core estimator
      estimator_spec = estimator.model_fn(
          features=wrapped_features,
          labels=wrapped_labels,
          mode=tf.estimator.ModeKeys.EVAL,
          config=estimator.config)
    else:
      # This is a contrib estimator
      model_fn_ops = estimator._call_model_fn(  # pylint: disable=protected-access
          features=wrapped_features,
          labels=wrapped_labels,
          mode=tf.estimator.ModeKeys.EVAL)
      estimator_spec = lambda x: None
      estimator_spec.predictions = model_fn_ops.predictions
      estimator_spec.eval_metric_ops = model_fn_ops.eval_metric_ops
      estimator_spec.scaffold = model_fn_ops.scaffold

    # Save metric using eval_metric_ops.
    for user_metric_key, (value_op, update_op) in (
        estimator_spec.eval_metric_ops.items()):
      tf.add_to_collection('%s/%s' % (encoding.METRICS_COLLECTION,
                                      encoding.KEY_SUFFIX),
                           encoding.encode_key(user_metric_key))
      tf.add_to_collection('%s/%s' % (encoding.METRICS_COLLECTION,
                                      encoding.VALUE_OP_SUFFIX),
                           encoding.encode_tensor_node(value_op))
      tf.add_to_collection('%s/%s' % (encoding.METRICS_COLLECTION,
                                      encoding.UPDATE_OP_SUFFIX),
                           encoding.encode_tensor_node(update_op))

    # Save all prediction nodes.
    # Predictions can either be a Tensor, or a dict of Tensors.
    predictions = estimator_spec.predictions
    if not isinstance(predictions, dict):
      predictions = {encoding.DEFAULT_PREDICTIONS_DICT_KEY: predictions}

    for prediction_key, prediction_node in predictions.items():
      _encode_and_add_to_node_collection(encoding.PREDICTIONS_COLLECTION,
                                         prediction_key, prediction_node)

    ############################################################
    ## Features, label (and weight) graph

    # Placeholder for input example to label graph.
    tf.add_to_collection(encoding.INPUT_EXAMPLE_COLLECTION,
                         encoding.encode_tensor_node(
                             eval_input_receiver.receiver_tensors['examples']))

    # Save all label nodes.
    # Labels can either be a Tensor, or a dict of Tensors.
    labels = eval_input_receiver.labels
    if not isinstance(labels, dict):
      labels = {encoding.DEFAULT_LABELS_DICT_KEY: labels}

    for label_key, label_node in labels.items():
      _encode_and_add_to_node_collection(encoding.LABELS_COLLECTION, label_key,
                                         label_node)

    # Save features.
    for feature_name, feature_node in eval_input_receiver.features.items():
      _encode_and_add_to_node_collection(encoding.FEATURES_COLLECTION,
                                         feature_name, feature_node)

    ############################################################
    ## Export as normal

    if not checkpoint_path:
      checkpoint_path = tf.train.latest_checkpoint(estimator.model_dir)
      if not checkpoint_path:
        raise ValueError(
            'Could not find trained model at %s.' % estimator.model_dir)

    export_dir = _get_timestamped_export_dir(export_dir_base)
    temp_export_dir = _get_temp_export_dir(export_dir)

    if estimator.config.session_config is None:
      session_config = config_pb2.ConfigProto(allow_soft_placement=True)
    else:
      session_config = estimator.config.session_config

    with tf.Session(config=session_config) as session:
      if estimator_spec.scaffold and estimator_spec.scaffold.saver:
        saver_for_restore = estimator_spec.scaffold.saver
      else:
        saver_for_restore = tf.train.Saver(sharded=True)
      saver_for_restore.restore(session, checkpoint_path)

      if estimator_spec.scaffold and estimator_spec.scaffold.local_init_op:
        local_init_op = estimator_spec.scaffold.local_init_op
      else:
        local_init_op = tf.train.Scaffold._default_local_init_op()
      # pylint: enable=protected-access

      # Perform the export
      builder = tf.saved_model.builder.SavedModelBuilder(temp_export_dir)
      builder.add_meta_graph_and_variables(
          session,
          [tf.saved_model.tag_constants.SERVING],
          # Don't export any signatures, since this graph is not actually
          # meant for serving.
          signature_def_map=None,
          assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS),
          legacy_init_op=local_init_op)
      builder.save(False)

      gfile.Rename(temp_export_dir, export_dir)
      return export_dir
