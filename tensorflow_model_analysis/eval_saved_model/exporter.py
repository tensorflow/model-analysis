# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""`Exporter` class represents different flavors of model export."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import contextlib
import os
import types

import tensorflow as tf

from tensorflow_model_analysis import util as tfma_util
from tensorflow_model_analysis.eval_saved_model import export
from typing import Callable, Dict, List, Optional, Text
from tensorflow.python.estimator import gc
from tensorflow.python.framework import errors_impl
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging


# Largely copied from tensorflow.python.estimator.exporter
class _EvalSavedModelExporter(tf.estimator.Exporter):
  """This class exports the EvalSavedModel.

  This class provides a basic exporting functionality and serves as a
  foundation for specialized `Exporter`s.
  """

  @tfma_util.kwargs_only
  def __init__(
      self,
      name: Text,
      eval_input_receiver_fn: Callable[[], export.EvalInputReceiverType],
      serving_input_receiver_fn: Optional[
          Callable[[], tf.estimator.export.ServingInputReceiver]] = None,
      assets_extra: Optional[Dict[Text, Text]] = None):
    """Create an `Exporter` to use with `tf.estimator.EvalSpec`.

    Args:
      name: Unique name of this `Exporter` that is going to be used in the
        export path.
      eval_input_receiver_fn: Eval input receiver function.
      serving_input_receiver_fn: (Optional) Serving input receiver function. We
        recommend that you provide this as well, so that the exported SavedModel
        also contains the serving graph. If not provided, the serving graph will
        not be included in the exported SavedModel.
      assets_extra: An optional dict specifying how to populate the assets.extra
        directory within the exported SavedModel.  Each key should give the
        destination path (including the filename) relative to the assets.extra
        directory.  The corresponding value gives the full path of the source
        file to be copied.  For example, the simple case of copying a single
        file without renaming it is specified as
        `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.
    """
    self._name = name
    self._eval_input_receiver_fn = eval_input_receiver_fn
    self._serving_input_receiver_fn = serving_input_receiver_fn
    self._assets_extra = assets_extra

  @property
  def name(self) -> Text:
    return self._name

  def export(self, estimator: tf.estimator.Estimator, export_path: Text,
             checkpoint_path: Optional[Text], eval_result: Optional[bytes],
             is_the_final_export: bool) -> bytes:
    del is_the_final_export

    export_result = export.export_eval_savedmodel(
        estimator=estimator,
        export_dir_base=export_path,
        eval_input_receiver_fn=self._eval_input_receiver_fn,
        serving_input_receiver_fn=self._serving_input_receiver_fn,
        assets_extra=self._assets_extra,
        checkpoint_path=checkpoint_path,
    )

    return export_result


class FinalExporter(tf.estimator.Exporter):
  """This class exports the EvalSavedModel in the end.

  This class performs a single export in the end of training.
  """

  @tfma_util.kwargs_only
  def __init__(
      self,
      name: Text,
      eval_input_receiver_fn: Callable[[], export.EvalInputReceiverType],
      serving_input_receiver_fn: Optional[
          Callable[[], tf.estimator.export.ServingInputReceiver]] = None,
      assets_extra: Optional[Dict[Text, Text]] = None):
    """Create an `Exporter` to use with `tf.estimator.EvalSpec`.

    Args:
      name: Unique name of this `Exporter` that is going to be used in the
        export path.
      eval_input_receiver_fn: Eval input receiver function.
      serving_input_receiver_fn: (Optional) Serving input receiver function. We
        recommend that you provide this as well, so that the exported SavedModel
        also contains the serving graph. If not privded, the serving graph will
        not be included in the exported SavedModel.
      assets_extra: An optional dict specifying how to populate the assets.extra
        directory within the exported SavedModel.  Each key should give the
        destination path (including the filename) relative to the assets.extra
        directory.  The corresponding value gives the full path of the source
        file to be copied.  For example, the simple case of copying a single
        file without renaming it is specified as
        `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.
    """
    self._eval_saved_model_exporter = _EvalSavedModelExporter(
        name=name,
        eval_input_receiver_fn=eval_input_receiver_fn,
        serving_input_receiver_fn=serving_input_receiver_fn,
        assets_extra=assets_extra)

  @property
  def name(self) -> Text:
    return self._eval_saved_model_exporter.name

  def export(self, estimator: tf.estimator.Estimator, export_path: Text,
             checkpoint_path: Optional[Text], eval_result: Optional[bytes],
             is_the_final_export: bool) -> Optional[bytes]:
    if not is_the_final_export:
      return None

    tf_logging.info('Performing the final export in the end of training.')

    return self._eval_saved_model_exporter.export(estimator, export_path,
                                                  checkpoint_path, eval_result,
                                                  is_the_final_export)


class LatestExporter(tf.estimator.Exporter):
  """This class regularly exports the EvalSavedModel.

  In addition to exporting, this class also garbage collects stale exports.
  """

  @tfma_util.kwargs_only
  def __init__(
      self,
      name: Text,
      eval_input_receiver_fn: Callable[[], export.EvalInputReceiverType],
      serving_input_receiver_fn: Optional[
          Callable[[], tf.estimator.export.ServingInputReceiver]] = None,
      exports_to_keep: int = 5):
    """Create an `Exporter` to use with `tf.estimator.EvalSpec`.

    Args:
      name: Unique name of this `Exporter` that is going to be used in the
        export path.
      eval_input_receiver_fn: Eval input receiver function.
      serving_input_receiver_fn: (Optional) Serving input receiver function. We
        recommend that you provide this as well, so that the exported SavedModel
        also contains the serving graph. If not privded, the serving graph will
        not be included in the exported SavedModel.
      exports_to_keep: Number of exports to keep.  Older exports will be
        garbage-collected.  Defaults to 5.  Set to `None` to disable garbage
        collection.

    Raises:
      ValueError: if exports_to_keep is set to a non-positive value.
    """
    self._eval_saved_model_exporter = _EvalSavedModelExporter(
        name=name,
        eval_input_receiver_fn=eval_input_receiver_fn,
        serving_input_receiver_fn=serving_input_receiver_fn)
    self._exports_to_keep = exports_to_keep
    if exports_to_keep is not None and exports_to_keep <= 0:
      raise ValueError(
          '`exports_to_keep`, if provided, must be positive number')

  @property
  def name(self) -> Text:
    return self._eval_saved_model_exporter.name

  def export(self, estimator: tf.estimator.Estimator, export_path: Text,
             checkpoint_path: Optional[Text], eval_result: Optional[bytes],
             is_the_final_export: bool) -> bytes:
    export_result = self._eval_saved_model_exporter.export(
        estimator, export_path, checkpoint_path, eval_result,
        is_the_final_export)

    self._garbage_collect_exports(export_path)
    return export_result

  def _garbage_collect_exports(self, export_dir_base: Text):
    """Deletes older exports, retaining only a given number of the most recent.

    Export subdirectories are assumed to be named with monotonically increasing
    integers; the most recent are taken to be those with the largest values.

    Args:
      export_dir_base: the base directory under which each export is in a
        versioned subdirectory.
    """
    if self._exports_to_keep is None:
      return

    def _export_version_parser(path):
      # create a simple parser that pulls the export_version from the directory.
      filename = os.path.basename(path.path)
      if not (len(filename) == 10 and filename.isdigit()):
        return None
      return path._replace(export_version=int(filename))

    # pylint: disable=protected-access
    keep_filter = gc._largest_export_versions(self._exports_to_keep)
    delete_filter = gc._negation(keep_filter)
    for p in delete_filter(
        gc._get_paths(export_dir_base, parser=_export_version_parser)):
      try:
        gfile.DeleteRecursively(p.path)
      except errors_impl.NotFoundError as e:
        tf_logging.warn('Can not delete %s recursively: %s', p.path, e)
    # pylint: enable=protected-access


@contextlib.contextmanager
def _remove_metrics(estimator: tf.estimator.Estimator,
                    metrics_to_remove: List[Text]):
  """Modifies the Estimator to make its model_fn return less metrics in EVAL.

  Note that this only removes the metrics from the
  EstimatorSpec.eval_metric_ops. It does not remove them from the graph or
  undo any side-effects that they might have had (e.g. modifications to
  METRIC_VARIABLES collections).

  This is useful for when you use py_func, streaming metrics, or other metrics
  incompatible with TFMA in your trainer. To keep these metrics in your trainer
  (so they still show up in Tensorboard) and still use TFMA, you can call
  remove_metrics on your Estimator before calling export_eval_savedmodel.

  This is a context manager, so it can be used like:
    with _remove_metrics(estimator, ['streaming_auc']):
      tfma.export.export_eval_savedmodel(estimator, ...)

  Args:
    estimator: tf.estimator.Estimator to modify. Will be mutated in place.
    metrics_to_remove: List of names of metrics to remove.

  Yields:
    Nothing.
  """
  old_call_model_fn = estimator._call_model_fn  # pylint: disable=protected-access

  def wrapped_call_model_fn(unused_self, features, labels, mode, config):
    result = old_call_model_fn(features, labels, mode, config)
    if mode == tf.estimator.ModeKeys.EVAL:
      filtered_eval_metric_ops = {}
      for k, v in result.eval_metric_ops.items():
        if k in metrics_to_remove:
          continue
        filtered_eval_metric_ops[k] = v
      result = result._replace(eval_metric_ops=filtered_eval_metric_ops)
    return result

  estimator._call_model_fn = types.MethodType(  # pylint: disable=protected-access
      wrapped_call_model_fn, estimator)

  yield

  estimator._call_model_fn = old_call_model_fn  # pylint: disable=protected-access


def adapt_to_remove_metrics(exporter: tf.estimator.Exporter,
                            metrics_to_remove: List[Text]
                           ) -> tf.estimator.Exporter:
  """Modifies the given exporter to remove metrics before export.

  This is useful for when you use py_func, streaming metrics, or other metrics
  incompatible with TFMA in your trainer. To keep these metrics in your trainer
  (so they still show up in Tensorboard) and still use TFMA, you can call
  adapt_to_remove_metrics on your TFMA exporter.

  Args:
    exporter: Exporter to modify. Will be mutated in place.
    metrics_to_remove: List of names of metrics to remove.

  Returns:
    The mutated exporter, which will be modified in place. We also return it
    so that this can be used in an expression.
  """

  old_export = exporter.export

  def wrapped_export(unused_self, estimator: tf.estimator.Estimator,
                     export_path: Text, checkpoint_path: Optional[Text],
                     eval_result: Optional[bytes],
                     is_the_final_export: bool) -> bytes:
    with _remove_metrics(estimator, metrics_to_remove):
      return old_export(estimator, export_path, checkpoint_path, eval_result,
                        is_the_final_export)

  exporter.export = types.MethodType(wrapped_export, exporter)
  return exporter
