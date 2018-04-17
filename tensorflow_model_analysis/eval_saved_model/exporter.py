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

from __future__ import print_function

import os

import tensorflow as tf

from tensorflow_model_analysis.eval_saved_model import export
from tensorflow_model_analysis.types_compat import Callable, Optional
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

  def __init__(self, name,
               eval_input_receiver_fn):
    """Create an `Exporter` to use with `tf.estimator.EvalSpec`.

    Args:
      name: Unique name of this `Exporter` that is going to be used in the
        export path.
      eval_input_receiver_fn: Eval input receiver function..
    """
    self._name = name
    self._eval_input_receiver_fn = eval_input_receiver_fn

  @property
  def name(self):
    return self._name

  def export(self, estimator, export_path,
             checkpoint_path, eval_result,
             is_the_final_export):
    del is_the_final_export

    export_result = export.export_eval_savedmodel(
        estimator=estimator,
        export_dir_base=export_path,
        eval_input_receiver_fn=self._eval_input_receiver_fn,
        checkpoint_path=checkpoint_path)

    return export_result


class FinalExporter(tf.estimator.Exporter):
  """This class exports the EvalSavedModel in the end.

  This class performs a single export in the end of training.
  """

  def __init__(self, name,
               eval_input_receiver_fn):
    """Create an `Exporter` to use with `tf.estimator.EvalSpec`.

    Args:
      name: Unique name of this `Exporter` that is going to be used in the
        export path.
      eval_input_receiver_fn: Eval input receiver function.
    """
    self._eval_saved_model_exporter = _EvalSavedModelExporter(
        name, eval_input_receiver_fn)

  @property
  def name(self):
    return self._eval_saved_model_exporter.name

  def export(self, estimator, export_path,
             checkpoint_path, eval_result,
             is_the_final_export):
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

  def __init__(self,
               name,
               eval_input_receiver_fn,
               exports_to_keep = 5):
    """Create an `Exporter` to use with `tf.estimator.EvalSpec`.

    Args:
      name: Unique name of this `Exporter` that is going to be used in the
        export path.
      eval_input_receiver_fn: Eval input receiver function.
      exports_to_keep: Number of exports to keep.  Older exports will be
        garbage-collected.  Defaults to 5.  Set to `None` to disable garbage
        collection.

    Raises:
      ValueError: if exports_to_keep is set to a non-positive value.
    """
    self._eval_saved_model_exporter = _EvalSavedModelExporter(
        name, eval_input_receiver_fn)
    self._exports_to_keep = exports_to_keep
    if exports_to_keep is not None and exports_to_keep <= 0:
      raise ValueError(
          '`exports_to_keep`, if provided, must be positive number')

  @property
  def name(self):
    return self._saved_model_exporter.name

  def export(self, estimator, export_path,
             checkpoint_path, eval_result,
             is_the_final_export):
    export_result = self._eval_saved_model_exporter.export(
        estimator, export_path, checkpoint_path, eval_result,
        is_the_final_export)

    self._garbage_collect_exports(export_path)
    return export_result

  def _garbage_collect_exports(self, export_dir_base):
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
