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
"""Utils for working with models."""

# Standard __future__ imports

import datetime
import apache_beam as beam
import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.eval_saved_model import load

from typing import Callable, List, Optional, Text


def model_construct_fn(  # pylint: disable=invalid-name
    model_path: Optional[Text] = None,
    eval_saved_model_path: Optional[Text] = None,
    add_metrics_callbacks: Optional[List[types.AddMetricsCallbackType]] = None,
    include_default_metrics: Optional[bool] = None,
    additional_fetches: Optional[List[Text]] = None,
    blacklist_feature_fetches: Optional[List[Text]] = None,
    tag: Text = tf.saved_model.SERVING):
  """Returns function for constructing shared ModelTypes."""

  def construct_fn(model_load_seconds_callback: Callable[[int], None]):
    """Thin wrapper for the actual construct to allow for load time metrics."""

    def construct():  # pylint: disable=invalid-name
      """Function for constructing shared ModelTypes."""
      start_time = datetime.datetime.now()
      saved_model = None
      keras_model = None
      eval_saved_model = None
      if model_path:
        if tf.version.VERSION.split('.')[0] == '1':
          saved_model = tf.compat.v1.saved_model.load_v2(model_path, tags=[tag])
        else:
          saved_model = tf.saved_model.load(model_path, tags=[tag])
        try:
          keras_model = tf.keras.experimental.load_from_saved_model(model_path)
        except tf.errors.NotFoundError:
          pass
      if eval_saved_model_path:
        eval_saved_model = load.EvalSavedModel(
            eval_saved_model_path,
            include_default_metrics,
            additional_fetches=additional_fetches,
            blacklist_feature_fetches=blacklist_feature_fetches)
        if add_metrics_callbacks:
          eval_saved_model.register_add_metric_callbacks(
              add_metrics_callbacks)
        eval_saved_model.graph_finalize()
      end_time = datetime.datetime.now()
      model_load_seconds_callback(int((end_time - start_time).total_seconds()))
      return types.ModelTypes(
          saved_model=saved_model,
          keras_model=keras_model,
          eval_saved_model=eval_saved_model)

    return construct

  return construct_fn


class DoFnWithModel(beam.DoFn):
  """Abstract class for DoFns that need the shared model."""

  def __init__(self, model_loader: types.ModelLoader) -> None:
    self._model_loader = model_loader
    self._loaded_models = None  # types.ModelTypes
    self._model_load_seconds = None
    self._model_load_seconds_distribution = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE, 'model_load_seconds')

  def _set_model_load_seconds(self, model_load_seconds):
    self._model_load_seconds = model_load_seconds

  # TODO(yifanmai): Merge _setup_if_needed into setup
  # after Beam dependency is upgraded to Beam 2.14.
  def _setup_if_needed(self):
    if self._loaded_models is None:
      self._loaded_models = (
          self._model_loader.shared_handle.acquire(
              self._model_loader.construct_fn(self._set_model_load_seconds)))

  def setup(self):
    self._setup_if_needed()

  def start_bundle(self):
    self._setup_if_needed()

  def process(self, elem):
    raise NotImplementedError('not implemented')

  def finish_bundle(self):
    # Must update distribution in finish_bundle instead of setup
    # because Beam metrics are not supported in setup.
    if self._model_load_seconds is not None:
      self._model_load_seconds_distribution.update(self._model_load_seconds)
      self._model_load_seconds = None


class CombineFnWithModel(beam.CombineFn):
  """Abstract class for CombineFns that need the shared model.

  Until BEAM-3736 (Add SetUp() and TearDown() for CombineFns) is implemented
  users of this class are responsible for calling _setup_if_needed manually.
  """

  def __init__(self, model_loader: types.ModelLoader) -> None:
    self._model_loader = model_loader
    self._loaded_models = None  # types.ModelTypes
    self._model_load_seconds = None
    self._model_load_seconds_distribution = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE, 'model_load_seconds')

  def _set_model_load_seconds(self, model_load_seconds):
    self._model_load_seconds = model_load_seconds

  # TODO(yifanmai): Merge _setup_if_needed into setup
  # There's no initialisation method for CombineFns.
  # See BEAM-3736: Add SetUp() and TearDown() for CombineFns.
  def _setup_if_needed(self) -> None:
    if self._loaded_models is None:
      self._loaded_models = (
          self._model_loader.shared_handle.acquire(
              self._model_loader.construct_fn(self._set_model_load_seconds)))
      if self._model_load_seconds is not None:
        self._model_load_seconds_distribution.update(self._model_load_seconds)
        self._model_load_seconds = None
