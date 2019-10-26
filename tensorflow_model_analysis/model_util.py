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
from tensorflow_model_analysis import config
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.eval_saved_model import constants as eval_constants
from tensorflow_model_analysis.eval_saved_model import load

from typing import Callable, Dict, List, Optional, Text


def get_baseline_model_spec(
    eval_config: config.EvalConfig) -> Optional[config.ModelSpec]:
  """Returns baseline model spec."""
  for spec in eval_config.model_specs:
    if spec.is_baseline:
      return spec
  return None


def get_model_spec(eval_config: config.EvalConfig,
                   model_name: Text) -> Optional[config.ModelSpec]:
  """Returns model spec with given model name."""
  for spec in eval_config.model_specs:
    if spec.name == model_name:
      return spec
  return None


def model_construct_fn(  # pylint: disable=invalid-name
    eval_saved_model_path: Optional[Text] = None,
    add_metrics_callbacks: Optional[List[types.AddMetricsCallbackType]] = None,
    include_default_metrics: Optional[bool] = None,
    additional_fetches: Optional[List[Text]] = None,
    blacklist_feature_fetches: Optional[List[Text]] = None,
    tags: Optional[List[Text]] = None):
  """Returns function for constructing shared ModelTypes."""
  if tags is None:
    tags = [eval_constants.EVAL_TAG]

  def construct_fn(model_load_seconds_callback: Callable[[int], None]):
    """Thin wrapper for the actual construct to allow for load time metrics."""

    def construct():  # pylint: disable=invalid-name
      """Function for constructing shared ModelTypes."""
      start_time = datetime.datetime.now()
      saved_model = None
      keras_model = None
      eval_saved_model = None
      if tags == [eval_constants.EVAL_TAG]:
        eval_saved_model = load.EvalSavedModel(
            eval_saved_model_path,
            include_default_metrics,
            additional_fetches=additional_fetches,
            blacklist_feature_fetches=blacklist_feature_fetches)
        if add_metrics_callbacks:
          eval_saved_model.register_add_metric_callbacks(add_metrics_callbacks)
        eval_saved_model.graph_finalize()
      else:
        # TODO(b/141524386, b/141566408): TPU Inference is not supported
        # for Keras saved_model yet.
        try:
          keras_model = tf.keras.models.load_model(eval_saved_model_path)
          # In some cases, tf.keras.models.load_model can successfully load a
          # saved_model but it won't actually be a keras model.
          if not isinstance(keras_model, tf.keras.models.Model):
            keras_model = None
        except Exception:  # pylint: disable=broad-except
          keras_model = None
        if keras_model is None:
          if tf.saved_model.TPU in tags:
            tf.tpu.experimental.initialize_tpu_system()
          saved_model = tf.compat.v1.saved_model.load_v2(
              eval_saved_model_path, tags=tags)
      end_time = datetime.datetime.now()
      model_load_seconds_callback(int((end_time - start_time).total_seconds()))
      return types.ModelTypes(
          saved_model=saved_model,
          keras_model=keras_model,
          eval_saved_model=eval_saved_model)

    return construct

  return construct_fn


class DoFnWithModels(beam.DoFn):
  """Abstract class for DoFns that need the shared models."""

  def __init__(self, model_loaders: Dict[Text, types.ModelLoader]) -> None:
    """Initializes DoFn using dict of model loaders keyed by model location."""
    self._model_loaders = model_loaders
    self._loaded_models = None  # types.ModelTypes
    self._model_load_seconds = None
    self._model_load_seconds_distribution = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE, 'model_load_seconds')

  def _set_model_load_seconds(self, model_load_seconds):
    self._model_load_seconds = model_load_seconds

  def setup(self):
    self._loaded_models = {}
    for model_name, model_loader in self._model_loaders.items():
      self._loaded_models[model_name] = model_loader.shared_handle.acquire(
          model_loader.construct_fn(self._set_model_load_seconds))

  def process(self, elem):
    raise NotImplementedError('not implemented')

  def finish_bundle(self):
    # Must update distribution in finish_bundle instead of setup
    # because Beam metrics are not supported in setup.
    if self._model_load_seconds is not None:
      self._model_load_seconds_distribution.update(self._model_load_seconds)
      self._model_load_seconds = None


class CombineFnWithModels(beam.CombineFn):
  """Abstract class for CombineFns that need the shared models.

  Until BEAM-3736 (Add SetUp() and TearDown() for CombineFns) is implemented
  users of this class are responsible for calling _setup_if_needed manually.
  """

  def __init__(self, model_loaders: Dict[Text, types.ModelLoader]) -> None:
    """Initializes CombineFn using dict of loaders keyed by model location."""
    self._model_loaders = model_loaders
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
      self._loaded_models = {}
      for model_name, model_loader in self._model_loaders.items():
        self._loaded_models[model_name] = model_loader.shared_handle.acquire(
            model_loader.construct_fn(self._set_model_load_seconds))
      if self._model_load_seconds is not None:
        self._model_load_seconds_distribution.update(self._model_load_seconds)
        self._model_load_seconds = None
