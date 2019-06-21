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
"""DoFns that load EvalSavedModel and use it."""

# Standard __future__ imports

import datetime
import apache_beam as beam

from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.eval_saved_model import load

from typing import Callable, List, Optional, Text


def make_construct_fn(  # pylint: disable=invalid-name
    eval_saved_model_path: Text,
    add_metrics_callbacks: List[types.AddMetricsCallbackType],
    include_default_metrics: bool,
    additional_fetches: Optional[List[Text]],
    blacklist_feature_fetches: Optional[List[Text]] = None):
  """Returns construct function for Shared for constructing EvalSavedModel."""

  def construct_fn(model_load_seconds_callback: Callable[[int], None]):
    """Thin wrapper for the actual construct to allow for metrics."""

    def construct():  # pylint: disable=invalid-name
      """Function for constructing a EvalSavedModel."""
      start_time = datetime.datetime.now()
      result = load.EvalSavedModel(
          eval_saved_model_path,
          include_default_metrics,
          additional_fetches=additional_fetches,
          blacklist_feature_fetches=blacklist_feature_fetches)
      if add_metrics_callbacks:
        result.register_add_metric_callbacks(add_metrics_callbacks)
      result.graph_finalize()
      end_time = datetime.datetime.now()
      model_load_seconds_callback(int((end_time - start_time).total_seconds()))
      return result

    return construct

  return construct_fn


class EvalSavedModelDoFn(beam.DoFn):
  """Abstract class for DoFns that loads the EvalSavedModel and uses it."""

  def __init__(self, eval_shared_model: types.EvalSharedModel) -> None:
    self._eval_shared_model = eval_shared_model
    self._eval_saved_model = None  # type: load.EvalSavedModel
    self._model_load_seconds = None
    self._model_load_seconds_distribution = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE, 'model_load_seconds')

  def _set_model_load_seconds(self, model_load_seconds):
    self._model_load_seconds = model_load_seconds

  # TODO(yifanmai): Merge _setup_if_needed into setup
  # after Beam dependency is upgraded to Beam 2.14.
  def _setup_if_needed(self):
    if self._eval_saved_model is None:
      self._eval_saved_model = (
          self._eval_shared_model.shared_handle.acquire(
              self._eval_shared_model.construct_fn(
                  self._set_model_load_seconds)))

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
