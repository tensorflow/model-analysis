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

from typing import List, Optional, Text


def make_construct_fn(  # pylint: disable=invalid-name
    eval_saved_model_path: Text,
    add_metrics_callbacks: List[types.AddMetricsCallbackType],
    include_default_metrics: bool, additional_fetches: Optional[List[Text]]):
  """Returns construct function for Shared for constructing EvalSavedModel."""

  def construct_fn(model_load_seconds: beam.metrics.metricbase.Distribution):
    """Thin wrapper for the actual construct to allow for metrics."""

    def construct():  # pylint: disable=invalid-name
      """Function for constructing a EvalSavedModel."""
      start_time = datetime.datetime.now()
      result = load.EvalSavedModel(
          eval_saved_model_path,
          include_default_metrics,
          additional_fetches=additional_fetches)
      if add_metrics_callbacks:
        result.register_add_metric_callbacks(add_metrics_callbacks)
      result.graph_finalize()
      end_time = datetime.datetime.now()
      model_load_seconds.update(int((end_time - start_time).total_seconds()))
      return result

    return construct

  return construct_fn


class EvalSavedModelDoFn(beam.DoFn):
  """Abstract class for DoFns that load the EvalSavedModel and use it."""

  def __init__(self, eval_shared_model: types.EvalSharedModel) -> None:
    self._eval_shared_model = eval_shared_model
    self._eval_saved_model = None  # type: load.EvalSavedModel
    self._model_load_seconds = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE, 'model_load_seconds')

  def start_bundle(self):
    construct_fn = make_construct_fn(
        self._eval_shared_model.model_path,
        self._eval_shared_model.add_metrics_callbacks,
        self._eval_shared_model.include_default_metrics,
        self._eval_shared_model.additional_fetches)
    self._eval_saved_model = (
        self._eval_shared_model.shared_handle.acquire(
            construct_fn(self._model_load_seconds)))

  def process(self, elem):
    raise NotImplementedError('not implemented')
