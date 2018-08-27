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



import datetime

import apache_beam as beam
from tensorflow_model_analysis import types
from tensorflow_model_analysis.eval_saved_model import load
from tensorflow_model_analysis.eval_saved_model import util
from tensorflow_transform.beam import shared

from tensorflow_model_analysis.types_compat import Callable, Dict, List, Optional, Tuple

_METRICS_NAMESPACE = 'tensorflow_model_analysis'



def make_construct_fn(  # pylint: disable=invalid-name
    eval_saved_model_path,
    add_metrics_callbacks,
    model_load_seconds_distribution):
  """Returns construct function for Shared for constructing EvalSavedModel."""

  def construct():  # pylint: disable=invalid-name
    """Function for constructing a EvalSavedModel."""
    start_time = datetime.datetime.now()
    result = load.EvalSavedModel(eval_saved_model_path)
    if add_metrics_callbacks:
      features_dict, predictions_dict, labels_dict = (
          result.get_features_predictions_labels_dicts())
      features_dict = util.wrap_tensor_or_dict_of_tensors_in_identity(
          features_dict)
      predictions_dict = util.wrap_tensor_or_dict_of_tensors_in_identity(
          predictions_dict)
      labels_dict = util.wrap_tensor_or_dict_of_tensors_in_identity(labels_dict)
      with result.graph_as_default():
        metric_ops = {}
        for add_metrics_callback in add_metrics_callbacks:
          new_metric_ops = add_metrics_callback(features_dict, predictions_dict,
                                                labels_dict)
          overlap = set(new_metric_ops.keys()) & set(metric_ops.keys())
          if overlap:
            raise ValueError('metric keys should not conflict, but an '
                             'earlier callback already added the metrics '
                             'named %s' % overlap)
          metric_ops.update(new_metric_ops)
        result.register_additional_metric_ops(metric_ops)
    end_time = datetime.datetime.now()
    model_load_seconds_distribution.update(
        int((end_time - start_time).total_seconds()))
    return result

  return construct


class EvalSavedModelDoFn(beam.DoFn):
  """Abstract class for DoFns that load the EvalSavedModel and use it."""

  def __init__(self, eval_saved_model_path,
               add_metrics_callbacks,
               shared_handle):
    self._eval_saved_model_path = eval_saved_model_path
    self._add_metrics_callbacks = add_metrics_callbacks
    self._shared_handle = shared_handle
    self._eval_saved_model = None  # type: load.EvalSavedModel
    self._model_load_seconds = beam.metrics.Metrics.distribution(
        _METRICS_NAMESPACE, 'model_load_seconds')

  def start_bundle(self):
    self._eval_saved_model = self._shared_handle.acquire(
        make_construct_fn(self._eval_saved_model_path,
                          self._add_metrics_callbacks,
                          self._model_load_seconds))

  def process(self, elem):
    raise NotImplementedError('not implemented')
