# Lint as: python3
# Copyright 2020 Google LLC
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
"""Utils for evaluations using the keras."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

from typing import List, Text

import tensorflow as tf
from tensorflow_model_analysis import config
from tensorflow_model_analysis import types
from tensorflow_model_analysis.metrics import metric_specs


def metrics_specs_from_keras(
    model_name: Text,
    model_loader: types.ModelLoader,
) -> List[config.MetricsSpec]:
  """Returns metrics specs for metrics and losses associated with the model."""
  model = model_loader.construct_fn()
  if model is None:
    return []

  metric_names = []
  metrics = []
  if hasattr(model, 'loss_functions'):
    # Legacy keras metrics separate the losses from the metrics and store them
    # under loss_functions. The first name in metric_names is always 'loss'
    # followed by the loss_function names (prefixed by output_name if multiple
    # outputs) and then followed by the metric names (also prefixed by output
    # name). Note that names in loss_functions will not have any output name
    # prefixes (if used) while the metrics will so we need to use the names in
    # metric_names for matching with outputs not the names in the functions.
    metric_names = model.metrics_names
    metrics.extend(model.loss_functions)
    metrics.extend(model.metrics)
    if len(metric_names) > len(metrics) and metric_names[0] == 'loss':
      metric_names = metric_names[1:]
  elif hasattr(model, 'compiled_loss') and hasattr(model, 'compiled_metrics'):
    # In the new keras metric setup the metrics include the losses (in the form
    # of a metric type not a loss type) and the metrics_names align with the
    # names in the metric classes. The metrics itself contains compiled_loss,
    # compiled_metrics, and custom metrics (added via add_metric). Since we only
    # care about compiled metrics we use these APIs instead. Note that the
    # overall loss metric is an average of the other losses which doesn't take
    # y_true, y_pred as inputs so it can't be calculated via standard inputs so
    # we remove it.
    for m in model.compiled_loss.metrics:
      # TODO(b/143228390): Pure Mean metrics cannot be calculated using labels,
      # predictions, and example weights.
      if type(m) in (tf.keras.metrics.Mean,):  # pylint: disable=unidiomatic-typecheck
        continue
      metrics.append(m)
    metrics.extend(model.compiled_metrics.metrics)
    metric_names = [m.name for m in metrics]

  specs = []

  # Need to check if model.output_names exists because the keras Sequential
  # model doesn't always contain output_names (b/150510258).
  if hasattr(model, 'output_names') and len(model.output_names) > 1:
    unmatched_metrics = {m for m in metrics}
    for output_name in model.output_names:
      per_output_metrics = []
      for (name, metric) in zip(metric_names, metrics):
        if name.startswith(output_name + '_'):
          per_output_metrics.append(metric)
          unmatched_metrics.remove(metric)
      if per_output_metrics:
        specs.extend(
            metric_specs.specs_from_metrics(
                metrics=per_output_metrics,
                model_names=[model_name],
                output_names=[output_name],
                include_example_count=False,
                include_weighted_example_count=False))
    metrics = list(unmatched_metrics)

  if metrics:
    specs.extend(
        metric_specs.specs_from_metrics(
            metrics=metrics,
            model_names=[model_name],
            include_example_count=False,
            include_weighted_example_count=False))

  return specs
