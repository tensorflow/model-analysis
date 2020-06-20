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
"""Library for handling the model agnostic TensorFlow graph.

In particular, the class defined creates a graph with placeholders for
feeding in FeaturesPredictionsLabels and calculating metrics via metric
callbacks., Some care must be given when creating the input placeholders
and the feedlist as they match. To achieve this, the graph is created with
an example FPL to determine FPL feed structure.
"""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

from typing import List, Optional  # pytype: disable=not-supported-yet

# Standard Imports
import tensorflow as tf

from tensorflow_model_analysis import types
from tensorflow_model_analysis.eval_metrics_graph import eval_metrics_graph
from tensorflow_model_analysis.model_agnostic_eval import model_agnostic_predict


def make_construct_fn(  # pylint: disable=invalid-name
    add_metrics_callbacks: Optional[List[types.AddMetricsCallbackType]],
    config: model_agnostic_predict.ModelAgnosticConfig):
  """Returns a construct fn for constructing the model agnostic eval graph."""

  def construct_fn():  # pylint: disable=invalid-name
    """Function for constructing a model agnostic eval graph."""
    return ModelAgnosticEvaluateGraph(add_metrics_callbacks, config)

  return construct_fn


class ModelAgnosticEvaluateGraph(eval_metrics_graph.EvalMetricsGraph):
  """Class handler for using a ModelAgnosticEvaluation graph."""

  def __init__(self, add_metrics_callbacks: List[types.AddMetricsCallbackType],
               config: model_agnostic_predict.ModelAgnosticConfig):
    # Note that we do not actually initialize the graph here. The reason is we
    # wait until we get the first FeaturesPredictionsLabels to get
    # how the graph is to be constructed. Otherwise, we will need define a
    # config.
    self._add_metrics_callbacks = add_metrics_callbacks
    self._config = config
    super(ModelAgnosticEvaluateGraph, self).__init__()

  def _construct_graph(self):
    """Creates a graph which we instantiate FPL infeed and metric ops."""
    with self._graph.as_default():
      self.input_serialized_example = tf.compat.v1.placeholder(dtype=tf.string)
      features = tf.io.parse_example(
          serialized=self.input_serialized_example,
          features=self._config.feature_spec)
      self._features_map = features
      for label in self._config.label_keys:
        self._labels_map[label] = self._features_map[label]
      for pred in self._config.prediction_keys:
        self._predictions_map[pred] = self._features_map[pred]
      self.register_add_metric_callbacks(self._add_metrics_callbacks)
      self._perform_metrics_update_fn = self._session.make_callable(
          fetches=self._all_metric_update_ops,
          feed_list=[self.input_serialized_example])
