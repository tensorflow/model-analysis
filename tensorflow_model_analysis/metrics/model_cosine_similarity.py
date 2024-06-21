# Copyright 2024 Google LLC
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
"""Model cosine similiarty metrics."""

from collections.abc import Iterable
import dataclasses
from typing import Any, Optional

import apache_beam as beam
import numpy as np
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.utils import model_util

_COSINE_SIMILARITY_METRIC_NAME = 'model_cosine_similarity'


def _compute_cosine_similarity(
    baseline_prediction: np.ndarray[Any, Any],
    candidate_prediction: np.ndarray[Any, Any],
) -> float:
  """Computes cosine similarity between two predictions of np.ndarrays."""
  return np.dot(baseline_prediction, candidate_prediction) / (
      np.linalg.norm(baseline_prediction) * np.linalg.norm(candidate_prediction)
  )


@dataclasses.dataclass
class _CosineSimilarityAccumulator:
  """Accumulator for computing average CosineSimilarity."""

  num_examples: int = 0
  sum_cosine_similarity: float = 0.0

  def merge(self, other: '_CosineSimilarityAccumulator'):
    self.num_examples += other.num_examples
    self.sum_cosine_similarity += other.sum_cosine_similarity

  def get_average(self) -> float:
    if self.num_examples == 0:
      return np.nan
    return self.sum_cosine_similarity / self.num_examples


class ModelCosineSimilarity(metric_types.Metric):
  """ModelCosineSimilarity compares predictions from baseline and candidate models using cosine similarity."""

  def __init__(self, name: str = _COSINE_SIMILARITY_METRIC_NAME):
    super().__init__(self._metric_computation, name=name)

  def _metric_computation(
      self,
      name: str,
      eval_config: config_pb2.EvalConfig,
      model_names: Iterable[str],
      output_names: Optional[Iterable[str]] = ('',),
      sub_keys: Optional[Iterable[metric_types.SubKey]] = None,
  ) -> metric_types.MetricComputations:
    """Returns the metric computations for calculating the cosine similarity.

    Args:
      name: Metric name for individual flip rate.
      eval_config: The EvalConfig for this TFMA evaluation. This is used to
        identify which model is the baseline.
      model_names: The name of the baseline model and the candidate model.
      output_names: The set of output names for which to compute this metric.
      sub_keys: The set of sub_key settings for which to compute this metric.
    """
    computations = []

    # Get the baseline model name.
    baseline_spec = model_util.get_baseline_model_spec(eval_config)
    baseline_model_name = baseline_spec.name if baseline_spec else None

    for candidate_model_name in model_names:
      if candidate_model_name == baseline_model_name:
        continue
      for output_name in output_names:
        for sub_key in sub_keys or (None,):
          # Define the metric key.
          key = metric_types.MetricKey(
              name=name,
              model_name=candidate_model_name,
              output_name=output_name,
              sub_key=sub_key,
              is_diff=True,
          )

          # Append cosine similarity calculation to computations.
          computations.append(
              metric_types.MetricComputation(
                  keys=[key],
                  preprocessors=None,
                  combiner=_ModelCosineSimilarityCombiner(
                      metric_key=key,
                      eval_config=eval_config,
                      baseline_model_name=baseline_model_name,
                      model_name=candidate_model_name,
                      output_name=output_name,
                  ),
              )
          )

    return computations


class _ModelCosineSimilarityCombiner(beam.CombineFn):
  """A combiner for computing the cosine similarity between models."""

  def __init__(
      self,
      metric_key: metric_types.MetricKey,
      eval_config: config_pb2.EvalConfig,
      baseline_model_name: str,
      model_name: str,
      output_name: str,
  ):
    self._metric_key = metric_key
    self._eval_config = eval_config
    self._baseline_model_name = baseline_model_name
    self._model_name = model_name
    self._output_name = output_name

  def create_accumulator(self) -> _CosineSimilarityAccumulator:
    return _CosineSimilarityAccumulator()

  def add_input(
      self,
      accumulator: _CosineSimilarityAccumulator,
      element: metric_types.StandardMetricInputs,
  ) -> _CosineSimilarityAccumulator:
    _, baseline_prediction, _ = next(
        metric_util.to_label_prediction_example_weight(
            inputs=element,
            eval_config=self._eval_config,
            model_name=self._baseline_model_name,
            output_name=self._output_name,
            flatten=False,
            allow_none=True,
        )
    )

    _, candidate_prediction, _ = next(
        metric_util.to_label_prediction_example_weight(
            inputs=element,
            eval_config=self._eval_config,
            model_name=self._model_name,
            output_name=self._output_name,
            flatten=False,
            allow_none=True,
        )
    )
    accumulator.merge(
        _CosineSimilarityAccumulator(
            num_examples=1,
            sum_cosine_similarity=_compute_cosine_similarity(
                baseline_prediction, candidate_prediction
            ),
        )
    )

    return accumulator

  def merge_accumulators(
      self, accumulators: Iterable[_CosineSimilarityAccumulator]
  ) -> _CosineSimilarityAccumulator:
    result = next(iter(accumulators))
    for accumulator in accumulators:
      result.merge(accumulator)
    return result

  def extract_output(
      self, accumulator: _CosineSimilarityAccumulator
  ) -> dict[metric_types.MetricKey, float]:
    return {self._metric_key: accumulator.get_average()}


# Register Model Cosine Similarity metric.
metric_types.register_metric(ModelCosineSimilarity)
