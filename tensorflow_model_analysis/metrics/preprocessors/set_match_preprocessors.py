# Copyright 2019 Google LLC
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
"""Include preprocessors for set matching.

The following preprocssors are included:
  SetMatchPreprocessor: it transforms the sets of labels and
  predictions to a list of label prediction pair for binary classifiction.
"""

from typing import Iterator, Optional

import apache_beam as beam
import numpy as np
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import types
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.utils import util

# indices for the inputs, it should be arranged in the following format:
LEFT, TOP, RIGHT, BOTTOM, CLASS, CONFIDENCE = range(6)

_DEFAULT_SET_MATCH_PREPROCESSOR_NAME = 'set_match_preprocessor'


class SetMatchPreprocessor(metric_types.Preprocessor):
  """Computes label and prediction pairs for set matching."""

  def __init__(
      self,
      prediction_class_key: str,
      prediction_score_key: str,
      top_k: Optional[int] = None,
      name: Optional[str] = None,
      model_name: str = '',
  ):
    """Initialize the preprocessor for set matching.

    Args:
      prediction_class_key: the key name of the classes in prediction.
      prediction_score_key: the key name of the scores in prediction.
      top_k: (Optional) Used with a multi-class model to specify that the top-k
        values should be used to compute the confusion matrix. The net effect is
        that the non-top-k values are truncated and the matrix is then
        constructed from the average TP, FP, TN, FN across the classes. When
        top_k is used, metrics_specs.binarize settings must not be present.
      name: (Optional) name for the preprocessor.
      model_name: Optional model name (if multi-model evaluation).
    """
    if not name:
      name = metric_util.generate_private_name_from_arguments(
          _DEFAULT_SET_MATCH_PREPROCESSOR_NAME, top_k=top_k
      )
    super().__init__(name=name)
    self._top_k = top_k
    self._model_name = model_name
    self._prediction_class_key = prediction_class_key
    self._prediction_score_key = prediction_score_key
    self._model_top_k_exceeds_prediction_length_distribution = (
        beam.metrics.Metrics.distribution(
            constants.METRICS_NAMESPACE, 'top_k_exceeds_prediction_length'
        )
    )

  def process(
      self, extracts: types.Extracts
  ) -> Iterator[metric_types.StandardMetricInputs]:
    extracts = util.StandardExtracts(extracts)

    label_classes = extracts.get_labels(self._model_name)
    if label_classes is None or label_classes.ndim != 1:
      raise ValueError(
          f'Labels must be a 1d numpy array. The classes are {label_classes}.'
      )

    predictions = extracts.get_predictions(model_name=self._model_name)
    if not isinstance(predictions, dict):
      raise TypeError(
          'Predictions are expected to be a dictionary conatining '
          'classes and scores.'
      )

    example_weight = extracts.get_example_weights(model_name=self._model_name)

    pred_classes = util.get_by_keys(predictions, [self._prediction_class_key])
    if pred_classes is None:
      raise KeyError(
          f'Key {self._prediction_class_key} is not found under '
          'predictions of the extracts.'
      )

    if self._prediction_score_key:
      pred_scores = util.get_by_keys(predictions, [self._prediction_score_key])
      if pred_scores is None:
        raise KeyError(
            f'Key {self._prediction_score_key} is not found under '
            'predictions of the extracts.'
        )

    if pred_classes.shape != pred_scores.shape:
      raise ValueError(
          'Classes and scores must be of the same shape. Classes and scores '
          f'are {pred_classes} and {pred_scores}.'
      )

    if pred_classes.ndim != 1:
      raise ValueError(
          'Predicted classes must be a 1d numpy array. The classes are '
          f'{pred_classes}.'
      )

    if self._top_k is not None:
      if self._top_k > len(pred_classes):
        self._model_top_k_exceeds_prediction_length_distribution.update(
            len(pred_classes)
        )
      top_k = min(self._top_k, len(pred_classes))
      pred_classes = pred_classes[:top_k]
      pred_scores = pred_scores[:top_k]

    label_classes = set(label_classes)
    pred_classes_scores = dict(zip(pred_classes, pred_scores))
    pred_classes = set(pred_classes)

    # yield all true positives
    for class_name in label_classes & pred_classes:
      result = {}
      result[constants.LABELS_KEY] = np.array([1.0])
      result[constants.PREDICTIONS_KEY] = np.array(
          [pred_classes_scores[class_name]]
      )
      if example_weight is not None:
        result[constants.EXAMPLE_WEIGHTS_KEY] = example_weight
      yield metric_util.to_standard_metric_inputs(result)

    # yield all the false negatives
    for _ in label_classes - pred_classes:
      result = {}
      result[constants.LABELS_KEY] = np.array([1.0])
      result[constants.PREDICTIONS_KEY] = np.array([0.0])
      if example_weight is not None:
        result[constants.EXAMPLE_WEIGHTS_KEY] = example_weight
      yield metric_util.to_standard_metric_inputs(result)

    # yield all false positives
    for class_name in pred_classes - label_classes:
      result = {}
      result[constants.LABELS_KEY] = [0.0]
      result[constants.PREDICTIONS_KEY] = [pred_classes_scores[class_name]]
      if example_weight is not None:
        result[constants.EXAMPLE_WEIGHTS_KEY] = example_weight
      yield metric_util.to_standard_metric_inputs(result)
