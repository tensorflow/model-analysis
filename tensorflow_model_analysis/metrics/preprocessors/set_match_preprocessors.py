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
      class_key: Optional[str] = None,
      weight_key: Optional[str] = None,
      top_k: Optional[int] = None,
      name: Optional[str] = None,
      model_name: str = '',
  ):
    """Initialize the preprocessor for set matching.

    Example:
      Labels: ['sun', 'moon']
      Predictions: {
          'classes': ['sun', 'sea', 'light'],
      }

      The (label, prediction) tuples generated are:
        (1, 1) (TP for sun)
        (1, 0) (FN for moon)
        (0, 1) (FP for sea)
        (0, 1) (FP for light)

    Example with class weights:
    Note: The preporcessor supports class wise weights inside each example. The
    weight should be a numpy array stored in the features. The user could
    provide the corresponding classes of the weights. If it is not provided,
    then by default, the preprocessor assumes the weights are for labels. The
    classes and weights should be of the same length.

    For classes with specified weights, the final weights of the label
    prediction pair is class weight * example weight.
    For the classes not listed in the class-weight pairs, the weight will be
    the example_weight by default.

      'labels': ['sun', 'moon']
      'predictions': {
          'classes': ['sun', 'sea', 'light'],
          'scores': [1, 0.7, 0.3],
      }
      'example_weights': [0.1]
      'features': 'class_weights': [0.3, 0.4]

      The (label, prediction, example weight) tuples generated are:
        (1, 1, 0.03) (TP for sun with weight 0.3 * 0.1)
        (1, 0, 0.04) (FN for moon with weight 0.4 * 0.1)
        (0, 0.7, 0.1) (FP for sea with weight 0.1)
        (0, 0.3, 0.1) (FP for light with weight 0.1)

    Args:
      prediction_class_key: The key name of the classes in predictions.
      prediction_score_key: The key name of the scores in predictions.
      class_key: (Optional) The key name of the classes in class-weight pairs.
        If it is not provided, the classes are assumed to be the label classes.
      weight_key: (Optional) The key name of the weights of classes in
        class-weight pairs. The value in this key should be a numpy array of the
        same length as the classes in class_key. The key should be stored under
        the features key.
      top_k: (Optional) Used with a multi-class model to specify that the top-k
        values should be used to compute the confusion matrix. The net effect is
        that the non-top-k values are truncated and the matrix is then
        constructed from the average TP, FP, TN, FN across the classes. When
        top_k is used, metrics_specs.binarize settings must not be present.
      name: (Optional) name for the preprocessor.
      model_name: (Optional) model name (if multi-model evaluation).
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
    self._class_key = class_key
    self._weight_key = weight_key
    self._model_top_k_exceeds_prediction_length_distribution = (
        beam.metrics.Metrics.distribution(
            constants.METRICS_NAMESPACE, 'top_k_exceeds_prediction_length'
        )
    )

  def process(
      self, extracts: types.Extracts
  ) -> Iterator[metric_types.StandardMetricInputs]:
    extracts = util.StandardExtracts(extracts)

    labels = extracts.get_labels(self._model_name)

    if labels is None or labels.ndim != 1:
      raise ValueError(
          f'Labels must be a 1d numpy array. The classes are {labels}.'
      )

    # classes and weights are two lists representing the class wise weights.
    classes = None
    weights = None
    if self._weight_key:
      features = extracts.get_features()
      if features is None:
        raise ValueError(
            'Weights should be under "features" key. However, features is None'
        )
      weights = util.get_by_keys(features, [self._weight_key])
      if self._class_key:
        classes = util.get_by_keys(features, [self._class_key])
      else:
        classes = labels

      if classes is None or not isinstance(classes, np.ndarray):
        raise TypeError(
            'The classes for class-weight pair should be a numpy'
            f' array. The classes are {classes}.'
        )
      if weights is None or not isinstance(weights, np.ndarray):
        raise TypeError(
            'The classes for class_weight pair should be a numpy'
            f' array. The classes are {weights}.'
        )
      if classes.shape != weights.shape:
        raise ValueError(
            'Classes and weights must be of the same shape.'
            f' Classes and weights are {classes} and {weights}.'
        )

    predictions = extracts.get_predictions(model_name=self._model_name)
    if not isinstance(predictions, dict):
      raise TypeError(
          'Predictions are expected to be a dictionary conatining '
          'classes and scores.'
      )

    pred_classes = util.get_by_keys(predictions, [self._prediction_class_key])
    if pred_classes is None:
      raise KeyError(
          f'Key {self._prediction_class_key} is not found under '
          'predictions of the extracts.'
      )

    pred_scores = None
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
      if pred_scores is not None:
        pred_scores = pred_scores[:top_k]

    example_weight = extracts.get_example_weights(model_name=self._model_name)

    class_weights = None
    if classes is not None and weights is not None:
      class_weights = dict(zip(classes, weights))

    label_classes = set(labels)
    pred_classes_scores = dict(zip(pred_classes, pred_scores))
    pred_classes = set(pred_classes)

    def calculate_weights(class_name):
      weight = np.array([1.0])
      if not example_weight and not class_weights:
        return None
      if example_weight:
        weight *= example_weight
      if class_weights and class_name in class_weights:
        weight *= class_weights[class_name]
      return weight

    # yield all true positives
    for class_name in label_classes & pred_classes:
      result = {}
      result[constants.LABELS_KEY] = np.array([1.0])
      result[constants.PREDICTIONS_KEY] = np.array(
          [pred_classes_scores[class_name]]
      )
      result[constants.EXAMPLE_WEIGHTS_KEY] = calculate_weights(class_name)
      yield metric_util.to_standard_metric_inputs(result)

    # yield all the false negatives
    for class_name in label_classes - pred_classes:
      result = {}
      result[constants.LABELS_KEY] = np.array([1.0])
      # set the prediction score to float('-inf') such that it will always be
      # counted as negative
      result[constants.PREDICTIONS_KEY] = np.array([float('-inf')])
      result[constants.EXAMPLE_WEIGHTS_KEY] = calculate_weights(class_name)
      yield metric_util.to_standard_metric_inputs(result)

    # yield all false positives
    for class_name in pred_classes - label_classes:
      result = {}
      result[constants.LABELS_KEY] = [0.0]
      result[constants.PREDICTIONS_KEY] = [pred_classes_scores[class_name]]
      result[constants.EXAMPLE_WEIGHTS_KEY] = calculate_weights(class_name)
      yield metric_util.to_standard_metric_inputs(result)
