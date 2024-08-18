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
"""Includes preprocessors for log2 inversion transformation."""

from typing import Iterator, Optional

import numpy as np
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import types
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.utils import util

_INVERT_BINARY_LOGARITHM_PREPROCESSOR_BASE_NAME = (
    'invert_binary_logarithm_preprocessor'
)


def _invert_log2_values(
    log_values: np.ndarray,
) -> np.ndarray:
  """Invert the binary logarithm and return an ndarray."""
  # We invert the following formula: log_2(y_pred + 1.0)
  return np.power(2.0, log_values) - 1.0


class InvertBinaryLogarithmPreprocessor(metric_types.Preprocessor):
  """Read label and prediction from binary logarithm to numpy array."""

  def __init__(
      self,
      name: Optional[str] = None,
      model_name: str = '',
      prediction_winsorisation_limit_max: Optional[float] = None,
  ):
    """Initialize the preprocessor for binary logarithm inversion.

    Args:
      name: (Optional) name for the preprocessor.
      model_name: (Optional) model name (if multi-model evaluation).
      prediction_winsorisation_limit_max: should the winsorisation max limit be
        applied to the predictions.
    """
    if not name:
      name = metric_util.generate_private_name_from_arguments(
          _INVERT_BINARY_LOGARITHM_PREPROCESSOR_BASE_NAME
      )
    super().__init__(name=name)
    self._model_name = model_name
    self._prediction_winsorisation_limit_max = (
        prediction_winsorisation_limit_max
    )

  def _read_label_or_prediction_in_multiple_dicts(
      self,
      key: str,
      extracts: util.StandardExtracts,
  ) -> np.ndarray:
    """Reads and inverts the binary logarithm from extracts."""
    if key == constants.LABELS_KEY:
      value = extracts.get_labels(self._model_name)
    else:
      value = extracts.get_predictions(self._model_name)
    return _invert_log2_values(value)

  def process(
      self, extracts: types.Extracts
  ) -> Iterator[metric_types.StandardMetricInputs]:
    """Reads and inverts the binary logarithm from extracts.

    It will search in labels/predictions, features and transformed features.

    Args:
      extracts: A tfma extract contains the regression data.

    Yields:
      A standard metric input contains the following key and values:
       - {'labels'}: A numpy array represents the regressed values.
       - {'predictions'}: A numpy array represents the regression predictions.
       - {'example_weights'}: (Optional) A numpy array represents the example
         weights.
    """
    extracts = util.StandardExtracts(extracts)

    extracts[constants.LABELS_KEY] = (
        self._read_label_or_prediction_in_multiple_dicts(
            constants.LABELS_KEY, extracts
        )
    )

    predictions = self._read_label_or_prediction_in_multiple_dicts(
        constants.PREDICTIONS_KEY,
        extracts,
    )
    if self._prediction_winsorisation_limit_max is not None:
      np.clip(
          predictions,
          0.0,
          self._prediction_winsorisation_limit_max,
          out=predictions,
      )
    extracts[constants.PREDICTIONS_KEY] = predictions

    if (
        extracts[constants.LABELS_KEY].shape
        != extracts[constants.PREDICTIONS_KEY].shape
    ):
      raise ValueError(
          'The size of ground truth '
          f'{extracts[constants.LABELS_KEY].shape} does not match '
          'with the size of prediction '
          f'{extracts[constants.PREDICTIONS_KEY].shape}'
      )

    yield metric_util.to_standard_metric_inputs(extracts)
