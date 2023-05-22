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
"""Includes preprocessors for image realted functions."""

import collections
import io
from typing import Iterator, Optional, Union

import numpy as np
from PIL import Image
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import types
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.utils import util

_DECODE_IMAGE_PREPROCESSOR_BASE_NAME = 'decode_image_preprocessor'


def _image_bytes_to_numpy_array(
    image_bytes: Union[bytes, np.ndarray]
) -> np.ndarray:
  """Read bytes or a numpy scalar of bytes of image and return an ndarray."""
  if isinstance(image_bytes, np.ndarray):
    assert image_bytes.size == 1
    image_bytes = image_bytes.item()
  image = Image.open(io.BytesIO(image_bytes))
  return np.array(image, dtype=np.uint8)


class DecodeImagePreprocessor(metric_types.Preprocessor):
  """Read images of label and prediciton from bytes to numpy array."""

  def __init__(
      self,
      ground_truth_key: str,
      prediction_key: str,
      decode_ground_truth: bool = True,
      decode_prediction: bool = True,
      name: Optional[str] = None,
      model_name: str = '',
  ):
    """Initialize the preprocessor for image reading.

    Args:
      ground_truth_key: the key for storing the ground truth of encoded image
        with class ids.
      prediction_key: the key for storing the predictions of encoded image with
        class ids.
      decode_ground_truth: If true, the ground truth is assumed to be bytes of
        images and will be decoded.
      decode_prediction: If true, the prediction is assumed to be bytes of
        images and will be decoded.
      name: (Optional) name for the preprocessor.
      model_name: (Optional) model name (if multi-model evaluation).
    """
    if not name:
      name = metric_util.generate_private_name_from_arguments(
          _DECODE_IMAGE_PREPROCESSOR_BASE_NAME,
          ground_truth_key=ground_truth_key,
          prediction_key=prediction_key,
      )
    super().__init__(name=name)
    self._ground_truth_key = ground_truth_key
    self._prediction_key = prediction_key
    self._model_name = model_name
    self._decode_ground_truth = decode_ground_truth
    self._decode_prediction = decode_prediction

  def _read_image_in_mutliple_dicts(
      self,
      key: str,
      label_or_prediction: str,
      decode_image: bool,
      extracts: util.StandardExtracts,
  ):
    """Reads images from extracts."""
    if label_or_prediction not in ['label', 'prediction']:
      raise ValueError(
          'The function could only search in the lables or predictions.'
      )
    if label_or_prediction == 'label':
      one_dict_to_search = extracts.get_labels(self._model_name) or {}
    else:
      one_dict_to_search = extracts.get_predictions(self._model_name) or {}
    dict_to_search = collections.ChainMap(
        one_dict_to_search,
        extracts.get_features() or {},
        extracts.get_transformed_features(self._model_name) or {},
    )
    if not dict_to_search or key not in dict_to_search:
      raise ValueError(f'{key} is not found in {list(dict_to_search.keys())}')
    if decode_image:
      result = _image_bytes_to_numpy_array(dict_to_search[key])
    else:
      result = dict_to_search[key]
    return result

  def process(
      self, extracts: types.Extracts
  ) -> Iterator[metric_types.StandardMetricInputs]:
    """Reads and decodes images from extracts.

    It will search in labels/predictions, features and transformed features. It
    also support decoding image from bytes.

    Args:
      extracts: A tfma extract contains the image data.

    Yields:
      A standard metric input contains the following key and values:
       - {'labels'}: A numpy array represents the image of labels.
       - {'predictions'}: A numpy array represents the image of predictions.
       - {'example_weights'}: (Optional) A numpy array represents the example
         weights.
    """
    extracts = util.StandardExtracts(extracts)
    extracted_result = {}

    extracted_result[constants.LABELS_KEY] = self._read_image_in_mutliple_dicts(
        self._ground_truth_key, 'label', self._decode_ground_truth, extracts
    )

    extracted_result[constants.PREDICTIONS_KEY] = (
        self._read_image_in_mutliple_dicts(
            self._prediction_key,
            'prediction',
            self._decode_prediction,
            extracts,
        )
    )

    if (
        extracted_result[constants.LABELS_KEY].shape
        != extracted_result[constants.PREDICTIONS_KEY].shape
    ):
      raise ValueError(
          'The image size of ground truth '
          f'{extracted_result[constants.LABELS_KEY].shape} does not match '
          'with the image size of prediction '
          f'{extracted_result[constants.PREDICTIONS_KEY]}'
      )

    if constants.EXAMPLE_WEIGHTS_KEY in extracts.keys():
      extracted_result = extracts[constants.EXAMPLE_WEIGHTS_KEY]

    yield metric_util.to_standard_metric_inputs(extracted_result)
