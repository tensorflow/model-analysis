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
"""Include preprocessors for object detections.

The following preprocssors are included:
  BoundingBoxMatchPreprocessor: it transforms the list of ground truth and
  detections to a list of label prediction pair for binary classifiction.
"""

from typing import Iterator, Tuple, Optional

import apache_beam as beam
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.metrics.preprocessors.utils import bounding_box
from tensorflow_model_analysis.metrics.preprocessors.utils import box_match

# indices for the inputs, it should be arranged in the following format:
LEFT, TOP, RIGHT, BOTTOM, CLASS, CONFIDENCE = range(6)


class BoundingBoxMatchPreprocessor(beam.DoFn):
  """Computes label and prediction pairs for object detection."""

  def __init__(self,
               class_id: int,
               iou_threshold: float,
               area_range: Tuple[float, float] = (0, float('inf')),
               max_num_detections: Optional[int] = None,
               class_weight: Optional[float] = None):
    """Initialize the preprocessor for bounding box match.

    Args:
      class_id: Used for object detection, the class id for calculating metrics.
        It must be provided if use_object_detection is True.
      iou_threshold: (Optional) Used for object detection, threholds for a
        detection and ground truth pair with specific iou to be considered as a
        match.
      area_range: (Optional) Used for object detection, the area-range for
        objects to be considered for metrics.
      max_num_detections: (Optional) Used for object detection, the maximum
        number of detections for a single image.
      class_weight: (Optional) Used for object detection, the weight associated
        with the object class id.
    """
    super().__init__()
    self._threshold = iou_threshold
    self._class_id = class_id
    self._area_range = area_range
    self._max_num_detections = max_num_detections
    self._class_weight = class_weight

  def process(
      self,
      extracts: types.Extracts) -> Iterator[metric_types.StandardMetricInputs]:
    if extracts[constants.LABELS_KEY].ndim != 2 or extracts[
        constants.LABELS_KEY].shape[1] <= 4:
      raise ValueError('Raw data of ground truth should be a 2d array of shape '
                       '( <batch_size> , 5+), ground truth is '
                       f'{extracts[constants.LABELS_KEY]}')
    if extracts[constants.PREDICTIONS_KEY].ndim != 2 or extracts[
        constants.PREDICTIONS_KEY].shape[1] <= 5:
      raise ValueError('Raw data of prediction should be a 2d array of shape '
                       '( <batch_size> , 6+), prediction is '
                       f'{extracts[constants.PREDICTIONS_KEY]}')
    boxes_gt = extracts[constants.LABELS_KEY]
    boxes_pred = bounding_box.sort_boxes_by_confidence(
        extracts[constants.PREDICTIONS_KEY])
    if constants.EXAMPLE_WEIGHTS_KEY in extracts.keys():
      weight = extracts[constants.EXAMPLE_WEIGHTS_KEY]
    else:
      weight = None

    labels, predictions, weights = (
        box_match.boxes_to_label_prediction_example_weight(
            boxes_gt=boxes_gt,
            boxes_pred=boxes_pred,
            iou_threshold=self._threshold,
            area_range=self._area_range,
            class_id=self._class_id,
            max_num_detections=self._max_num_detections,
            class_weight=self._class_weight,
            weight=weight))
    for l, p, w in zip(labels, predictions, weights):
      result = {}
      result[constants.LABELS_KEY] = [l]
      result[constants.PREDICTIONS_KEY] = [p]
      result[constants.EXAMPLE_WEIGHTS_KEY] = [w]
      yield metric_util.to_standard_metric_inputs(result)
