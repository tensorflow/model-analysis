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

from typing import Iterator, List, Optional, Tuple

from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.metrics.preprocessors.utils import bounding_box
from tensorflow_model_analysis.metrics.preprocessors.utils import box_match
from tensorflow_model_analysis.metrics.preprocessors.utils import object_detection_format
from tensorflow_model_analysis.utils import util

# indices for the inputs, it should be arranged in the following format:
LEFT, TOP, RIGHT, BOTTOM, CLASS, CONFIDENCE = range(6)

_DEFAULT_BOUNDING_BOX_MATCH_PREPROCESSOR_NAME = 'bounding_box_match_preprocessor'
_DEFAULT_IOU_THRESHOLD = 0.5
_DEFAULT_AREA_RANGE = (0, float('inf'))


class BoundingBoxMatchPreprocessor(metric_types.Preprocessor):
  """Computes label and prediction pairs for object detection."""

  def __init__(self,
               class_id: int,
               iou_threshold: Optional[float] = 0.5,
               area_range: Tuple[float, float] = (0, float('inf')),
               max_num_detections: Optional[int] = None,
               class_weight: Optional[float] = None,
               name: Optional[str] = None,
               labels_to_stack: Optional[List[str]] = None,
               predictions_to_stack: Optional[List[str]] = None,
               num_detections_key: Optional[str] = None,
               model_name=''):
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
      name: (Optional) name for the preprocessor.
      labels_to_stack: (Optional) Keys for columns to be stacked as a single
        numpy array as the labels. It is searched under the key labels, features
        and transformed features. The desired format is [left bounadary, top
        boudnary, right boundary, bottom boundary, class id]. e.g. ['xmin',
        'ymin', 'xmax', 'ymax', 'class_id']
      predictions_to_stack: (Optional) Keys for columns to be stacked as a
        single numpy array as the prediction. It should be the model's output
        names. The desired format is [left bounadary, top boudnary, right
        boundary, bottom boundary, class id, confidence score]. e.g. ['xmin',
        'ymin', 'xmax', 'ymax', 'class_id', 'scores']
      num_detections_key: (Optional) Number of detections in each column except
        the paddings.
      model_name: Optional model name (if multi-model evaluation).
    """
    if not name:
      name = metric_util.generate_private_name_from_arguments(
          _DEFAULT_BOUNDING_BOX_MATCH_PREPROCESSOR_NAME,
          class_id=class_id,
          iou_threshold=iou_threshold,
          area_range=area_range,
          max_num_detections=max_num_detections,
          class_weight=class_weight)
    if not area_range:
      area_range = _DEFAULT_AREA_RANGE
    if iou_threshold is None:
      iou_threshold = _DEFAULT_IOU_THRESHOLD
    super().__init__(name=name)
    self._threshold = iou_threshold
    self._class_id = class_id
    self._area_range = area_range
    self._max_num_detections = max_num_detections
    self._class_weight = class_weight
    self._labels_to_stack = labels_to_stack
    self._predictions_to_stack = predictions_to_stack
    self._num_detections_key = num_detections_key
    self._model_name = model_name

  def process(
      self,
      extracts: types.Extracts) -> Iterator[metric_types.StandardMetricInputs]:
    # stack all the columns of labels and predictions(e.g. xmin, xmax, ymin,
    # ymax, and class_id of bounding boxes) into one single numpy array
    extracts = util.StandardExtracts(extracts)
    if self._labels_to_stack:
      extracts[constants.LABELS_KEY] = object_detection_format.stack_labels(
          extracts, self._labels_to_stack, model_name=self._model_name)
    if self._predictions_to_stack:
      predictions = object_detection_format.stack_predictions(
          extracts, self._predictions_to_stack, model_name=self._model_name)
    else:
      predictions = extracts.get_predictions(model_name=self._model_name)
    if self._num_detections_key:
      predictions = (
          object_detection_format.truncate_by_num_detections(
              extracts, self._num_detections_key, predictions,
              self._model_name))
    extracts[constants.PREDICTIONS_KEY] = predictions

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
