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
"""Contains functions to computes pairwise ious for two lists of boxes.

It includes functions for pairwise iou calculations and box matching utilities.
"""

from typing import Callable, Iterable, Union, Tuple, Optional
import numpy as np

from tensorflow_model_analysis.metrics.preprocessors.utils import bounding_box

# indices for the inputs, it should be arranged in the following format:
LEFT, TOP, RIGHT, BOTTOM, CLASS, CONFIDENCE = range(6)


def _match_boxes(
    ious: np.ndarray,
    thresholds: Union[float, Iterable[float]]) -> Tuple[np.ndarray, np.ndarray]:
  """Match predictions and ground_truth through the pairwise IoUs.

  Args:
   ious: a numpy array, ious[i,j] is the iou between the i th prediction and the
     j th ground truth.
   thresholds: the minimum IoU for a pair to be considered match.

  Returns:
   (matches_gt, matches_pred): a tuple of ndarray of the following,
    matches_gt: a numpy array with shape [T, G], the matched prediction index
      at each iou threshold (-1 means unmatched)
     matches_pred: a numpy array with shape [T, P], the matched ground truth
      index at each iou threshold (-1 means unmatched)
   where,
    T: num of thresholds
    P: num of predictions
    G: num of ground truth
  """
  if ious.ndim != 2:
    raise ValueError('Input ious list should be a 2d array')

  if isinstance(thresholds, float):
    thresholds = [thresholds]
  for threshold in thresholds:
    if threshold < 0:
      raise ValueError(f'Invalid input of threshold = {threshold}, should be'
                       ' greater than or equal to 0')

  num_iou_thresholds = len(thresholds)
  num_pred = ious.shape[0]
  num_gt = ious.shape[1]

  # initialize the matching, and set the type to int.
  matches_gt = -1 * np.ones((num_iou_thresholds, num_gt), dtype=int)
  matches_pred = -1 * np.ones((num_iou_thresholds, num_pred), dtype=int)

  for i, threshold in enumerate(thresholds):
    # find the matched ground truth, for each prediction
    for pred_idx in range(num_pred):
      # initialize the index of ground truth which will match with the
      # prediction.
      match_gt_index = -1
      # make sure the threshold is < 1.0
      iou = np.minimum(threshold, 1.0 - 1e-10)

      for gt_idx in range(num_gt):
        # if this ground truth is already matched, skip it
        if matches_gt[i, gt_idx] != -1:
          continue

        if ious[pred_idx, gt_idx] < iou:
          continue

        iou = ious[pred_idx, gt_idx]
        match_gt_index = gt_idx

      matches_pred[i, pred_idx] = match_gt_index
      if match_gt_index != -1:
        matches_gt[i, match_gt_index] = pred_idx
  return (matches_gt, matches_pred)


def compute_ious_for_image(boxes1: np.ndarray,
                           boxes2: np.ndarray) -> np.ndarray:
  """Computes pairwise ious for two lists of boxes.

  Args:
   boxes1: numpy array, containing a list of bounding boxes in 'corners' format.
   boxes2: numpy array, containing a list of bounding boxes in 'corners' format.
     Bounding boxes are expected to be in the corners format of [LEFT, TOP,
     RIGHT, BOTTOM]  For example, the bounding box with it's left bound at 20,
     right bound at 100, TOP_bound at 110, BOTTOM bound at 300 is be represented
     as [20, 110, 100, 300]

  Returns:
     iou_lookup_table: a vector containing the pairwise ious of boxes1 and
     boxes2. The (i,j) element of the table is the iou of the i th box of
     boxes1 and the j th element of boxes2.
  """

  # Sanity check of the dimension and shape of inputs
  if boxes1.ndim != 2 or boxes1.shape[
      1] != 4 or boxes2.ndim != 2 or boxes2.shape[1] != 4:
    raise ValueError('Input boxes lists should be a 2d array of shape '
                     f'(<batch_size>, 4), Input shapes are {boxes1.shape},'
                     f' {boxes2.shape}')

  # Split each dimension
  boxes1_xmin, boxes1_ymin, boxes1_xmax, boxes1_ymax = np.split(
      boxes1, 4, axis=1)
  boxes2_xmin, boxes2_ymin, boxes2_xmax, boxes2_ymax = np.split(
      boxes2, 4, axis=1)

  # Calculate the area of each box
  boxes1_area = bounding_box.bounding_box_area(boxes1)
  boxes2_area = bounding_box.bounding_box_area(boxes2)

  # Calculate the intersection area for each boxes pair
  intersect_ymin = np.maximum(boxes1_ymin, boxes2_ymin.transpose())
  intersect_xmin = np.maximum(boxes1_xmin, boxes2_xmin.transpose())
  intersect_ymax = np.minimum(boxes1_ymax, boxes2_ymax.transpose())
  intersect_xmax = np.minimum(boxes1_xmax, boxes2_xmax.transpose())

  intersect_width = np.maximum(intersect_xmax - intersect_xmin, 0)
  intersect_height = np.maximum(intersect_ymax - intersect_ymin, 0)
  intersect_area = intersect_width * intersect_height

  # Calculate the union area of each boxes pair
  union_area = boxes1_area[..., np.newaxis] + boxes2_area[np.newaxis,
                                                          ...] - intersect_area
  # Return with a out arg to avoid divide by zero
  return np.divide(
      intersect_area,
      union_area,
      out=np.zeros_like(intersect_area, dtype=float),
      where=union_area != 0)


def boxes_to_label_prediction_example_weight(
    boxes_gt: np.ndarray,
    boxes_pred: np.ndarray,
    class_id: int,
    iou_threshold: float,
    area_range: Optional[Tuple[float, float]] = (0, float('inf')),
    max_num_detections: Optional[int] = None,
    class_weight: Optional[float] = None,
    weight: Optional[float] = None,
    match_boxes_func: Optional[
        Callable[
            [np.ndarray, Union[float, Iterable[float]]],
            Tuple[np.ndarray, np.ndarray],
        ]
    ] = None,
    iou_func: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Generate label prediction weight tuple from ground truths and detections.

  Args:
   boxes_gt: a numpy array representing the bounding boxes in the following
     format [LEFT, TOP, RIGHT, BOTTOM, CLASS]
   boxes_pred: a numpy array representing the bounding boxes in the following
     format [LEFT, TOP, RIGHT, BOTTOM, CLASS, CONFIDENCE]
   class_id: the class to consider classification.
   iou_threshold: the threshold for two bounding boxes to be considered as a
     Match.
   area_range: objects outside of this arange will be excluded.
   max_num_detections: maximum number of detections in a single image.
   class_weight: the weight associated with this class.
   weight: weight of this image/example.
   match_boxes_func: optional alternative function to compute match_boxes.
   iou_func: optional alternative function to compute box ious.

  Returns:
   (label, prediction, weight): three lists of numpy array for binary
   classfication.
  """
  # Sanity check of the dimension and shape of inputs
  if len(boxes_gt.shape) != 2 or boxes_gt.shape[1] != 5 or len(
      boxes_pred.shape) != 2 or boxes_pred.shape[1] != 6:
    raise ValueError('Input boxes list should be a 2d array of shape ( ,5)'
                     'for ground truth and ( ,6) for prediction, where'
                     f'boxes_gt.shape = {boxes_gt.shape}, '
                     f'boxes_pred.shape = {boxes_pred.shape}')
  if max_num_detections is not None and max_num_detections <= 0:
    raise ValueError(
        f'max_num_detections = {max_num_detections} must be positive.')
  if class_id < 0:
    raise ValueError(f'class_id = {class_id} must be 0 or positive.')
  if class_weight is not None and class_weight < 0.:
    raise ValueError(f'class_weight = {class_weight} must be 0 or positive.')

  # Filter all bounding boxes with a specific Class
  boxes_gt = bounding_box.filter_boxes_by_class(boxes_gt, [class_id])
  boxes_pred = bounding_box.filter_boxes_by_class(boxes_pred, [class_id])

  # Filter ground truth bounding boxes within an area range
  boxes_gt = bounding_box.filter_boxes_by_area_range(boxes_gt, area_range)

  # Sort predictions with confidence(larger ones first)
  boxes_pred = bounding_box.sort_boxes_by_confidence(boxes_pred)

  # Limit detection numbers to max_num_detections
  if (max_num_detections
      is not None) and (boxes_pred.shape[0] > max_num_detections):
    boxes_pred = boxes_pred[:max_num_detections]

  if not iou_func:
    iou_func = compute_ious_for_image
  ious = iou_func(boxes_pred[:, :CLASS], boxes_gt[:, :CLASS])
  if not match_boxes_func:
    match_boxes_func = _match_boxes
  matches_gt, matches_pred = match_boxes_func(ious, iou_threshold)

  # It is assumed that it only takes one iou_threshold result while it
  # returns a list of results, so the matches only needs to take the first
  # entry of the results
  matches_gt = matches_gt[0]
  matches_pred = matches_pred[0]

  # Ignore the unmatched predictions which is out of the area range
  boxes_pred_area_tag = bounding_box.check_boxes_in_area_range(
      boxes_pred, area_range)
  # Only count unmatched predictions within the area range
  boxes_pred_num_unmatched = np.count_nonzero(boxes_pred_area_tag
                                              & (matches_pred == -1))

  # If the index is -1, it means unmatched
  labels = np.append(
      np.ones(boxes_gt.shape[0]), np.zeros(boxes_pred_num_unmatched))

  predictions = np.concatenate(
      (boxes_pred[matches_gt[matches_gt != -1],
                  CONFIDENCE], np.zeros(np.count_nonzero((matches_gt == -1))),
       boxes_pred[boxes_pred_area_tag & (matches_pred == -1), CONFIDENCE]))

  if weight is None:
    weight = 1.0
  if class_weight is None:
    class_weight = 1.0
  weights = np.ones_like(labels) * weight * class_weight

  return (labels, predictions, weights)
