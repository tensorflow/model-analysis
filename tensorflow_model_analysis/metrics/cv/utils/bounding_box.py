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
"""Contains functions to operate bounding boxes.

It includes functions for bounding boxes filtering, area calculations, sorting,
etc.
"""

from typing import Iterable, Tuple
import warnings
import numpy as np

# indices for the inputs, it should be arranged in the following format:
# [LEFT, TOP, RIGHT, BOTTOM, CLASS_ID, CONFIDENCE]
LEFT, TOP, RIGHT, BOTTOM, CLASS, CONFIDENCE = range(6)


def bounding_box_area(boxes: np.ndarray) -> np.ndarray:
  """Compute areas for a list of boxes.

  Args:
   boxes: a numpy array with dimenstion [ <batch_size> , 4+] in corners format
     [LEFT, TOP, RIGHT, BOTTOM]

  Returns:
   boxes_areas: a numpy array with dimension len(boxes)
  """
  if boxes.ndim != 2 or boxes.shape[1] < 4:
    raise ValueError('Input boxes list should be a 2d array of shape '
                     '( <batch_size> , 4+)')
  # Calculate the height and width for each dimension
  if np.any(boxes[:, RIGHT] - boxes[:, LEFT] < 0):
    raise ValueError('The right boundary is less than the left boundary '
                     'right = {}, left = {}.'.format(boxes[:, RIGHT],
                                                     boxes[:, LEFT]))
  if np.any(boxes[:, BOTTOM] - boxes[:, TOP] < 0):
    raise ValueError('The BOTTOM boundary is less than the TOP boundary '
                     'BOTTOM = {}, TOP = {}.'.format(boxes[:, BOTTOM],
                                                     boxes[:, TOP]))
  boxes_width = boxes[:, RIGHT] - boxes[:, LEFT]
  boxes_height = boxes[:, BOTTOM] - boxes[:, TOP]

  # Calculate the area of each box
  boxes_areas = boxes_width * boxes_height
  return boxes_areas


def filter_boxes_by_class(boxes: np.ndarray,
                          class_id: Iterable[int]) -> np.ndarray:
  """Select boxes for a given set of classes.

  Args:
   boxes: a numpy array representing the bounding boxes in the following format
     [LEFT, TOP, RIGHT, BOTTOM, CLASS, CONFIDENCE(Optional)]
   class_id: id of target num_classes

  Returns:
   filtered_boxes: the filtered bounding boxes
  """
  if boxes.ndim != 2 or boxes.shape[1] <= CLASS:
    raise ValueError(f'Input boxes list should be a 2d array of shape '
                     f'(<batch_size>, {CLASS}+)')
  if isinstance(class_id, int):
    class_id = [class_id]
  return boxes[np.isin(boxes[:, CLASS], class_id), :]


def check_boxes_in_area_range(boxes: np.ndarray,
                              area_range: Tuple[float, float]) -> np.ndarray:
  """Check boxes whether their areas fall in a given range.

  Args:
   boxes: a numpy array representing the bounding boxes in the following format
     [LEFT, TOP, RIGHT, BOTTOM, CLASS, CONFIDENCE(Optional)]
   area_range: [lowerbound(inclusive), upperbound(inclusive)] of the box area

  Returns:
   A numpy array of bool indicates whether the box is in the range.
  """
  if boxes.ndim != 2 or boxes.shape[1] <= 3:
    raise ValueError('Input boxes list should be a 2d array of shape '
                     '(<batch_size>,4+)')

  if len(area_range) != 2:
    raise ValueError('Invalid shape of area_range')
  if area_range[1] <= area_range[0]:
    raise ValueError('Invalid input of area range: lower bound is greater'
                     'or equal to upperbound')

  area = bounding_box_area(boxes)
  return (area_range[0] <= area) & (area <= area_range[1])


def filter_boxes_by_area_range(boxes: np.ndarray,
                               area_range: Tuple[float, float]) -> np.ndarray:
  """Select boxes whose areas fall in a given range.

  Args:
   boxes: a numpy array representing the bounding boxes in the following format
     [LEFT, TOP, RIGHT, BOTTOM, CLASS, CONFIDENCE(Optional)]
   area_range: [lowerbound(inclusive), upperbound(exclusive)] of the box area

  Returns:
   filtered_boxes: the filtered bounding Boxes
  """
  return boxes[check_boxes_in_area_range(boxes, area_range), :]


def sort_boxes_by_confidence(boxes: np.ndarray) -> np.ndarray:
  """Sort boxes according the confidence in descending order.

    It is using merge sort to agree with COCO metrics.

  Args:
   boxes: a numpy array representing the bounding boxes in the following format
     [LEFT, TOP, RIGHT, BOTTOM, CLASS, CONFIDENCE]

  Returns:
   sorted_boxes: the sorted list of bounding boxes
  """
  if boxes.ndim != 2:
    raise ValueError(f'Input boxes list should be a 2d array of shape '
                     f'( <batch_size>, {CONFIDENCE}+)')

  if boxes.shape[1] <= CONFIDENCE:
    warnings.warn('The axis for sort does not exist, return the original data')
    return boxes
  inds = np.argsort(-boxes[:, CONFIDENCE], kind='mergesort')
  return boxes[inds]
