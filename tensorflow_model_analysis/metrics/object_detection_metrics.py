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
"""COCO object detection metrics."""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from tensorflow_model_analysis.metrics import confusion_matrix_metrics
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util

AVERAGE_RECALL_NAME = 'average_recall'
AVERAGE_PRECISION_NAME = 'average_precision'
MEAN_AVERAGE_PRECISION_NAME = 'mean_average_precision'
MEAN_AVERAGE_RECALL_NAME = 'mean_average_recall'


def _validate_object_detection_arguments(
    object_class_id: Optional[Union[int, List[int]]],
    object_class_weight: Optional[Union[float, List[float]]],
    area_range: Optional[Tuple[float, float]] = None,
    max_num_detections: Optional[int] = None) -> None:
  """Validate the arguments for object detection related functions."""
  if object_class_id is None:
    raise ValueError('object_class_id must be provided if use object'
                     ' detection.')
  if isinstance(object_class_id, int):
    object_class_id = [object_class_id]
  if object_class_weight is not None:
    if isinstance(object_class_weight, float):
      object_class_weight = [object_class_weight]
    for weight in object_class_weight:
      if weight < 0:
        raise ValueError(f'object_class_weight = {object_class_weight} must '
                         'not be negative.')
    if len(object_class_id) != len(object_class_weight):
      raise ValueError('Mismatch of length between object_class_id = '
                       f'{object_class_id} and object_class_weight = '
                       f'{object_class_weight}.')
  if area_range is not None:
    if len(area_range) != 2 or area_range[0] > area_range[1]:
      raise ValueError(f'area_range = {area_range} must be a valid interval.')
  if max_num_detections is not None and max_num_detections <= 0:
    raise ValueError(f'max_num_detections = {max_num_detections} must be '
                     'positive.')


class COCOAveragePrecision(metric_types.Metric):
  """Confusion matrix at thresholds.

  It computes the average precision of object detections for a single class and
  a single iou_threshold.
  """

  def __init__(self,
               num_thresholds: Optional[int] = None,
               iou_threshold: Optional[float] = None,
               object_class_id: Optional[int] = None,
               object_class_weight: Optional[float] = None,
               area_range: Optional[Tuple[float, float]] = None,
               max_num_detections: Optional[int] = None,
               recalls: Optional[List[float]] = None,
               num_recalls: Optional[int] = None,
               name: Optional[str] = None):
    """Initialize average precision metric.

    This metric is only used in object-detection setting. use_object_detection
    is by default set to be True. It does not support sub_key parameters due to
    the matching algorithm of bounding boxes.

    Args:
      num_thresholds: (Optional) Number of thresholds to use for calculating the
        matrices and finding the precision at given recall.
      iou_threshold: (Optional) Used for object detection, threholds for a
        detection and ground truth pair with specific iou to be considered as a
        match.
      object_class_id: (Optional) Used for object detection, the class id for
        calculating metrics. It must be provided if use_object_detection is
        True.
      object_class_weight: (Optional) Used for object detection, the weight
        associated with the object class id.
      area_range: (Optional) Used for object detection, the area-range for
        objects to be considered for metrics.
      max_num_detections: (Optional) Used for object detection, the maximum
        number of detections for a single image.
      recalls: (Optional) recalls at which precisions will be calculated.
      num_recalls: (Optional) Used for objecth detection, the number of recalls
        for calculating average precision, it equally generates points bewteen 0
        and 1. (Only one of recalls and num_recalls should be used).
      name: (Optional) string name of the metric instance.
    """
    if recalls is not None:
      recall_thresholds = recalls
    elif num_recalls is not None:
      recall_thresholds = np.linspace(0.0, 1.0, num_recalls)
    else:
      # by default set recall_thresholds to [0.0:0.01:1.0].
      recall_thresholds = np.linspace(0.0, 1.0, 101)

    super().__init__(
        metric_util.merge_per_key_computations(self._metric_computations),
        num_thresholds=num_thresholds,
        use_object_detection=True,
        iou_threshold=iou_threshold,
        object_class_id=object_class_id,
        object_class_weight=object_class_weight,
        area_range=area_range,
        max_num_detections=max_num_detections,
        recall_thresholds=recall_thresholds,
        name=name)

  def _default_name(self) -> str:
    return AVERAGE_PRECISION_NAME

  def _metric_computations(
      self,
      num_thresholds: Optional[int] = None,
      use_object_detection: Optional[bool] = True,
      iou_threshold: Optional[float] = None,
      object_class_id: Optional[int] = None,
      object_class_weight: Optional[float] = None,
      max_num_detections: Optional[int] = None,
      area_range: Optional[Tuple[float, float]] = None,
      recall_thresholds: Optional[List[float]] = None,
      name: Optional[str] = None,
      model_name: str = '',
      output_name: str = '',
      example_weighted: bool = False) -> metric_types.MetricComputations:
    """Returns computations for confusion matrix metric."""
    _validate_object_detection_arguments(
        object_class_id=object_class_id,
        object_class_weight=object_class_weight,
        area_range=area_range,
        max_num_detections=max_num_detections)

    key = metric_types.MetricKey(
        name=name,
        model_name=model_name,
        output_name=output_name,
        sub_key=None,
        example_weighted=example_weighted,
        aggregation_type=None)

    if recall_thresholds is None:
      # If recall thresholds is not defined, initialize it as [0.0]
      recall_thresholds = [0.0]

    if num_thresholds is None:
      num_thresholds = 10000
    thresholds = [1.e-12] + [
        (i + 1) * 1.0 / (num_thresholds - 1) for i in range(num_thresholds - 2)
    ] + [1.0 - 1.e-12]
    # PrecisionAtRecall is a public function. To hide it from users who do not
    # need it, we make the name private with '_'.
    precision_at_recall_name = metric_util.generate_private_name_from_arguments(
        confusion_matrix_metrics.PRECISION_AT_RECALL_NAME,
        recall=recall_thresholds,
        num_thresholds=num_thresholds,
        iou_threshold=iou_threshold,
        object_class_id=object_class_id,
        object_class_weight=object_class_weight,
        area_range=area_range,
        max_num_detections=max_num_detections)

    pr = confusion_matrix_metrics.PrecisionAtRecall(
        recall=recall_thresholds,
        thresholds=thresholds,
        use_object_detection=use_object_detection,
        iou_threshold=iou_threshold,
        object_class_id=object_class_id,
        object_class_weight=object_class_weight,
        area_range=area_range,
        max_num_detections=max_num_detections,
        name=precision_at_recall_name)
    computations = pr.computations()
    precisions_key = computations[-1].keys[-1]

    def result(
        metrics: Dict[metric_types.MetricKey, Any]
    ) -> Dict[metric_types.MetricKey, Union[float, np.ndarray]]:
      value = np.nanmean(metrics[precisions_key])
      return {key: value}

    derived_computation = metric_types.DerivedMetricComputation(
        keys=[key], result=result)
    computations.append(derived_computation)
    return computations


metric_types.register_metric(COCOAveragePrecision)


class COCOMeanAveragePrecision(metric_types.Metric):
  """Mean average precision for object detections.

  It calculates the mean average precision metric for object detections. It
  averages COCOAveragePrecision over multiple classes and IoU thresholds.
  """

  def __init__(self,
               num_thresholds: Optional[int] = None,
               iou_thresholds: Optional[List[float]] = None,
               object_class_ids: Optional[List[int]] = None,
               object_class_weights: Optional[List[float]] = None,
               area_range: Optional[Tuple[float, float]] = None,
               max_num_detections: Optional[int] = None,
               recalls: Optional[List[float]] = None,
               num_recalls: Optional[int] = None,
               name: Optional[str] = None):
    """Initializes mean average precision metric.

    This metric is only used in object-detection setting. use_object_detection
    is by default set to be Ture. It does not support sub_key parameters due to
    the matching algorithm of bounding boxes.

    Args:
      num_thresholds: (Optional) Number of thresholds to use for calculating the
        matrices and finding the precision at given recall.
      iou_thresholds: (Optional) Used for object detection, threholds for a
        detection and ground truth pair with specific iou to be considered as a
        match.
      object_class_ids: (Optional) Used for object detection, the class ids for
        calculating metrics. It must be provided if use_object_detection is
        True.
      object_class_weights: (Optional) Used for object detection, the weight
        associated with the object class ids. If it is provided, it should have
        the same length as object_class_ids.
      area_range: (Optional) Used for object detection, the area-range for
        objects to be considered for metrics.
      max_num_detections: (Optional) Used for object detection, the maximum
        number of detections for a single image.
      recalls: (Optional) recalls at which precisions will be calculated.
      num_recalls: (Optional) Used for objecth detection, the number of recalls
        for calculating average precision, it equally generates points bewteen 0
        and 1. (Only one of recalls and num_recalls should be used).
      name: (Optional) Metric name.
    """

    super().__init__(
        metric_util.merge_per_key_computations(self._metric_computations),
        num_thresholds=num_thresholds,
        use_object_detection=True,
        iou_thresholds=iou_thresholds,
        object_class_ids=object_class_ids,
        object_class_weights=object_class_weights,
        area_range=area_range,
        max_num_detections=max_num_detections,
        recalls=recalls,
        num_recalls=num_recalls,
        name=name)

  def _default_name(self) -> str:
    return MEAN_AVERAGE_PRECISION_NAME

  def _metric_computations(self,
                           num_thresholds: Optional[int] = None,
                           use_object_detection: Optional[bool] = True,
                           iou_thresholds: Optional[List[float]] = None,
                           object_class_ids: Optional[List[int]] = None,
                           object_class_weights: Optional[List[float]] = None,
                           max_num_detections: Optional[int] = None,
                           area_range: Optional[Tuple[float, float]] = None,
                           recalls: Optional[List[float]] = None,
                           num_recalls: Optional[int] = None,
                           name: Optional[str] = None,
                           model_name: str = '',
                           output_name: str = '',
                           example_weighted: bool = False,
                           **kwargs) -> metric_types.MetricComputations:
    """Returns computations for confusion matrix metric."""

    _validate_object_detection_arguments(
        object_class_id=object_class_ids,
        object_class_weight=object_class_weights,
        area_range=area_range,
        max_num_detections=max_num_detections)

    # set default value according to COCO metrics
    if iou_thresholds is None:
      iou_thresholds = np.linspace(0.5, 0.95, 10)
    if object_class_weights is None:
      object_class_weights = [1.0] * len(object_class_ids)

    key = metric_types.MetricKey(
        name=name,
        model_name=model_name,
        output_name=output_name,
        sub_key=None,
        example_weighted=example_weighted,
        aggregation_type=None)

    computations = []
    precisions_keys = []
    for iou_threshold in iou_thresholds:
      for object_class_id, object_class_weight in zip(object_class_ids,
                                                      object_class_weights):

        average_precision_name = (
            metric_util.generate_private_name_from_arguments(
                AVERAGE_PRECISION_NAME,
                recall=recalls,
                num_recalls=num_recalls,
                num_thresholds=num_thresholds,
                iou_threshold=iou_threshold,
                object_class_id=object_class_id,
                object_class_weight=object_class_weight,
                area_range=area_range,
                max_num_detections=max_num_detections))

        ap = COCOAveragePrecision(
            num_thresholds=num_thresholds,
            iou_threshold=iou_threshold,
            object_class_id=object_class_id,
            object_class_weight=object_class_weight,
            area_range=area_range,
            max_num_detections=max_num_detections,
            recalls=recalls,
            num_recalls=num_recalls,
            name=average_precision_name)
        computations.extend(ap.computations())
        precisions_keys.append(computations[-1].keys[-1])

    def result(
        metrics: Dict[metric_types.MetricKey, Any]
    ) -> Dict[metric_types.MetricKey, Union[float, np.ndarray]]:
      precisions = [
          metrics[precisions_key] for precisions_key in precisions_keys
      ]
      value = np.nanmean(precisions)
      return {key: value}

    derived_computation = metric_types.DerivedMetricComputation(
        keys=[key], result=result)
    computations.append(derived_computation)
    return computations


metric_types.register_metric(COCOMeanAveragePrecision)


class COCOAverageRecall(metric_types.Metric):
  """Average recall metric for object detection.

  It computes the average precision metric for object detections for a single
  class. It averages MaxRecall metric over mulitple IoU thresholds.
  """

  def __init__(self,
               iou_thresholds: Optional[List[float]] = None,
               object_class_id: Optional[int] = None,
               object_class_weight: Optional[float] = None,
               area_range: Optional[Tuple[float, float]] = None,
               max_num_detections: Optional[int] = None,
               name: Optional[str] = None):
    """Initializes average recall metric.

    This metric is only used in object-detection setting. use_object_detection
    is by default set to be Ture. It does not support sub_key parameters due to
    the matching algorithm of bounding boxes.

    Args:
      iou_thresholds: (Optional) Used for object detection, threholds for a
        detection and ground truth pair with specific iou to be considered as a
        match.
      object_class_id: (Optional) Used for object detection, the class ids for
        calculating metrics. It must be provided if use_object_detection is
        True.
      object_class_weight: (Optional) Used for object detection, the weight
        associated with the object class ids. If it is provided, it should have
        the same length as object_class_ids.
      area_range: (Optional) Used for object detection, the area-range for
        objects to be considered for metrics.
      max_num_detections: (Optional) Used for object detection, the maximum
        number of detections for a single image.
      name: (Optional) Metric name.
    """

    super().__init__(
        metric_util.merge_per_key_computations(self._metric_computations),
        use_object_detection=True,
        iou_thresholds=iou_thresholds,
        object_class_id=object_class_id,
        object_class_weight=object_class_weight,
        area_range=area_range,
        max_num_detections=max_num_detections,
        name=name)

  def _default_name(self) -> str:
    return AVERAGE_RECALL_NAME

  def _metric_computations(
      self,
      use_object_detection: Optional[bool] = True,
      iou_thresholds: Optional[Union[float, List[float]]] = None,
      object_class_id: Optional[int] = None,
      object_class_weight: Optional[float] = None,
      max_num_detections: Optional[int] = None,
      area_range: Optional[Tuple[float, float]] = None,
      name: Optional[str] = None,
      model_name: str = '',
      output_name: str = '',
      example_weighted: bool = False) -> metric_types.MetricComputations:
    """Returns computations for confusion matrix metric."""

    _validate_object_detection_arguments(
        object_class_id=object_class_id,
        object_class_weight=object_class_weight,
        area_range=area_range,
        max_num_detections=max_num_detections)

    # set default value according to COCO metrics
    if iou_thresholds is None:
      iou_thresholds = np.linspace(0.5, 0.95, 10)
    if object_class_weight is None:
      object_class_weight = 1.0

    key = metric_types.MetricKey(
        name=name,
        model_name=model_name,
        output_name=output_name,
        sub_key=None,
        example_weighted=example_weighted,
        aggregation_type=None)

    computations = []
    recalls_keys = []
    for iou_threshold in iou_thresholds:
      max_recall_name = metric_util.generate_private_name_from_arguments(
          confusion_matrix_metrics.MAX_RECALL_NAME,
          iou_threshold=iou_threshold,
          object_class_id=object_class_id,
          object_class_weight=object_class_weight,
          area_range=area_range,
          max_num_detections=max_num_detections)

      mr = confusion_matrix_metrics.MaxRecall(
          use_object_detection=use_object_detection,
          iou_threshold=iou_threshold,
          object_class_id=object_class_id,
          object_class_weight=object_class_weight,
          area_range=area_range,
          max_num_detections=max_num_detections,
          name=max_recall_name)
      computations.extend(mr.computations())
      recalls_keys.append(computations[-1].keys[-1])

    def result(
        metrics: Dict[metric_types.MetricKey, Any]
    ) -> Dict[metric_types.MetricKey, Union[float, np.ndarray]]:
      for recalls_key in recalls_keys:
        if math.isnan(metrics[recalls_key]):
          logging.warning(
              'Recall with metric key %s is NaN, it will be'
              ' ignored in the following calculation.', recalls_key)
      recalls = [metrics[recalls_key] for recalls_key in recalls_keys]
      value = np.nanmean(recalls)
      return {key: value}

    derived_computation = metric_types.DerivedMetricComputation(
        keys=[key], result=result)
    computations.append(derived_computation)
    return computations


metric_types.register_metric(COCOAverageRecall)


class COCOMeanAverageRecall(metric_types.Metric):
  """Mean Average recall metric for object detection.

  It computes the mean average precision metric for object detections for a
  single class. It averages COCOAverageRecall metric over mulitple classes.
  """

  def __init__(self,
               iou_thresholds: Optional[List[float]] = None,
               object_class_ids: Optional[List[int]] = None,
               object_class_weights: Optional[List[float]] = None,
               area_range: Optional[Tuple[float, float]] = None,
               max_num_detections: Optional[int] = None,
               name: Optional[str] = None):
    """Initializes average recall metric.

    This metric is only used in object-detection setting. use_object_detection
    is by default set to be Ture. It does not support sub_key parameters due to
    the matching algorithm of bounding boxes.

    Args:
      iou_thresholds: (Optional) Used for object detection, threholds for a
        detection and ground truth pair with specific iou to be considered as a
        match.
      object_class_ids: (Optional) Used for object detection, the class ids for
        calculating metrics. It must be provided if use_object_detection is
        True.
      object_class_weights: (Optional) Used for object detection, the weight
        associated with the object class ids. If it is provided, it should have
        the same length as object_class_ids.
      area_range: (Optional) Used for object detection, the area-range for
        objects to be considered for metrics.
      max_num_detections: (Optional) Used for object detection, the maximum
        number of detections for a single image.
      name: (Optional) Metric name.
    """

    super().__init__(
        metric_util.merge_per_key_computations(self._metric_computations),
        use_object_detection=True,
        iou_thresholds=iou_thresholds,
        object_class_ids=object_class_ids,
        object_class_weights=object_class_weights,
        area_range=area_range,
        max_num_detections=max_num_detections,
        name=name)

  def _default_name(self) -> str:
    return MEAN_AVERAGE_RECALL_NAME

  def _metric_computations(
      self,
      use_object_detection: Optional[bool] = True,
      iou_thresholds: Optional[List[float]] = None,
      object_class_ids: Optional[Union[int, List[int]]] = None,
      object_class_weights: Optional[Union[float, List[float]]] = None,
      max_num_detections: Optional[int] = None,
      area_range: Optional[Tuple[float, float]] = None,
      name: Optional[str] = None,
      model_name: str = '',
      output_name: str = '',
      example_weighted: bool = False) -> metric_types.MetricComputations:
    """Returns computations for confusion matrix metric."""

    _validate_object_detection_arguments(
        object_class_id=object_class_ids,
        object_class_weight=object_class_weights,
        area_range=area_range,
        max_num_detections=max_num_detections)

    if object_class_weights is None:
      object_class_weights = [1.0] * len(object_class_ids)

    key = metric_types.MetricKey(
        name=name,
        model_name=model_name,
        output_name=output_name,
        sub_key=None,
        example_weighted=example_weighted,
        aggregation_type=None)

    computations = []
    recalls_keys = []
    for object_class_id, object_class_weight in zip(object_class_ids,
                                                    object_class_weights):
      max_recall_name = metric_util.generate_private_name_from_arguments(
          AVERAGE_RECALL_NAME,
          iou_thresholds=iou_thresholds,
          object_class_id=object_class_id,
          object_class_weight=object_class_weight,
          area_range=area_range,
          max_num_detections=max_num_detections)

      mr = COCOAverageRecall(
          iou_thresholds=iou_thresholds,
          object_class_id=object_class_id,
          object_class_weight=object_class_weight,
          area_range=area_range,
          max_num_detections=max_num_detections,
          name=max_recall_name)
      computations.extend(mr.computations())
      recalls_keys.append(computations[-1].keys[-1])

    def result(
        metrics: Dict[metric_types.MetricKey, Any]
    ) -> Dict[metric_types.MetricKey, Union[float, np.ndarray]]:
      recalls = [metrics[recalls_key] for recalls_key in recalls_keys]
      value = np.nanmean(recalls)
      return {key: value}

    derived_computation = metric_types.DerivedMetricComputation(
        keys=[key], result=result)
    computations.append(derived_computation)
    return computations


metric_types.register_metric(COCOMeanAverageRecall)
